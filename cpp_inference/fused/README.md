# Fused CUDA runtime (torch-free)

`gaussifier_fused_infer` runs the v0.12.0 Gaussifier inference pipeline with hand-written
CUDA kernels and nothing but the CUDA runtime: no LibTorch, TensorRT or cuDNN. It implements
the GroupNorm-free ("frozen-GN") release of the model, in which every GroupNorm is a
calibrated per-channel affine folded into its conv. It is the default profile in the [C++ build](../README.md).

## Architecture

One `FusedPipeline::run(K)` (`pipeline.cu`) enqueues every stage on one stream, out of one
device arena, with no host synchronisation, allocation or copies. `run_graph(K)` captures the
whole run as one CUDA graph per (H, W, K) and replays it for every image of that size. In
video mode (`PipelineOptions::video`) each run advances one frame, with one graph for the
first frame and one for every warm frame after it.

1. **Forward map** (`forward_map.cu`, `conv_fused.cuh`, `conv_block.cuh`, `epilogues.cuh`).
   The residual U-Net backbone (base 32, depth 4) runs on one implicit-GEMM conv kernel
   (`launch_conv2`: `mma.sync` FP16 with FP32 accumulation, or e4m3 FP8 per layer from a
   precision plan). Max-pooling and bilinear-upsample + concat (resized to the skip's
   extent at odd decoder levels) happen inside the input staging, each block's 1x1
   projection is an extra K segment of its second conv (in the up blocks only the skip half:
   a 1x1 conv commutes with the bilinear upsample, so the upsampled half runs as a 1x1 conv at
   low resolution whose up2x conv2 adds before its epilogue, `ConvParams::up_add`), the pooled
   skip copies are written by the producer's epilogue, and the density / log-cov /
   RGB heads run in the epilogue of the last conv. When a block's conv2 is FP8, its conv1
   stores the intermediate as e4m3 with conv2's activation scale (`F8StoreEpilogue`,
   `InMode::kPlainF8`): half the bytes, bit-identical to quantising an FP16 copy. From 1024 x 1024 pixels the stem and down0
   run as single-kernel residual blocks and the last block runs in 128-row bands for L2 reuse.
2. **Point count and sampler** (`pipeline_sampler.cu`). n = floor(sum rate), capped at 0.40
   points/pixel and 1e6. The equal-mass Voronoi (EMV) sampler (error-diffusion oversample,
   mass-preserving merge rounds with an exact-count contract, Lloyd polish, cell-mass
   weights) keeps every count-dependent quantity on the device, so one launch sequence
   serves any count up to the planned capacity. Voronoi cells are exact nearest-seed
   assignments from a grid hash. `sampler_assign_check.cu` checks them against brute force.
3. **K-loop renderer** (`kloop_render.cu`). It is a raw-pointer port of gaussifier's native
   simple-sum renderer (it includes its device helpers from `third_party/raster`). The
   projection pass also writes the point-count map. The raster writes the clamped render, its
   squared error against the input, and the per-state half of the packed head input.
4. **Correction head** (`correction_head.cu`, `head_conv.cuh`, `head_fullres.cuh`). The head
   is a main U-Net (hidden 32) plus an aux U-Net (hidden 24), with every GroupNorm folded.
   The first conv of both nets runs as one persistent launch from the packed FP16 input
   (planar halo, the aux net at native width 24, partial edge tiles). Up to 512 x 1024
   pixels, the aux net runs on a forked stream and the full-resolution enc0.b and dec1.b + 1x1
   output layers use lean persistent kernels. Larger images fuse the first conv with enc0.b
   and use the generic conv core. The aux net runs at native width 24 unless a precision plan
   makes its enc1.a FP8. In that case it runs the copy padded to 32. The K-1 head calls
   update the points' log-cov, colour and position (xy step 2.56 / max(H, W)).
5. **CUDA graph**. Fixed K is a straight-line graph. Adaptive K (`--psnr-target` /
   `--min-gain`, off by default) puts the head-update + render step in a conditional WHILE
   node that the GPU decides after each state (CUDA 12.4+).

## Build

The runtime needs CUDA 12.8 or 13.x (`nvcc`) and CMake 3.18 or newer. The renderer device
helpers ship in `third_party/raster` (`-DGAUSSIFIER_SRC_DIR` can point at a training checkout
instead). It is tuned for sm_89 (L40S) and sm_120 (RTX 5090); the correction head
keeps a tile table per architecture, and every other device uses the sm_89 one.

```bash
# standalone (no LibTorch / TensorRT)
cmake -S cpp_inference/fused -B build-fused -G Ninja \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-fused --target gaussifier_fused_infer gaussifier_fused
# RTX 5090: -DCMAKE_CUDA_ARCHITECTURES=120 (also builds the FP16-accumulate kernels)
# Also built by default from cpp_inference/.
```

`-DGFUSED_BUILD_CHECKS=ON` also builds `sampler_assign_check`, which compares the sampler's
Voronoi assignment, cell sums and merge matching with brute force.

### FP16 accumulation (`-DGFUSED_ACC16`, default on sm_120)

`ConvParams::acc16` switches a conv-core launch to `mma ... f16.f16.f16.f16`: mode 1 accumulates
each K chunk (9 x BK products) in FP16 and adds the chunk partials in FP32, mode 2 accumulates
the whole K (9 x Cin, projection segment included) in FP16. It exists for GPUs whose
FP16-accumulate MMA rate is higher than the FP32-accumulate one (GeForce); on the L40S both run
at the same rate. The RTX 5090 runs `m16n8k16` at 510 dense TFLOP/s with FP16 accumulation and
256 with FP32 (e4m3 with FP32 accumulation: 510). The CMake option compiles the variants
(about twice the conv kernels) and is on by default when `CMAKE_CUDA_ARCHITECTURES` includes
120. On an sm_120 device the loaders then default every FP16 conv-core layer to mode 1;
everywhere else they keep FP32 accumulation, and the sm_89 code is unchanged. The layers can
be picked at load time with `GFUSED_FM_ACC16` (one digit for all 14 forward-map
convs, or 14 digits: stem.conv1, stem.conv2, ..., up2.conv2) and `GFUSED_HEAD_ACC16` (one
digit, or 18: main layers 1..9, then aux 1..9); either one replaces the default, and `0` turns
it off. FP8 layers, the head's first conv and the fused
single-kernel blocks always accumulate in FP32; head layers set to FP16 accumulation leave the
lean full-resolution kernels for the conv core. L40S Kodak (24, K=7, FP16 blobs): 40.818 dB
off, 40.818 with every layer in mode 1, 40.820 in mode 2 (run-to-run noise is about 0.001 dB);
the largest pre-activation magnitude is about 40, far from the FP16 range.

RTX 5090, v0.12.0 sources, Kodak (24, K=7, reference 40.820 dB): FP16 blobs 40.8188 off, 40.8195 in mode 1,
40.8188 in mode 2; FP8 blobs 40.8004, 40.8007 and 40.7995. Counts with reference maps match
on 24 of 24 in every case. Per conv-core layer at 512 x 768, mode 1 is 1.3x to 1.6x faster
than FP32 accumulation, mode 2 another 2% at 3x to 5x the error; at 1024 x 1536 the
32-channel full-resolution layers are bandwidth-bound and gain nothing. End to end the head
gains most, even with its full-resolution layers moved off the lean kernels: K=7 FP16 at
512 x 512 runs 4.31 ms off, 3.56 ms with only the head in mode 1, 3.39 ms with every layer in
mode 1 and 3.34 ms in mode 2. Mode 1 is the default because its FP32 chunk sums bound the
FP16 rounding to one chunk.

## Weight export

The runtime reads the GFFM forward map and GFCH correction head. Both precision
variants ship in `src/gaussifier_sampler/weights/frozen_gn/` alongside the
frozen-GN source weights and FP8 precision plan.

```bash
bash scripts/build_release_artifacts.sh src/gaussifier_sampler/weights/frozen_gn /tmp/fused-export
```

The script calls `export_fused_weights.py` and `export_fused_head.py` and prints
hashes for comparison with [the runtime manifest](../../release/runtime_bundle.json).

FP8 layers are quantised exactly as `fp8_sensitivity.py` does it: per-output-channel e4m3
weights and the plan's static activation scale. Some plan layers have channel counts the FP8
kernel cannot take (input segments must be multiples of 32), so they run FP16. The exporters
print these fallbacks.

## Run

```bash
gaussifier_fused_infer --fm fm_fp16.gffm --head head_fp16.gfch --K 7 input.npy render.npy \
    [--gaussians out.npy] [--seed 12345] [--n-points N] [--repeat R] [--sustained R] \
    [--size-mode pad|exact]
gaussifier_fused_infer --fm ... --head ... --manifest pairs.txt [--inflight N]
gaussifier_fused_infer --fm ... --head ... --video FRAMES --video-out DIR \
    [--count-ema 0.9] [--warm-attribute-blend 0.1]
```

The input is a float32 NPY `[1, 3, H, W]` or `[3, H, W]` in [0, 1], with H and W at least
16. The output is the final render `[3, H, W]`. `--gaussians` also writes the points, log-cov,
latent RGB, premultiplied colour and weights. Each image prints one JSON line with the count,
per-state PSNR and timings, plus sustained back-to-back replay statistics with `--sustained`.

A manifest line is `INPUT RENDER [GAUSSIANS [MAPS_STEM]]` (render `DUMMY` or gaussians `-`
skips writing). The model loads once; each resolution gets a prepared workspace (arena and
graphs) in a small LRU cache that evicts idle sizes when an allocation fails. `--inflight N`
keeps up to N images in flight on separate streams; `--repeat` and `--sustained` need
`--inflight 1`.

`--video` takes a directory of `.npy` frames (lexical order) or a manifest of frame paths
relative to it; every frame must have the first frame's size, and the frames run in order in
one workspace (`--inflight 1`, no `--repeat` / `--sustained` / external maps). The first frame
uses EMV placement; later frames keep the previous Gaussians as tracks (count smoothed by
`--count-ema`, split / kill / birth repair against the new density) and blend
`--warm-attribute-blend` of each track's previous log-cov and colour into the initial
attributes. `DIR/NNNNNN.npy` gets the Gaussian outputs plus row-aligned `.ids.npy` (int64,
stable per track) and `.slots.npy` (int32); the JSON line adds the frame index and track
statistics. Video works with both size modes.

The C ABI in [`python_api.h`](python_api.h) opens an independent tracking
workspace with `gfused_video_create` and shares the source session's weights.
`gfused_video_push` takes a fixed-size CHW float32 frame and returns its point
count and statistics. Read the image and Gaussian arrays with `gfused_read`
and the row-aligned IDs and slots with `gfused_video_read_tracks` before pushing
another frame. Release the workspace with `gfused_video_destroy`.
Python exposes this through [`sampler.video(...)`](../../README.md#fused-video-tracking).

Sizes that are not multiples of 16 run in one of two modes. Both run the correction head,
sampling, rendering and PSNR at the real H and W.

- `exact` (default) runs the forward map and the head at the real extent, like the PyTorch
  reference: zero padding at the real border, floor pooling, and the reference's up2x +
  resize at decoder levels whose extent is odd. The aligned kernels take partial edge tiles,
  so a multiple of 8 (forward map) or 4 (head) needs no resize at all: 1920 x 1080 runs as
  fast as 1920 x 1088 (K=7 FP16, 24 Kodak crops: 46.9 ms, -0.006 dB vs the reference). At
  other sizes the conv of each odd level stages its input with the up2x resized to the
  skip's extent (`InMode::kUpCatRs`), so no resized tensor is written. Its interpolation is
  bit-identical to a materialised up2x + resize + concat. K=7, 24 Kodak crops, sustained
  median:

  | Size | FP16 | FP8 | FP16 dB vs ref |
  |---|---|---|---|
  | 1919 x 1079 | 51.7 ms | 47.6 ms | -0.004 |
  | 1920 x 1080 | 47.1 ms | 44.1 ms | -0.006 |
  | 1279 x 719 | 20.2 ms | 20.0 ms | -0.003 |
  | 1280 x 720 | 19.9 ms | 18.7 ms | -0.004 |
- `pad` runs the forward map on the image reflected at the bottom and right to the next
  multiple of 16 and crops its maps; density, rate and count come from the real crop. It
  saves the forward map's resizes at odd sizes, but the reflected border shifts the point
  count (-0.06% at 1920 x 1080, up to a few percent on small images) and costs 0.016 dB
  (FP16) at 1920 x 1080, at the same speed as exact there.
- Multiples of 16 use the same aligned path in both modes.

The correction head must not see a padded border: it was trained with zero padding at the
image edge, and with a reflected (or constant) border it mispredicts the last ~16 rows
(10x their MSE, -0.5 dB at 1920 x 1080 and 47 dB).

In C++, set `PipelineOptions::size_mode` to `SizeMode::Pad` or `SizeMode::Exact`. In the C ABI,
`gfused_run` defaults to exact. `gfused_run_with_size_mode` takes the same arguments followed by
an `int size_mode` (0 for pad, 1 for exact). Each call selects the mode for that run. In Python,
`GaussifierSampler.bundled(size_mode="exact" | "pad")` (CLI `--size-mode`).

The validation flags are:

- `--pin-count N`: a fixed count without the covariance rescale.
- `--external-maps` / `--maps STEM`: feed reference forward-map outputs.
- `--check-head-io`: verify the packed head input on the GPU.
- `--no-graph`: eager launches.
- `--psnr-target DB` / `--min-gain DB`: adaptive K, where `--K` is the cap.

## Tests

The tests live in `tests/fused/`. Each one is a pytest test that skips without CUDA or its
inputs, and also a benchmark script (`--help`). `tests/fused/fused_env.py` lists the
environment variables they read.

```bash
W=src/gaussifier_sampler/weights/frozen_gn
export GFUSED_KODAK=/path/to/kodak GFUSED_STAGE1=$W/stage1.pt \
       GFUSED_HEAD_MODULE=/path/to/head_module.pt \
       GFUSED_PRECISION_PLAN=$W/precision_plan_release.json \
       GFUSED_BIN=build-fused/gaussifier_fused_infer \
       GFUSED_LIB=build-fused/libgaussifier_fused.so \
       GAUSSIFIER_NATIVE_SIMPLE_SUM_BUILD_DIR=/path/to/scratch  # PyTorch reference renderer
uv run pytest tests/fused -q
```

| Test | Checks | Script mode adds |
|---|---|---|
| `test_conv_fused.py` | every forward-map / head conv shape and tile config (FP16, FP8, native 24-wide), fused projections, fused blocks, row bands | per-layer timing vs cuDNN, `--sweep` |
| `test_forward_map_fused.py` | outputs vs the FP32 PyTorch model; banded up2 bit-identical; FP8 plan vs `fp8_sensitivity.py` fake-quant | latency, `--flags` |
| `test_correction_head_fused.py` | head vs the FP32 PyTorch head on captured K-loop inputs (both schedules, padded aux, packed input, edge crops); FP8 blobs vs fake-quant | latency, `--tune` per-layer config sweep |
| `test_pipeline_fused.py` | end to end vs the PyTorch reference (counts, per-state PSNR), odd sizes, FP8 vs FP16 at unaligned sizes in both modes, C ABI mode switches, manifest cache / `--inflight`, video mode, input errors | `--latency`, `--adaptive-sweep` |

## Validated results

These are the mean PSNR (dB) at K=7 with the release weights, seed 1234. Here "fused" is this
runtime with FP16 blobs, "ref" is the PyTorch frozen-GN release model run through gaussifier's
reference pipeline, and "v0.11.1" is the original GroupNorm model.

| Set | fused FP16 | ref | v0.11.1 | fused FP8 (budget_0.03dB) |
|---|---|---|---|---|
| Kodak (24) | 40.819 | 40.820 | 40.780 | 40.793 |
| OpenImages-v7 val 512 crops (64) | 42.616 | 42.621 | 42.611 | 42.592 |

Latency is the sustained K=7 median on an L40S (CUDA graph, back-to-back replays, adaptive
count):

| Size | FP16 | FP8 |
|---|---|---|
| 512 x 512 | 5.27 ms | 4.91 ms |
| 512 x 768 | 8.01 ms | 7.47 ms |
| 1024 x 1536 | 35.2 ms | 33.0 ms |
| 1280 x 720 | 20.0 ms | 18.6 ms |
| 1920 x 1080 (exact) | 46.9 ms | 43.6 ms |

The 1280 x 720 and 1920 x 1080 rows are the median over the 24 Kodak images (portraits
rotated, bicubic resize to cover, centre crop).

For comparison, the production LibTorch harness at 512 x 768 spends 2.32 ms in its TensorRT
forward map (GroupNorm model) and 3.34 ms per correction-head call.

### RTX 5090 (sm_120)

Measured on an RTX 5090 (driver 595.99.02, CUDA 13.0, 600 W limit; it runs at the limit, about
2.7 GHz). The sm_120 defaults are FP16 accumulation (mode 1, above) and the sm_120 head tile
tables. Kodak (24, K=7, reference 40.820 dB): FP16 40.821 dB, FP8 40.801 (upstream build
without the sm_120 defaults: 40.818 and 40.799); counts with reference maps match on 24 of 24.

Sustained K=7 medians over the 24 Kodak crops (portraits rotated, bicubic resize to cover,
centre crop), configurations interleaved over two rounds. "Before" is the same source without
the sm_120 changes, "FP32 acc." the new build with `GFUSED_FM_ACC16=0 GFUSED_HEAD_ACC16=0`.
Other jobs shared the GPU for part of the 1280 x 720 runs, so that row takes the faster round
of each configuration:

| Size | FP16 before | FP16 FP32 acc. | **FP16 sm_120** | FP8 before | FP8 sm_120 |
|---|---|---|---|---|---|
| 512 x 512 | 4.28 ms | 4.24 ms | **3.33 ms** (-22%) | 4.02 ms | 3.44 ms |
| 1280 x 720 | 14.84 ms | 15.04 ms | **12.15 ms** (-18%) | 13.82 ms | 12.16 ms |
| 1920 x 1080 | 32.77 ms | 32.96 ms | **27.58 ms** (-16%) | 30.27 ms | 27.19 ms |

FP16 accumulation runs at the e4m3 rate, and FP8 layers cannot use it, so on sm_120 FP8 is no
faster than FP16 below 1080p and still costs 0.02 dB; FP16 is the profile to use. The head is
71% of the K=7 time at 512 x 768 (nsys: forward map 0.64 ms, sampler 0.37, render 0.31, head
3.29, gaps 0.03), down from 4.48 of 6.04 ms before. On the v0.12.0 sources and the same 24
crops at 512 x 512 and K=7, the legacy TRT + AOTI harness (v0.11.1 GN model) takes 13.76 ms
and fused FP16 3.34 ms (4.1x).

Tried on sm_120 without a gain: removing the head first conv's register spills (one block per
SM), the 32-padded aux net (+5% at 512 x 768, +3.5% at 1080p, more FLOPs than the better tiles
save) and running the two output layers concurrently (both nets already share the SMs).
