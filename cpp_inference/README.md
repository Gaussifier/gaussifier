# gaussifier_sampler C++ inference harness

## Fused CUDA profile

v0.12.0 defaults to `gaussifier_fused_infer` in C++ and
`libgaussifier_fused.so.1` for the Python binding. Both run the forward map,
equal-mass Voronoi placement, renderer and shared correction head with CUDA
kernels. The default CMake build needs only a CUDA toolkit; the renderer device
helpers it includes ship in `fused/third_party/raster`.

```bash
cmake -S cpp_inference -B cpp_inference/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build cpp_inference/build --parallel 4
ctest --test-dir cpp_inference/build --output-on-failure

weights=src/gaussifier_sampler/weights/frozen_gn
cpp_inference/build/fused/gaussifier_fused_infer \
  --fm "$weights/fm_fp16.gffm" --head "$weights/head_fp16.gfch" \
  input.npy render.npy
```

Inputs are contiguous float32 NPY RGB images in `[0, 1]` with height and width
of at least 16 (`--size-mode exact|pad` for sizes that are not multiples of 16).
K defaults to 7 and point count defaults to adaptive. For FP8, pass both FP8
blobs. `--manifest` runs many images with one model load (`--inflight N`
overlaps them) and `--video` runs a frame sequence with GPU warm start. The
[fused runtime guide](fused/README.md) documents binary options, weight export
and validation.

## Legacy LibTorch, TensorRT and AOTI profile

The rest of this guide covers the deprecated v0.11.1 model and its
`gaussifier_infer` harness. The legacy profile links LibTorch, can dispatch
the forward map through raw TensorRT and the correction head through AOTI, and
builds when `GAUSSIFIER_BUILD_LEGACY` is enabled. Removal is scheduled for the
release after v0.12.0.

## Prerequisites

- CUDA toolkit (matching the LibTorch CUDA version)
- LibTorch (we pull it from a `uv` venv via `torch.utils.cmake_prefix_path`)
- CMake 3.18+
- The gaussifier source repo — only for native simple-sum renderer sources.
  The Voronoi sampler is compiled from the snapshot shipped in this package.
  Point CMake at the gaussifier checkout via
  `-DGAUSSIFIER_SRC_DIR=<path>` (default: sibling `../gaussifier` from the
  sampler repository root).

## Build

```bash
# From the gaussifier-sampler repo root. Assumes ../gaussifier exists.
cmake -S cpp_inference -B cpp_inference/build \
  -DGAUSSIFIER_BUILD_LEGACY=ON -DGAUSSIFIER_BUILD_FUSED=OFF \
  -DCMAKE_PREFIX_PATH=$(uv run python -c \
    "import torch; print(torch.utils.cmake_prefix_path)") \
  -DGAUSSIFIER_SRC_DIR=$(realpath ../gaussifier)
cmake --build cpp_inference/build -j
ctest --test-dir cpp_inference/build --output-on-failure
```

The first build takes 5-10 minutes (the vendored rasterizer is ~30 CUDA TUs).

## Run

```bash
# 1. Export TorchScript modules from a Gaussifier checkpoint (one-time per ckpt).
#    Done from the gaussifier repo because it has the tracing tool.
cd ../gaussifier
uv run python tools/export_torchscript.py \
  --ckpt artifacts/releases/v0.11.1/gaussifier_best.pt \
  --out artifacts/exports/v0.11.1_torchscript
# Produces forward_map.pt and correction_head.pt in <out>/
# Legacy checkpoints without an embedded config may add --config <config.yaml>.

# 2. Use Gaussifier's canonical image converter. It edge-pads to a multiple
#    of 32 and writes input.npy plus original-size metadata in input.npy.json.
uv run python tools/cpp_inference_io.py to_tensor \
  my_image.png ../gaussifier-sampler/input.npy

# 3. Run C++ inference.
cd ../gaussifier-sampler
LD_LIBRARY_PATH=$(uv run python -c \
  "import torch; print(torch.__file__.rsplit('/',1)[0]+'/lib')") \
  ./cpp_inference/build/gaussifier_infer \
  ../gaussifier/artifacts/exports/v0.11.1_torchscript/forward_map.pt \
  ../gaussifier/artifacts/exports/v0.11.1_torchscript/correction_head.pt \
  input.npy \
  output.npy \
  15000 \
  7
# The last two arguments are optional; this example fixes n_points at 15000
# and selects K=7 for quality.

# 4. Convert output back to PNG and crop away the input padding via its sidecar.
cd ../gaussifier
uv run python tools/cpp_inference_io.py to_image \
  ../gaussifier-sampler/output.npy rendered.png \
  --crop-original-from ../gaussifier-sampler/input.npy
```

The command above exercises the TorchScript fallback. To run the locked
production profile, point the same binary at a complete release bundle:

```bash
source tensorrt_sdk/setup_env.sh
BUNDLE=/path/to/release-bundle
LD_LIBRARY_PATH="$(uv run python -c \
  'import torch; print(torch.__file__.rsplit("/", 1)[0] + "/lib")'):${LD_LIBRARY_PATH:-}" \
GAUSSIFIER_FWDMAP_TRT_ENGINE="$BUNDLE/forward_map.engine" \
GAUSSIFIER_HEAD_AOTI="$BUNDLE/correction_head_aoti.pt2" \
GAUSSIFIER_FUSED_KLOOP=1 \
GAUSSIFIER_VORONOI_CAPTURABLE=1 \
  ./cpp_inference/build/gaussifier_infer \
    "$BUNDLE/forward_map.pt" "$BUNDLE/correction_head.pt" \
    input.npy output.npy 15000 7
```

Build that self-consistent bundle using the
[release script index](../scripts/README.md). The harness still
loads both TorchScript files before selecting optimized backends, so they must
be present even when TRT and AOTI do the actual forward calls.

NPY is the only input/final-render interchange format; a data path that does
not end in `.npy` is an error. A Gaussian output basename such as `sample.npy`
produces `sample.points.npy`, `sample.cov.npy`, and `sample.rgb.npy`. Model
artifacts are separate: `forward_map.pt` and `correction_head.pt` remain
required because the only complete executable still initializes both through
LibTorch.

Image conversion belongs to the sibling
`gaussifier/tools/cpp_inference_io.py` tool. Its maintained `to_tensor` path
writes C-contiguous float32 `[1,3,H,W]` NPY and an `<input>.npy.json` sidecar;
`to_image --crop-original-from <input>.npy` validates that sidecar before
cropping. The C++ harness consumes only the NPY tensor, not the sidecar.

### Positional argument contract

The first four paths are required. `n_points` is an optional integer followed
by optional `K`:

- A positive `n_points` fixes the point budget. When it is omitted or `0`, a
  rate-bearing forward-map artifact uses the integer value of `rate.sum()`.
  A legacy three-output map has no rate tensor and falls back to 15,000 points.
- With a rate-bearing map, a fixed `n_points` above the adaptive count
  `n_auto` scales the predicted covariance map by `n_auto / n_points`
  before the Gaussians and the correction head read it, as
  `GaussifierSampler.sample(count=...)` does. The shape map is predicted for
  `n_auto`, so without this `n_points` Gaussians render about
  `n_points / n_auto` times too bright. A lower `n_points` leaves the map
  alone: the correction loop recovers the darker start itself.
- `K` defaults to `5` and must be at least `1`. It counts rendered states, so
  `K=1` produces the initializer render without a correction-head update.

The release commands above pass `15000 7` explicitly. In batch-manifest mode,
the input and final-render paths remain required placeholders even though the
manifest supplies each input and Gaussian-output path.

### Runtime environment variables

Production profile:

| Variable | Effect |
|---|---|
| `GAUSSIFIER_FWDMAP_TRT_ENGINE=/path/forward_map.engine` | Use the raw NvInfer forward-map engine. This is the v0.11.1 production path. |
| `GAUSSIFIER_HEAD_AOTI=/path/correction_head_aoti.pt2` | Replace the JIT correction head with the AOTI package. |
| `GAUSSIFIER_FUSED_KLOOP=1` | Use the fused pre/post-render CUDA kernels. |
| `GAUSSIFIER_VORONOI_CAPTURABLE=1` | Use the exact-count, synchronization-reduced Voronoi path. |
| `GAUSSIFIER_VORONOI_OVERSAMPLE`, `GAUSSIFIER_VORONOI_MERGE_ROUNDS`, `GAUSSIFIER_VORONOI_LLOYD_ITERS` | Override sampler defaults `1.5`, `6`, and `3`; changing them invalidates release baselines. |

Operation and diagnostics:

| Variable | Effect |
|---|---|
| `GAUSSIFIER_REPEAT=N` | Repeat one input; the first run is warmup and later runs form the latency report. |
| `GAUSSIFIER_REPORT_MEMORY=1` | Log per-repetition LibTorch allocator peak allocated/current allocated/peak reserved bytes. Counter reset and reads are outside the latency timer; discard warm-up rows when reporting steady-state peaks. TensorRT-managed allocations and CUDA context are excluded. |
| `GAUSSIFIER_EXPORT_CUDA_IPC=1` | On the final single-image repetition, publish `IPC_READY` JSON on stdout with a monotonic start timestamp and one shared CUDA allocation containing final positions, effective scales/rotation/weighted colors, and the render. GPU packing is timed. The consumer must close its imported handle, then send `release` on stdin before the producer frees the allocation. Image-file output is skipped; batch mode is unsupported. Intended for a local downstream optimizer. |
| `GAUSSIFIER_BATCH_MANIFEST=/path/manifest.txt` | Reuse loaded models for same-resolution `<input.npy> <gaussians_out.npy>` pairs, one pair per line. Positional input/output arguments remain required. |
| `GAUSSIFIER_GAUSSIANS_OUT=/path/output.npy` | Save final point, covariance, and RGB arrays as `<stem>.points.npy`, `<stem>.cov.npy`, and `<stem>.rgb.npy` in single-image mode. |
| `GAUSSIFIER_PROFILE=1` | Print synchronized per-stage GPU timings. |
| `GAUSSIFIER_SHIM_TRACE=1` | Trace torch-free AOTI shim calls for ABI debugging. |

Fallback and experimental controls:

| Variable | Effect |
|---|---|
| `GAUSSIFIER_FP16=1`, `GAUSSIFIER_FWDMAP_FP16=1`, `GAUSSIFIER_HEAD_FP16=1` | Cast both or one TorchScript module to FP16. |
| `GAUSSIFIER_AUTOCAST=1` | Use autocast instead of static FP16 for fallback diagnosis. |
| `GAUSSIFIER_TRT=1`, `GAUSSIFIER_FWDMAP_TRT=1`, `GAUSSIFIER_HEAD_TRT=1` | Select hybrid TRT `.pt` modules; this is a legacy/debug path, not the raw-engine production path. |
| `GAUSSIFIER_TRT_RUNTIME=/path/libtorchtrt_runtime.so` | Load the custom-op runtime required by hybrid TRT `.pt` modules. |
| `GAUSSIFIER_FWDMAP_AOTI=/path/forward_map_aoti.pt2` | Use AOTI for the forward map; raw TRT is the validated production backend. |
| `GAUSSIFIER_HEAD_TRT_ENGINE=/path/head.engine` | Select the staged raw-TRT head backend if a compatible engine exists; no such release engine is currently shipped. |
| `GAUSSIFIER_USE_DIRECT_AOTI=1` plus `GAUSSIFIER_DIRECT_AOTI_SO=/path/model.so` | Exercise the direct torch-free AOTI head loader. |
| `GAUSSIFIER_VORONOI_GRAPH=1` | Capture the capturable Voronoi path; requires `GAUSSIFIER_VORONOI_CAPTURABLE=1` and is not the latency-default release profile. |
| `GAUSSIFIER_NO_CUDA_GRAPH=1` | Disable head-call CUDA graph capture for debugging. |
| `GAUSSIFIER_CHANNELS_LAST=1`, `GAUSSIFIER_NO_CHANNELS_LAST=1` | Force channels-last layout on or off. |
| `GAUSSIFIER_NO_JIT_OPT=1` | Skip `torch::jit::optimize_for_inference`. |

## Current limitations

1. **Direct PNG I/O**: the harness uses float32 NPY tensors. The canonical
   sibling `gaussifier/tools/cpp_inference_io.py` converter is still a Python
   pre/post-processing step; embedding an image codec would remove it.
2. **Tensor batching**: each inference call processes one image. Manifest mode
   amortizes model startup across same-resolution images but does not form a
   batched tensor.
3. **Legacy dependencies**. The legacy runner links LibTorch across the pipeline.
   The direct AOTI shim supports a standalone correction-head test.

## Performance reference

<!-- BEGIN GENERATED RUNTIME REGRESSION FACTS -->
Runtime release `v0.12.0` pins the following gates.

| Profile | Fixture | GPU | Timing | Points | States | Mean | P99 | Stddev | PSNR |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `fused_fp16` | `bench_input_512` | NVIDIA L40S | sustained CUDA graph | auto | 7 | 5.2086 ms | 5.2818 ms | 0.0368 ms | 38.7462 dB |
| `fused_fp16` | `bench_input_512` | NVIDIA L40S | sustained CUDA graph | auto | 5 | 3.7721 ms | 3.8390 ms | 0.0433 ms | 38.1391 dB |
| `fused_fp8` | `bench_input_512` | NVIDIA L40S | sustained CUDA graph | auto | 7 | 4.9147 ms | 5.0115 ms | 0.1057 ms | 38.7186 dB |
| `fused_fp8` | `bench_input_512` | NVIDIA L40S | sustained CUDA graph | auto | 5 | 3.5725 ms | 3.6270 ms | 0.0534 ms | 38.1164 dB |
| `fused_fp16` | `bench_input_512` | NVIDIA GeForce RTX 5090 | sustained CUDA graph | auto | 7 | 3.3238 ms | 3.3536 ms | 0.0331 ms | 38.7510 dB |
| `fused_fp8` | `bench_input_512` | NVIDIA GeForce RTX 5090 | sustained CUDA graph | auto | 7 | 3.4555 ms | 3.4857 ms | 0.0236 ms | 38.7356 dB |
| `trt_aoti_fused` | `bench_input_512` | NVIDIA GeForce RTX 5090 | legacy end-to-end | auto | 7 | 14.8860 ms | 15.1072 ms | 0.0284 ms | 37.7464 dB |
| `trt_aoti_fused` | `bench_input_512` | NVIDIA GeForce RTX 5090 | legacy end-to-end | auto | 5 | 11.1869 ms | 11.3861 ms | 0.0244 ms | 37.2323 dB |
<!-- END GENERATED RUNTIME REGRESSION FACTS -->

The source-of-truth baselines are `tests/expected_latency.json` and
`tests/expected_psnr.json`. AOTI fuses the correction head into a `.pt2`
package, while the forward map uses a raw TensorRT engine. These compiled
artifacts are machine-stack-specific and must be regenerated after material
CUDA, driver, PyTorch, or TensorRT changes.

Do not compare the current row directly with old milestone numbers whose
weights, point-count rules, or fixtures differ.

## TensorRT boundaries

The current builder uses the `dynamo` frontend. The forward map compiles to a
single extractable engine for the legacy C++ profile.

**The correction_head is not a single raw TRT engine.** A direct full-graph
compile fails on the U-Net + aux_net residual chain. With TensorRT 10.16,
`gaussifier-sample-build-trt` succeeds by keeping `aten.cat` in Torch and emits
a hybrid `correction_head_trt.pt`; its measured random-input parity is
max-absolute error 0.0147 in FP16. Because the raw C++ loader cannot extract
that hybrid as one engine, production C++ uses `correction_head_aoti.pt2`.
If hybrid compilation fails on another validated stack, Python callers fall
back to eager/AOTI. See [compatibility.md](../docs/compatibility.md) for the
locked versions.

## Architecture decisions

- **TorchScript via trace (not script).** `JointGaussianModel` overrides
  `forward_map`/`forward_features` instead of `forward`, which blocks
  scripting. Tracing works and is bit-exact at multiple resolutions
  (verified in `tools/probe_torchscript_export.py`).
- **Tensor I/O via NPY.** The C++ harness reads and writes C-contiguous float32
  NPY directly. This avoids both an image-library dependency and deprecated
  TorchScript-as-data serialization; Python only converts PNG at the edge.
- **`.cu` sources compiled once.** One CMake object target compiles the local
  Voronoi snapshot plus the sibling renderer, then links those objects into
  the harness and native C-API tests. This avoids compiling the same CUDA
  translation units three times.
- **PYBIND11_MODULE symbols left in.** The Python module init functions
  (e.g. `PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)`) compile fine in a C++
  binary — they define entry points that are never called. `jfa_kernel.cu`'s
  init symbol is renamed via `TORCH_EXTENSION_NAME` to avoid a duplicate
  with the rasterizer's `ext.cpp` (which we instead exclude from the source
  glob, since it has only bindings and no kernel code).
- **Explicit cross-repo boundary.** `-DGAUSSIFIER_SRC_DIR=` selects only the
  renderer source checkout. `scripts/sync_native_voronoi.py` controls the
  packaged sampler snapshot and its `--check` mode prevents silent drift.
