# Gaussifier

Inference code and weights for **Gaussifier: Fast 2D Gaussian Decomposition**, an
anonymous submission under double-blind review.

- Project page: <https://gaussifier.github.io/>
- Live browser demo (WebGPU): <https://gaussifier.github.io/demo/>
- Weights: [release v0.12.0](https://github.com/Gaussifier/gaussifier/releases/tag/v0.12.0)

## Installation

Requirements: Linux, an NVIDIA GPU (tuned for `sm_89` and `sm_120`), CUDA 12.8 or 13.x with
`nvcc`, CMake 3.18 or newer, Python 3.12 and [uv](https://docs.astral.sh/uv/). Tested
versions are listed in [docs/compatibility.md](docs/compatibility.md).

```bash
git clone https://github.com/Gaussifier/gaussifier.git
cd gaussifier

base=https://github.com/Gaussifier/gaussifier/releases/download/v0.12.0
w=src/gaussifier_sampler/weights
curl -L --create-dirs -o $w/gaussifier_best.pt $base/gaussifier_best.pt
for f in fm_fp16.gffm fm_fp8.gffm head_fp16.gfch head_fp8.gfch stage1.pt head.pt \
         precision_plan_release.json; do
  curl -L --create-dirs -o $w/frozen_gn/$f $base/$f
done
sha256sum -c release/SHA256SUMS

uv sync --no-dev
```

Download the weights before `uv sync`, because the package build includes them.
`release/SHA256SUMS` lists the hashes of every weight file.

## Build

The default runtime is a CUDA-only pipeline (forward map, equal-mass Voronoi placement,
renderer and correction head, captured as one CUDA graph). It needs no LibTorch or TensorRT.
Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (89 for Ada, 120 for Blackwell GeForce):

```bash
cmake -S cpp_inference -B cpp_inference/build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build cpp_inference/build --parallel
```

This builds `cpp_inference/build/fused/gaussifier_fused_infer` and
`libgaussifier_fused.so.1`, which the Python package finds there. To use a library built
elsewhere, set `GAUSSIFIER_FUSED_LIBRARY` to its absolute path. Build options and the binary's
flags are in [cpp_inference/fused/README.md](cpp_inference/fused/README.md).

## Inference

The release model is the frozen-GN `v0.12.0` model in FP16, with its own (adaptive) point
count and K=7 rendered states (the initial render plus six correction-head updates). Lower K
is faster and lower quality.

### Command line

```bash
uv run --no-dev gaussifier-sample --image image.png --output sample.npz \
  --density-png density.png --points-png points.png
```

`--precision fp8` selects the optional mixed-precision plan, `--count N` fixes the number of
Gaussians, `--seed` fixes the placement seed, `--resize SIZE` or `WIDTHxHEIGHT` resizes the
input first, and `--size-mode exact|pad` picks how sizes that are not multiples of 16 run.

### Python

```python
from gaussifier_sampler import GaussifierSampler
from gaussifier_sampler.image_io import load_image_array

sampler = GaussifierSampler.bundled()  # precision="fp8", size_mode="pad" are optional
image = load_image_array("image.png")  # float32 [3, H, W] in [0, 1]
result = sampler.sample(image)
print(result.predicted_count, result.final_points.shape, result.final_rendered.shape)
```

NumPy input returns NumPy arrays; a PyTorch CUDA tensor returns tensors on the sampler's
device. `correction_iterations` (1 to 16, default 7) sets the number of rendered states.
`sample_batch` runs a list of images in order.

Video: open a session for a fixed frame size and push frames in order. Later frames keep the
previous Gaussians as tracks, with stable `ids` and `slots`:

```python
with sampler.video(height, width) as video:
    for image in frames:  # [3, H, W] or [1, 3, H, W]
        frame = video.push(image)
        use(frame.final_rendered, frame.final_points, frame.ids)
```

`backend="torch"` (install with `uv sync --no-dev --extra torch`) runs the original
v0.11.1 GroupNorm checkpoint `gaussifier_best.pt` in PyTorch, with a renderer you supply.

### C++ binary

```bash
w=src/gaussifier_sampler/weights/frozen_gn
cpp_inference/build/fused/gaussifier_fused_infer \
  --fm $w/fm_fp16.gffm --head $w/head_fp16.gfch --K 7 input.npy render.npy \
  --gaussians gaussians.npy
```

The input is a float32 NPY `[1, 3, H, W]` or `[3, H, W]` in `[0, 1]` (H, W at least 16), and the
output is the final render `[3, H, W]`. Each image prints one JSON line with the point count,
per-state PSNR and timings. `--manifest` runs many images with one model load, `--video` a
frame sequence, and `--sustained R` times R back-to-back runs.

### Outputs

`sample.npz` contains the normalized `[x, y]` points (`[N, 2]`), the density map, the initial
attributes (`init_log_covariance_channels` `[xx, xy, yy]`, `init_covariance`, `init_scale`,
`init_rotation`, `init_color`) and the corrected Gaussians (`final_points`,
`final_log_covariance_channels`, `final_color`, `point_weights`) with `final_rendered`. Colors
are renderer-ready (weights applied). Multiply points by `[width, height]` for pixel
coordinates.

## Performance

C++ binary on an RTX 5090, FP16 weights with full FP16 accumulation
(`GFUSED_FM_ACC16=2 GFUSED_HEAD_ACC16=2`; the default, mode 1, is about 1% slower at the same
PSNR). Latency is the median over the images of each image's GPU time (CUDA graph, median of
5 runs) and PSNR the mean, over 24 Kodak images at 512×512 and 768×512 and 24
high-resolution photos at 1280×720 and 1920×1080, seed 1234.

| Resolution | K=5 Latency (ms) | K=5 FPS | K=5 PSNR (dB) | K=7 Latency (ms) | K=7 FPS | K=7 PSNR (dB) |
|---|---|---|---|---|---|---|
| 512×512 | 2.3 | 431 | 40.23 | 3.1 | 321 | 40.82 |
| 768×512 | 3.1 | 318 | 40.11 | 4.3 | 234 | 40.82 |
| 1280×720 | 7.9 | 126 | 40.61 | 11.1 | 90 | 41.28 |
| 1920×1080 | 17.5 | 57 | 41.06 | 24.5 | 41 | 41.66 |

## Browser demo

The browser engine runs the same pipeline with WebGPU. Export the model for it, then build and
serve the demo (needs Node.js and npm):

```bash
uv sync --extra web
uv run python scripts/export_web_bundle.py --weights-dir src/gaussifier_sampler/weights/frozen_gn \
  --out-dir web/apps/demo/public/models
cd web
npm ci && npm run build && npm run build:demo
npm run serve
```

Open <http://localhost:8080/> in a browser with WebGPU (current Chrome or Edge, Safari 26, or
Firefox on Windows). See [web/README.md](web/README.md) and [docs/web.md](docs/web.md).
