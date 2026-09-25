# Gaussifier

Inference code and weights for **Gaussifier: Fast 2D Gaussian Decomposition**, an
anonymous submission under double-blind review.

- Project page: <https://gaussifier.github.io/>
- Live browser demo (WebGPU): <https://gaussifier.github.io/demo/>
- Weights: [release v0.11.1](https://github.com/Gaussifier/gaussifier/releases/tag/v0.11.1)

## Installation

Requirements: Linux, an NVIDIA GPU with a CUDA toolkit, Python 3.12, and
[uv](https://docs.astral.sh/uv/). Tested versions are listed in
[docs/compatibility.md](docs/compatibility.md).

```bash
git clone https://github.com/Gaussifier/gaussifier.git
cd gaussifier

curl -L --create-dirs -o src/gaussifier_sampler/weights/gaussifier_best.pt \
  https://github.com/Gaussifier/gaussifier/releases/download/v0.11.1/gaussifier_best.pt
echo "ceacdf9c9465b09bc932787848a8dc2a177757db466f6b1bd2d26e9813b7e841  src/gaussifier_sampler/weights/gaussifier_best.pt" | sha256sum -c

uv sync
```

Download the weights before `uv sync`, because the package build includes them. The first
inference call compiles the native placement kernels (Torch extensions).

## Inference

### Command line

```bash
uv run gaussifier-sample --image image.png --output sample.npz \
  --density-png density.png --points-png points.png
```

This predicts the density map, places the model's own number of Gaussians, and writes their
initial parameters to `sample.npz`: `points` (`[N, 2]`, normalized `[x, y]`), `density`,
and the `init_*` covariance, scale, rotation, and color arrays. `--count N` fixes the number
of Gaussians, `--seed` fixes the placement seed, and `--checkpoint` loads other weights.

### Python

The correction loop renders the Gaussians after every update, so it takes a renderer. The
package includes an exact reference renderer. It needs nothing else but is slower than the
engines below.

```python
from gaussifier_sampler import GaussifierSampler
from gaussifier_sampler.image_io import edge_pad, load_image_tensor
from gaussifier_sampler.web import psnr, reference_render

sampler = GaussifierSampler.bundled(device="cuda")
image = load_image_tensor("image.png")
padded, height, width = edge_pad(image, 32)  # sizes must be multiples of 32
H, W = padded.shape[-2:]


def renderer(xy, cov, rgb):
    return reference_render(xy, cov, rgb, H, W, deterministic=False)


result = sampler.sample(
    padded,
    renderer=renderer,
    correction_iterations=7,  # seven rendered states, six updates
    inference_profile="production",
)
output = result.final_rendered[:, :height, :width]
print(result.predicted_count, "Gaussians,", psnr(output, image.to(output.device)), "dB")
```

The final Gaussians are `result.final_points`, `result.final_log_covariance_channels`, and
`result.final_color`. Without a renderer, `sampler.sample(image)` returns only the initial
Gaussians. `count=` fixes the number of Gaussians.

### TensorRT and AOT Inductor engines (optional)

The engines are FP16 and specific to the GPU, driver, and CUDA, PyTorch, and TensorRT
versions, so build them on the machine that runs them.

```bash
uv sync --extra tensorrt
CKPT=src/gaussifier_sampler/weights/gaussifier_best.pt

# TensorRT forward map for one input size (here 512 x 512)
uv run gaussifier-sample-build-trt --ckpt $CKPT --out-dir trt_engines --crop-size 512

# AOT Inductor forward map and correction head for any size that is a multiple of 16
uv run gaussifier-sample-build-aoti --ckpt $CKPT --out-dir aoti_packages --dynamic --dict-forward-map
```

Load them in place of `GaussifierSampler.bundled(...)`:

```python
sampler = GaussifierSampler.bundled_trt("trt_engines", device="cuda")
sampler = GaussifierSampler.bundled_aoti("aoti_packages", device="cuda")
```

## Build

### Browser demo

The browser engine runs the whole pipeline with WebGPU. Export the model for it, then build
and serve the demo (needs Node.js and npm):

```bash
uv sync --extra web  # add --extra tensorrt to keep the TensorRT packages
uv run python scripts/export_web_bundle.py --out-dir web/apps/demo/public/models
cd web
npm ci && npm run build && npm run build:demo
npm run serve
```

Open <http://localhost:8080/> in a browser with WebGPU (current Chrome or Edge, Safari 26, or
Firefox on Windows). WebGPU needs a secure context, so other machines should use the HTTPS
port 8443. See [web/README.md](web/README.md) and [docs/web.md](docs/web.md).

### C++ harness

The production C++ harness runs the TensorRT forward map and the AOT Inductor correction
head. It also compiles a CUDA renderer from the training code, which is not included in
this repository; set `GAUSSIFIER_SRC_DIR` to a checkout that provides it. Build steps are in
[cpp_inference/README.md](cpp_inference/README.md) and [scripts/README.md](scripts/README.md).
