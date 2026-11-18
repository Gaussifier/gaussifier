# Supported versions

Versions tested green by the regression suite. Anything else is best-effort.

<!-- BEGIN GENERATED RUNTIME COMPATIBILITY FACTS -->
Runtime `v0.12.0` defaults to the `torch-free` fused profile. The Python package supports `>=3.12,<3.13`. PyTorch `2.12.*` is optional.

The deprecated `gaussifier_infer` C++ profile builds with `-DGAUSSIFIER_BUILD_LEGACY=ON` and is scheduled for removal in the next release. The Python GN backend uses the v0.11.1 checkpoint with SHA-256 `ceacdf9c9465b09bc932787848a8dc2a177757db466f6b1bd2d26e9813b7e841`.
<!-- END GENERATED RUNTIME COMPATIBILITY FACTS -->

## Tested green

| Fused component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04 x86-64 | ELF executable and ctypes library |
| CUDA toolkit | 12.8 | Builds `sm_89` by default |
| NVIDIA driver | 580.178.04 | Used for the release verification |
| GPU | NVIDIA L40S (`sm_89`) | FP16 and FP8 quality and latency gates |
| Python | 3.12 | NumPy input works without PyTorch |

The fused source also targets `sm_120` (RTX 5090, CUDA 13.0, driver 595.99.02)
with its own FP16-accumulate default and head tile tables. The release gates were
measured on `sm_89`; the RTX 5090 has K=7 fused gates of its own.

### Legacy profile and Python GN backend

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04 | aarch64 untested |
| Python | 3.12 | pinned by `uv.lock` |
| PyTorch | 2.12.0 | `+cu130` wheel |
| torch-tensorrt | 2.12.0 | matches the PyTorch minor |
| CUDA toolkit | 13.0 (V13.0.88) | `nvcc --version` |
| cuDNN | 9.20.0.48 | shipped via `nvidia-cudnn-cu13` |
| TensorRT (Python pkg) | 10.16.1.11 | pinned by the `tensorrt` extra |
| TensorRT (dev SDK) | 10.16.1.11 | `cpp_inference/scripts/install_trt_sdk.sh` matches the Python runtime |
| GPU compute capability | 12.0 (Blackwell, RTX 5090) | historical legacy regression baselines |

The torch-free AOTI shim advertises the validated range
`2.10.0..2.12.x`; the v0.11.1 load smoke and FP16 output-parity validation
use PyTorch 2.12.0.

## Not validated

Fused quality and latency are verified on L40S and RTX 5090. The historical legacy
benchmark used RTX 5090. The legacy toolchain and GPU combinations listed
above describe the scope of the published baselines.

## Known broken

| Component | Versions | Issue |
|---|---|---|
| TensorRT | < 10.16.1 or ≥ 10.17 | torch-tensorrt 2.12 requires TensorRT ≥ 10.16.1 and < 10.17 |
| PyTorch | ≤ 2.4 | `aoti_torch_*` shim ABI predates several functions our shim depends on |
| GPU compute capability | ≤ 7.0 | unsupported by the CUDA 13 toolchain used for this release |
| TensorRT 10.15.1, 10.16.1, or 11.0 | correction head as one raw engine | A direct full-graph compile can fail at the 17→24 convolution. TensorRT 10.16.1 succeeds as a hybrid module when `aten.cat` remains in Torch, but that artifact is not extractable as the single raw engine required by the C++ loader. |
| TensorRT 11.0 | torch-tensorrt 2.12 | ABI break: torch-tensorrt 2.12 was built against TRT 10's SONAME (libnvinfer_plugin.so.10). Upgrading TRT to 11 requires also upgrading torch-tensorrt, which requires upgrading PyTorch. |

## Adding a version to the matrix

1. Update `uv.lock` (`uv lock --upgrade-package <name>`)
2. Reserve the GPU exclusively; concurrent GPU work invalidates the latency gate
   and can perturb the PSNR gate.
3. Run the full regression suite: `uv run pytest tests/ -v`
4. Verify the harness still meets perf SLA: `uv run pytest tests/test_latency_regression.py -v`
5. If green, add the new version to "Tested green" above.
6. If red, document the failure mode under "Known broken" and file an issue.

## How to check what you have

```bash
uv run python -c "import torch, torch_tensorrt, tensorrt; print(
    f'pytorch={torch.__version__}\ntorch_tensorrt={torch_tensorrt.__version__}\n'
    f'tensorrt={tensorrt.__version__}')"
nvcc --version | grep release
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```
