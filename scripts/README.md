# Script index

These scripts build the release artifacts, the native C++ harness, and the browser export.
The inference entry points are installed from `pyproject.toml`; see the root
[README](../README.md) for their usage. The C++ harness reads and writes float32 NPY, so convert
images with NumPy and Pillow.

| Script | Purpose | Main inputs | Output or check |
|---|---|---|---|
| `audit_binary_dependencies.py` | Inspect direct ELF dependencies and enforce a declared LibTorch boundary | Binary and `--contract` | Pass/fail `DT_NEEDED` report |
| `sync_native_voronoi.py` | Synchronize the packaged Voronoi release snapshot from Gaussifier | A sibling checkout or `--gaussifier-root` | Updates three native files, or reports drift with `--check` |
| `full_build.sh` | Prepare dependencies, build C++/native targets, and run strict regressions | Gaussifier checkout and bundled weights | CMake targets, CTest results, and pytest quality/latency gates |
| `build_release_artifacts.sh` | Export FP16 and FP8 runtime blobs or an opt-in legacy bundle | Frozen-GN weights directory or `--legacy` training checkpoint, output directory | GFFM and GFCH blobs or legacy TorchScript, TensorRT and AOTI assets |
| `build_torch_free_aoti.sh` | Patch one AOTI correction-head package to use the repository shim | AOTI `.pt2`, output directory, built shim | Head-only torch-free bundle plus dependency check |
| `export_web_bundle.py` | Export single-file fp32 ONNX graphs plus the validated web bundle manifest for the browser engine | Frozen-GN weights directory, output directory, optional static sizes | `forward_map.onnx`, `correction_head.onnx`, static variants, WGSL weights and layout, `web_bundle.json` |
| `export_web_goldens.py` | Record stage-wise production-profile goldens for browser parity tests | Frozen-GN weights directory, fixture image, name, output directory, seed, state count | Raw `.bin` arrays, `shapes.json`, `scene.npz`, `manifest.json` |
| `export_fused_weights.py` | Fold a frozen-GN forward map into the fused runtime's weight blob, optionally with an FP8 precision plan | Stage-1 state dict, optional `--precision-plan` | GFFM blob for `gaussifier_fused_infer` |
| `export_fused_head.py` | Fold a frozen-GN correction head into the fused runtime's weight blob, optionally with an FP8 precision plan | Frozen-GN head state dict or trusted research module, optional `--precision-plan` | GFCH blob for `gaussifier_fused_infer` |
| `extract_trt_engine.py` | Extract the raw NvInfer plan from a Torch-TensorRT forward-map module | Hybrid `.pt` module | Raw `.engine` file |
| `install_trt_sdk.sh` | Install the matching local TensorRT headers/tools without system mutation | Pinned SDK version and optional mode | `tensorrt_sdk/` checkout marker, headers, and tools |

## Common workflows

Verify that the packaged native snapshot still matches training:

```bash
uv run python scripts/sync_native_voronoi.py --check
```

Build and verify the fused profile with the shipped weights:

```bash
bash scripts/full_build.sh
```

Re-export the shipped fused blobs from their source weights:

```bash
bash scripts/build_release_artifacts.sh src/gaussifier_sampler/weights/frozen_gn <bundle_dir>
```

The fused runtime builds from this repository alone. The weight exporters, the native
snapshot check and the legacy profile (`--legacy`) also need the training code, which is not
included here; `GAUSSIFIER_SRC_DIR` points at it.

Generated engines, build trees, SDK files, and compiled artifact bundles are ignored and
must not be committed. The schema-backed `release/runtime_bundle.json` manifest is tracked.
