# Browser engine and viewer

The `web/` tree holds a browser port of the production inference profile and a
framework-free viewer for 2D Gaussian splat scenes. The TypeScript packages,
their build and test commands, and the WebGPU conventions are documented in
[web/README.md](../web/README.md). This page covers the Python side that feeds
them.

## Export the model bundle

The browser uses the v0.12.0 frozen-GN model in FP32. Set `WEIGHTS` to the
release directory containing `stage1.pt` and `head.pt`. Both files are
tensor-only state dicts.

```bash
WEIGHTS=/path/to/weights/release_frozen_gn
uv run --extra web python scripts/export_web_bundle.py --weights-dir "$WEIGHTS" \
  --out-dir web/apps/demo/public/models
```

The exporter folds each calibrated affine into the preceding convolution's
weights and bias. It writes dynamic `forward_map.onnx` and
`correction_head.onnx` graphs and static 512 variants for graph capture.
It checks the graphs against the unfused eager model on the CPU before
writing `web_bundle.json`, validated by [web_bundle.schema.json](../schemas/web_bundle.schema.json).

The same command writes `correction_head_wgsl.bin`, a little-endian float32
array containing the folded head's convolution weights and biases in
`state_dict` order. `correction_head_wgsl.json` records each tensor's name,
shape, element offset, and count. The engine prefers the WGSL head and offers
the folded ONNX head through the `ort` backend. The output directory is ignored
by git. Regenerate it when upgrading a v0.11.1 browser demo to the new bundle
format.

## Write goldens

The golden exporter loads the same weights with the calibrated affines
intact. It runs eager FP32 on CUDA and uses the reference renderer from
`gaussifier_sampler.web`. It records the five forward-map outputs, CUDA
oversampling, merged and polished points, owner map, masses, weights, initial
attributes, and the states of the render and correction loop.

```bash
uv run python scripts/export_web_goldens.py --weights-dir "$WEIGHTS" \
  --image tests/fixtures/bench_input_512.png --store-head yes \
  --name bench_input_512 --out web/fixtures/bench_input_512
uv run python scripts/export_web_goldens.py --weights-dir "$WEIGHTS" \
  --image tests/fixtures/bench_input_512.png --crop 128 \
  --name bench_128 --out web/fixtures/bench_128
uv run python scripts/export_web_goldens.py --weights-dir "$WEIGHTS" \
  --image tests/fixtures/synthetic_gradient.png \
  --name synthetic_gradient --out web/fixtures/synthetic_gradient
```

Each fixture directory contains raw little-endian `.bin` arrays described by
`shapes.json` and a `scene.npz` for the viewer. Its `manifest.json` records
per-state PSNR and the native placement's run-to-run envelope. The browser
parity tests compare both input weight hashes with the bundle before scoring.
The binary outputs are ignored by git. The head tests use the recorded head
inputs and deltas at 128² and 512², so keep `--store-head yes` for the 512²
fixture.

## Viewer module and demo

`web/apps/demo` is the browser demo; `npm run build:demo` then `npm run serve`
in `web/` serves it over HTTP and HTTPS. `npm run build:bundle` produces the
single-file viewer module `gaussifier-viewer.js`, which plays a `.splat2d` or
`.npz` asset from a URL through the `<gaussifier-viewer src="...">` element;
`embed.html` in the demo shows it. The demo exports Gaussians as `.splat2d`,
the Splat2D format that the training code's `data/splat2d.py` reads, written at
the original image size (padding removed, xy rescaled), and
`gaussifier_sampler.web.load_scene` reads the viewer's NPZ export.

The viewer overlays are the source image, the difference, the density map,
the Voronoi cells of the current centers (tinted, with dark borders), the
centers, and the 1σ ellipses. The oversampler wasm ships prebuilt under
`web/packages/wasm-oversampler/prebuilt/` with a provenance record that the
web unit tests check against the C++ source, and `.github/workflows/web.yml`
runs the GPU-free web checks on every change under `web/`.

## Reference renderer

`gaussifier_sampler.web.reference_render` evaluates the production simple-sum
formula without tile culling. It accumulates each Gaussian only inside the box
where its alpha can reach the `1/255` cutoff, which is identical to a full
per-pixel loop up to summation order. `save_scene` writes the NPZ key set the
viewer's loader understands.
