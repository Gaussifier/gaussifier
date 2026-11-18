"""Web bundle manifest schema and the ONNX export entry point."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jsonschema
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (REPOSITORY_ROOT / "schemas/web_bundle.schema.json").read_text(encoding="utf-8")
)
PRODUCED = REPOSITORY_ROOT / "web/apps/demo/public/models/web_bundle.json"


def _minimal_manifest() -> dict:
    file = {
        "path": "forward_map.onnx",
        "sha256": "a" * 64,
        "bytes": 10,
        "role": "forward_map",
        "precision": "f32",
        "shape": "dynamic",
    }
    head = {**file, "path": "correction_head.onnx", "role": "correction_head"}
    return {
        "schema_version": 2,
        "runtime_version": "0.11.1",
        "model_version": "0.12.0",
        "weights_sha256": {"stage1": "b" * 64, "head": "c" * 64},
        "model": {
            "correction_iterations": 7,
            "correction_predict_xy": True,
            "correction_output_channels": 8,
            "correction_input_channels": 17,
            "correction_xy_step_pixels": 2.56,
            "decoder_density_aware": False,
        },
        "files": [file, head],
        "io": {
            "forward_map": {
                "input": "image",
                "outputs": ["density", "log_cov", "rgb", "raw_density", "rate"],
            },
            "correction_head": {"input": "features", "output": "delta"},
        },
        "sampler": {
            "oversample": 1.5,
            "merge_rounds": 6,
            "per_round_fraction": 1 / 3,
            "knn_k": 8,
            "lloyd_iters": 3,
            "tile_cap": 32,
            "seed_default": 12345,
        },
        "renderer": {
            "tile": 16,
            "large_tile_threshold": 32,
            "alpha_cutoff": 1 / 255,
            "capacity": "max(8N,4096)",
        },
        "input": {"pad_multiple": 32, "pad_mode": "edge"},
    }


def test_schema_accepts_minimal_manifest_and_rejects_fp16() -> None:
    jsonschema.validate(_minimal_manifest(), SCHEMA)
    broken = _minimal_manifest()
    broken["files"][0]["precision"] = "f16"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, SCHEMA)
    broken = _minimal_manifest()
    broken["model"]["decoder_density_aware"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, SCHEMA)


def test_produced_manifest_validates_and_matches_files() -> None:
    if not PRODUCED.exists():
        pytest.skip("no exported bundle present")
    manifest = json.loads(PRODUCED.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, SCHEMA)
    for entry in manifest["files"]:
        path = PRODUCED.parent / entry["path"]
        assert path.exists() and path.stat().st_size == entry["bytes"]


@pytest.fixture(scope="module")
def frozen_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch

    from gaussifier_sampler.model import FirstStageModel, ModelConfig

    model = FirstStageModel(
        ModelConfig(
            conv_block_type="residual",
            predict_decoder_attributes=True,
            predict_decoder_color=True,
            enable_correction_head=True,
            correction_predict_xy=True,
            correction_aux_hidden=24,
        )
    )
    state = model.state_dict()
    for name, gn in model.named_modules():
        if isinstance(gn, torch.nn.GroupNorm):
            channels = gn.num_channels
            state[f"{name}.weight"] = torch.linspace(0.8, 1.2, channels)
            state[f"{name}.bias"] = torch.linspace(-0.1, 0.1, channels)
            state[f"{name}.mean"] = torch.linspace(-0.5, 0.5, channels)
            state[f"{name}.var"] = torch.linspace(0.5, 2.0, channels)
    weights = tmp_path_factory.mktemp("frozen_weights")
    torch.save(
        {k: v for k, v in state.items() if not k.startswith("correction_head.")},
        weights / "stage1.pt",
    )
    torch.save(
        {
            f"net.{k.removeprefix('correction_head.')}": v
            for k, v in state.items()
            if k.startswith("correction_head.")
        },
        weights / "head.pt",
    )
    return weights


def test_export_bundle_writes_validated_single_file_models(frozen_weights, tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from gaussifier_sampler.web import web_weight_hashes
    from scripts.export_web_bundle import export_bundle

    manifest = export_bundle(
        frozen_weights,
        tmp_path,
        static_sizes=(),
        trace_size=64,
        verify_sizes=(64, 96),
    )
    assert manifest["weights_sha256"] == web_weight_hashes(frozen_weights)
    for name in ("forward_map.onnx", "correction_head.onnx"):
        ops = {node.op_type for node in onnx.load(tmp_path / name).graph.node}
        assert "Conv" in ops
        assert not any("Normalization" in op for op in ops)
    assert {entry["role"] for entry in manifest["files"]} == {
        "forward_map",
        "correction_head",
        "correction_head_wgsl",
        "correction_head_wgsl_layout",
    }
    assert (tmp_path / "correction_head_wgsl.bin").exists()
    assert (tmp_path / "correction_head_wgsl.json").exists()
    assert not list(tmp_path.glob("*.data"))
    jsonschema.validate(manifest, SCHEMA)


def test_head_weight_layout_matches_folded_state_dict(frozen_weights) -> None:
    import numpy as np
    import torch

    from gaussifier_sampler.model import fold_frozen_gn, load_frozen_gn_release
    from scripts.export_web_bundle import HEAD_TENSOR_COUNT, head_weight_layout

    eager = load_frozen_gn_release(frozen_weights)
    model = fold_frozen_gn(eager)
    # The backbone stem conv originally had no bias. The head conv already had one.
    for original, folded in (
        (eager.backbone.stem.conv1.block, model.backbone.stem.conv1.block),
        (eager.correction_head.net.enc_blocks[0], model.correction_head.net.enc_blocks[0]),
    ):
        norm = original[1]
        scale = norm.weight * torch.rsqrt(norm.var + norm.eps)
        conv_bias = 0.0 if original[0].bias is None else original[0].bias
        torch.testing.assert_close(
            folded[0].weight, original[0].weight * scale[:, None, None, None]
        )
        torch.testing.assert_close(folded[0].bias, (conv_bias - norm.mean) * scale + norm.bias)
        assert isinstance(folded[1], torch.nn.Identity)

    layout, arrays = head_weight_layout(model)
    tensors = layout["tensors"]
    assert len(tensors) == HEAD_TENSOR_COUNT == len(arrays)
    assert layout["format"] == 2
    assert layout["arch"] == {
        "in_channels": 17,
        "out_channels": 8,
        "hidden": 32,
        "aux_hidden": 24,
        "depth": 2,
        "kernel_size": 3,
    }
    assert tensors[0]["offset"] == 0
    for previous, entry in itertools.pairwise(tensors):
        assert entry["offset"] == previous["offset"] + previous["count"]
    total = tensors[-1]["offset"] + tensors[-1]["count"]
    assert total == sum(array.size for array in arrays)
    state = model.correction_head.state_dict()
    assert [entry["name"] for entry in tensors] == list(state.keys())
    bias = next(entry for entry in tensors if entry["name"] == "net.out.bias")
    assert bias["shape"] == [8] and bias["count"] == 8

    assert all(entry["name"].endswith((".weight", ".bias")) for entry in tensors)
    for entry, array in zip(tensors, arrays, strict=True):
        assert np.array_equal(array, state[entry["name"]].numpy().reshape(-1))
