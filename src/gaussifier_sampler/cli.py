"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gaussifier_sampler.image_io import (
    load_image_array,
    load_image_tensor,
    parse_resize,
    save_density_png,
    save_points_png,
)
from gaussifier_sampler.inference import GaussifierSampler, as_numpy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run first-stage Gaussifier density inference and sample points."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Training checkpoint; defaults to the bundled weights.",
    )
    parser.add_argument(
        "--count",
        type=float,
        default=None,
        help=(
            "Target point count; defaults to the model's own predicted count. A higher count "
            "scales the predicted covariance by own/target to keep the splatted mass."
        ),
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional inference resize: SIZE or WIDTHxHEIGHT.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--backend", choices=("fused", "torch"), help="Default: fused.")
    parser.add_argument("--precision", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument(
        "--size-mode",
        choices=("exact", "pad"),
        default="exact",
        help="Fused sizes that are not multiples of 16: real extent (exact) or reflect pad.",
    )
    parser.add_argument("--density-png", type=Path, default=None, help="Optional density preview.")
    parser.add_argument("--points-png", type=Path, default=None, help="Optional point preview.")
    parser.add_argument("--json", type=Path, default=None, help="Optional metadata JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    backend = args.backend or ("torch" if args.checkpoint else "fused")
    if backend == "fused" and args.checkpoint:
        parser.error("--checkpoint requires --backend torch")
    if backend == "torch" and args.precision != "fp16":
        parser.error("--precision fp8 requires --backend fused")
    if backend == "torch" and args.size_mode != "exact":
        parser.error("--size-mode pad requires --backend fused")
    load_image = load_image_array if backend == "fused" else load_image_tensor
    image = load_image(args.image, resize=parse_resize(args.resize))
    if args.checkpoint:
        sampler = GaussifierSampler.from_checkpoint(args.checkpoint, device=args.device)
    else:
        sampler = GaussifierSampler.bundled(
            backend=backend, precision=args.precision, device=args.device, size_mode=args.size_mode
        )
    result = sampler.sample(
        image,
        count=args.count,
        seed=args.seed,
        image_key=str(args.image) if backend == "torch" else None,
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    points_np = as_numpy(result.points).astype(np.float32, copy=False)
    density_np = as_numpy(result.density).astype(np.float32, copy=False)
    payload = {"points": points_np, "density": density_np, **result.init_arrays()}
    for name in (
        "final_points",
        "final_log_covariance_channels",
        "final_color",
        "final_rendered",
        "final_unweighted_color",
        "point_weights",
    ):
        if (value := getattr(result, name)) is not None:
            payload[name] = as_numpy(value)
    np.savez_compressed(output, **payload)

    height, width = density_np.shape
    if args.density_png is not None:
        args.density_png.parent.mkdir(parents=True, exist_ok=True)
        save_density_png(result.density, args.density_png)
    if args.points_png is not None:
        args.points_png.parent.mkdir(parents=True, exist_ok=True)
        save_points_png(result.points, args.points_png, height=height, width=width)

    metadata = {
        "backend": backend,
        "checkpoint": "bundled" if args.checkpoint is None else str(args.checkpoint),
        "count": result.predicted_count,
        "image": str(args.image),
        "output": str(output),
        "image_height": int(height),
        "image_width": int(width),
        "has_init_covariance": result.init_covariance is not None,
        "has_init_color": result.init_color is not None,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
