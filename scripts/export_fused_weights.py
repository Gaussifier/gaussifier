"""Export a frozen-GroupNorm Gaussifier forward map to the fused runtime's GFFM weight blob.

Every frozen GroupNorm (per-channel mean/var buffers) is folded into the preceding conv:
w' = w * s, b' = beta - mean * s with s = gamma / sqrt(var + eps). Conv weights are written
as [taps][cin][cout] FP16 (stem input channels zero-padded 3 -> 16); biases as FP32.

    uv run python scripts/export_fused_weights.py --state stage1.pt --out forward_map.gffm

``--state`` is a state_dict with ``backbone.*`` (frozen-GN buffers ``mean``/``var``),
``density_head.head.*``, ``decoder_log_covariance_head.*`` and ``decoder_rgb_head.*``.

FP8 precision plan (GFFM v2):

    ... --precision-plan precision_plan.json --plan budget_0.03dB

appends, for each 3x3 conv (stem, down0..2, up0..2; conv1 then conv2), three tensors:
meta f32 [is_fp8, act_scale], e4m3 weights u8 [taps][cout][cin] (cin contiguous) and the
per-output-channel weight scales f32 [cout] (both empty for FP16 layers). The recipe matches
tools/fp8_sensitivity.py in the gaussifier repo exactly: FrozenGN folded in FP32 first,
w_scale = amax_c(folded W) / 448, w8 = e4m3(clamp(W / w_scale, +-448)); the activation scale
is the plan's static ``act_scale`` (recalibrate with ``fp8_sensitivity.py calib`` when the
weights change). The plan's ``density_bias_offset`` is added to the density-head bias so the
predicted point count matches FP16. Without a plan the blob is GFFM v1 (unchanged).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC, VERSION, VERSION_FP8 = 0x4D464647, 1, 2
GN_EPS = 1e-5
E4M3_MAX = 448.0
PLAN_NAMES = ["stem", "down0", "down1", "down2", "up0", "up1", "up2"]
BLOCKS = [
    "stem",
    *(f"down_blocks.{i}.conv" for i in range(3)),
    *(f"up_blocks.{i}.conv" for i in range(3)),
]


def _fold(sd: dict, conv_key: str, gn_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    w = sd[f"{conv_key}.weight"].double()
    gamma, beta = sd[f"{gn_key}.weight"].double(), sd[f"{gn_key}.bias"].double()
    mean, var = sd[f"{gn_key}.mean"].double(), sd[f"{gn_key}.var"].double()
    s = gamma / torch.sqrt(var + GN_EPS)
    b = beta - mean * s
    if f"{conv_key}.bias" in sd:
        b = b + sd[f"{conv_key}.bias"].double() * s
    return w * s[:, None, None, None], b


def _taps(w: torch.Tensor, cin_pad: int | None = None) -> np.ndarray:
    """[cout, cin, kh, kw] -> [kh*kw, cin, cout] FP16, optionally zero-padding cin."""
    cout, cin, kh, kw = w.shape
    t = w.permute(2, 3, 1, 0).reshape(kh * kw, cin, cout)
    if cin_pad is not None and cin_pad > cin:
        t = torch.cat([t, t.new_zeros(kh * kw, cin_pad - cin, cout)], 1)
    return t.float().numpy().astype(np.float16)


def tensors(sd: dict) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, name in enumerate(BLOCKS):
        pad = 16 if i == 0 else None
        p = f"backbone.{name}"
        w1, b1 = _fold(sd, f"{p}.conv1.block.0", f"{p}.conv1.block.1")
        w2, b2 = _fold(sd, f"{p}.conv2.0", f"{p}.conv2.1")
        proj = sd[f"{p}.projection.weight"].double()
        out += [_taps(w1, pad), b1.float().numpy(), _taps(w2), b2.float().numpy(), _taps(proj, pad)]
    # heads: [32][8] = density, log-cov (3), rgb (3), zero pad
    hw = torch.cat(
        [
            sd["density_head.head.weight"].flatten(1),
            sd["decoder_log_covariance_head.weight"].flatten(1),
            sd["decoder_rgb_head.weight"].flatten(1),
            torch.zeros(1, sd["density_head.head.weight"].shape[1]),
        ]
    )  # [8, 32]
    hb = torch.cat(
        [
            sd["density_head.head.bias"],
            sd["decoder_log_covariance_head.bias"],
            sd["decoder_rgb_head.bias"],
            torch.zeros(1),
        ]
    )
    out += [hw.t().contiguous().float().numpy().astype(np.float16), hb.float().numpy()]
    return out


def _fold_fp32(sd: dict, conv_key: str, gn_key: str) -> torch.Tensor:
    """Folded weight exactly as fp8_sensitivity.QConv computes it (FP32 ops)."""
    w = sd[f"{conv_key}.weight"].float()
    s = sd[f"{gn_key}.weight"].float() * torch.rsqrt(sd[f"{gn_key}.var"].float() + GN_EPS)
    return w * s[:, None, None, None]


def fp8_tensors(sd: dict, plan: dict) -> list[np.ndarray]:
    """Per 3x3 conv: meta [is_fp8, act_scale], e4m3 [taps][cout][cin], w_scale [cout]."""
    out: list[np.ndarray] = []
    layers = plan["layers"]
    for name, blk in zip(PLAN_NAMES, BLOCKS, strict=True):
        p = f"backbone.{blk}"
        for conv, ck, gk in (
            ("conv1", f"{p}.conv1.block.0", f"{p}.conv1.block.1"),
            ("conv2", f"{p}.conv2.0", f"{p}.conv2.1"),
        ):
            lay = layers[f"bb.{name}.{conv}"]
            if lay["precision"] != "fp8":
                out += [
                    np.array([0.0, 0.0], np.float32),
                    np.zeros(0, np.uint8),
                    np.zeros(0, np.float32),
                ]
                continue
            w = _fold_fp32(sd, ck, gk)  # [cout, cin, kh, kw]
            ws = w.abs().amax((1, 2, 3)).clamp_min(1e-12) / E4M3_MAX
            q = (w / ws[:, None, None, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            cout, cin, kh, kw = q.shape
            q8 = q.view(torch.uint8).permute(2, 3, 0, 1).reshape(kh * kw, cout, cin).contiguous()
            out += [
                np.array([1.0, lay["act_scale"]], np.float32),
                q8.numpy(),
                ws.numpy().astype(np.float32),
            ]
    return out


def write_blob(arrays: list[np.ndarray], path: Path, version: int = VERSION) -> None:
    buf = bytearray(struct.pack("<III", MAGIC, version, len(arrays)))
    for a in arrays:
        dtype = {np.dtype(np.float16): 0, np.dtype(np.float32): 1, np.dtype(np.uint8): 2}[a.dtype]
        buf += struct.pack("<II", dtype, a.size)
        buf += b"\0" * ((-len(buf)) % 16)
        buf += np.ascontiguousarray(a).tobytes()
    path.write_bytes(bytes(buf))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision-plan", type=Path, help="FP8 plan JSON (fp8_sensitivity.py)")
    ap.add_argument("--plan", default="budget_0.03dB", help="plan name inside the JSON")
    a = ap.parse_args()
    sd = torch.load(a.state, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    arrays = tensors(sd)
    if a.precision_plan is None:
        write_blob(arrays, a.out)
    else:
        plan = json.loads(a.precision_plan.read_text())["plans"][a.plan]
        arrays[-1] = arrays[-1].copy()
        arrays[-1][0] += np.float32(plan["density_bias_offset"])  # keeps the point count
        arrays += fp8_tensors(sd, plan)
        n8 = sum(1 for x in arrays[37::3] if x[0] > 0)
        print(
            f"plan {a.plan}: {n8} of 14 backbone 3x3 convs FP8, density bias "
            f"+{plan['density_bias_offset']}"
        )
        write_blob(arrays, a.out, VERSION_FP8)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {len(arrays)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
