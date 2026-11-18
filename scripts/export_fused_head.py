"""Export a frozen-GroupNorm Gaussifier correction head to the fused runtime's GFCH weight blob.

Every frozen GroupNorm (per-channel mean/var buffers) is folded into the preceding conv:
w' = w * s, b' = (b - mean) * s + beta with s = gamma / sqrt(var + eps).

The 17-channel head input contains ``diff = image - rendered``, which is linear in two other
input channels, so the first conv of each U-Net is rewritten to take 14 channels:

    image (3) | density (1) | log_cov (3) | rgb (3) | rendered (3) | count (1)

with W_image += W_diff and W_rendered -= W_diff (exact; zero padding is consistent because
diff is zero wherever image and rendered are). The fused runtime builds these 14 channels
directly from the pipeline's FP32 buffers.

Blob layout (little endian), GFCH v2 (v1 = same without FP8 layers; still loadable):

    u32 magic 'GFCH' (0x48434647), u32 version (2), u32 n_convs (22)
    per conv: u32 ks, u32 cin, u32 cout, u32 precision (0 = FP16, 1 = FP8 e4m3)
              <pad to 16> FP16 weight [ks*ks][cin][cout]   (always present: reference/fallback)
              <pad to 16> FP32 bias [cout]
      if FP8: <pad to 16> FP32 a_scale (static per-tensor scale of the conv input)
              <pad to 16> e4m3 weight [ks*ks][cout][cin]   (Cin contiguous)
              <pad to 16> FP32 w_scale [cout]               (per output channel)

Channel counts are the real (unpadded) ones; the loader pads them (to 32 at level 0 for the
main net, to multiples of 8 for the native-width aux net, see correction_head.cu). Conv order:
main net (hidden 32) then aux net (hidden 24), each as
enc0.a, enc0.b, enc1.a, enc1.b, bottleneck.a, bottleneck.b, merge0.a, merge0.b, merge1.a,
merge1.b, out (1x1, 8 channels). Merge convs take concat([up2x(deeper), skip]).

    uv run python scripts/export_fused_head.py --head head_module.pt --out head.gfch

FP8 (``--precision-plan precision_plan.json --plan-name budget_0.03dB``): the plan names convs
head.{main|aux}.{enc0.a, enc0.b, enc1.a, enc1.b, mid.a, mid.b, dec0.a, dec0.b, dec1.a, dec1.b,
out} (dec0 = lowest-resolution merge block), i.e. exactly the per-net conv order above. FP8
layers are quantized exactly like gaussifier/tools/fp8_sensitivity.py (GN folded in FP32, then
e4m3 per-output-channel scale amax_c / 448, saturating); the activation scale comes from the
plan. The runtime's FP8 conv needs every (16-padded) input segment to be a multiple of 32, so
plan layers with 48-channel inputs fall back to FP16 (printed).

``--head`` is a pickled CorrectionHead module whose GroupNorms are ``head_arch_screen.FrozenGN``
(needs ``--gaussifier-src`` for unpickling) or a state_dict of such a module.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC, VERSION = 0x48434647, 2
E4M3_MAX = 448.0
PLAN_NAMES = [
    "enc0.a",
    "enc0.b",
    "enc1.a",
    "enc1.b",
    "mid.a",
    "mid.b",
    "dec0.a",
    "dec0.b",
    "dec1.a",
    "dec1.b",
    "out",
]
GN_EPS = 1e-5
CONV_SEQ = [
    ("enc_blocks.0", 0),
    ("enc_blocks.0", 3),
    ("enc_blocks.1", 0),
    ("enc_blocks.1", 3),
    ("bottleneck", 0),
    ("bottleneck", 3),
    ("merge_blocks.0", 0),
    ("merge_blocks.0", 3),
    ("merge_blocks.1", 0),
    ("merge_blocks.1", 3),
]
# 17-ch input index -> 14-ch fused input index (diff channels 13..15 are folded away)
_KEEP = [*range(13), 16]


def _fold(sd: dict, conv: str, gn: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    w = sd[f"{conv}.weight"].double()
    s = sd[f"{gn}.weight"].double() / torch.sqrt(sd[f"{gn}.var"].double() + eps)
    b = sd.get(f"{conv}.bias", torch.zeros(w.shape[0])).double()
    b = (b - sd[f"{gn}.mean"].double()) * s + sd[f"{gn}.bias"].double()
    return w * s[:, None, None, None], b


def fold_diff(w: torch.Tensor) -> torch.Tensor:
    """[cout, 17, kh, kw] -> [cout, 14, kh, kw] with the diff channels folded in."""
    w = w.clone()
    w[:, 0:3] += w[:, 13:16]  # image
    w[:, 10:13] -= w[:, 13:16]  # rendered
    return w[:, _KEEP]


def net_convs(sd: dict, prefix: str, eps: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    convs = []
    for i, (blk, idx) in enumerate(CONV_SEQ):
        w, b = _fold(sd, f"{prefix}.{blk}.{idx}", f"{prefix}.{blk}.{idx + 1}", eps)
        convs.append((fold_diff(w) if i == 0 else w, b))
    convs.append((sd[f"{prefix}.out.weight"].double(), sd[f"{prefix}.out.bias"].double()))
    return convs


def fp8_weights(sd: dict, conv: str, gn: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 weights [cout, cin, kh, kw] + per-output-channel scales, as fp8_sensitivity.QConv."""
    w = sd[f"{conv}.weight"].float()
    s = sd[f"{gn}.weight"].float() * torch.rsqrt(sd[f"{gn}.var"].float() + eps)
    w = w * s[:, None, None, None]
    wscale = w.abs().amax((1, 2, 3)).clamp_min(1e-12) / E4M3_MAX
    q = (w / wscale[:, None, None, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, wscale


def fp8_eligible(prefix: str, i: int, sd: dict) -> bool:
    """Runtime FP8 conv: every 16-padded input segment must be a multiple of 32."""
    pad = lambda c: (c + 15) // 16 * 16  # noqa: E731
    w = [sd[f"{prefix}.{blk}.{idx}.weight"].shape[0] for blk, idx in CONV_SEQ[::2]][:3]
    segs = [
        [16],
        [w[0]],
        [w[0]],
        [w[1]],
        [w[1]],
        [w[2]],
        [w[2], w[1]],
        [w[1]],
        [w[1], w[0]],
        [w[0]],
    ][i]
    return all(pad(c) % 32 == 0 for c in segs)


def plan_fp8(sd: dict, eps: float, plan: dict | None) -> list[dict | None]:
    """Per exported conv: None (FP16) or {a_scale, q, wscale}. Prints fallbacks."""
    out: list[dict | None] = []
    for prefix, tag in (("net", "main"), ("aux_net", "aux")):
        for i, nm in enumerate(PLAN_NAMES):
            name = f"head.{tag}.{nm}"
            ent = None if plan is None else plan["layers"][name]
            if ent is None or ent["precision"] != "fp8":
                out.append(None)
                continue
            if i == 0 or i == 10 or not fp8_eligible(prefix, i, sd):
                print(f"FP8 -> FP16 fallback: {name} (cin {ent['cin']}, cout {ent['cout']})")
                out.append(None)
                continue
            blk, idx = CONV_SEQ[i]
            q, wscale = fp8_weights(sd, f"{prefix}.{blk}.{idx}", f"{prefix}.{blk}.{idx + 1}", eps)
            out.append({"a_scale": float(ent["act_scale"]), "q": q, "wscale": wscale})
    return out


def write_blob(
    convs: list[tuple[torch.Tensor, torch.Tensor]], path: Path, fp8: list[dict | None] | None = None
) -> None:
    fp8 = fp8 or [None] * len(convs)
    pad = lambda: b"\0" * ((-len(buf)) % 16)  # noqa: E731
    buf = bytearray(struct.pack("<III", MAGIC, VERSION, len(convs)))
    for (w, b), f8 in zip(convs, fp8, strict=True):
        cout, cin, kh, kw = w.shape
        buf += struct.pack("<IIII", kh, cin, cout, 0 if f8 is None else 1)
        buf += pad()
        taps = w.permute(2, 3, 1, 0).reshape(kh * kw, cin, cout)
        buf += taps.float().numpy().astype(np.float16).tobytes()
        buf += pad()
        buf += b.float().numpy().astype(np.float32).tobytes()
        if f8 is not None:
            buf += pad()
            buf += struct.pack("<f", f8["a_scale"])
            buf += pad()
            q8 = f8["q"].permute(2, 3, 0, 1).reshape(kh * kw, cout, cin).contiguous()
            buf += q8.view(torch.uint8).numpy().tobytes()
            buf += pad()
            buf += f8["wscale"].float().numpy().astype(np.float32).tobytes()
    path.write_bytes(bytes(buf))


def load_state(path: Path, gaussifier_src: Path) -> tuple[dict, float]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Compatibility with the trusted research head_module.pt export.
        sys.path[:0] = [str(gaussifier_src / "src"), str(gaussifier_src / "tools")]
        obj = torch.load(path, map_location="cpu", weights_only=False)
    eps = GN_EPS
    if isinstance(obj, torch.nn.Module):
        for m in obj.modules():
            if hasattr(m, "mean") and hasattr(m, "var") and hasattr(m, "eps"):
                eps = float(m.eps)
        obj = obj.state_dict()
    return {k: v.detach().cpu() for k, v in obj.items()}, eps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gaussifier-src", type=Path, default=Path.home() / "projects/gaussifier")
    ap.add_argument(
        "--plan",
        "--precision-plan",
        type=Path,
        default=None,
        help="precision_plan.json (FP8 layers); default: all FP16",
    )
    ap.add_argument("--plan-name", default="budget_0.03dB")
    a = ap.parse_args()
    sd, eps = load_state(a.head, a.gaussifier_src)
    if any(k.startswith("correction_head.") for k in sd):
        sd = {
            k[len("correction_head.") :]: v
            for k, v in sd.items()
            if k.startswith("correction_head.")
        }
    if "net.net.enc_blocks.0.0.weight" in sd:
        # The release state dict was saved from the research HeadWrapper.
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
    convs = net_convs(sd, "net", eps) + net_convs(sd, "aux_net", eps)
    plan = json.loads(a.plan.read_text())["plans"][a.plan_name] if a.plan else None
    fp8 = plan_fp8(sd, eps, plan)
    write_blob(convs, a.out, fp8)
    n8 = sum(f is not None for f in fp8)
    print(
        f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {len(convs)} convs, {n8} FP8"
        f"{f' [{a.plan_name}]' if plan else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
