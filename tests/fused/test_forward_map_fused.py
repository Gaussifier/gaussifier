"""Fused forward map vs the PyTorch frozen-GN model: outputs and latency.

    uv run python tests/fused/test_forward_map_fused.py --state STAGE1.pt --images KODAK_DIR \\
        [--flags 0,1] [--precision-plan PLAN.json [--plan budget_0.03dB] --head-module HEAD.pt]

``--flags`` runs the accuracy check once per ForwardMapWorkspace flags value (forward_map.cuh:
1 = single-kernel stem / down0 blocks at any size, 2 = banded up2 at any size, 4 = never
banded, 16 = up blocks project their upsampled input at full resolution instead of at low
resolution). Every run also checks that the banded up2 schedule (the default from 1024 x 1024)
gives log-cov and rgb outputs bit-identical to the unbanded one, at Kodak size, on a crop with
partial edge tiles and at 1024 x 1536 (density / rate sum use float atomics and are not
bitwise stable).

``--precision-plan`` also exports the FP8-plan blob (GFFM v2) and checks the fused FP8 forward
map against the fake-quant model of the gaussifier repo's tools/fp8_sensitivity.py (same
quantizer, static scales from the JSON) built from the same ``--state``, then times FP16 vs
FP8. fp8_sensitivity builds the whole model, so it also needs the head module. It also checks
that FP8 activation storage (conv1 writes e4m3 for an FP8 conv2; flag 8 turns it off) leaves
log-cov and rgb bit-identical under every schedule, and times it.

Under pytest: GFUSED_STAGE1 and GFUSED_KODAK (plus GFUSED_PRECISION_PLAN and
GFUSED_HEAD_MODULE for the FP8 check), see fused_env.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fused_env import (
    FUSED_SRC,
    GAUSSIFIER_SRC,
    HERE,
    ROOT,
    build_extension,
    import_gaussifier,
    require,
    require_gaussifier,
    work_dir,
)
from PIL import Image

# ForwardMapWorkspace flags (forward_map.cuh)
FUSED_BLOCKS, BANDED, UNBANDED, NO_F8_STORAGE, FULL_RES_PROJ = 1, 2, 4, 8, 16
# Max errors vs the FP32 PyTorch model (the fused map runs FP16 storage, FP32 accumulation).
TOLERANCE = {"density": 2e-2, "log_cov": 1e-2, "rgb": 1e-2, "rate": 1e-2}


def build():
    return build_extension(
        "gfused_fwd_test", [HERE / "fwd_binding.cu", FUSED_SRC / "forward_map.cu"]
    )


def export(state: Path, out: Path, plan: Path | None = None, plan_name: str = "") -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/export_fused_weights.py"),
        "--state",
        str(state),
        "--out",
        str(out),
    ]
    if plan is not None:
        cmd += ["--precision-plan", str(plan), "--plan", plan_name]
    subprocess.run(cmd, check=True)
    return out


def read_image(path: Path) -> torch.Tensor:
    img = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(img).float().div(255).cuda().contiguous()


def bench(fn, reps=7, n=20):
    for _ in range(5):
        fn()
    runs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t) / n * 1e3)
    return statistics.median(runs)


def rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return ((x - y).abs().max() / y.abs().max()).item()


def reference_outputs(model, img: torch.Tensor):
    with torch.inference_mode():
        r = model.forward_map(img.permute(2, 0, 1)[None], derive_decoder_outputs=False)
    return (
        r["density"][0, 0],
        r["decoder_log_covariance"][0],
        r["decoder_rgb"][0],
        r["logits"].double().exp().sum().item(),
    )


def max_errors(fm, cases) -> dict[str, float]:
    """cases: (image, (density, log_cov, rgb, rate sum) reference)."""
    errs = {k: [] for k in TOLERANCE}
    for img, (r_d, r_lc, r_rgb, r_rate) in cases:
        with torch.inference_mode():
            d, lc, rgb, rate = fm.run(img)
        errs["density"].append(rel(d, r_d))
        errs["log_cov"].append(rel(lc, r_lc))
        errs["rgb"].append((rgb - r_rgb).abs().max().item())
        errs["rate"].append(abs(rate.item() - r_rate) / r_rate)
    return {k: max(v) for k, v in errs.items()}


def check_banding(fm, images: list[torch.Tensor]) -> None:
    """Banded vs unbanded up2: log-cov and rgb must be bit-identical."""
    gen = torch.Generator("cuda").manual_seed(0)
    big = torch.rand(1024, 1536, 3, device="cuda", generator=gen)
    inputs = [*images, big]
    outs = {}
    for flags in (BANDED, UNBANDED):
        fm.set_flags(flags)
        with torch.inference_mode():
            outs[flags] = [[t.clone() for t in fm.run(x)[1:3]] for x in inputs]
    fm.set_flags(0)
    same = all(
        torch.equal(u, v)
        for o0, o1 in zip(outs[BANDED], outs[UNBANDED], strict=True)
        for u, v in zip(o0, o1, strict=True)
    )
    print(f"banded vs unbanded up2: log_cov / rgb bit-identical: {same}")
    assert same, "banded up2 changed the outputs"


def check_accuracy(
    state: Path, images: list[Path], gsrc: Path, flags_list=(0,), timing: bool = False
) -> None:
    blob = export(state, work_dir() / "forward_map.gffm")
    fm = build().ForwardMap(str(blob))
    h = import_gaussifier(gsrc)
    model, _ = h.load(torch.device("cuda"), None, stage1_path=state)
    model.float().eval()

    imgs = [read_image(f) for f in images]
    # plus one crop whose pyramid widths (W/2 .. W/8) are not multiples of 16: partial tiles
    imgs.append(imgs[0][:400, :464].contiguous())
    cases = [(img, reference_outputs(model, img)) for img in imgs]
    check_banding(fm, [imgs[0], imgs[-1]])
    for flags in flags_list:
        fm.set_flags(flags)
        errs = max_errors(fm, cases)
        print(
            f"[flags {flags}] {len(images)} images + 400x464 crop: max rel err density "
            f"{errs['density']:.2e}, log_cov {errs['log_cov']:.2e}, rgb abs {errs['rgb']:.2e},"
            f" rate sum {errs['rate']:.2e}"
        )
        for key, tol in TOLERANCE.items():
            assert errs[key] < tol, (flags, key, errs[key])
        if timing:
            for H, W in ((512, 512), (512, 768), (1024, 1536)):
                x = torch.rand(H, W, 3, device="cuda")
                ms = bench(lambda x=x: fm.run(x))
                print(f"[flags {flags}] fused forward map {H}x{W}: {ms:.3f} ms")
    fm.set_flags(0)


def check_f8_storage(fm8, images: list[torch.Tensor], timing: bool = False) -> None:
    """FP8 activation storage on vs off (flag NO_F8_STORAGE): log-cov / rgb bit-identical."""
    gen = torch.Generator("cuda").manual_seed(0)
    big = torch.rand(1024, 1536, 3, device="cuda", generator=gen)
    inputs = [*images, images[0][:400, :464].contiguous(), big]
    same = True
    for base in (0, FUSED_BLOCKS, BANDED):
        outs = {}
        for flags in (base, base | NO_F8_STORAGE):
            fm8.set_flags(flags)
            with torch.inference_mode():
                outs[flags] = [[t.clone() for t in fm8.run(x)[1:3]] for x in inputs]
        same &= all(
            torch.equal(u, v)
            for o0, o1 in zip(outs[base], outs[base | NO_F8_STORAGE], strict=True)
            for u, v in zip(o0, o1, strict=True)
        )
    print(f"FP8 activation storage vs FP16 t_y1: log_cov / rgb bit-identical: {same}")
    assert same, "FP8 activation storage changed the outputs"
    if timing:
        for H, W in ((512, 768), (1024, 1536)):
            x = torch.rand(H, W, 3, device="cuda", generator=gen)
            t = {}
            for flags in (NO_F8_STORAGE, 0):
                fm8.set_flags(flags)
                t[flags] = bench(lambda x=x: fm8.run(x))
            print(
                f"FP8 forward map {H}x{W}: FP16 t_y1 {t[NO_F8_STORAGE]:.3f} ms, e4m3 t_y1 "
                f"{t[0]:.3f} ms"
            )
    fm8.set_flags(0)


def check_fp8_plan(
    state: Path,
    head_module: Path,
    images: list[Path],
    gsrc: Path,
    plan_path: Path,
    plan_name: str,
    timing: bool = False,
) -> None:
    blob16 = export(state, work_dir() / "forward_map.gffm")
    blob8 = export(state, work_dir() / "forward_map.fp8.gffm", plan_path, plan_name)
    ext = build()
    fm16, fm8 = ext.ForwardMap(str(blob16)), ext.ForwardMap(str(blob8))

    import_gaussifier(gsrc)
    import fp8_sensitivity as fs

    fs.STAGE1, fs.HEAD = state, head_module  # fs.build() loads these module-level paths
    plan = json.loads(plan_path.read_text())["plans"][plan_name]
    model, _cfg, layers = fs.build(torch.device("cuda"))
    # stem.conv2 runs FP16 in the fused map: its fused projection reads the 3-channel image
    # (16 padded channels), which the BK-32 FP8 kernel cannot take; FP16 is never worse.
    fp8 = {
        n: "w8a8"
        for n, v in plan["layers"].items()
        if v["precision"] == "fp8" and n.startswith("bb.") and n != "bb.stem.conv2"
    }
    scales = {n: v["act_scale"] for n, v in plan["layers"].items()}
    print(f"plan {plan_name}: backbone FP8 layers ({len(fp8)}): {sorted(fp8)}")

    def ref_outputs(img, quant):
        fs.set_plan(layers, fp8 if quant else {}, "static", scales)
        dens = layers["bb.head.density"]
        b0 = dens.b.clone()
        if quant:
            dens.b += plan["density_bias_offset"]
        out = reference_outputs(model, img)
        dens.b.copy_(b0)
        return out

    fq_orig = fs.fq_e4m3

    def fq_half_first(x, scale):  # the fused map stores activations as FP16 before quantising
        return fq_orig(x.half().float(), scale)

    imgs = [read_image(f) for f in images]
    results = {}
    for tag, fm, quant in (
        ("FP16 map vs fp16 model", fm16, False),
        ("FP8 map vs fake-quant model", fm8, True),
        ("FP8 map vs fake-quant, FP16-rounded act", fm8, "h"),
        ("FP8 map vs fp16 model", fm8, False),
    ):
        fs.fq_e4m3 = fq_half_first if quant == "h" else fq_orig
        try:
            errs = max_errors(fm, [(img, ref_outputs(img, bool(quant))) for img in imgs])
        finally:
            fs.fq_e4m3 = fq_orig
        results[tag] = errs
        print(
            f"{tag}: {len(imgs)} images, max rel err density {errs['density']:.2e}, "
            f"log_cov {errs['log_cov']:.2e}, rgb abs {errs['rgb']:.2e}, rate sum "
            f"{errs['rate']:.2e}"
        )
    # The fused FP8 map must sit closer to the fake-quant model than to the FP16 one.
    exact = results["FP8 map vs fake-quant, FP16-rounded act"]
    vs16 = results["FP8 map vs fp16 model"]
    assert exact["log_cov"] < vs16["log_cov"] and exact["rgb"] < vs16["rgb"], (exact, vs16)
    check_f8_storage(fm8, imgs, timing)
    if timing:
        for H, W in ((512, 512), (512, 768), (1024, 1536)):
            x = torch.rand(H, W, 3, device="cuda")
            t16, t8 = bench(lambda x=x: fm16.run(x)), bench(lambda x=x: fm8.run(x))
            print(f"forward map {H}x{W}: FP16 {t16:.3f} ms, FP8 plan {t8:.3f} ms ({t16 / t8:.2f}x)")


N_IMAGES = int(os.environ.get("GFUSED_N_IMAGES", "4"))


def _kodak() -> list[Path]:
    (kodak,) = require("GFUSED_KODAK")
    return sorted(kodak.glob("*.png"))[:N_IMAGES]


def test_forward_map_accuracy():
    (state,) = require("GFUSED_STAGE1")
    check_accuracy(
        state, _kodak(), require_gaussifier(), flags_list=(0, FUSED_BLOCKS, FULL_RES_PROJ)
    )


def test_forward_map_fp8_plan():
    state, head, plan = require("GFUSED_STAGE1", "GFUSED_HEAD_MODULE", "GFUSED_PRECISION_PLAN")
    check_fp8_plan(
        state,
        head,
        _kodak(),
        require_gaussifier(),
        plan,
        os.environ.get("GFUSED_PLAN", "budget_0.03dB"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--gaussifier-src", type=Path, default=GAUSSIFIER_SRC)
    ap.add_argument("--flags", default="0,1", help="comma-separated ForwardMapWorkspace flags")
    ap.add_argument("--precision-plan", type=Path)
    ap.add_argument("--plan", default="budget_0.03dB")
    ap.add_argument("--head-module", type=Path, help="head module (needed with --precision-plan)")
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    images = sorted(a.images.glob("*.png"))[: a.n_images]
    if a.precision_plan is not None:
        if a.head_module is None:
            ap.error("--precision-plan needs --head-module")
        check_fp8_plan(
            a.state,
            a.head_module,
            images,
            a.gaussifier_src,
            a.precision_plan,
            a.plan,
            timing=not a.no_timing,
        )
        return
    check_accuracy(
        a.state,
        images,
        a.gaussifier_src,
        flags_list=[int(x) for x in a.flags.split(",")],
        timing=not a.no_timing,
    )


if __name__ == "__main__":
    main()
