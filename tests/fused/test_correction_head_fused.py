"""Fused correction head vs the PyTorch frozen-GN head: accuracy on real K-loop inputs, latency.

Real inputs: each Kodak image runs through the gaussifier v0.11.1 model (forward map + K-loop
render) with the frozen-GN head installed; a forward pre-hook on the head captures every
17-channel correction input. The fused head (FP16 storage, FP32 accumulation) is compared with
the FP32 PyTorch head on the same inputs, under both schedules (small / large image), with the
aux net padded and native, with the packed FP16 input (whole, or with the per-state half
re-gathered from FP32), and on crops with partial edge tiles.
Latency is measured eagerly and under CUDA-graph replay at 512x512, 512x768 and 1024x1536.

    uv run python tests/fused/test_correction_head_fused.py --head HEAD_MODULE.pt \\
        --images KODAK_DIR [--n-images 6] [--plan PLAN.json] [--tune]

Under pytest: GFUSED_HEAD_MODULE and GFUSED_KODAK (plus GFUSED_PRECISION_PLAN for the FP8
blobs), see fused_env.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

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

# head_tuning.cuh knobs
SCHEDULE, AUX_NATIVE, LAYER_CFG, N_CFGS = 0, 1, 100, 31
# (name, knobs, crop); crops have widths that are not multiples of 64 (partial edge tiles at
# the coarse levels). Kodak (512 x 768) runs the small-image schedule by default.
VARIANTS = (
    ("default", {}, None),
    ("large schedule", {SCHEDULE: 1}, None),
    ("padded aux", {AUX_NATIVE: 0}, None),
    ("packed input", {"packed": True}, None),
    ("packed static half", {"packed": False}, None),
    ("crop 400x464", {}, (400, 464)),
    ("crop 496x752 large schedule", {SCHEDULE: 1}, (496, 752)),
    ("crop 496x752 padded aux", {AUX_NATIVE: 0}, (496, 752)),
)


def build():
    return build_extension(
        "gfused_head_test",
        [HERE / "head_binding.cu", FUSED_SRC / "correction_head.cu"],
        ["-lineinfo"],
    )


def split_inputs(x: torch.Tensor) -> list[torch.Tensor]:
    """[1, 17, H, W] head input -> the fused interface's six FP32 buffers."""
    x = x[0].float()
    return [
        x[0:3].permute(1, 2, 0).contiguous(),
        x[3].contiguous(),
        x[4:7].contiguous(),
        x[7:10].contiguous(),
        x[10:13].contiguous(),
        x[16].contiguous(),
    ]


def pack_inputs(ins: list[torch.Tensor], dynamic: bool) -> torch.Tensor:
    """Six FP32 buffers -> the packed [2, H, W, 8] FP16 unit planes (contract C1). Without
    `dynamic`, unit 1 halves 0..3 are garbage placeholders (the head must re-gather them)."""
    img, dens, lc, rgb, ren, cnt = ins
    u0 = torch.cat([img, dens[..., None], lc.permute(1, 2, 0), rgb[0][..., None]], -1)
    rc = torch.cat([ren.permute(1, 2, 0), cnt[..., None]], -1)
    if not dynamic:
        rc = torch.full_like(rc, 7.0)
    u1 = torch.cat([rc, rgb[1:].permute(1, 2, 0), torch.zeros_like(rgb[:2].permute(1, 2, 0))], -1)
    return torch.stack([u0, u1]).half().contiguous()


def bench(fn, reps=9, n=20):
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


def bench_graph(fn, reps=9, n=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return bench(g.replay, reps, n)


def random_inputs(H: int, W: int, dev) -> list[torch.Tensor]:
    g = torch.Generator(device=dev).manual_seed(0)
    return [
        torch.rand(H, W, 3, device=dev, generator=g),
        torch.rand(H, W, device=dev, generator=g) * 4,
        torch.randn(3, H, W, device=dev, generator=g) - 4,
        torch.rand(3, H, W, device=dev, generator=g),
        torch.rand(3, H, W, device=dev, generator=g),
        (torch.rand(H, W, device=dev, generator=g) < 0.1).float(),
    ]


def tune(fused, sizes: str) -> None:
    """Per-layer sweep of the conv_fused tile configs (knob LAYER_CFG + net * 10 + layer)."""
    for size in sizes.split(","):
        H, W = map(int, size.split("x"))
        fused.prepare(H, W)
        ins = random_inputs(H, W, torch.device("cuda"))
        print(f"{size} first conv: {fused.time_layer(ins, 0, 0, 20) * 1e3:.1f} us")
        for net in (0, 1):
            for layer in range(1, 10):
                key, res = LAYER_CFG + net * 10 + layer, []
                for c in range(N_CFGS):
                    if fused.set_knob(key, c):
                        t = fused.time_layer(ins, net, layer, 20)
                        if t > 0:
                            res.append((t * 1e3, c))
                fused.set_knob(key, -1)
                res.sort()
                tuned = fused.time_layer(ins, net, layer, 20) * 1e3
                print(
                    f"  net{net} layer{layer}: tuned {tuned:.1f} us | "
                    + " ".join(f"c{c}:{t:.1f}" for t, c in res),
                    flush=True,
                )


def export(
    head: Path, out: Path, gsrc: Path, plan: Path | None = None, plan_name: str | None = None
) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/export_fused_head.py"),
        "--head",
        str(head),
        "--out",
        str(out),
        "--gaussifier-src",
        str(gsrc),
    ]
    if plan is not None:
        cmd += ["--plan", str(plan), "--plan-name", plan_name]
    log = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    print(log, end="")
    return log


def fake_quant_head(ref_head, plan: dict, fp8_layers: list[str]):
    """PyTorch W8A8 fake-quant head (gaussifier tools/fp8_sensitivity.py QConv): FP8 layers
    as the fused runtime runs them (static plan act scales), every other conv FP16."""
    import fp8_sensitivity as fq

    head = copy.deepcopy(ref_head)
    layers = fq._fold_unet(head.net, "head.main.") | fq._fold_unet(head.aux_net, "head.aux.")
    scales = {n: v["act_scale"] for n, v in plan["layers"].items()}
    fq.set_plan(layers, dict.fromkeys(fp8_layers, "w8a8"), "static", scales)
    return head.eval()


def capture_inputs(images: list[Path], head_path: Path, gsrc: Path, dev) -> list[torch.Tensor]:
    h = import_gaussifier(gsrc)
    model, cfg = h.load(dev, "module", module_path=head_path)
    cfg.optimizer.checkpoint_correction_iters = False
    xs: list[torch.Tensor] = []
    model.correction_head.register_forward_pre_hook(lambda m, a: xs.append(a[0].detach().clone()))
    with torch.inference_mode():
        for f in images:
            h.run_image(model, cfg, h.read_rgb(f).to(dev), seed=0)
    del model
    return xs


def check_variants(fused, ref_head, xs: list[torch.Tensor], n_images: int, dev) -> None:
    """Every schedule variant vs the FP32 head; must stay within ~2x the PyTorch FP16 head's
    error."""
    ref16_head = copy.deepcopy(ref_head).half()
    for name, knobs, crop in VARIANTS:
        knobs = dict(knobs)
        packed = knobs.pop("packed", None)  # None: FP32 gather; True / False: dynamic
        for k, v in knobs.items():
            assert fused.set_knob(k, v)
        worst, worst_mean, worst16 = 0.0, 0.0, 0.0
        for x in xs:
            if crop is not None:
                h0, w0 = crop if x.shape[-2] < x.shape[-1] else crop[::-1]
                x = x[..., :h0, :w0].contiguous()
            assert (x[0, 13:16] - (x[0, 0:3] - x[0, 10:13])).abs().max() < 1e-5, "diff"
            H, W = x.shape[-2:]
            fused.prepare(H, W)
            ins = split_inputs(x)
            out = torch.empty(8, H, W, device=dev)
            with torch.inference_mode():
                if packed is not None:
                    fused.run_packed(*ins, pack_inputs(ins, packed), packed, out)
                else:
                    fused.run(*ins, out)
                ref = ref_head(x.float())[0]
                o16 = ref16_head(x.half())[0].float()
            scale = ref.abs().amax(dim=(1, 2), keepdim=True)  # per output channel
            err = (out - ref).abs() / scale
            worst = max(worst, err.amax().item())
            worst_mean = max(worst_mean, err.mean().item())
            worst16 = max(worst16, ((o16 - ref).abs() / scale).amax().item())
        print(
            f"[{name}] {n_images} Kodak images x {len(xs) // n_images} K-loop calls: "
            f"max rel err {worst:.2e} (per-channel max-abs scale), worst mean "
            f"{worst_mean:.2e}; PyTorch FP16 head: max rel err {worst16:.2e}"
        )
        for k in knobs:
            fused.set_knob(k, -1)
        assert worst < 2e-2 and worst < 2 * worst16 + 1e-3, (name, worst, worst16)


def export_plans(head: Path, blob: Path, gsrc: Path, plan: Path, names: list[str], ext):
    """name -> (runner, FP8 layers actually run, FP16 fallbacks)."""
    plans = {}
    for name in names:
        pblob = blob.with_name(f"{blob.stem}_{name}.gfch")
        log = export(head, pblob, gsrc, plan, name)
        fallback = [
            ln.split(": ")[1].split(" ")[0]
            for ln in log.splitlines()
            if ln.startswith("FP8 -> FP16 fallback")
        ]
        layers = json.loads(plan.read_text())["plans"][name]["layers"]
        fp8 = sorted(
            n
            for n, v in layers.items()
            if n.startswith("head.") and v["precision"] == "fp8" and n not in fallback
        )
        plans[name] = (ext.CorrectionHead(str(pblob)), fp8, fallback)
        print(f"[{name}] FP8 layers: {', '.join(fp8)}; FP16 fallback: {', '.join(fallback)}")
    return plans


def check_fp8(plans, plan: Path, ref_head, xs: list[torch.Tensor], dev) -> None:
    for name, (runner, fp8, _) in plans.items():
        fq = fake_quant_head(ref_head, json.loads(plan.read_text())["plans"][name], fp8)
        worst, worst32, worst16, mean_fq, mean16 = 0.0, 0.0, 0.0, 0.0, 0.0
        for x in xs:
            H, W = x.shape[-2:]
            runner.prepare(H, W)
            out = torch.empty(8, H, W, device=dev)
            with torch.inference_mode():
                runner.run(*split_inputs(x), out)
                ref = ref_head(x.float())[0]
                rq = fq(x.float())[0]
            scale = ref.abs().amax(dim=(1, 2), keepdim=True)
            e_fq, e16 = (out - rq).abs() / scale, (rq - ref).abs() / scale
            worst, mean_fq = max(worst, e_fq.amax().item()), max(mean_fq, e_fq.mean().item())
            worst32 = max(worst32, ((out - ref).abs() / scale).amax().item())
            worst16, mean16 = max(worst16, e16.amax().item()), max(mean16, e16.mean().item())
        # FP8 rounding flips (e4m3 steps are ~6%) make max errors noisy; the fused head must
        # carry the same quantization error as the fake-quant model (vs FP32) and stay well
        # inside it (vs fake-quant).
        print(
            f"[{name}] fused FP8 vs PyTorch fake-quant head: max rel err {worst:.2e}, "
            f"worst mean {mean_fq:.2e} | fused vs FP32 head: max {worst32:.2e} | "
            f"fake-quant vs FP32: max {worst16:.2e}, worst mean {mean16:.2e}"
        )
        assert worst32 < 1.25 * worst16 + 5e-3 and mean_fq < 0.5 * mean16, (
            name,
            worst,
            worst32,
            worst16,
            mean_fq,
            mean16,
        )


def latency(runners: dict, ref_head, sizes: str, dev) -> None:
    ref16 = ref_head.half().to(memory_format=torch.channels_last)
    for size in sizes.split(","):
        H, W = map(int, size.split("x"))
        ins = random_inputs(H, W, dev)
        out = torch.empty(8, H, W, device=dev)
        line = []
        for name, r in runners.items():
            r.prepare(H, W)
            fn = partial(r.run, *ins, out)
            line.append(f"{name}: eager {bench(fn):.3f} / graph {bench_graph(fn):.3f} ms")
        x16 = torch.rand(1, 17, H, W, device=dev).half().to(memory_format=torch.channels_last)
        with torch.inference_mode():
            t_torch = bench(partial(ref16, x16), reps=5, n=10)
        print(
            f"fused head {H}x{W}: {' | '.join(line)} | (torch eager fp16 cudnn: {t_torch:.2f} ms)",
            flush=True,
        )


def run_checks(
    head: Path,
    images: list[Path],
    gsrc: Path,
    plan: Path | None = None,
    plan_names: tuple[str, ...] = ("budget_0.03dB",),
    accuracy: bool = True,
    variants: bool = True,
    sizes: str | None = None,
    tune_sizes: str | None = None,
) -> None:
    dev = torch.device("cuda")
    import_gaussifier(gsrc)  # the head module pickle references head_arch_screen
    blob = work_dir() / "correction_head.gfch"
    export(head, blob, gsrc)
    ext = build()
    fused = ext.CorrectionHead(str(blob))
    ref_head = torch.load(head, map_location=dev, weights_only=False).float().eval()
    plans = export_plans(head, blob, gsrc, plan, list(plan_names), ext) if plan else {}
    if tune_sizes:
        tune(next(iter(plans.values()))[0] if plans else fused, tune_sizes)
        return
    if accuracy:
        xs = capture_inputs(images, head, gsrc, dev)
        if variants:
            check_variants(fused, ref_head, xs, len(images), dev)
        check_fp8(plans, plan, ref_head, xs, dev)
    if sizes:
        latency({"fp16": fused} | {n: r for n, (r, _, _) in plans.items()}, ref_head, sizes, dev)


N_IMAGES = int(os.environ.get("GFUSED_N_IMAGES", "2"))


def test_correction_head_accuracy():
    head, kodak = require("GFUSED_HEAD_MODULE", "GFUSED_KODAK")
    run_checks(head, sorted(kodak.glob("*.png"))[:N_IMAGES], require_gaussifier())


def test_correction_head_fp8_plan():
    head, kodak, plan = require("GFUSED_HEAD_MODULE", "GFUSED_KODAK", "GFUSED_PRECISION_PLAN")
    run_checks(
        head,
        sorted(kodak.glob("*.png"))[:N_IMAGES],
        require_gaussifier(),
        plan,
        (os.environ.get("GFUSED_PLAN", "budget_0.03dB"),),
        variants=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True, help="pickled frozen-GN head module")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--gaussifier-src", type=Path, default=GAUSSIFIER_SRC)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--sizes", default="512x512,512x768,1024x1536")
    ap.add_argument("--tune", action="store_true", help="per-layer tile-config sweep only")
    ap.add_argument("--plan", type=Path, default=None, help="precision_plan.json (FP8 blobs)")
    ap.add_argument("--plan-names", default="budget_0.03dB,budget_0.1dB")
    a = ap.parse_args()
    run_checks(
        a.head,
        sorted(a.images.glob("*.png"))[: a.n_images],
        a.gaussifier_src,
        a.plan,
        tuple(a.plan_names.split(",")),
        accuracy=not a.skip_accuracy,
        sizes=a.sizes,
        tune_sizes=a.sizes if a.tune else None,
    )


if __name__ == "__main__":
    main()
