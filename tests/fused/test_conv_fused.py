"""Fused conv kernel vs PyTorch: correctness on every forward-map and head layer shape, timing.

    uv run python tests/fused/test_conv_fused.py            # correctness + tuned timing table
    uv run python tests/fused/test_conv_fused.py --sweep    # also print every config's time
    uv run python tests/fused/test_conv_fused.py --only aux --hw 1024x1536
                                     # native-width (24-channel) aux layers vs 32-padded
    uv run pytest tests/fused/test_conv_fused.py            # correctness only (needs CUDA)

Checks launch_conv2 on every layer for every applicable tile config (FP16 and FP8), the fused
conv2 + 1x1-projection path for every residual block, the single-kernel residual block
(conv_block.cuh) and row-band launches. Timings are CUDA-graph replays (no Python launch
overhead), median over repeats, next to cuDNN on the materialised input.
"""

from __future__ import annotations

import argparse
import statistics
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from fused_env import HERE, build_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = build_extension("gfused_conv_test", [HERE / "conv_binding.cu"])
    return _ext


def to_x(in0, in1, mode):
    x = in0.float().permute(2, 0, 1)[None]
    if mode == 1:
        x = F.max_pool2d(x, 2)
    elif mode in (2, 6):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = in1.float().permute(2, 0, 1)[None]
        if mode == 6:  # odd decoder level: the U-Net resizes its up2x to the skip's size
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], 1)
    return x


def conv_ref(x, w, ks):
    _, cin, cout = w.shape
    wt = w.float().view(ks, ks, cin, cout).permute(3, 2, 0, 1)
    return F.conv2d(x, wt, None, padding=ks // 2)


def reference(in0, in1, w, b, res, mode, ks, silu, H, W):
    """FP32 reference on NCHW tensors built from the NHWC inputs."""
    y = conv_ref(to_x(in0, in1, mode), w, ks) + b.view(1, -1, 1, 1)
    if res is not None:
        y = y + res.float().permute(2, 0, 1)[None]
    if silu:
        y = F.silu(y)
    return y[0].permute(1, 2, 0)


# (name, mode, ks, H, W, c0, c1, cout) for the forward map at 512 x 768.
H0, W0 = 512, 768
LAYERS = [
    ("stem.conv1 (3->32, cin pad 16)", 0, 3, H0, W0, 16, 0, 32),
    ("stem.conv2", 0, 3, H0, W0, 32, 0, 32),
    ("down0.conv1 pool", 1, 3, H0 // 2, W0 // 2, 32, 0, 64),
    ("down0.proj pool 1x1", 1, 1, H0 // 2, W0 // 2, 32, 0, 64),
    ("down0.conv2", 0, 3, H0 // 2, W0 // 2, 64, 0, 64),
    ("down1.conv1 pool", 1, 3, H0 // 4, W0 // 4, 64, 0, 128),
    ("down1.conv2", 0, 3, H0 // 4, W0 // 4, 128, 0, 128),
    ("down2.conv1 pool", 1, 3, H0 // 8, W0 // 8, 128, 0, 256),
    ("down2.conv2", 0, 3, H0 // 8, W0 // 8, 256, 0, 256),
    ("up0.conv1 upcat 256+128", 2, 3, H0 // 4, W0 // 4, 256, 128, 128),
    ("up0.proj upcat 1x1", 2, 1, H0 // 4, W0 // 4, 256, 128, 128),
    ("up0.conv2", 0, 3, H0 // 4, W0 // 4, 128, 0, 128),
    ("up1.conv1 upcat 128+64", 2, 3, H0 // 2, W0 // 2, 128, 64, 64),
    ("up1.conv2", 0, 3, H0 // 2, W0 // 2, 64, 0, 64),
    ("up2.conv1 upcat 64+32", 2, 3, H0, W0, 64, 32, 32),
    ("up2.conv2", 0, 3, H0, W0, 32, 0, 32),
]

# Residual blocks for the fused conv2 + projection path: (name, X mode, H, W, c0, c1, cout).
BLOCKS = [
    ("stem", 0, H0, W0, 16, 0, 32),
    ("down0", 1, H0 // 2, W0 // 2, 32, 0, 64),
    ("down1", 1, H0 // 4, W0 // 4, 64, 0, 128),
    ("down2", 1, H0 // 8, W0 // 8, 128, 0, 256),
    ("up0", 2, H0 // 4, W0 // 4, 256, 128, 128),
    ("up1", 2, H0 // 2, W0 // 2, 128, 64, 64),
    ("up2", 2, H0, W0, 64, 32, 32),
]


# Correction-head convs at 512 x 768 (cpp_inference/fused/correction_head.cu layer_spec):
# main U-Net widths 32/64/128, aux 24->32 (padded)/48/96; conv 9's OutEpilogue is replaced
# by the default epilogue here.
def head_layers(net, c0, c1, c2):
    H, W = H0, W0
    return [
        (f"{net}.1 {c0}->{c0}", 0, 3, H, W, c0, 0, c0),
        (f"{net}.2 pool {c0}->{c1}", 1, 3, H // 2, W // 2, c0, 0, c1),
        (f"{net}.3 {c1}->{c1}", 0, 3, H // 2, W // 2, c1, 0, c1),
        (f"{net}.4 pool {c1}->{c2}", 1, 3, H // 4, W // 4, c1, 0, c2),
        (f"{net}.5 {c2}->{c2}", 0, 3, H // 4, W // 4, c2, 0, c2),
        (f"{net}.6 upcat {c2}+{c1}", 2, 3, H // 2, W // 2, c2, c1, c1),
        (f"{net}.7 {c1}->{c1}", 0, 3, H // 2, W // 2, c1, 0, c1),
        (f"{net}.8 upcat {c1}+{c0}", 2, 3, H, W, c1, c0, c0),
        (f"{net}.9 {c0}->{c0}", 0, 3, H, W, c0, 0, c0),
    ]


HEAD_LAYERS = head_layers("head.main", 32, 64, 128) + head_layers("head.aux", 32, 48, 96)
# The aux net at its native level-0 width (BK 24 / 8 configs, BN 24).
HEAD_AUX_NATIVE = head_layers("head.aux24", 24, 48, 96)

# Native-width (8-channel K granularity) and odd-BN shapes, including partial edge tiles.
K8_LAYERS = [
    ("k8 plain 3x3 24x40 24->24", 0, 3, 24, 40, 24, 0, 24),
    ("k8 plain 3x3 30x56 40->24", 0, 3, 30, 56, 40, 0, 24),
    ("k8 plain 1x1 22x66 40->48", 0, 1, 22, 66, 40, 0, 48),
    ("k8 pool 3x3 26x50 24->48", 1, 3, 26, 50, 24, 0, 48),
    ("k8 pool 1x1 26x50 24->24", 1, 1, 26, 50, 24, 0, 24),
    ("k8 upcat 3x3 28x52 48+24", 2, 3, 28, 52, 48, 24, 24),
    ("k8 upcat 3x3 12x36 24+24", 2, 3, 12, 36, 24, 24, 48),
    ("k8 upcat 1x1 20x36 48+24", 2, 1, 20, 36, 48, 24, 24),
    ("bn56 plain 3x3 24x40 16->56", 0, 3, 24, 40, 16, 0, 56),
    ("bn56 plain 3x3 40x64 32->56", 0, 3, 40, 64, 32, 0, 56),
]
K8_BLOCKS = [  # conv2 + fused 1x1 projection with 24-channel segments
    ("k8 plain", 0, 40, 56, 24, 0, 24),
    ("k8 pool", 1, 26, 50, 24, 0, 48),
    ("k8 upcat", 2, 28, 52, 48, 24, 24),
]

# Partial edge tiles: H, W not multiples of 16 / TH (pyramid levels of a W % 16 == 0 image).
EDGE_LAYERS = [
    ("edge plain 3x3 24x40", 0, 3, 24, 40, 32, 0, 32),
    ("edge plain 1x1 22x66", 0, 1, 22, 66, 64, 0, 64),
    ("edge pool 3x3 26x50", 1, 3, 26, 50, 32, 0, 64),
    ("edge upcat 3x3 28x52", 2, 3, 28, 52, 64, 32, 32),
    ("edge upcat 3x3 12x36", 2, 3, 12, 36, 128, 64, 64),
]

# Odd decoder levels (InMode::kUpCatRs, mode 6): the up2x of the floor(H/2) x floor(W/2) input
# resized to H x W, as the U-Net does when an extent is odd (one or both axes; partial tiles).
RS_LAYERS = [
    ("rs 3x3 27x37 64+32", 6, 3, 27, 37, 64, 32, 32),
    ("rs 3x3 34x41 128+64", 6, 3, 34, 41, 128, 64, 64),
    ("rs 3x3 33x48 256+128", 6, 3, 33, 48, 256, 128, 128),
    ("rs 3x3 9x17 32+32", 6, 3, 9, 17, 32, 32, 64),
    ("rs 3x3 45x23 48+24", 6, 3, 45, 23, 48, 24, 24),
    ("rs 1x1 27x37 64+32", 6, 1, 27, 37, 64, 32, 32),
    ("rs 1x1 19x50 48+24", 6, 1, 19, 50, 48, 24, 24),
]
RS_BLOCKS = [
    ("rs up", 6, 27, 37, 64, 32, 32),
    ("rs up k8", 6, 34, 21, 48, 24, 24),
    ("rs up 128", 6, 19, 30, 128, 64, 64),
]


def make(mode, ks, H, W, c0, c1, cout, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    shape0 = {0: (H, W, c0), 1: (2 * H, 2 * W, c0), 2: (H // 2, W // 2, c0)}.get(
        mode, (H // 2, W // 2, c0)
    )
    in0 = r(*shape0).half()
    in1 = r(H, W, c1).half() if mode in (2, 6) else None
    cin = c0 + c1
    w = (r(ks * ks, cin, cout) / (ks * ks * cin) ** 0.5).half()
    b = r(cout).float() * 0.1
    res = r(H, W, cout).half()
    return in0, in1, w, b, res


def bench(fn, reps=9, n=20):
    """Median ms per call of a CUDA-graph replay of n calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        runs.append(a.elapsed_time(b) / n)
    del g
    return statistics.median(runs)


def valid_cfgs(cfgs, c0, c1, cout, pc0=0, pc1=0):
    out = []
    for i, (bk, _th, bn, _wm, _wn) in enumerate(cfgs):
        if (c0 + c1) % bk or c0 % bk or (pc0 + pc1) % bk or pc0 % bk or cout % bn:
            continue
        out.append(i)
    return out


def try_run(fn):
    """Run a launch; None if this config cannot launch (e.g. over the smem limit)."""
    try:
        return fn()
    except RuntimeError as ex:
        if "invalid argument" in str(ex) or "not a multiple" in str(ex):
            return None
        raise


def rel_err(out, ref):
    return (out.float() - ref).abs().max().item() / ref.abs().max().item()


def conv_fn(e, in0, in1, w, b, res, mode, ks, H, W, c):
    """One launch_conv2 call (no projection) as a closure over its arguments."""
    return lambda: e.conv_v2(in0, in1, w, b, res, mode, ks, True, H, W, c, None, None, None, -1)


def cudnn_fn(in0, in1, w, b, mode, ks):
    """cuDNN channels_last FP16 conv on the materialised input (pool / upsample + concat)."""
    c0 = in0.shape[2]
    cout = w.shape[2]
    x0 = in0.permute(2, 0, 1)[None].contiguous(memory_format=torch.channels_last)
    if mode == 1:
        xin = F.max_pool2d(x0, 2)
    elif mode == 2:
        x1 = in1.permute(2, 0, 1)[None].contiguous(memory_format=torch.channels_last)
        xin = torch.cat([F.interpolate(x0, scale_factor=2, mode="bilinear"), x1], 1)
    else:
        xin = x0
    cin = c0 + (in1.shape[2] if mode == 2 else 0)
    wt = w.view(ks, ks, cin, cout).permute(3, 2, 0, 1).contiguous(memory_format=torch.channels_last)
    bh = b.half()
    return lambda: F.conv2d(xin, wt, bh, padding=ks // 2)


def check_layers(e, sweep: bool, timing: bool, layers=None, edge=True):
    cfgs = e.configs()
    rows = []
    worst = 0.0
    best_cfg = {}
    for name, mode, ks, H, W, c0, c1, cout in layers or LAYERS:
        in0, in1, w, b, res = make(mode, ks, H, W, c0, c1, cout)
        ref = reference(in0, in1, w, b, res, mode, ks, True, H, W)
        times = {}
        for c in valid_cfgs(cfgs, c0, c1, cout):
            fn = conv_fn(e, in0, in1, w, b, res, mode, ks, H, W, c)
            out = try_run(fn)
            if out is None:
                continue
            err = rel_err(out, ref)
            worst = max(worst, err)
            assert err < 1e-2, f"{name} cfg {c}: rel err {err:.2e}"
            if timing:
                times[c] = bench(fn)
        if not timing:
            continue
        best = min(times, key=times.get)
        best_cfg[name] = (best, times[best])
        tc = bench(cudnn_fn(in0, in1, w, b, mode, ks))
        rows.append((name, best, times[best], tc))
        if sweep:
            print(f"  {name}: " + "  ".join(f"c{c}={t:.3f}" for c, t in sorted(times.items())))
    for name, mode, ks, H, W, c0, c1, cout in EDGE_LAYERS if edge else []:
        in0, in1, w, b, res = make(mode, ks, H, W, c0, c1, cout)
        ref = reference(in0, in1, w, b, res, mode, ks, True, H, W)
        for c in valid_cfgs(cfgs, c0, c1, cout):
            out = try_run(conv_fn(e, in0, in1, w, b, res, mode, ks, H, W, c))
            if out is None:
                continue
            err = rel_err(out, ref)
            worst = max(worst, err)
            assert err < 1e-2, f"{name} cfg {c}: rel err {err:.2e}"
    print(f"worst rel err over all layers x configs (incl. edge tiles): {worst:.2e}")
    if timing:
        print(
            f"\n{'layer':<32}{'ms':>8}{'cfg':>5}  {'(BK,TH,BN,WM,WN)':<20}{'cuDNN':>7}"
            f"{'cuDNN/fused':>12}"
        )
        t2s = tcs = 0.0
        for name, c, t2, tc in rows:
            t2s, tcs = t2s + t2, tcs + tc
            print(f"{name:<32}{t2:8.3f}{c:5d}  {tuple(cfgs[c])!s:<20}{tc:7.3f}{tc / t2:11.2f}x")
        print(f"{'TOTAL':<32}{t2s:8.3f}{'':27}{tcs:7.3f}{tcs / t2s:11.2f}x")
    return best_cfg


def best_time(fns) -> float:
    """Fastest of the launches that can run (configs over the smem limit are skipped)."""
    return min(bench(fn) for fn in fns if try_run(fn) is not None)


def check_blocks(e, sweep: bool, timing: bool, blocks=None):
    """conv2(y1) + proj(X) fused vs proj kernel + conv2-with-residual."""
    cfgs = e.configs()
    if timing:
        print(f"\n{'block conv2+proj':<18}{'separate ms':>12}{'fused ms':>10}{'cfg':>5}")
    for name, mode, H, W, c0, c1, cout in blocks or BLOCKS:
        in0, in1, wp, _, _ = make(mode, 1, H, W, c0, c1, cout, seed=1)
        y1, _, w2, b2, _ = make(0, 3, H, W, cout, 0, cout, seed=2)
        x = to_x(in0, in1, mode)
        ref = F.silu(
            conv_ref(y1.float().permute(2, 0, 1)[None], w2, 3)
            + b2.view(1, -1, 1, 1)
            + conv_ref(x, wp, 1)
        )[0].permute(1, 2, 0)
        times = {}
        for c in valid_cfgs(cfgs, cout, 0, cout, c0, c1):
            fn = partial(e.conv_v2, y1, None, w2, b2, None, 0, 3, True, H, W, c, in0, in1, wp, mode)
            out = try_run(fn)
            if out is None:
                continue
            err = rel_err(out, ref)
            assert err < 1e-2, f"block {name} cfg {c}: rel err {err:.2e}"
            if timing:
                times[c] = bench(fn)
        if not timing:
            continue
        # separate: best proj config + best conv2-with-residual config
        res = torch.empty(H, W, cout, device="cuda", dtype=torch.half)
        t_proj = best_time(
            partial(
                e.conv_v2, in0, in1, wp, None, None, mode, 1, False, H, W, c, None, None, None, -1
            )
            for c in valid_cfgs(cfgs, c0, c1, cout)
        )
        t_c2 = best_time(
            partial(e.conv_v2, y1, None, w2, b2, res, 0, 3, True, H, W, c, None, None, None, -1)
            for c in valid_cfgs(cfgs, cout, 0, cout)
        )
        best = min(times, key=times.get)
        print(f"{name:<18}{t_proj + t_c2:12.3f}{times[best]:10.3f}{best:5d}")
        if sweep:
            print("  " + "  ".join(f"c{c}={t:.3f}" for c, t in sorted(times.items())))


# Fused residual blocks (conv_block.cuh): (name, X mode, H, W, c0, c1, cm). Mode 4 = image.
FUSED_BLOCKS = [
    ("stem (image)", 4, H0, W0, 16, 0, 32),
    ("down0 (pool)", 1, H0 // 2, W0 // 2, 32, 0, 64),
    ("up1 (upcat)", 2, H0 // 2, W0 // 2, 128, 64, 64),
    ("up2 (upcat)", 2, H0, W0, 64, 32, 32),
    ("edge plain 40x56", 0, 40, 56, 32, 0, 32),
    ("edge upcat 28x52", 2, 28, 52, 64, 32, 32),
]


def check_fused_blocks(e, timing: bool):
    """conv_block vs conv1 kernel + (conv2 + projection) kernel, and vs PyTorch."""
    if timing:
        print(f"\n{'fused block':<20}{'2 kernels ms':>13}{'1 kernel ms':>12}{'cfg':>5}")
    for name, mode, H, W, c0, c1, cm in FUSED_BLOCKS:
        g = torch.Generator(device="cuda").manual_seed(5)
        if mode == 4:
            x0 = torch.rand(H, W, 3, device="cuda", generator=g)
            x16 = torch.zeros(H, W, 16, device="cuda")
            x16[..., :3] = x0
            x = x16.half().float().permute(2, 0, 1)[None]
            in1 = None
        else:
            x0, in1, _, _, _ = make(mode, 3, H, W, c0, c1, cm, seed=6)
            x = to_x(x0, in1, mode).half().float()
        cin = c0 + c1
        w1 = (torch.randn(9, cin, cm, device="cuda", generator=g) / (9 * cin) ** 0.5).half()
        w2 = (torch.randn(9, cm, cm, device="cuda", generator=g) / (9 * cm) ** 0.5).half()
        wp = (torch.randn(1, cin, cm, device="cuda", generator=g) / cin**0.5).half()
        b1 = torch.randn(cm, device="cuda", generator=g) * 0.1
        b2 = torch.randn(cm, device="cuda", generator=g) * 0.1
        y1 = F.silu(conv_ref(x, w1, 3) + b1.view(1, -1, 1, 1)).half().float()
        ref = F.silu(conv_ref(y1, w2, 3) + b2.view(1, -1, 1, 1) + conv_ref(x, wp, 1))
        ref = ref[0].permute(1, 2, 0)
        times = {}
        for c in range(4):
            fn = partial(e.conv_block, x0, in1, w1, b1, w2, b2, wp, mode, H, W, c)
            out = try_run(fn)
            if out is None:
                continue
            err = rel_err(out, ref)
            assert err < 1e-2, f"fused block {name} cfg {c}: rel err {err:.2e}"
            if timing:
                times[c] = bench(fn)
        if not timing or name.startswith("edge"):
            continue
        cfgs = e.configs()
        y1h = y1[0].permute(1, 2, 0).contiguous().half()
        c1s = (5, 10) if mode == 4 else valid_cfgs(cfgs, c0, c1, cm)
        c2s = (5, 10) if mode == 4 else valid_cfgs(cfgs, cm, 0, cm, c0, c1)
        t1 = best_time(
            partial(e.conv_v2, x0, in1, w1, b1, None, mode, 3, True, H, W, c, None, None, None, -1)
            for c in c1s
        )
        t2 = best_time(
            partial(e.conv_v2, y1h, None, w2, b2, None, 0, 3, True, H, W, c, x0, in1, wp, mode)
            for c in c2s
        )
        best = min(times, key=times.get)
        print(f"{name:<20}{t1 + t2:13.3f}{times[best]:12.3f}{best:5d}")
    print("fused residual blocks: ok")


def check_image(e):
    """kImageF32: the stem reads the FP32 H x W x 3 image directly (as 16 channels)."""
    H, W = 64, 80
    g = torch.Generator(device="cuda").manual_seed(3)
    img = torch.rand(H, W, 3, device="cuda", generator=g)
    x16 = torch.zeros(H, W, 16, device="cuda")
    x16[..., :3] = img
    x16 = x16.half()
    _, _, w, b, res = make(0, 3, H, W, 16, 0, 32)
    ref = reference(x16, None, w, b, res, 0, 3, True, H, W)
    for c in (5, 10):
        err = rel_err(
            e.conv_v2(img, None, w, b, res, 4, 3, True, H, W, c, None, None, None, -1), ref
        )
        assert err < 1e-2, f"image conv cfg {c}: {err:.2e}"
    y1, _, w2, b2, _ = make(0, 3, H, W, 32, 0, 32, seed=4)
    wp = (torch.randn(1, 16, 32, device="cuda") / 4).half()
    ref = F.silu(
        conv_ref(y1.float().permute(2, 0, 1)[None], w2, 3)
        + b2.view(1, -1, 1, 1)
        + conv_ref(x16.float().permute(2, 0, 1)[None], wp, 1)
    )[0].permute(1, 2, 0)
    for c in (5, 10):
        err = rel_err(e.conv_v2(y1, None, w2, b2, None, 0, 3, True, H, W, c, img, None, wp, 4), ref)
        assert err < 1e-2, f"image proj cfg {c}: {err:.2e}"
    print("kImageF32 (conv + fused projection): ok")


E4M3_MAX = 448.0


def fq(x, scale):
    """Fake-quantise to e4m3 with the given scale (broadcast), saturating."""
    return (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


def quant_weights(w):
    """[taps, cin, cout] -> (e4m3 bytes [taps, cout, cin], per-cout scale, fake-quant w)."""
    wf = w.float()
    ws = (wf.abs().amax(dim=(0, 1)) / E4M3_MAX).clamp_min(1e-12)
    w8 = (wf / ws).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return w8.permute(0, 2, 1).contiguous().view(torch.uint8), ws.contiguous(), w8.float() * ws


def check_fp8(e, layers, fp16_best, timing: bool, blocks=None):
    """FP8 main segment vs a fake-quant PyTorch reference; speedup vs the best FP16 config."""
    f8 = e.f8_configs()
    if timing:
        print(
            f"\n{'FP8 layer':<32}{'err fq':>8}{'err fp16':>9}{'fp16 ms':>9}{'fp8 ms':>8}"
            f"{'cfg':>5}{'speedup':>9}"
        )
    t16s = t8s = 0.0
    worst = 0.0
    for name, mode, ks, H, W, c0, c1, cout in layers:
        if ks != 3 or (c0 + c1) % 32 or c0 % 32:
            continue
        in0, in1, w, b, res = make(mode, ks, H, W, c0, c1, cout)
        x = to_x(in0, in1, mode).half().float()  # the kernel quantises the FP16 input
        a_scale = x.abs().max().item() / E4M3_MAX
        w8, ws, wq = quant_weights(w)
        ref = conv_ref(fq(x, a_scale), wq, 3) + b.view(1, -1, 1, 1)
        ref = F.silu(ref + res.float().permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        ref16 = reference(in0, in1, w, b, res, mode, ks, True, H, W)
        times = {}
        for c, (bk, _th, bn, _wm, _wn) in enumerate(f8):
            if (c0 + c1) % bk or c0 % bk or cout % bn:
                continue
            fn = partial(
                e.conv_v2_f8,
                in0,
                in1,
                w,
                w8,
                ws,
                a_scale,
                b,
                res,
                mode,
                ks,
                True,
                H,
                W,
                c,
                None,
                None,
                None,
                -1,
            )
            out = try_run(fn)
            if out is None:
                continue
            err = rel_err(out, ref)
            worst = max(worst, err)
            assert err < 1e-2, f"FP8 {name} cfg {c}: rel err vs fake-quant {err:.2e}"
            if timing:
                times[c] = (bench(fn), rel_err(out, ref16), err)
        if timing and times and name in fp16_best:
            best = min(times, key=lambda k: times[k][0])
            t8, e16, efq = times[best]
            t16 = fp16_best[name][1]
            t16s, t8s = t16s + t16, t8s + t8
            print(f"{name:<32}{efq:8.1e}{e16:9.1e}{t16:9.3f}{t8:8.3f}{best:5d}{t16 / t8:8.2f}x")
    if timing and t8s:
        print(
            f"{'TOTAL (FP8-eligible layers)':<32}{'':17}{t16s:9.3f}{t8s:8.3f}{'':5}"
            f"{t16s / t8s:8.2f}x"
        )
    # FP8 conv2 + FP16 fused projection (the dequant scale is applied before the projection)
    for name, mode, H, W, c0, c1, cout in blocks or BLOCKS:
        if cout % 32 or (c0 + c1) % 32 or c0 % 32:
            continue
        in0, in1, wp, _, _ = make(mode, 1, H, W, c0, c1, cout, seed=1)
        y1, _, w2, b2, _ = make(0, 3, H, W, cout, 0, cout, seed=2)
        yf = y1.float().permute(2, 0, 1)[None]
        a_scale = yf.abs().max().item() / E4M3_MAX
        w8, ws, wq = quant_weights(w2)
        ref = F.silu(
            conv_ref(fq(yf, a_scale), wq, 3)
            + b2.view(1, -1, 1, 1)
            + conv_ref(to_x(in0, in1, mode), wp, 1)
        )[0].permute(1, 2, 0)
        for c, (bk, _th, bn, _wm, _wn) in enumerate(f8):
            if cout % bk or (c0 + c1) % bk or c0 % bk or cout % bn:
                continue
            out = try_run(
                partial(
                    e.conv_v2_f8,
                    y1,
                    None,
                    w2,
                    w8,
                    ws,
                    a_scale,
                    b2,
                    None,
                    0,
                    3,
                    True,
                    H,
                    W,
                    c,
                    in0,
                    in1,
                    wp,
                    mode,
                )
            )
            if out is None:
                continue
            err = rel_err(out, ref)
            worst = max(worst, err)
            assert err < 1e-2, f"FP8 block {name} cfg {c}: rel err {err:.2e}"
    print(f"FP8 worst rel err vs fake-quant reference (incl. FP8 conv2 + FP16 proj): {worst:.2e}")


def check_bands(e):
    """Row-band launches (ConvParams::row0 / rows) are bit-identical to one full launch."""
    cfgs = e.configs()
    cases = [  # (mode, ks, H, W, c0, c1, cout, cfg indices)
        (0, 3, 72, 80, 32, 0, 32, (0, 4, 15)),
        (1, 3, 40, 48, 32, 0, 64, (1, 2)),
        (2, 3, 48, 64, 64, 32, 32, (5, 10)),
        (0, 3, 72, 56, 24, 0, 24, (20, 21, 26)),
        (2, 3, 40, 48, 48, 24, 24, (20, 27)),
        (0, 1, 50, 40, 64, 0, 64, (1, 3)),
    ]
    n = 0
    for mode, ks, H, W, c0, c1, cout, cl in cases:
        in0, in1, w, b, res = make(mode, ks, H, W, c0, c1, cout, seed=7)
        for c in cl:
            th = cfgs[c][1]
            full = e.conv_v2(in0, in1, w, b, res, mode, ks, True, H, W, c, None, None, None, -1)
            out = torch.full_like(full, float("nan"))
            step = 2 * th
            r0 = 0
            while r0 < H:
                rows = min(step, H - r0) if r0 + step < H else 0  # last band: rows = 0 (to H)
                assert e.conv_v2_band(in0, in1, w, b, res, mode, ks, True, H, W, c, r0, rows, out)
                r0 += step
            assert torch.equal(out, full), f"band mode {mode} cfg {c}: differs from full launch"
            n += 1
            # misaligned bands are rejected, out-of-range ones are a no-op
            assert not e.conv_v2_band(
                in0, in1, w, b, res, mode, ks, True, H, W, c, th // 2 or 1, th, out
            )
            assert not e.conv_v2_band(in0, in1, w, b, res, mode, ks, True, H, W, c, 0, th + 1, out)
            assert e.conv_v2_band(
                in0, in1, w, b, res, mode, ks, True, H, W, c, H // th * th + th, 0, out
            )
    # fused residual block in bands
    x0, in1, _, _, _ = make(2, 3, 40, 52, 64, 32, 32, seed=8)
    g = torch.Generator(device="cuda").manual_seed(9)
    w1 = (torch.randn(9, 96, 32, device="cuda", generator=g) / 30).half()
    w2 = (torch.randn(9, 32, 32, device="cuda", generator=g) / 17).half()
    wp = (torch.randn(1, 96, 32, device="cuda", generator=g) / 10).half()
    b1 = torch.randn(32, device="cuda", generator=g) * 0.1
    b2 = torch.randn(32, device="cuda", generator=g) * 0.1
    full = e.conv_block(x0, in1, w1, b1, w2, b2, wp, 2, 40, 52, 1)
    out = torch.full_like(full, float("nan"))
    for r0 in range(0, 40, 16):
        e.conv_block_band(
            x0, in1, w1, b1, w2, b2, wp, 2, 40, 52, 1, r0, 16 if r0 + 16 < 40 else 0, out
        )
    assert torch.equal(out, full), "fused block bands differ from full launch"
    print(f"row bands: {n} conv configs + fused block bit-identical to full launches")


def check_aux_native(e, hw=(H0, W0), sweep=False):
    """Aux-net layers at native width 24 vs the current 24 -> 32 padded build: best config of
    each over the whole table, FP16, default epilogue (layer 9's OutEpilogue needs BN == cout,
    so its row also shows the best BN == cout config)."""
    cfgs = e.configs()
    h, w = hw

    def sc(layers):
        return [
            (n, m, k, H * h // H0, W * w // W0, c0, c1, co) for n, m, k, H, W, c0, c1, co in layers
        ]

    pad, nat = sc(head_layers("aux", 32, 48, 96)), sc(head_layers("aux", 24, 48, 96))
    print(f"\naux layers at {h}x{w}: padded (24->32) vs native 24 (best config each, ms)")
    print(
        f"{'layer':<26}{'padded':>8}{'cfg':>4}{'native':>8}{'cfg':>4}{'speedup':>8}"
        f"{'TFLOPS p/n (real)':>19}{'n BN=cout':>10}{'cfg':>4}"
    )
    tp = tn = 0.0
    for (name, mode, ks, H, W, c0, c1, cout), nl in zip(pad, nat, strict=True):
        name, _, _, _, _, n0, n1, nco = nl
        if n0 == c0 and n1 == c1 and nco == cout:
            continue  # layers 3-7 do not change
        res = {}
        for tag, (a0, a1, aco) in (("p", (c0, c1, cout)), ("n", (n0, n1, nco))):
            in0, in1, wt, b, _ = make(mode, ks, H, W, a0, a1, aco)
            ref = reference(in0, in1, wt, b, None, mode, ks, True, H, W)
            times = {}
            for c in valid_cfgs(cfgs, a0, a1, aco):
                fn = conv_fn(e, in0, in1, wt, b, None, mode, ks, H, W, c)
                out = try_run(fn)
                if out is None:
                    continue
                assert rel_err(out, ref) < 1e-2
                times[c] = bench(fn)
            res[tag] = times
            if sweep:
                print(
                    f"  {tag} {name}: "
                    + "  ".join(f"c{c}={t * 1e3:.1f}" for c, t in sorted(times.items()))
                )
        bp = min(res["p"], key=res["p"].get)
        bn = min(res["n"], key=res["n"].get)
        full = {c: t for c, t in res["n"].items() if cfgs[c][2] == nco}
        bf = min(full, key=full.get) if full else None
        real = 2 * H * W * ks * ks * (n0 + n1) * nco
        padf = 2 * H * W * ks * ks * (c0 + c1) * cout
        t_p, t_n = res["p"][bp], res["n"][bn]
        tp, tn = tp + t_p, tn + t_n
        t_full = full[bf] if bf is not None else float("nan")
        tflops = f"{padf / t_p / 1e9:8.0f} /{real / t_n / 1e9:4.0f} ({real / t_p / 1e9:3.0f})"
        print(
            f"{name:<26}{t_p:8.4f}{bp:4d}{t_n:8.4f}{bn:4d}{t_p / t_n:7.2f}x{tflops}"
            f"{t_full:10.4f}{bf if bf is not None else -1:4d}"
        )
    print(f"{'TOTAL (changed layers)':<26}{tp:8.4f}{'':4}{tn:8.4f}{'':4}{tp / tn:7.2f}x")


def test_correctness():
    e = ext()
    check_layers(e, sweep=False, timing=False)
    check_layers(e, sweep=False, timing=False, layers=HEAD_LAYERS, edge=False)
    check_blocks(e, sweep=False, timing=False)
    check_image(e)
    check_fused_blocks(e, timing=False)
    check_fp8(e, LAYERS + HEAD_LAYERS, {}, timing=False)


def test_native_widths():
    """BK an odd multiple of 8 (24 / 8), BN 24 / 48 / 56: layers, edge tiles, projections."""
    e = ext()
    check_layers(e, sweep=False, timing=False, layers=HEAD_AUX_NATIVE + K8_LAYERS, edge=False)
    check_blocks(e, sweep=False, timing=False, blocks=K8_BLOCKS)


def test_row_bands():
    check_bands(ext())


def test_upcat_resize():
    """kUpCatRs (odd decoder levels): every config vs up2x + F.interpolate(size) + concat,
    fused projections, FP8, and row bands bit-identical to a full launch."""
    e = ext()
    check_layers(e, sweep=False, timing=False, layers=RS_LAYERS, edge=False)
    check_blocks(e, sweep=False, timing=False, blocks=RS_BLOCKS)
    check_fp8(e, RS_LAYERS, {}, timing=False, blocks=RS_BLOCKS)
    cfgs = e.configs()
    for H, W, c0, c1, cout, cl in (
        (45, 37, 64, 32, 32, (0, 4, 5, 10)),
        (37, 26, 48, 24, 24, (20, 21)),
    ):
        in0, in1, w, b, res = make(6, 3, H, W, c0, c1, cout, seed=7)
        for c in cl:
            th = cfgs[c][1]
            full = e.conv_v2(in0, in1, w, b, res, 6, 3, True, H, W, c, None, None, None, -1)
            out = torch.full_like(full, float("nan"))
            for r0 in range(0, H, 2 * th):
                rows = 2 * th if r0 + 2 * th < H else 0
                assert e.conv_v2_band(in0, in1, w, b, res, 6, 3, True, H, W, c, r0, rows, out)
            assert torch.equal(out, full), f"rs band cfg {c}: differs from full launch"


def test_acc16():
    """ConvParams::acc16 (FP16-accumulate MMAs, per chunk / whole K): every FP16 config of the
    forward-map and head shapes stays close to the FP32 reference (unit-variance inputs and
    outputs, K up to 9 x 384)."""
    e = ext()
    cfgs = e.configs()
    for mode, tol in ((1, 5e-3), (2, 1e-2)):
        e.set_acc16(mode)
        worst = 0.0
        try:
            for name, m, ks, H, W, c0, c1, cout in LAYERS + HEAD_AUX_NATIVE + K8_LAYERS:
                in0, in1, w, b, res = make(m, ks, H, W, c0, c1, cout)
                ref = reference(in0, in1, w, b, res, m, ks, True, H, W)
                for c in valid_cfgs(cfgs, c0, c1, cout)[:3]:
                    out = try_run(conv_fn(e, in0, in1, w, b, res, m, ks, H, W, c))
                    if out is None:
                        continue
                    err = rel_err(out, ref)
                    worst = max(worst, err)
                    assert err < tol, f"acc16={mode} {name} cfg {c}: rel err {err:.2e}"
        finally:
            e.set_acc16(0)
        print(f"acc16={mode}: worst rel err {worst:.2e}")


def test_single_stage():
    """ConvParams::single_stage: same results with one smem stage on single-chunk layers."""
    e = ext()
    e.set_single_stage(True)
    try:
        check_layers(
            e,
            sweep=False,
            timing=False,
            layers=[
                L for L in HEAD_LAYERS + HEAD_AUX_NATIVE + K8_LAYERS + EDGE_LAYERS if L[1] != 2
            ],
            edge=False,
        )
    finally:
        e.set_single_stage(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="print every config's timing")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--only", choices=["blocks", "aux", "bands"], help="run one section")
    ap.add_argument("--hw", default=None, help="override layer resolution, e.g. 1024x1536")
    ap.add_argument(
        "--single-stage",
        action="store_true",
        help="ConvParams::single_stage = 1 (one smem stage for single-chunk layers)",
    )
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    e = ext()
    e.set_single_stage(a.single_stage)
    t = not a.no_timing
    if a.only == "blocks":
        check_fused_blocks(e, t)
        return
    if a.only == "bands":
        check_bands(e)
        return
    if a.only == "aux":
        hw = tuple(int(v) for v in a.hw.split("x")) if a.hw else (H0, W0)
        check_aux_native(e, hw, a.sweep)
        return
    best = check_layers(e, a.sweep, t) or {}
    print("\ncorrection-head layers:")
    best.update(check_layers(e, a.sweep, t, layers=HEAD_LAYERS, edge=False) or {})
    print("\nnative-width layers (BK 24 / 8, BN 24 / 48 / 56):")
    check_layers(e, a.sweep, False, layers=HEAD_AUX_NATIVE + K8_LAYERS, edge=False)
    check_blocks(e, a.sweep, t)
    check_blocks(e, False, False, blocks=K8_BLOCKS)
    check_bands(e)
    check_image(e)
    check_fused_blocks(e, t)
    check_fp8(e, LAYERS + HEAD_LAYERS, best, t)


if __name__ == "__main__":
    main()
