"""End-to-end check of the torch-free fused runner (gaussifier_fused_infer) against the PyTorch
reference pipeline of the gaussifier repo (render_loop._correction_render_loss) with the same
frozen-GN weights, on Kodak.

    uv run python tests/fused/test_pipeline_fused.py --bin build-fused/gaussifier_fused_infer \
        --state STAGE1.pt --head-module HEAD.pt --kodak DIR --work DIR [--K 7] [--n-images 24]
        [--fm FM.gffm] [--head-weights HEAD.gfch] [--bin-args='--check-head-io']
    ... --latency          (runner latency at 512x512 / 512x768, K 1 / 5 / 7, graph vs eager)
    ... --adaptive-sweep   (adaptive K settings vs fixed K)

Without --fm / --head-weights the blobs are exported from --state / --head-module (FP16).
Reported per state k (mean over images):
  fused          fused runner end to end (fused forward map + sampler + renderer + head)
  fused/refmaps  fused runner fed the PyTorch forward-map outputs (isolates sampler/render/head)
  ref            PyTorch reference, seed 1234
  ref/seed2      PyTorch reference, seed 4321 (placement noise floor)
With --check-head-io in --bin-args the runner verifies on the GPU, before every head call,
that the packed head input equals __float2half of the FP32 buffers; summary["io_mismatch"]
must then be 0.

Under pytest: GFUSED_BIN, GFUSED_STAGE1, GFUSED_HEAD_MODULE and GFUSED_KODAK; optional
GFUSED_FM_BLOB / GFUSED_HEAD_BLOB (else exported), GFUSED_BIN_ARGS, GFUSED_N_IMAGES (default
4), see fused_env.py.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fused_env import (
    GAUSSIFIER_SRC,
    ROOT,
    env_path,
    import_gaussifier,
    require,
    require_gaussifier,
    work_dir,
)

REF_SEED = 1234
REF_SEED2 = 4321


def to_npy(png: Path, out: Path, crop: tuple[int, int] | None = None) -> Path:
    from PIL import Image

    a = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
    if crop is not None:
        h, w = crop
        top, left = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
        a = a[top : top + h, left : left + w]
    np.save(out, np.ascontiguousarray(a.transpose(2, 0, 1)[None]))
    return out


def run_bin(
    binary: Path, fm: Path, head: Path, manifest: Path, K: int, seed: int, extra: list[str]
) -> list[dict]:
    cmd = [
        str(binary),
        "--fm",
        str(fm),
        "--head",
        str(head),
        "--K",
        str(K),
        "--seed",
        str(seed),
        "--manifest",
        str(manifest),
        *extra,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def blobs(a) -> tuple[Path, Path]:
    """Forward-map and head blobs: given, or exported (FP16) from --state / --head-module."""
    fm = a.fm or a.work / "forward_map.gffm"
    if not fm.exists():
        if a.state is None:
            sys.exit("need --fm or --state")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_fused_weights.py"),
                "--state",
                str(a.state),
                "--out",
                str(fm),
            ],
            check=True,
        )
    head = a.head_weights or a.work / "correction_head.gfch"
    if not head.exists():
        if a.head_module is None:
            sys.exit("need --head-weights or --head-module")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_fused_head.py"),
                "--head",
                str(a.head_module),
                "--out",
                str(head),
                "--gaussifier-src",
                str(a.gaussifier_src),
            ],
            check=True,
        )
    return fm, head


class Reference:
    """PyTorch reference: head_arch_screen.load(...) + run_image(...)."""

    def __init__(self, gaussifier_src: Path, state: Path | None, head_module: Path | None):
        import torch

        h = import_gaussifier(gaussifier_src)
        self.h, self.torch = h, torch
        self.dev = torch.device("cuda")
        if state is None:  # retained v0.11.1 GroupNorm model, as shipped
            self.model, self.cfg = h.load(self.dev, None)
        else:
            self.model, self.cfg = h.load(self.dev, "module", None, head_module, state)
        self.cfg.optimizer.checkpoint_correction_iters = False

    def run(self, img_chw: np.ndarray, K: int, seed: int) -> tuple[list[float], int]:
        torch = self.torch
        self.model.correction_iterations = K
        with torch.inference_mode():
            img = torch.from_numpy(img_chw).to(self.dev)
            _l, _r, pts, per_iter, _it, _m = self.h.run_image(self.model, self.cfg, img, seed=seed)
        return [float(v) for v in per_iter], int(pts.shape[0])

    def dump_maps(self, img_chw: np.ndarray, stem: Path) -> None:
        from gaussifier.losses import rate_from_logits

        torch = self.torch
        with torch.inference_mode():
            maps = self.model.forward_map(torch.from_numpy(img_chw).to(self.dev)[None])
            rate = rate_from_logits(maps["logits"]).flatten(1).sum(1)

        def save(suffix, t):
            np.save(f"{stem}{suffix}", t.float().contiguous().cpu().numpy())

        save(".density.npy", maps["density"][0, 0])
        save(".log_cov.npy", maps["decoder_log_covariance"][0])
        save(".rgb.npy", maps["decoder_rgb"][0])
        save(".rate.npy", rate)


def mean_curve(rows: list[list[float]]) -> list[float]:
    return list(np.mean(np.array(rows), axis=0))


def evaluate(a) -> dict:
    if a.state is None or a.head_module is None:
        sys.exit("the reference comparison needs --state and --head-module")
    work = a.work
    (work / "io").mkdir(parents=True, exist_ok=True)
    fm, head = blobs(a)
    # images: (name, npy) from Kodak PNGs (converted, optionally cropped) or a dir of NPYs
    if a.npy_dir is not None:
        images = [(f.stem, f) for f in sorted(a.npy_dir.glob("*.npy"))[: a.n_images]]
    else:
        images = []
        for png in sorted(a.kodak.glob("*.png"))[: a.n_images]:
            npy = work / "io" / f"{png.stem}{a.tag}.npy"
            if not npy.exists():
                to_npy(png, npy, a.crop_hw)
            images.append((png.stem, npy))
    ref = Reference(a.gaussifier_src, a.state, a.head_module)
    rows = []
    manifest, manifest_maps = [], []
    for name, npy in images:
        stem = work / "io" / f"{name}{a.tag}.ref"
        ref.dump_maps(np.load(npy)[0], stem)
        manifest.append(f"{npy} DUMMY -")
        manifest_maps.append(f"{npy} DUMMY - {stem}")
    (work / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (work / "manifest_maps.txt").write_text("\n".join(manifest_maps) + "\n")
    fused = run_bin(a.bin, fm, head, work / "manifest.txt", a.K, REF_SEED, a.bin_args)
    fused_maps = run_bin(
        a.bin, fm, head, work / "manifest_maps.txt", a.K, REF_SEED, [*a.bin_args, "--external-maps"]
    )
    for (name, npy), f, fmaps in zip(images, fused, fused_maps, strict=True):
        img = np.load(npy)[0]
        r1, n1 = ref.run(img, a.K, REF_SEED)
        r2, _ = ref.run(img, a.K, REF_SEED2)
        rows.append(
            {
                "image": name,
                "fused": f["psnr"],
                "fused_refmaps": fmaps["psnr"],
                "ref": r1,
                "ref_seed2": r2,
                "n_fused": f["n"],
                "n_fused_refmaps": fmaps["n"],
                "n_ref": n1,
                "forced_nn": f["forced_nn"],
                "forced_adj": f["forced_adj"],
                "count_error": f["count_error"],
                "io_mismatch": [f.get("io_mismatch", -1), fmaps.get("io_mismatch", -1)],
            }
        )
        print(
            f"{name}: n fused={f['n']} ref={n1} | k1 fused={f['psnr'][0]:.3f} "
            f"refmaps={fmaps['psnr'][0]:.3f} ref={r1[0]:.3f} seed2={r2[0]:.3f} | "
            f"kK fused={f['psnr'][-1]:.3f} refmaps={fmaps['psnr'][-1]:.3f} ref={r1[-1]:.3f}",
            flush=True,
        )
    keys = ["fused", "fused_refmaps", "ref", "ref_seed2"]
    if a.retained_ref:  # the original v0.11.1 GroupNorm model, same images and seed
        del ref
        retained = Reference(a.gaussifier_src, None, None)
        for (_name, npy), row in zip(images, rows, strict=True):
            row["ref_retained"], row["n_ref_retained"] = retained.run(
                np.load(npy)[0], a.K, REF_SEED
            )
        keys.append("ref_retained")
    summary = {}
    for key in keys:
        summary[key] = mean_curve([r[key] for r in rows])

    def d(k1, k2, k):
        return [abs(r[k1][k] - r[k2][k]) for r in rows]

    summary["mean_abs_diff_k1"] = {
        "fused_vs_ref": float(np.mean(d("fused", "ref", 0))),
        "refmaps_vs_ref": float(np.mean(d("fused_refmaps", "ref", 0))),
        "seed2_vs_ref": float(np.mean(d("ref_seed2", "ref", 0))),
    }
    summary["mean_abs_diff_kK"] = {
        "fused_vs_ref": float(np.mean(d("fused", "ref", -1))),
        "refmaps_vs_ref": float(np.mean(d("fused_refmaps", "ref", -1))),
        "seed2_vs_ref": float(np.mean(d("ref_seed2", "ref", -1))),
    }
    summary["n_mean"] = {
        k: float(np.mean([r[k] for r in rows])) for k in ("n_fused", "n_fused_refmaps", "n_ref")
    }
    summary["n_exact_match_refmaps"] = int(sum(r["n_fused_refmaps"] == r["n_ref"] for r in rows))
    summary["n_rel_diff_fused_mean"] = float(
        np.mean([abs(r["n_fused"] - r["n_ref"]) / r["n_ref"] for r in rows])
    )
    summary["forced_nn_mean"] = float(np.mean([r["forced_nn"] for r in rows]))
    summary["count_errors"] = int(sum(r["count_error"] for r in rows))
    # --check-head-io: packed head input bit check; -1 = not run
    io = [v for r in rows for v in r["io_mismatch"]]
    summary["io_mismatch"] = -1 if all(v < 0 for v in io) else int(sum(max(v, 0) for v in io))
    out = {"K": a.K, "rows": rows, "summary": summary}
    (work / f"pipeline_eval_K{a.K}{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    return out


def latency(a) -> dict:
    work = a.work
    (work / "io").mkdir(parents=True, exist_ok=True)
    fm, head = blobs(a)
    png = sorted(a.kodak.glob("*.png"))[0]
    res = {}
    for h, w in ((512, 512), (512, 768)):
        npy = to_npy(png, work / "io" / f"lat_{h}x{w}.npy", crop=(h, w))
        for K in (1, 5, 7):
            for graph in (True, False):
                cmd = [
                    str(a.bin),
                    "--fm",
                    str(fm),
                    "--head",
                    str(head),
                    "--K",
                    str(K),
                    "--repeat",
                    str(a.reps),
                    *a.bin_args,
                    *([] if graph else ["--no-graph"]),
                    str(npy),
                    str(work / "lat_out.npy"),
                ]
                out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
                j = json.loads(out.splitlines()[-1])
                key = f"{h}x{w}_K{K}_{'graph' if graph else 'eager'}"
                res[key] = {
                    k: j[k]
                    for k in (
                        "n",
                        "gpu_ms_median",
                        "wall_ms_median",
                        "first_ms",
                        "graph_nodes",
                        "h2d_ms_median",
                    )
                }
                print(key, res[key], flush=True)
    (work / "latency.json").write_text(json.dumps(res, indent=1))
    return res


ADAPTIVE_SETTINGS = (
    ("fixed K=7", []),
    ("target 39 dB", ["--psnr-target", "39"]),
    ("target 40 dB", ["--psnr-target", "40"]),
    ("target 41 dB", ["--psnr-target", "41"]),
    ("gain < 0.1 dB", ["--min-gain", "0.1"]),
    ("gain < 0.2 dB", ["--min-gain", "0.2"]),
)


def adaptive_sweep(a) -> dict:
    """Adaptive K (CUDA conditional WHILE node) vs fixed K on Kodak at native size."""
    work = a.work
    (work / "io").mkdir(parents=True, exist_ok=True)
    fm, head = blobs(a)
    lines = []
    for png in sorted(a.kodak.glob("*.png"))[: a.n_images]:
        npy = work / "io" / f"{png.stem}.npy"
        if not npy.exists():
            to_npy(png, npy)
        lines.append(f"{npy} DUMMY")
    (work / "manifest_adaptive.txt").write_text("\n".join(lines) + "\n")
    res = {}
    for name, flags in ADAPTIVE_SETTINGS:
        rows = run_bin(
            a.bin,
            fm,
            head,
            work / "manifest_adaptive.txt",
            a.K,
            REF_SEED,
            [*a.bin_args, "--repeat", str(a.reps), *flags],
        )
        ks = [r["k_used"] for r in rows]
        ms = [r["gpu_ms_median"] for r in rows]
        res[name] = {
            "k_hist": {k: ks.count(k) for k in sorted(set(ks))},
            "k_mean": float(np.mean(ks)),
            "psnr_mean": float(np.mean([r["psnr"][-1] for r in rows])),
            "psnr_min": float(np.min([r["psnr"][-1] for r in rows])),
            "ms_mean": float(np.mean(ms)),
            "ms_median": float(np.median(ms)),
        }
        print(name, res[name], flush=True)
    (work / "adaptive_sweep.json").write_text(json.dumps(res, indent=1))
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--kodak", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None, help="frozen-GN stage-1 state_dict")
    ap.add_argument("--head-module", type=Path, default=None, help="frozen-GN head module")
    ap.add_argument("--fm", type=Path, default=None, help="forward-map GFFM blob to use")
    ap.add_argument("--head-weights", type=Path, default=None, help="head GFCH blob to use")
    ap.add_argument("--gaussifier-src", type=Path, default=GAUSSIFIER_SRC)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--adaptive-sweep", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--crop", default=None, help="HxW centre crop")
    ap.add_argument(
        "--npy-dir",
        type=Path,
        default=None,
        help="evaluate these [1,3,H,W] NPYs instead of the Kodak PNGs",
    )
    ap.add_argument("--tag", dest="set_tag", default="", help="suffix for output files")
    ap.add_argument(
        "--retained-ref",
        action="store_true",
        help="also run the retained v0.11.1 GroupNorm PyTorch model",
    )
    ap.add_argument("--bin-args", default="", help="extra runner flags, e.g. '--check-head-io'")
    a = ap.parse_args(argv)
    a.bin_args = a.bin_args.split()
    a.crop_hw = tuple(int(v) for v in a.crop.split("x")) if a.crop else None
    a.tag = (f"_{a.crop}" if a.crop else "") + (f"_{a.set_tag}" if a.set_tag else "")
    return a


def reference_args(work: Path) -> list[str]:
    binary, state, head_module, kodak = require(
        "GFUSED_BIN", "GFUSED_STAGE1", "GFUSED_HEAD_MODULE", "GFUSED_KODAK"
    )
    gsrc = require_gaussifier()
    argv = [
        "--bin",
        str(binary),
        "--state",
        str(state),
        "--head-module",
        str(head_module),
        "--kodak",
        str(kodak),
        "--work",
        str(work),
        "--gaussifier-src",
        str(gsrc),
        "--n-images",
        os.environ.get("GFUSED_N_IMAGES", "4"),
        f"--bin-args={os.environ.get('GFUSED_BIN_ARGS', '--check-head-io')}",
    ]
    for flag, name in (("--fm", "GFUSED_FM_BLOB"), ("--head-weights", "GFUSED_HEAD_BLOB")):
        if env_path(name) is not None:
            argv += [flag, str(env_path(name))]
    return argv


def test_pipeline_fused():
    result = evaluate(parse(reference_args(work_dir() / "pipeline")))
    s = result["summary"]
    assert s["count_errors"] == 0
    assert s["io_mismatch"] <= 0  # -1: not checked; 0: packed head input bit-exact
    # Same maps: only placement/accumulation-order differences remain.
    assert s["n_exact_match_refmaps"] == len(result["rows"])
    assert s["mean_abs_diff_k1"]["refmaps_vs_ref"] < 0.05
    assert abs(s["fused"][-1] - s["ref"][-1]) < 0.03


def test_pipeline_odd_sizes(tmp_path):
    from PIL import Image

    args = parse(reference_args(tmp_path))
    args.bin_args.extend(["--size-mode", "exact"])
    args.npy_dir = tmp_path / "images"
    args.npy_dir.mkdir()
    # Odd at every level, mixed, thin strips; multiples of 8 (no resize, partial 16-tiles);
    # a multiple of 4 plus 2 (one odd head level).
    shapes = [(511, 767), (333, 500), (17, 900), (1000, 17), (264, 520), (518, 764)]
    args.n_images = len(shapes)
    for png, (h, w) in zip(sorted(args.kodak.glob("*.png"))[: len(shapes)], shapes, strict=True):
        image = Image.open(png).convert("RGB")
        if image.width < w or image.height < h:
            image = image.resize((max(w, image.width), max(h, image.height)))
        x, y = (image.width - w) // 2, (image.height - h) // 2
        data = np.asarray(image.crop((x, y, x + w, y + h)), dtype=np.float32) / 255
        np.save(args.npy_dir / f"{h}x{w}.npy", np.ascontiguousarray(data.transpose(2, 0, 1)[None]))

    summary = evaluate(args)["summary"]
    assert summary["count_errors"] == 0
    assert summary["io_mismatch"] <= 0
    assert summary["n_exact_match_refmaps"] == len(shapes)
    assert summary["n_rel_diff_fused_mean"] < 0.002
    # Thin strips are sensitive to placement (the seed2 curve measures that noise).
    assert summary["mean_abs_diff_k1"]["refmaps_vs_ref"] < 0.05
    assert summary["mean_abs_diff_kK"]["refmaps_vs_ref"] < 0.05
    assert abs(summary["fused"][-1] - summary["ref"][-1]) < max(
        0.03, summary["mean_abs_diff_kK"]["seed2_vs_ref"]
    )


def test_pipeline_size_modes(tmp_path):
    binary, fm, head = require("GFUSED_BIN", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    rng = np.random.default_rng(42)
    for h, w in [(512, 512), (720, 1280), (1080, 1920)]:
        image = rng.random((3, h, w), dtype=np.float32)
        path, render = tmp_path / "image.npy", tmp_path / "render.npy"
        np.save(path, image)
        manifest = tmp_path / "manifest.txt"
        manifest.write_text(f"{path} {render} {tmp_path / 'g.npy'}\n")

        results = []
        for extra in ([], ["--size-mode", "pad"], ["--size-mode", "exact"]):
            row = run_bin(binary, fm, head, manifest, 7, REF_SEED, extra)[0]
            actual = np.load(render)
            points = np.load(tmp_path / "g.points.npy")
            assert actual.shape == image.shape
            assert points.shape == (row["n"], 2)
            assert np.all((points >= 0) & (points <= 1))
            assert row["count_error"] == 0 and row["n"] == row["n_requested"]
            psnr = -10 * np.log10(max(np.mean(np.square(actual - image)), 1e-10))
            assert psnr == pytest.approx(row["psnr"][-1], abs=1e-4)
            results.append(row)
        default, pad, exact = results
        for other in (pad, exact) if h % 16 == w % 16 == 0 else (exact,):
            assert default["n"] == other["n"]
            assert default["graph_nodes"] == other["graph_nodes"]
            # Independent sampler launches can accumulate cell mass in different orders.
            np.testing.assert_allclose(default["psnr"], other["psnr"], atol=0.02, rtol=0)
        if h % 16:
            assert pad["graph_nodes"] != exact["graph_nodes"]


@pytest.mark.parametrize("size_mode", ["exact", "pad"])
def test_pipeline_fp8_unaligned(tmp_path, size_mode):
    from PIL import Image, ImageOps

    binary, kodak = require("GFUSED_BIN", "GFUSED_KODAK")
    weights = ROOT / "src/gaussifier_sampler/weights/frozen_gn"
    png = sorted(kodak.glob("*.png"))[0]
    source = Image.open(png).convert("RGB")
    shapes = [(518, 764), (264, 520), (1080, 1920)]
    manifest = tmp_path / "manifest.txt"
    paths = []
    for h, w in shapes:
        image = ImageOps.fit(source, (w, h), Image.Resampling.BICUBIC)
        data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
        path = tmp_path / f"{h}x{w}.npy"
        np.save(path, np.ascontiguousarray(data))
        paths.append(path)
    manifest.write_text("".join(f"{path} DUMMY\n" for path in paths))

    fp16, fp8 = (
        run_bin(
            binary,
            weights / f"fm_{p}.gffm",
            weights / f"head_{p}.gfch",
            manifest,
            7,
            REF_SEED,
            ["--size-mode", size_mode, "--check-head-io"],
        )
        for p in ("fp16", "fp8")
    )
    for (h, w), ref, mixed in zip(shapes, fp16, fp8, strict=True):
        for row in (ref, mixed):
            assert (row["H"], row["W"], row["k_used"]) == (h, w, 7)
            assert row["count_error"] == row["io_mismatch"] == 0
            assert row["n"] == row["n_requested"] > 0
            assert np.isfinite(row["psnr"]).all()
        # The FP8 forward map can shift the predicted count slightly.
        assert mixed["n"] == pytest.approx(ref["n"], rel=0.005)
        # Allow 0.005 dB for independent EMV/render runs around the 0.06 dB budget.
        assert abs(mixed["psnr"][-1] - ref["psnr"][-1]) <= 0.065


def test_size_mode_c_abi():
    lib, fm, head = require("GFUSED_LIB", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    api = ctypes.CDLL(str(lib))
    session = ctypes.c_void_p()
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    api.gfused_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, integer, ctypes.POINTER(ptr)]
    api.gfused_destroy.argtypes = [ptr]
    api.gfused_destroy.restype = None
    api.gfused_run.argtypes = [ptr, ptr, integer, integer, integer, ctypes.c_int64, integer]
    api.gfused_run_with_size_mode.argtypes = [*api.gfused_run.argtypes, integer]
    api.gfused_read.argtypes = [ptr, *([ptr] * 10)]
    api.gfused_error.restype = ctypes.c_char_p
    assert api.gfused_create(os.fsencode(fm), os.fsencode(head), 0, ctypes.byref(session)) == 0, (
        api.gfused_error()
    )
    h, w = 1080, 1920
    image = np.random.default_rng(42).random((3, h, w), dtype=np.float32)

    def run(img, mode=None):
        _, height, width = img.shape
        args = (session, img.ctypes.data, height, width, 7, REF_SEED, 0)
        n = api.gfused_run(*args) if mode is None else api.gfused_run_with_size_mode(*args, mode)
        assert n > 0, api.gfused_error()
        density = np.empty((height, width), np.float32)
        render = np.empty_like(img)
        counts = (2, 3, 3, 2, 3, 3, 3, 1)
        attrs = [np.empty((n, c), np.float32) for c in counts]
        assert (
            api.gfused_read(
                session, density.ctypes.data, render.ctypes.data, *(a.ctypes.data for a in attrs)
            )
            == 0
        ), api.gfused_error()
        assert np.all((attrs[0] >= 0) & (attrs[0] <= 1))
        return density, n

    try:
        default, _ = run(image)
        pad, count = run(image, 0)
        exact, _ = run(image, 1)
        np.testing.assert_allclose(default, exact, rtol=1e-5, atol=1e-6)
        assert not np.array_equal(pad, exact)
        again, count_again = run(image, 0)
        assert count == count_again
        np.testing.assert_allclose(pad, again, rtol=1e-5, atol=1e-6)
        aligned, _ = run(np.pad(image, ((0, 0), (0, 8), (0, 0)), mode="reflect"), 0)
        crop = aligned[:h, :]
        np.testing.assert_allclose(pad, crop / crop.mean(), rtol=1e-5, atol=1e-6)
        assert (
            api.gfused_run_with_size_mode(session, image.ctypes.data, h, w, 7, REF_SEED, 0, 2) == -1
        )
        assert b"size mode" in api.gfused_error()
    finally:
        api.gfused_destroy(session)


def test_manifest_inflight_and_cache(tmp_path):
    binary, fm, head = require("GFUSED_BIN", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    rng = np.random.default_rng(42)
    shapes = [(17, 31), (32, 48), (17, 31), (16, 16), (48, 32), (17, 31)]
    jobs = []
    for index, (h, w) in enumerate(shapes):
        input_path = tmp_path / f"input_{index}.npy"
        np.save(input_path, rng.random((3, h, w), dtype=np.float32))
        jobs.append(
            f"{input_path} {tmp_path / f'render_{index}.npy'} {tmp_path / f'g_{index}.npy'}"
        )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(jobs) + "\n")
    serial = run_bin(binary, fm, head, manifest, 3, REF_SEED, ["--check-head-io"])
    assert [r["graph_captured"] for r in serial] == [True, True, False, True, True, True]
    parallel = run_bin(
        binary, fm, head, manifest, 3, REF_SEED, ["--inflight", "3", "--check-head-io"]
    )
    for index, ((h, w), one, many) in enumerate(zip(shapes, serial, parallel, strict=True)):
        assert (one["input"], h, w) == (many["input"], many["H"], many["W"])
        assert one["n"] == many["n"] == many["n_requested"]
        assert many["count_error"] == many["io_mismatch"] == 0
        np.testing.assert_allclose(one["psnr"], many["psnr"], atol=0.02)
        render = np.load(tmp_path / f"render_{index}.npy")
        assert render.shape == (3, h, w)
        assert np.all(np.isfinite(render) & (render >= 0) & (render <= 1))
        image = np.load(many["input"])
        psnr = -10 * np.log10(max(np.mean(np.square(render - image)), 1e-10))
        assert psnr == pytest.approx(many["psnr"][-1], abs=1e-4)
        points = np.load(tmp_path / f"g_{index}.points.npy")
        assert points.shape == (many["n"], 2)
        assert np.all((points >= 0) & (points <= 1))


def test_odd_adaptive_graph(tmp_path):
    binary, fm, head = require("GFUSED_BIN", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    image = tmp_path / "odd.npy"
    np.save(image, np.full((3, 17, 31), 0.5, dtype=np.float32))
    manifest = tmp_path / "adaptive.txt"
    manifest.write_text(f"{image} DUMMY\n")
    adaptive = ["--psnr-target", "1", "--check-head-io"]
    graph = run_bin(binary, fm, head, manifest, 7, REF_SEED, adaptive)[0]
    eager = run_bin(binary, fm, head, manifest, 7, REF_SEED, [*adaptive, "--no-graph"])[0]
    assert graph["k_used"] == eager["k_used"] == 1
    assert graph["n"] == eager["n"]
    np.testing.assert_allclose(graph["psnr"], eager["psnr"], atol=0.02)


def run_video(binary, fm, head, frames: Path, out: Path, extra: list[str]) -> list[dict]:
    cmd = [binary, "--fm", fm, "--head", head, "--K", "3", "--seed", str(REF_SEED)]
    cmd += ["--video", frames, "--video-out", out, *extra]
    proc = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def test_video_mode(tmp_path):
    """Video warm start on an unaligned size: graph vs eager, both size modes, track outputs."""
    binary, fm, head = require("GFUSED_BIN", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    h, w, n_frames = 97, 130, 4
    frames = tmp_path / "frames"
    frames.mkdir()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for f in range(n_frames):  # smooth background with a slowly moving blob
        blob = np.exp(-((xx - 40 - 3 * f) ** 2 + (yy - 50) ** 2) / 200.0)
        image = np.stack([xx / w, yy / h, 0.5 * (1 + np.sin(xx / 9))]) * 0.6 + 0.4 * blob
        np.save(frames / f"{f:03d}.npy", np.clip(image, 0, 1).astype(np.float32))

    graph = run_video(binary, fm, head, frames, tmp_path / "graph", ["--check-head-io"])
    eager = run_video(binary, fm, head, frames, tmp_path / "eager", ["--no-graph"])
    pad = run_video(binary, fm, head, frames, tmp_path / "pad", ["--size-mode", "pad"])
    assert len(graph) == len(eager) == len(pad) == n_frames
    previous_ids = None
    for f, (g, e) in enumerate(zip(graph, eager, strict=True)):
        assert g["frame"] == e["frame"] == f and (g["H"], g["W"]) == (h, w)
        assert g["io_mismatch"] == 0
        assert g["n"] == e["n"] > 0
        np.testing.assert_allclose(g["psnr"], e["psnr"], atol=0.02)
        stem = tmp_path / "graph" / f"{f:06d}"
        points = np.load(f"{stem}.points.npy")
        ids, slots = np.load(f"{stem}.ids.npy"), np.load(f"{stem}.slots.npy")
        assert points.shape == (g["n"], 2) and np.all((points >= 0) & (points <= 1))
        assert ids.dtype == np.int64 and slots.dtype == np.int32
        assert ids.shape == slots.shape == (g["n"],)
        assert len(np.unique(ids)) == len(ids) and len(np.unique(slots)) == len(slots)
        if previous_ids is not None:  # a nearly static clip keeps most tracks
            assert g["retained_fraction"] > 0.5
            assert len(np.intersect1d(ids, previous_ids)) > 0.5 * len(previous_ids)
        previous_ids = ids
    assert all(p["n"] > 0 and np.isfinite(p["psnr"]).all() for p in pad)

    np.save(frames / "999.npy", np.zeros((3, h + 1, w), np.float32))
    for extra, error in (
        ([], "first frame's dimensions"),
        (["--inflight", "2"], "--inflight"),
    ):
        cmd = [binary, "--fm", fm, "--head", head, "--video", frames]
        cmd += ["--video-out", tmp_path / "bad", *extra]
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        assert proc.returncode != 0 and error in proc.stderr


def test_pipeline_input_errors(tmp_path):
    binary, fm, head = require("GFUSED_BIN", "GFUSED_FM_BLOB", "GFUSED_HEAD_BLOB")
    image = tmp_path / "image.npy"
    valid = io.BytesIO()
    np.save(valid, np.zeros((3, 16, 16), np.float32))
    long_header = (
        b"{'descr':" + b" " * 63_000 + b"'<f4', 'fortran_order': False, 'shape': (3,16,16), }\n"
    )
    for data, error in (
        (np.zeros((2, 16, 16), np.float32), "expected NPY shape"),
        (np.zeros((3, 15, 16), np.float32), "at least 16"),
        (np.zeros((3, 16, 16), np.float64), "little-endian float32"),
        (np.asfortranarray(np.zeros((3, 16, 16), np.float32)), "C order"),
        (np.full((3, 16, 16), np.nan, np.float32), "finite and in [0,1]"),
        (b"\x93NUMPY\x02", "truncated NPY version"),
        (b"garbage", "not an NPY file"),
        (valid.getvalue()[:-4], "payload size does not match shape"),
        (
            b"\x93NUMPY\x02\x00" + struct.pack("<I", len(long_header)) + long_header,
            "payload size does not match shape",
        ),
    ):
        if isinstance(data, bytes):
            image.write_bytes(data)
        else:
            np.save(image, data)
        proc = subprocess.run(
            [binary, "--fm", fm, "--head", head, image, "DUMMY"], capture_output=True, text=True
        )
        assert proc.returncode != 0
        assert error in proc.stderr and str(image) in proc.stderr

    missing = tmp_path / "missing.gfch"
    proc = subprocess.run(
        [binary, "--fm", fm, "--head", missing, image, "DUMMY"], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "cannot open" in proc.stderr and str(missing) in proc.stderr


def main():
    a = parse()
    if a.latency:
        latency(a)
    elif a.adaptive_sweep:
        adaptive_sweep(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
