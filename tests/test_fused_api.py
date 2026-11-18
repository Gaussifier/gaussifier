"""Verify the shipped fused Python profile and its native result contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gaussifier_sampler import GaussifierSampler, SampleResult, VideoFrameResult
from gaussifier_sampler.image_io import load_image_array
from tests.harness import (
    FIXTURES,
    FUSED_BIN,
    FUSED_WEIGHTS,
    REPO,
    REQUIRE_FUSED,
    require_fused,
    run_fused,
)


@pytest.fixture
def fused_available():
    require_fused()
    if (
        not os.environ.get("GAUSSIFIER_FUSED_LIBRARY")
        and not (REPO / "cpp_inference/build/fused/libgaussifier_fused.so.1").is_file()
    ):
        if REQUIRE_FUSED:
            pytest.fail("build the Python fused library or set GAUSSIFIER_FUSED_LIBRARY")
        pytest.skip("Python fused library unavailable")


@pytest.mark.parametrize("precision", ["fp16", "fp8"])
def test_bundled_fused_result(precision, tmp_path, fused_available):
    image = load_image_array(FIXTURES / "bench_input_512.png")
    sampler = (
        GaussifierSampler.bundled()
        if precision == "fp16"
        else GaussifierSampler.bundled(backend="fused", precision=precision)
    )
    result = sampler.sample(image)
    assert isinstance(result, SampleResult)
    n = result.predicted_count
    assert result.points.shape == result.final_points.shape == (n, 2)
    assert result.density.shape == image.shape[1:]
    assert result.init_covariance.shape == (n, 2, 2)
    assert result.init_scale.shape == result.init_rotation_vector.shape == (n, 2)
    assert result.init_rotation.shape == (n, 1)
    assert np.isfinite(result.final_rendered).all()
    assert np.allclose(
        result.final_color,
        np.clip(result.final_unweighted_color * result.point_weights[:, None], 0, 1),
        atol=1e-6,
    )

    initial = sampler.sample(image, correction_iterations=1)
    np.testing.assert_array_equal(initial.points, initial.final_points)
    np.testing.assert_array_equal(
        initial.init_log_covariance_channels, initial.final_log_covariance_channels
    )
    assert not np.array_equal(result.points, result.final_points)

    input_path, output_path = tmp_path / "input.npy", tmp_path / "render.npy"
    np.save(input_path, image, allow_pickle=False)
    stats = run_fused(
        input_path=input_path, output_path=output_path, profile=f"fused_{precision}", k_iters=7
    )
    assert stats["n"] == n
    # Independent EMV runs can differ slightly in cell assignment and float accumulation.
    python_psnr = -10 * np.log10(np.mean(np.square(result.final_rendered - image)))
    np.testing.assert_allclose(python_psnr, stats["psnr"][-1], atol=0.05)
    assert python_psnr > -10 * np.log10(np.mean(np.square(initial.final_rendered - image)))

    (pinned,) = sampler.sample_batch(torch.from_numpy(image[None]), seeds=[42])
    assert pinned.final_rendered.device.type == "cuda"
    assert (
        sampler.sample(image, count=1000, seed=42, correction_iterations=1).predicted_count == 1000
    )


def test_fused_concurrent_sessions(fused_available):
    def sample(precision):
        sampler = GaussifierSampler.bundled(precision=precision)
        for size in (16, 32, 64, 32):
            image = np.full((3, size, size), 0.25, dtype=np.float32)
            result = sampler.sample(image, count=8)
            assert result.predicted_count == 8
            assert result.final_rendered.shape == image.shape
            assert np.isfinite(result.final_rendered).all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(sample, ("fp16", "fp8")))


@pytest.mark.parametrize("size_mode", ["exact", "pad"])
def test_fused_unaligned_size_modes(size_mode, fused_available):
    image = load_image_array(FIXTURES / "bench_input_512.png")[:, :97, :130]
    sampler = GaussifierSampler.bundled(size_mode=size_mode)
    assert sampler.size_mode == size_mode
    result = sampler.sample(np.ascontiguousarray(image), seed=7)
    n = result.predicted_count
    assert n > 0 and result.final_points.shape == (n, 2)
    assert result.density.shape == image.shape[1:]
    assert result.final_rendered.shape == image.shape
    assert np.isfinite(result.final_rendered).all()
    with pytest.raises(ValueError, match="at least 16"):
        sampler.sample(np.zeros((3, 15, 64), dtype=np.float32))


def test_fused_size_mode_validation():
    with pytest.raises(ValueError, match="size_mode"):
        GaussifierSampler.bundled(backend="torch", size_mode="pad")


@pytest.mark.parametrize("size_mode", ["exact", "pad"])
def test_fused_video_matches_cli(size_mode, tmp_path, fused_available):
    h, w = 97, 130
    frames = tmp_path / "frames"
    frames.mkdir()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    background = np.stack((xx / w, yy / h, 0.5 * (1 + np.sin(xx / 9)))) * 0.6
    images = []
    for f in range(4):
        image = background + 0.4 * np.exp(-((xx - 40 - 3 * f) ** 2 + (yy - 50) ** 2) / 200)
        images.append(image)
        np.save(frames / f"{f:03d}.npy", image)

    out = tmp_path / "cli"
    cli = subprocess.run(
        [
            str(FUSED_BIN),
            "--fm",
            str(FUSED_WEIGHTS / "fm_fp16.gffm"),
            "--head",
            str(FUSED_WEIGHTS / "head_fp16.gfch"),
            "--video",
            str(frames),
            "--video-out",
            str(out),
            "--size-mode",
            size_mode,
            "--K",
            "3",
            "--seed",
            "1234",
            "--count-ema",
            "0.7",
            "--warm-attribute-blend",
            "0.2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in cli.stdout.splitlines() if line.startswith("{")]
    # Exercise both inheriting the sampler size mode and overriding it for a session.
    sampler = GaussifierSampler.bundled(size_mode="pad")
    with sampler.video(
        h,
        w,
        size_mode="exact" if size_mode == "exact" else None,
        seed=1234,
        correction_iterations=3,
        count_ema=0.7,
        warm_attribute_blend=0.2,
    ) as stream:
        previous = None
        for f, (image, expected) in enumerate(zip(images, rows, strict=True)):
            if f == 1:
                with pytest.raises(ValueError, match="frame dimensions"):
                    stream.push(image[:, :-1])
            result = stream.push(image[None])
            assert isinstance(result, VideoFrameResult)
            n = result.predicted_count
            assert result.stats.frame == f and n == expected["n"] > 0
            assert result.density.shape == (h, w) and result.final_rendered.shape == image.shape
            assert result.points.shape == result.final_points.shape == (n, 2)
            assert result.ids.dtype == np.int64 and result.slots.dtype == np.int32
            assert result.ids.shape == result.slots.shape == (n,)
            assert len(np.unique(result.ids)) == len(np.unique(result.slots)) == n
            if previous is not None:
                ids, before, now = np.intersect1d(previous.ids, result.ids, return_indices=True)
                assert len(ids) > 0.5 * previous.predicted_count
                assert result.stats.retained_fraction > 0.5
                np.testing.assert_array_equal(previous.slots[before], result.slots[now])
            previous = result
            for key, value in asdict(result.stats).items():
                if key == "psnr":
                    np.testing.assert_allclose(value, expected[key], atol=0.1, rtol=0)
                elif key == "rate_sum":
                    # GPU reduction order varies between the two processes.
                    assert value == pytest.approx(expected[key], rel=3e-6)
                elif key == "retained_fraction":
                    assert value == pytest.approx(expected[key], abs=1e-6)
                else:
                    assert value == expected[key]
            mse = np.mean(np.square(result.final_rendered - image), dtype=np.float64)
            assert result.stats.psnr[-1] == pytest.approx(-10 * np.log10(mse), abs=1e-4)
            stem = out / f"{f:06d}"
            for array, suffix in [(result.ids, "ids"), (result.slots, "slots")]:
                np.testing.assert_array_equal(array, np.load(f"{stem}.{suffix}.npy"))
            # Rounding can change which track owns a boundary pixel and cause larger local
            # differences even between CLI runs. Compare the mean error of each attribute.
            for array, suffix, limit in [
                (result.final_points, "points", 0.0003),
                (result.final_log_covariance_channels, "cov", 0.006),
                (result.final_unweighted_color, "rgb", 0.001),
                (result.final_color, "color", 0.001),
                (result.point_weights, "weights", 0.008),
            ]:
                expected_array = np.load(f"{stem}.{suffix}.npy")
                assert array.shape == expected_array.shape
                error = np.mean(np.abs(array - expected_array))
                assert error <= limit, (f, suffix, error)
    with pytest.raises(RuntimeError, match="closed"):
        stream.push(images[0])


def test_fused_video_session_lifetime_and_arguments(fused_available):
    sampler = GaussifierSampler.bundled(precision="fp8")
    for options, match in [
        ({"size_mode": "other"}, "size_mode"),
        ({"count_ema": float("nan")}, "count_ema"),
        ({"warm_attribute_blend": 1.1}, "warm_attribute_blend"),
        ({"correction_iterations": 17}, "correction_iterations"),
        ({"count": 1_000_001}, "points"),
        ({"seed": 1 << 63}, "seed"),
    ]:
        with pytest.raises(ValueError, match=match):
            sampler.video(24, 32, **options)
    with pytest.raises(ValueError, match="at least 16"):
        sampler.video(15, 32)
    image = torch.full((3, 24, 32), 0.25)
    with sampler.video(24, 32, count=8, correction_iterations=1) as first:
        with pytest.raises(ValueError, match="Expected"):
            first.push(image[:2])
        second = sampler.video(24, 32, count=8, correction_iterations=1)
        assert sampler.sample(np.zeros((3, 32, 32), dtype=np.float32), count=4).predicted_count == 4
        del sampler
        for stream in (first, second):
            result = stream.push(image)
            assert result.stats.frame == 0
            assert result.predicted_count == 8
            assert result.ids.dtype == torch.int64 and result.ids.is_cuda
            assert result.slots.dtype == torch.int32 and result.slots.is_cuda
            torch.testing.assert_close(result.points, result.final_points)
        assert first.push(image).stats.frame == 1
        second.close()
        second.close()
        with pytest.raises(RuntimeError, match="closed"):
            second.push(image)


def test_fused_import_has_no_torch_dependency():
    code = """
import sys
from concurrent.futures import ThreadPoolExecutor
sys.modules['torch'] = None
from gaussifier_sampler import GaussifierSampler, SampleResult
from gaussifier_sampler.cli import build_parser
assert GaussifierSampler and SampleResult and build_parser()
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
