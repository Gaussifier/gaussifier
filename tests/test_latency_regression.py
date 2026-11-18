"""Latency regression test for the C++ inference harness.

Runs 100-rep steady-state benchmark, compares mean/stddev/p99 against a
baseline JSON committed to the repo. Fails CI if any metric drifts beyond
the configured tolerance.

Layout:
  tests/expected_latency.json     baseline metrics per config
  tests/test_latency_regression.py (this file)

Run:
  # Requires exclusive access to the benchmark GPU.
  uv run pytest tests/test_latency_regression.py -v

Update baselines:
  uv run pytest tests/test_latency_regression.py --update-baselines
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

from tests.harness import (
    CONFIGS,
    FIXTURES,
    FUSED_GATE_DEVICES,
    FUSED_PROFILES,
    REPO,
    baseline_key,
    load_baselines,
    png_to_input_tensor,
    require_fused,
    require_fused_gate,
    run_fused,
    run_harness,
    save_baselines,
    save_npy_tensor,
)

BASELINE_FILE = REPO / "tests" / "expected_latency.json"

# Latency tolerances. Tight enough to catch ~1 ms regressions, loose enough
# to absorb GPU scheduler noise (~0.5 ms p99 stddev in practice).
MEAN_TOL_MS = 1.0  # absolute drift either direction
STDDEV_TOL_FACTOR = 2.5  # observed stddev must stay < factor * baseline stddev
P99_TOL_MS = 2.0  # p99 absolute drift

# Only the production config — latency on torchscript_eager is not a
# perf SLA target. Add more here if other paths become perf-critical.
LATENCY_CONFIGS = ["trt_aoti_fused", *FUSED_PROFILES]

# Use the smaller PNG (smoke test should be fast)
LATENCY_FIXTURE = "bench_input_512"
LATENCY_REPS = 100


_STEADY_RE = re.compile(
    r"\[e2e\] steady-state \(n=\d+\): mean=([\d.]+) min=([\d.]+) max=([\d.]+) "
    r"stddev=([\d.]+) p99=([\d.]+) ms"
)


def _run_bench(config: dict[str, str], input_path: Path, k_iters: int = 5) -> dict[str, float]:
    stderr = run_harness(
        input_path=input_path,
        output_path=input_path.parent / "out.npy",
        config=config,
        k_iters=k_iters,
        repeat=LATENCY_REPS,
    )
    match = _STEADY_RE.search(stderr)
    if not match:
        pytest.fail(
            "could not parse [e2e] steady-state line. stderr tail:\n"
            + "\n".join(stderr.splitlines()[-15:])
        )
    return {
        "mean_ms": float(match.group(1)),
        "min_ms": float(match.group(2)),
        "max_ms": float(match.group(3)),
        "stddev_ms": float(match.group(4)),
        "p99_ms": float(match.group(5)),
    }


@pytest.mark.parametrize("k_iters", [5, 7])
@pytest.mark.parametrize("config_name", LATENCY_CONFIGS)
def test_latency_regression(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    config_name: str,
    k_iters: int,
) -> None:
    if not torch.cuda.is_available():
        if config_name in FUSED_PROFILES:
            require_fused()
        pytest.skip("CUDA unavailable for latency regression")
    gpus = FUSED_GATE_DEVICES if config_name in FUSED_PROFILES else ["NVIDIA GeForce RTX 5090"]
    if torch.cuda.get_device_name() not in gpus:
        pytest.skip(f"{config_name} latency baselines require one of {', '.join(gpus)}")
    require_fused_gate(config_name, k_iters)
    png = FIXTURES / f"{LATENCY_FIXTURE}.png"
    gt = png_to_input_tensor(png)
    input_npy = tmp_path / "input.npy"
    save_npy_tensor(gt, input_npy)

    if config_name in FUSED_PROFILES:
        measured = run_fused(
            input_path=input_npy,
            output_path="DUMMY",
            profile=config_name,
            k_iters=k_iters,
            sustained=LATENCY_REPS,
        )["sustained_latency"]
    else:
        measured = _run_bench(CONFIGS[config_name], input_npy, k_iters)

    baselines = load_baselines(BASELINE_FILE)
    key = baseline_key(config_name, k_iters)

    if request.config.getoption("--update-baselines"):
        baselines[key] = {k: round(v, 4) for k, v in measured.items()}
        save_baselines(BASELINE_FILE, baselines)
        sys.stderr.write(
            f"\n[update-baselines] {key}: "
            f"mean={measured['mean_ms']:.3f} stddev={measured['stddev_ms']:.3f} "
            f"p99={measured['p99_ms']:.3f} ms\n"
        )
        return

    expected = baselines.get(key)
    if expected is None:
        pytest.fail(
            f"no baseline for {key}: measured mean={measured['mean_ms']:.3f} ms. "
            f"Run with --update-baselines to seed."
        )

    # Mean drift either direction
    mean_drift = measured["mean_ms"] - expected["mean_ms"]
    # Stddev: only fail if it grew (smaller stddev is fine)
    stddev_ratio = measured["stddev_ms"] / max(expected["stddev_ms"], 0.01)
    # p99 drift either direction
    p99_drift = measured["p99_ms"] - expected["p99_ms"]

    fails = []
    if abs(mean_drift) > MEAN_TOL_MS:
        fails.append(
            f"mean drift {mean_drift:+.3f} ms > ±{MEAN_TOL_MS:.2f} ms "
            f"(measured={measured['mean_ms']:.3f}, baseline={expected['mean_ms']:.3f})"
        )
    if stddev_ratio > STDDEV_TOL_FACTOR:
        fails.append(
            f"stddev grew by {stddev_ratio:.2f}x > {STDDEV_TOL_FACTOR:.1f}x "
            f"(measured={measured['stddev_ms']:.3f}, baseline={expected['stddev_ms']:.3f})"
        )
    if abs(p99_drift) > P99_TOL_MS:
        fails.append(
            f"p99 drift {p99_drift:+.3f} ms > ±{P99_TOL_MS:.2f} ms "
            f"(measured={measured['p99_ms']:.3f}, baseline={expected['p99_ms']:.3f})"
        )
    if fails:
        pytest.fail(f"latency regression on {key}:\n  " + "\n  ".join(fails))
