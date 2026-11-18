"""Shared setup for the fused-runtime tests (tests/fused/test_*.py).

The tests build their CUDA bindings with ``torch.utils.cpp_extension`` and, for the reference
comparisons, import the gaussifier training repo. Inputs come from environment variables
(pytest) or the scripts' command lines; a pytest test skips when one it needs is missing:

    GFUSED_KODAK          directory of Kodak PNGs
    GFUSED_STAGE1         frozen-GN stage-1 state_dict (stage1.pt)
    GFUSED_HEAD_MODULE    pickled frozen-GN correction head module (head_module.pt)
    GFUSED_PRECISION_PLAN FP8 precision plan JSON (optional FP8 checks)
    GFUSED_BIN            gaussifier_fused_infer binary (pipeline test)
    GFUSED_LIB            libgaussifier_fused.so (C ABI size-mode test)
    GAUSSIFIER_SRC_DIR    gaussifier checkout (default: ../gaussifier next to this repo)
    GFUSED_WORK           scratch directory for exported blobs and dumps (default: a temp dir)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSED_SRC = ROOT / "cpp_inference" / "fused"
GAUSSIFIER_SRC = Path(os.environ.get("GAUSSIFIER_SRC_DIR", ROOT.parent / "gaussifier"))
# Architecture config for head_arch_screen.load (the frozen-GN weights are loaded over it).
RELEASE_CKPT = "artifacts/releases/v0.11.1/gaussifier_best.pt"


def work_dir() -> Path:
    path = Path(os.environ.get("GFUSED_WORK", Path(tempfile.gettempdir()) / "gfused_tests"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def require_cuda() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def require(*names: str) -> list[Path]:
    """Paths from the named environment variables; skips the test if one is unset/missing."""
    import pytest

    require_cuda()
    paths = []
    for name in names:
        path = env_path(name)
        if path is None or not path.exists():
            pytest.skip(f"set {name} to an existing path")
        paths.append(path)
    return paths


def require_gaussifier(src: Path = GAUSSIFIER_SRC) -> Path:
    import pytest

    if not (src / "src" / "gaussifier").is_dir() or not (src / "tools").is_dir():
        pytest.skip(f"gaussifier checkout not found at {src} (set GAUSSIFIER_SRC_DIR)")
    return src


def import_gaussifier(src: Path = GAUSSIFIER_SRC):
    """Puts the gaussifier sources and tools on sys.path; returns head_arch_screen with its
    architecture checkpoint pointed at the release config."""
    for p in (str(src / "tools"), str(src / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("GAUSSIFIER_NATIVE_SIMPLE_SUM_BUILD_DIR", str(work_dir() / "native"))
    import head_arch_screen

    head_arch_screen.CKPT = src / RELEASE_CKPT
    return head_arch_screen


def build_extension(name: str, sources: list[Path], extra_cflags: list[str] | None = None):
    """Compiles a test binding for the current GPU (cached by torch under TORCH_EXTENSIONS_DIR)."""
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    # The launchers cache per-kernel setup (cudaFuncSetAttribute) in function-local statics.
    # g++ makes those STB_GNU_UNIQUE, which unifies them across every extension loaded into the
    # process while each extension has its own kernels; -fno-gnu-unique keeps them per module.
    flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-Xcompiler=-fno-gnu-unique",
        "-DGFUSED_ACC16=1",  # FP16-accumulate conv variants (ConvParams::acc16) for the tests
        *(extra_cflags or []),
    ]
    return load(
        name=name, sources=[str(s) for s in sources], extra_cuda_cflags=flags, verbose=False
    )
