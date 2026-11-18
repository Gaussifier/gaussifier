"""ctypes binding for the bundled frozen-GN CUDA profile."""

from __future__ import annotations

import ctypes as ct
import os
import re
from collections.abc import Mapping
from ctypes.util import find_library
from operator import index
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Self
from weakref import finalize

import numpy as np

from gaussifier_sampler.inference import (
    GaussifierSampler,
    SampleResult,
    VideoFrameResult,
    VideoFrameStats,
)
from gaussifier_sampler.utils import (
    BUNDLED_WEIGHTS_PATH,
    PRODUCTION_DEFAULT_SEED,
    stable_seed_from_key,
)

if TYPE_CHECKING:
    from gaussifier_sampler.inference import Array, FusedSizeMode, InferenceProfile, RendererFn


_WEIGHTS = BUNDLED_WEIGHTS_PATH.parent / "frozen_gn"
_LIBRARY = "libgaussifier_fused.so.1"
_FLOAT_PTR = ct.POINTER(ct.c_float)
_SIZE_MODES = {"pad": 0, "exact": 1}  # gfused::SizeMode
_MIN_SIZE = 16
_MAX_STATES = 16


class _VideoStats(ct.Structure):
    # GfusedVideoStats in python_api.h
    _fields_ = [
        ("frame", ct.c_uint64),
        ("n_requested", ct.c_int32),
        ("n_target", ct.c_int32),
        ("n_target_limited", ct.c_int32),
        ("survivors", ct.c_int32),
        ("kills_for_next", ct.c_int32),
        ("births_for_next", ct.c_int32),
        ("shortfall_for_next", ct.c_int32),
        ("k_used", ct.c_int32),
        ("retained_fraction", ct.c_double),
        ("rate_sum", ct.c_double),
        ("psnr", ct.c_double * _MAX_STATES),
    ]

    def to_result(self) -> VideoFrameStats:
        values = {name: getattr(self, name) for name, _ in self._fields_}
        values["psnr"] = tuple(self.psnr[: self.k_used])
        return VideoFrameStats(**values)


def _load_library() -> ct.CDLL:
    explicit = os.environ.get("GAUSSIFIER_FUSED_LIBRARY")
    if explicit:
        path = str(Path(explicit).expanduser().resolve())
    else:
        package = Path(__file__).resolve().parent
        candidates = (
            package / _LIBRARY,
            package.parents[1] / "cpp_inference/build/fused" / _LIBRARY,
        )
        path = next(
            (str(candidate) for candidate in candidates if candidate.is_file()),
            find_library("gaussifier_fused") or _LIBRARY,
        )
    try:
        lib = ct.CDLL(path)
    except OSError as exc:
        raise RuntimeError(
            "Build or install libgaussifier_fused with CMake and set "
            f"GAUSSIFIER_FUSED_LIBRARY to its path ({exc})."
        ) from exc

    lib.gfused_error.argtypes = []
    lib.gfused_error.restype = ct.c_char_p
    lib.gfused_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.POINTER(ct.c_void_p)]
    lib.gfused_create.restype = ct.c_int
    lib.gfused_destroy.argtypes = [ct.c_void_p]
    lib.gfused_destroy.restype = None
    lib.gfused_run.argtypes = [
        ct.c_void_p,
        _FLOAT_PTR,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int64,
        ct.c_int,
    ]
    lib.gfused_run.restype = ct.c_int
    lib.gfused_run_with_size_mode.argtypes = [*lib.gfused_run.argtypes, ct.c_int]
    lib.gfused_run_with_size_mode.restype = ct.c_int
    lib.gfused_read.argtypes = [ct.c_void_p, *([_FLOAT_PTR] * 10)]
    lib.gfused_read.restype = ct.c_int
    if hasattr(lib, "gfused_video_create"):
        lib.gfused_video_create.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int64,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_void_p),
        ]
        lib.gfused_video_create.restype = ct.c_int
        lib.gfused_video_push.argtypes = [
            ct.c_void_p,
            _FLOAT_PTR,
            ct.c_int,
            ct.c_int,
            ct.POINTER(_VideoStats),
        ]
        lib.gfused_video_push.restype = ct.c_int
        lib.gfused_video_read_tracks.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_int32),
        ]
        lib.gfused_video_read_tracks.restype = ct.c_int
        lib.gfused_video_destroy.argtypes = [ct.c_void_p]
        lib.gfused_video_destroy.restype = None
    return lib


def _covariance_attributes(channels: np.ndarray) -> dict[str, np.ndarray]:
    a, b, c = channels.T
    log_cov = np.stack((a, b, b, c), axis=-1).reshape(-1, 2, 2)
    center = (a + c) * 0.5
    radius = np.hypot((a - c) * 0.5, b)
    eigenvalues = np.stack((center + radius, center - radius), axis=-1)
    scale = np.exp(0.5 * eigenvalues)
    rotation = np.where(radius > 1e-8, -0.5 * np.arctan2(2 * b, a - c), 0)
    cos, sin = np.cos(rotation), np.sin(rotation)
    basis = np.stack((cos, sin, -sin, cos), axis=-1).reshape(-1, 2, 2)
    variance = np.exp(np.clip(eigenvalues, -20.0, 20.0))
    return {
        "init_log_covariance": log_cov,
        "init_covariance": (basis * variance[:, None, :]) @ basis.transpose(0, 2, 1),
        "init_scale": scale,
        "init_rotation": rotation[:, None],
        "init_rotation_vector": np.stack((cos, sin), axis=-1),
    }


class FusedGaussifierSampler(GaussifierSampler):
    """One CUDA session with reusable weights and graphs for the current resolution."""

    def __init__(
        self, *, precision: str, device: object, size_mode: FusedSizeMode = "exact"
    ) -> None:
        if precision not in ("fp16", "fp8"):
            raise ValueError("Fused precision must be 'fp16' or 'fp8'.")
        if size_mode not in _SIZE_MODES:
            raise ValueError("Fused size_mode must be 'exact' or 'pad'.")
        self.size_mode = size_mode
        name = "cuda" if device is None or str(device) == "auto" else str(device)
        match = re.fullmatch(r"cuda(?::([0-9]+))?", name)
        if match is None:
            raise ValueError("The fused backend requires CUDA. For CPU use backend='torch'.")
        ordinal = int(match[1] or 0)
        self.device = f"cuda:{ordinal}"
        self.precision = precision
        self._lock = Lock()
        self._lib = _load_library()
        self._session = ct.c_void_p()
        _check(
            self._lib,
            self._lib.gfused_create(
                os.fsencode(_WEIGHTS / f"fm_{precision}.gffm"),
                os.fsencode(_WEIGHTS / f"head_{precision}.gfch"),
                ordinal,
                ct.byref(self._session),
            ),
        )
        self._finalize = finalize(self, self._lib.gfused_destroy, self._session)

    def sample(
        self,
        image: Array,
        *,
        count: float | None = None,
        seed: int | None = None,
        image_key: str | None = None,
        renderer: RendererFn | None = None,
        correction_iterations: int | None = None,
        inference_profile: InferenceProfile = "production",
    ) -> SampleResult:
        """Run K rendered states with the built-in renderer and download the result."""
        if renderer is not None or inference_profile != "production":
            raise ValueError("The fused backend uses its built-in renderer and production profile.")
        source, tensor_input = _image_source(image)
        height, width = source.shape[1:]
        if seed is None:
            seed = PRODUCTION_DEFAULT_SEED if not image_key else stable_seed_from_key(image_key)
        states, requested, seed = _sampling_options(correction_iterations, count, seed)

        with self._lock:
            n = _check(
                self._lib,
                self._lib.gfused_run_with_size_mode(
                    self._session,
                    source.ctypes.data_as(_FLOAT_PTR),
                    height,
                    width,
                    states,
                    seed,
                    requested,
                    _SIZE_MODES[self.size_mode],
                ),
            )
            arrays = _read_arrays(self._lib, self._session, height, width, n)
        return SampleResult(
            predicted_count=n,
            inference_profile="production",
            **_result_arrays(arrays, self.device, tensor_input),
        )

    def video(
        self,
        height: int,
        width: int,
        *,
        count: float | None = None,
        seed: int = PRODUCTION_DEFAULT_SEED,
        correction_iterations: int | None = None,
        size_mode: FusedSizeMode | None = None,
        count_ema: float = 0.9,
        warm_attribute_blend: float = 0.1,
    ) -> FusedVideoSession:
        """Open a fixed-size video stream. Push frames in order and close it when done.

        By default the stream inherits this sampler's size mode. Each stream owns
        its tracking state and shares the sampler's immutable model weights.
        """
        if size_mode is None:
            size_mode = self.size_mode
        if size_mode not in _SIZE_MODES:
            raise ValueError("Fused size_mode must be 'exact' or 'pad'.")
        height, width = index(height), index(width)
        _validate_size(height, width)
        if not 0 <= count_ema <= 1:
            raise ValueError("Video count_ema must be in [0, 1].")
        if not 0 <= warm_attribute_blend <= 1:
            raise ValueError("Video warm_attribute_blend must be in [0, 1].")
        states, requested, seed = _sampling_options(correction_iterations, count, seed)
        if not hasattr(self._lib, "gfused_video_create"):
            raise RuntimeError("Video sessions require an updated libgaussifier_fused.")
        session = ct.c_void_p()
        _check(
            self._lib,
            self._lib.gfused_video_create(
                self._session,
                height,
                width,
                states,
                seed,
                requested,
                _SIZE_MODES[size_mode],
                count_ema,
                warm_attribute_blend,
                ct.byref(session),
            ),
        )
        return FusedVideoSession(self._lib, session, self.device, (height, width))

    def sample_batch(
        self,
        images: Array,
        *,
        renderer: RendererFn | None = None,
        correction_iterations: int | None = None,
        seeds: list[int] | None = None,
        inference_profile: InferenceProfile = "production",
    ) -> list[SampleResult]:
        """Run each image through the fused session in input order."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {images.shape}.")
        if seeds is not None and len(seeds) != len(images):
            raise ValueError("Expected one seed per image.")
        return [
            self.sample(
                image,
                renderer=renderer,
                correction_iterations=correction_iterations,
                seed=None if seeds is None else seeds[i],
                inference_profile=inference_profile,
            )
            for i, image in enumerate(images)
        ]


class FusedVideoSession:
    """An ordered sequence of frames with its own tracking state."""

    def __init__(
        self, lib: ct.CDLL, session: ct.c_void_p, device: str, shape: tuple[int, int]
    ) -> None:
        self._finalize = finalize(self, lib.gfused_video_destroy, session)
        self._lib = lib
        self._session = session
        self._device = device
        self._shape = shape
        self._lock = Lock()

    def __enter__(self) -> Self:
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release this stream and its CUDA workspace."""
        with self._lock:
            self._finalize()

    def _require_open(self) -> None:
        if not self._finalize.alive:
            raise RuntimeError("The fused video session is closed.")

    def push(self, image: Array) -> VideoFrameResult:
        """Advance the stream by one frame and return its Gaussians, render and stats."""
        with self._lock:
            self._require_open()
            source, tensor_input = _image_source(image)
            height, width = self._shape
            if source.shape[1:] != self._shape:
                raise ValueError(f"Expected video frame dimensions {self._shape}.")
            stats = _VideoStats()
            try:
                n = _check(
                    self._lib,
                    self._lib.gfused_video_push(
                        self._session,
                        source.ctypes.data_as(_FLOAT_PTR),
                        height,
                        width,
                        ct.byref(stats),
                    ),
                )
                arrays = _read_arrays(self._lib, self._session, height, width, n)
                ids, slots = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int32)
                _check(
                    self._lib,
                    self._lib.gfused_video_read_tracks(
                        self._session,
                        ids.ctypes.data_as(ct.POINTER(ct.c_int64)),
                        slots.ctypes.data_as(ct.POINTER(ct.c_int32)),
                    ),
                )
                arrays.update(ids=ids, slots=slots)
                return VideoFrameResult(
                    predicted_count=n,
                    inference_profile="production",
                    stats=stats.to_result(),
                    **_result_arrays(arrays, self._device, tensor_input),
                )
            except BaseException:
                # A failed read or conversion can follow a successful tracking update.
                self._finalize()
                raise


def _check(lib: ct.CDLL, result: int) -> int:
    if result < 0:
        raise RuntimeError(f"Fused CUDA runtime: {lib.gfused_error().decode()}")
    return result


def _validate_size(height: int, width: int) -> None:
    if min(height, width) < _MIN_SIZE:
        raise ValueError(f"Fused image height and width must be at least {_MIN_SIZE}.")
    if max(height, width) >= 1 << 31:
        raise ValueError("Fused image is too large.")


def _image_source(image: Array) -> tuple[np.ndarray, bool]:
    tensor_input = not isinstance(image, np.ndarray)
    source = image.detach().float().cpu().numpy() if tensor_input else image
    if source.ndim == 4 and source.shape[0] == 1:
        source = source[0]
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] or [1,3,H,W], got {source.shape}.")
    _validate_size(*source.shape[1:])
    return np.ascontiguousarray(np.clip(source, 0.0, 1.0), dtype=np.float32), tensor_input


def _sampling_options(
    correction_iterations: int | None, count: float | None, seed: int
) -> tuple[int, int, int]:
    states = 7 if correction_iterations is None else int(correction_iterations)
    if not 1 <= states <= _MAX_STATES:
        raise ValueError(f"Fused correction_iterations must be between 1 and {_MAX_STATES}.")
    requested = 0 if count is None else max(1, int(count))
    if requested > 1_000_000:
        raise ValueError("The fused backend supports up to 1,000,000 points.")
    if not -(1 << 63) <= seed < (1 << 63):
        raise ValueError("Fused seed must fit in a signed 64-bit integer.")
    return states, requested, int(seed)


def _read_arrays(
    lib: ct.CDLL, session: ct.c_void_p, height: int, width: int, n: int
) -> dict[str, np.ndarray]:
    shapes = {
        "density": (height, width),
        "final_rendered": (3, height, width),
        "points": (n, 2),
        "init_log_covariance_channels": (n, 3),
        "init_unweighted_color": (n, 3),
        "final_points": (n, 2),
        "final_log_covariance_channels": (n, 3),
        "final_unweighted_color": (n, 3),
        "final_color": (n, 3),
        "point_weights": (n,),
    }
    arrays = {name: np.empty(shape, dtype=np.float32) for name, shape in shapes.items()}
    _check(
        lib,
        lib.gfused_read(session, *(array.ctypes.data_as(_FLOAT_PTR) for array in arrays.values())),
    )
    return arrays


def _result_arrays(
    arrays: dict[str, np.ndarray], device: str, tensor_input: bool
) -> Mapping[str, Array]:
    arrays["init_color"] = np.clip(
        arrays["init_unweighted_color"] * arrays["point_weights"][:, None], 0, 1
    )
    arrays.update(_covariance_attributes(arrays["init_log_covariance_channels"]))
    if tensor_input:
        import torch

        return {name: torch.from_numpy(array).to(device) for name, array in arrays.items()}
    return arrays
