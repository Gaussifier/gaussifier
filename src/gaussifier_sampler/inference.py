"""High-level Gaussifier inference API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from gaussifier_sampler.utils import BUNDLED_WEIGHTS_PATH

if TYPE_CHECKING:
    import torch

    from gaussifier_sampler._fused_inference import FusedGaussifierSampler
    from gaussifier_sampler._torch_inference import TorchGaussifierSampler

    Array = np.ndarray | torch.Tensor

RendererFn = Callable[["torch.Tensor", "torch.Tensor", "torch.Tensor"], "torch.Tensor"]
InferenceProfile = Literal["reference", "production"]
# Fused sizes that are not multiples of 16: "exact" runs the networks at the real extent
# (matches the reference), "pad" reflect-pads the forward-map input to the aligned extent.
FusedSizeMode = Literal["exact", "pad"]


def as_numpy(value: Array) -> np.ndarray:
    """Read a NumPy array or a tensor from the optional torch backend."""
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


@dataclass(slots=True)
class SampleResult:
    """First-stage sampler output for one image.

    `init_*` fields are stage-1 outputs (bilinear-sampled at point xy).
    The fused backend always returns the final state. Torch returns the final state
    when the correction head is enabled and a renderer is supplied.
    `init_color` and `final_color` can always be passed directly to a renderer.
    In production mode they already include clamped initial Voronoi-mass
    weights; `*_unweighted_color` exposes the latent RGB correction state.
    """

    # What was sampled.
    points: Array
    density: Array
    predicted_count: int
    inference_profile: InferenceProfile = "reference"
    # Initializer attributes, bilinear-sampled from the decoder maps at the point centers.
    init_log_covariance_channels: Array | None = None
    init_log_covariance: Array | None = None
    init_covariance: Array | None = None
    init_scale: Array | None = None
    init_rotation: Array | None = None
    init_rotation_vector: Array | None = None
    init_color: Array | None = None
    # Final state, including the initial state when fused runs with K=1.
    # ``final_points`` holds the moved positions when the head predicts xy; else it is ``points``.
    final_points: Array | None = None
    final_log_covariance_channels: Array | None = None
    final_color: Array | None = None
    final_rendered: Array | None = None
    # Production only: the fixed Voronoi-mass weights and the latent colors before weighting,
    # so callers can inspect or replay the recurrent state without weighting twice.
    point_weights: Array | None = None
    init_unweighted_color: Array | None = None
    final_unweighted_color: Array | None = None

    def init_arrays(self) -> dict[str, np.ndarray]:
        """The ``init_*`` attributes that are present, as NumPy arrays keyed by field name."""
        return {
            f.name: as_numpy(getattr(self, f.name))
            for f in fields(self)
            if f.name.startswith("init_") and getattr(self, f.name) is not None
        }


@dataclass(frozen=True, slots=True)
class VideoFrameStats:
    """Frame index, point counts, track repair statistics and per-state PSNR.

    Repair prepares the next frame. The retained fraction is relative to the
    previous frame's count and is zero on the first frame.
    """

    frame: int
    n_requested: int
    n_target: int
    n_target_limited: int
    survivors: int
    kills_for_next: int
    births_for_next: int
    shortfall_for_next: int
    k_used: int
    retained_fraction: float
    rate_sum: float
    psnr: tuple[float, ...]


@dataclass(slots=True, kw_only=True)
class VideoFrameResult(SampleResult):
    """Sample output with row-aligned int64 track IDs and int32 reusable slots."""

    ids: Array
    slots: Array
    stats: VideoFrameStats


class GaussifierSampler:
    """Run the bundled v0.12.0 fused runtime or load a PyTorch checkpoint."""

    def __new__(cls, *args: object, **kwargs: object) -> GaussifierSampler:
        if cls is GaussifierSampler:
            from gaussifier_sampler._torch_inference import TorchGaussifierSampler

            return object.__new__(TorchGaussifierSampler)
        return super().__new__(cls)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> TorchGaussifierSampler:
        """Load a training checkpoint into the optional PyTorch backend."""
        from gaussifier_sampler._torch_inference import TorchGaussifierSampler

        return TorchGaussifierSampler.from_checkpoint(checkpoint_path, device=device)

    @classmethod
    @overload
    def bundled(
        cls,
        *,
        backend: Literal["fused"] = "fused",
        precision: Literal["fp16", "fp8"] = "fp16",
        device: torch.device | str | None = None,
        size_mode: FusedSizeMode = "exact",
    ) -> FusedGaussifierSampler: ...

    @classmethod
    @overload
    def bundled(
        cls,
        *,
        backend: Literal["torch"],
        precision: Literal["fp16"] = "fp16",
        device: torch.device | str | None = None,
        size_mode: Literal["exact"] = "exact",
    ) -> TorchGaussifierSampler: ...

    @classmethod
    def bundled(
        cls,
        *,
        backend: Literal["fused", "torch"] = "fused",
        precision: Literal["fp16", "fp8"] = "fp16",
        device: torch.device | str | None = None,
        size_mode: FusedSizeMode = "exact",
    ) -> GaussifierSampler:
        """Open fused frozen-GN weights or the v0.11.1 PyTorch GN checkpoint.

        ``size_mode`` selects how the fused backend runs sizes that are not multiples of 16.
        """
        if backend == "fused":
            from gaussifier_sampler._fused_inference import FusedGaussifierSampler

            return FusedGaussifierSampler(precision=precision, device=device, size_mode=size_mode)
        if backend == "torch":
            if precision != "fp16":
                raise ValueError("The precision option applies to the fused backend.")
            if size_mode != "exact":
                raise ValueError("The size_mode option applies to the fused backend.")
            return cls.from_checkpoint(BUNDLED_WEIGHTS_PATH, device=device)
        raise ValueError(f"Unknown Gaussifier backend: {backend!r}")

    @classmethod
    def bundled_trt(
        cls, engine_dir: str | Path, *, device: torch.device | str | None = None
    ) -> TorchGaussifierSampler:
        """Open the legacy TensorRT-compiled PyTorch profile."""
        from gaussifier_sampler._torch_inference import TorchGaussifierSampler

        return TorchGaussifierSampler.bundled_trt(engine_dir, device=device)

    @classmethod
    def bundled_aoti(
        cls, package_dir: str | Path, *, device: torch.device | str | None = None
    ) -> TorchGaussifierSampler:
        """Open the legacy AOTI-compiled PyTorch profile."""
        from gaussifier_sampler._torch_inference import TorchGaussifierSampler

        return TorchGaussifierSampler.bundled_aoti(package_dir, device=device)
