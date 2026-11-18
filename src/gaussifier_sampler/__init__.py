"""Inference for the Gaussifier model: density map, equal-mass Voronoi placement, and the
recurrent correction head, from a bundled checkpoint or a compiled backend."""

from importlib import import_module

from gaussifier_sampler.inference import (
    FusedSizeMode,
    GaussifierSampler,
    InferenceProfile,
    RendererFn,
    SampleResult,
    VideoFrameResult,
    VideoFrameStats,
)

_TORCH_EXPORTS = {
    "FirstStageModel": "model",
    "LoadReport": "model",
    "ModelConfig": "model",
    "load_first_stage_model": "model",
    "equal_mass_voronoi": "voronoi_samplers",
    "equal_mass_voronoi_capturable": "voronoi_samplers",
}


def __getattr__(name: str):
    if name in _TORCH_EXPORTS:
        module = import_module(f"gaussifier_sampler.{_TORCH_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FirstStageModel",
    "FusedSizeMode",
    "GaussifierSampler",
    "InferenceProfile",
    "LoadReport",
    "ModelConfig",
    "RendererFn",
    "SampleResult",
    "VideoFrameResult",
    "VideoFrameStats",
    "equal_mass_voronoi",
    "equal_mass_voronoi_capturable",
    "load_first_stage_model",
]
