"""Canonical layer-control normalization via napari enums.

The client and server both import these helpers to ensure the same
vocabulary is enforced for every layer control surfaced to the UI.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from napari.layers.base._base_constants import Blending as NapariBlending
from napari.layers.image._image_constants import (
    ImageProjectionMode as NapariProjectionMode,
    ImageRendering as NapariImageRendering,
    Interpolation as NapariInterpolation,
)


@dataclass(frozen=True)
class ControlSpec:
    """Describe how to normalise and compare a layer control."""

    key: str
    normalize: Callable[[Any], Any]
    equals: Callable[[Any, Any], bool]
    default: Any | None


def _normalize_bool(value: Any) -> bool:
    return bool(value)


def _normalize_float(value: Any) -> float:
    return float(value)


def _normalize_clim(value: Any) -> tuple[float, float]:
    lo, hi = value
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0
    return (lo_f, hi_f)


def _normalize_colormap(value: Any) -> str:
    return str(value)


def _normalize_blending(value: Any) -> str:
    return NapariBlending(value).value


def _normalize_interpolation(value: Any) -> str:
    return NapariInterpolation(value).value


def _normalize_rendering(value: Any) -> str:
    return NapariImageRendering(value).value


def _normalize_projection_mode(value: Any) -> str:
    return NapariProjectionMode(value).value


def _float_equals(a: Any, b: Any, *, tol: float = 1e-5) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _tuple_float_equals(a: Iterable[Any], b: Iterable[Any], *, tol: float = 1e-5) -> bool:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        return False
    return all(_float_equals(x, y, tol=tol) for x, y in zip(a_list, b_list, strict=False))


LAYER_CONTROL_SPECS: dict[str, ControlSpec] = {
    "visible": ControlSpec(
        key="visible",
        normalize=_normalize_bool,
        equals=lambda a, b: bool(a) is bool(b),
        default=True,
    ),
    "opacity": ControlSpec(
        key="opacity",
        normalize=_normalize_float,
        equals=_float_equals,
        default=1.0,
    ),
    "blending": ControlSpec(
        key="blending",
        normalize=_normalize_blending,
        equals=lambda a, b: str(a) == str(b),
        default=NapariBlending.OPAQUE.value,
    ),
    "interpolation": ControlSpec(
        key="interpolation",
        normalize=_normalize_interpolation,
        equals=lambda a, b: str(a) == str(b),
        default=NapariInterpolation.LINEAR.value,
    ),
    "colormap": ControlSpec(
        key="colormap",
        normalize=_normalize_colormap,
        equals=lambda a, b: str(a) == str(b),
        default="gray",
    ),
    "rendering": ControlSpec(
        key="rendering",
        normalize=_normalize_rendering,
        equals=lambda a, b: str(a) == str(b),
        default=NapariImageRendering.MIP.value,
    ),
    "gamma": ControlSpec(
        key="gamma",
        normalize=_normalize_float,
        equals=lambda a, b: _float_equals(a, b, tol=1e-4),
        default=1.0,
    ),
    "contrast_limits": ControlSpec(
        key="contrast_limits",
        normalize=_normalize_clim,
        equals=lambda a, b: _tuple_float_equals(a, b, tol=1e-4),
        default=None,
    ),
    "attenuation": ControlSpec(
        key="attenuation",
        normalize=_normalize_float,
        equals=_float_equals,
        default=None,
    ),
    "iso_threshold": ControlSpec(
        key="iso_threshold",
        normalize=_normalize_float,
        equals=_float_equals,
        default=None,
    ),
    "projection_mode": ControlSpec(
        key="projection_mode",
        normalize=_normalize_projection_mode,
        equals=lambda a, b: str(a) == str(b),
        default=NapariProjectionMode.MEAN.value,
    ),
}


def normalize_control(key: str, value: Any) -> Any:
    """Normalise ``value`` for the given control ``key`` using napari enums."""

    spec = LAYER_CONTROL_SPECS.get(key)
    if spec is None:
        raise KeyError(f"Unsupported layer control: {key}")
    if value is None:
        raise ValueError(f"Control '{key}' cannot be normalised from None")
    return spec.normalize(value)


def controls_defaults() -> dict[str, Any | None]:
    """Return the canonical defaults for layer controls."""

    return {key: spec.default for key, spec in LAYER_CONTROL_SPECS.items()}


__all__ = [
    "LAYER_CONTROL_SPECS",
    "ControlSpec",
    "controls_defaults",
    "normalize_control",
]
