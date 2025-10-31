"""Layer application helpers for viewport-managed napari layers."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from napari.layers.base._base_constants import Blending as NapariBlending
from napari.layers.image._image_constants import (
    ImageRendering as NapariImageRendering,
)
from napari_cuda.server.data.roi import plane_scale_for_level
from napari_cuda.server.data.roi_applier import SliceDataApplier
from napari_cuda.server.data import SliceROI

logger = logging.getLogger(__name__)


def apply_slice_layer_data(
    *,
    layer: Any,
    source: Any,
    level: int,
    slab: Any,
    roi: Optional[SliceROI],
    update_contrast: bool,
) -> tuple[float, float]:
    """Apply the requested slice slab to the active napari image layer."""

    sy, sx = plane_scale_for_level(source, int(level))
    roi_to_apply = roi or SliceROI(0, int(slab.shape[0]), 0, int(slab.shape[1]))
    SliceDataApplier(layer=layer).apply(slab=slab, roi=roi_to_apply, scale=(sy, sx))

    layer.visible = True  # type: ignore[assignment]
    layer.opacity = 1.0  # type: ignore[assignment]
    if not getattr(layer, "blending", None):
        layer.blending = NapariBlending.OPAQUE.value  # type: ignore[assignment]

    if update_contrast:
        smin = float(np.nanmin(slab)) if hasattr(np, "nanmin") else float(np.min(slab))
        smax = float(np.nanmax(slab)) if hasattr(np, "nanmax") else float(np.max(slab))
        if not math.isfinite(smin) or not math.isfinite(smax) or smax <= smin or (0.0 <= smin <= 1.0 and 0.0 <= smax <= 1.1):
            layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
        else:
            layer.contrast_limits = [smin, smax]  # type: ignore[assignment]

    return float(sy), float(sx)


def apply_volume_layer_data(
    *,
    layer: Any,
    volume: Any,
    contrast: tuple[float, float],
    scale: tuple[float, float, float],
    ensure_volume_visual: Optional[Callable[[], Any]] = None,
) -> tuple[tuple[int, int], Optional[int]]:
    """Apply the provided volume data to the active napari layer."""

    if ensure_volume_visual is not None:
        ensure_volume_visual()

    if layer is not None:
        layer.depiction = "volume"  # type: ignore[assignment]
        layer.rendering = NapariImageRendering.MIP.value  # type: ignore[assignment]
        layer.data = volume

        lo = float(contrast[0])
        hi = float(contrast[1])
        if hi <= lo:
            hi = lo + 1.0

        layer.translate = tuple(0.0 for _ in range(int(volume.ndim)))  # type: ignore[assignment]

        data_min = float(np.nanmin(volume)) if hasattr(np, "nanmin") else float(np.min(volume))
        data_max = float(np.nanmax(volume)) if hasattr(np, "nanmax") else float(np.max(volume))
        normalized = (-0.05 <= data_min <= 1.05) and (-0.05 <= data_max <= 1.05)
        logger.debug(
            "volume.apply stats: min=%.6f max=%.6f contrast=(%.6f, %.6f) normalized=%s",
            data_min,
            data_max,
            lo,
            hi,
            normalized,
        )
        if normalized:
            layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
        else:
            layer.contrast_limits = [lo, hi]  # type: ignore[assignment]

        scale_vals: tuple[float, ...] = tuple(float(s) for s in scale)
        if len(scale_vals) < int(volume.ndim):
            pad = int(volume.ndim) - len(scale_vals)
            scale_vals = tuple(1.0 for _ in range(pad)) + scale_vals
        scale_vals = tuple(scale_vals[-int(volume.ndim):])
        layer.scale = scale_vals  # type: ignore[assignment]

    depth = int(volume.shape[0])
    height = int(volume.shape[1]) if int(volume.ndim) >= 2 else int(volume.shape[-1])
    width = int(volume.shape[2]) if int(volume.ndim) >= 3 else int(volume.shape[-1])
    return (int(width), int(height)), depth


__all__ = ["apply_slice_layer_data", "apply_volume_layer_data"]
