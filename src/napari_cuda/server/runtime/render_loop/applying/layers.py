"""Layer visual application helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from napari._vispy.layers.image import _napari_cmap_to_vispy
from napari.layers import Image as NapariImage
from napari.utils.colormaps.colormap_utils import ensure_colormap

from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import LayerVisualState

logger = logging.getLogger(__name__)


def _active_visual(worker: Any) -> Any:
    mode = worker.viewport_state.mode  # type: ignore[attr-defined]
    if mode is RenderMode.VOLUME:
        handle = getattr(worker, "_volume_visual_handle", None)
    else:
        handle = getattr(worker, "_plane_visual_handle", None)
    assert handle is not None, "visual handle must be registered before applying layer state"
    return handle.node


def _set_visible(worker: Any, layer: Any, value: Any) -> bool:
    visual = _active_visual(worker)
    val = bool(value)
    layer.visible = val  # type: ignore[assignment]
    visual.visible = val  # type: ignore[attr-defined]
    return True


def _set_opacity(worker: Any, layer: Any, value: Any) -> bool:
    layer.opacity = float(max(0.0, min(1.0, float(value))))  # type: ignore[assignment]
    return True


def _set_blending(worker: Any, layer: Any, value: Any) -> bool:
    layer.blending = str(value)  # type: ignore[assignment]
    return True


def _set_interpolation(worker: Any, layer: Any, value: Any) -> bool:
    """Apply unified interpolation token to both 2D and 3D on Image layers.

    napari exposes separate properties (`interpolation2d`/`interpolation3d`).
    Our state uses a single `interpolation` token. For correctness across
    2D/3D mode switches we set both on Image.
    """
    assert isinstance(layer, NapariImage), "interpolation apply requires napari Image layer"
    token = str(value)
    layer.interpolation2d = token  # type: ignore[assignment]
    layer.interpolation3d = token  # type: ignore[assignment]
    return True


def _set_colormap(worker: Any, layer: Any, value: Any) -> bool:
    cmap = ensure_colormap(value)
    layer.colormap = cmap  # type: ignore[assignment]
    visual = _active_visual(worker)
    logger.debug("layer updates: updating visual colormap to %s", cmap.name)
    visual.cmap = _napari_cmap_to_vispy(cmap)
    return True


def _set_gamma(worker: Any, layer: Any, value: Any) -> bool:
    gamma = float(value)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    layer.gamma = gamma  # type: ignore[assignment]
    visual = _active_visual(worker)
    visual.gamma = gamma  # type: ignore[attr-defined]
    return True


def _set_contrast_limits(worker: Any, layer: Any, value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("contrast_limits must be a length-2 sequence")
    lo = float(value[0])
    hi = float(value[1])
    if hi < lo:
        lo, hi = hi, lo
    layer.contrast_limits = [lo, hi]  # type: ignore[assignment]
    visual = _active_visual(worker)
    visual.clim = (lo, hi)  # type: ignore[attr-defined]
    return True


def _set_depiction(worker: Any, layer: Any, value: Any) -> bool:
    layer.depiction = str(value)  # type: ignore[assignment]
    return True


# 'rendering' is a volume-only control in our model; layer path ignores it.

def _set_projection_mode(worker: Any, layer: Any, value: Any) -> bool:
    layer.projection_mode = str(value)  # type: ignore[assignment]
    return True

def _set_plane_thickness(worker: Any, layer: Any, value: Any) -> bool:
    layer.plane.thickness = float(value)  # type: ignore[assignment]
    return True


def _set_attenuation(worker: Any, layer: Any, value: Any) -> bool:
    layer.attenuation = float(value)  # type: ignore[assignment]
    return True


def _set_iso_threshold(worker: Any, layer: Any, value: Any) -> bool:
    layer.iso_threshold = float(value)  # type: ignore[assignment]
    return True


def _set_metadata(worker: Any, layer: Any, value: Any) -> bool:
    if value is None:
        layer.metadata = {}
        return True
    if not isinstance(value, Mapping):
        raise ValueError("metadata updates require mapping payloads")
    layer.metadata = {str(m_key): m_val for m_key, m_val in value.items()}
    return True


def _noop_setter(worker: Any, layer: Any, value: Any) -> bool:
    return False


_PLANE_ONLY_PROPS = {"projection_mode"}
_ALLOW_NONE_PROPS = {"metadata", "thumbnail"}
_LAYER_SETTERS = {
    "visible": _set_visible,
    "opacity": _set_opacity,
    "blending": _set_blending,
    "interpolation": _set_interpolation,
    "colormap": _set_colormap,
    "gamma": _set_gamma,
    "contrast_limits": _set_contrast_limits,
    "depiction": _set_depiction,
    "projection_mode": _set_projection_mode,
    "plane_thickness": _set_plane_thickness,
    "attenuation": _set_attenuation,
    "iso_threshold": _set_iso_threshold,
    "metadata": _set_metadata,
    "thumbnail": _noop_setter,
}


def apply_layer_visual_state(worker: Any, layer_state: LayerVisualState, *, mode: RenderMode) -> bool:
    """Apply a single layer visual state to the active napari layer."""

    layer = worker._napari_layer  # type: ignore[attr-defined]

    changed = False
    active_keys = layer_state.keys()
    for key in active_keys:
        # Skip plane-only props in volume
        if key in _PLANE_ONLY_PROPS and mode is RenderMode.VOLUME:
            continue
        # depiction is 3D-only; ignore in plane mode
        if key == "depiction" and mode is not RenderMode.VOLUME:
            continue
        # plane_thickness is 3D-only (slicing plane)
        if key == "plane_thickness" and mode is not RenderMode.VOLUME:
            continue
        setter = _LAYER_SETTERS.get(key)
        if setter is None:
            continue
        value = layer_state.get(key)
        if value is None and key not in _ALLOW_NONE_PROPS:
            continue
        if setter(worker, layer, value):
            changed = True
    return changed


def apply_layer_visual_updates(worker: Any, updates: Mapping[str, LayerVisualState]) -> bool:
    """Apply a mapping of layer visual updates."""

    if not updates:
        return False

    mode = worker.viewport_state.mode  # type: ignore[attr-defined]
    changed = False
    for layer_state in updates.values():
        if apply_layer_visual_state(worker, layer_state, mode=mode):
            changed = True

    if changed:
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]
    return changed


__all__ = ["apply_layer_visual_state", "apply_layer_visual_updates"]
