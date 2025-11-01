"""Helpers for applying viewport-related state updates on the worker."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from vispy.scene.cameras import TurntableCamera

from napari._vispy.layers.image import _napari_cmap_to_vispy
from napari.layers.image._image_constants import (
    ImageRendering as NapariImageRendering,
)
from napari.utils.colormaps.colormap_utils import ensure_colormap
from napari_cuda.server.runtime.viewport.state import RenderMode
from napari_cuda.server.scene import (
    RenderLedgerSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrainOutcome:
    """Result of applying viewer/camera updates."""

    z_index: Optional[int] = None
    data_wh: Optional[tuple[int, int]] = None
    render_marked: bool = False


def apply_dims_step(worker: Any, current_step: Sequence[int]) -> tuple[Optional[int], bool]:
    """Update viewer dims and source slice from the provided step."""

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        return None, False

    steps = tuple(int(x) for x in current_step)
    viewer = worker._viewer  # type: ignore[attr-defined]
    assert viewer is not None, "viewer must exist before applying dims"
    viewer.dims.current_step = steps  # type: ignore[attr-defined]

    source = worker._ensure_scene_source()  # type: ignore[attr-defined]
    axes = getattr(source, "axes", ())
    if isinstance(axes, str):
        axes = tuple(axes)

    z_index: Optional[int] = None
    if isinstance(axes, (list, tuple)) and "z" in axes:
        zi = axes.index("z")
        if zi < len(steps):
            z_index = int(steps[zi])

    if z_index is None:
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]
        return None, True

    current_z = getattr(worker, "_z_index", None)
    if current_z is not None and int(z_index) == int(current_z):
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]
        return int(z_index), True

    zi = axes.index("z") if isinstance(axes, (list, tuple)) and "z" in axes else 0
    base = list(getattr(source, "current_step", steps) or steps)
    level_idx = int(worker._current_level_index())  # type: ignore[attr-defined]
    lvl_shape = source.level_shape(level_idx)
    if len(base) < len(lvl_shape):
        base.extend(0 for _ in range(len(lvl_shape) - len(base)))
    base[zi] = int(z_index)

    with worker._state_lock:  # type: ignore[attr-defined]
        source.set_current_slice(tuple(int(x) for x in base), level_idx)

    request_idr = getattr(worker, "_request_encoder_idr", None)
    if worker._idr_on_z and request_idr is not None:  # type: ignore[attr-defined]
        request_idr()

    worker._mark_render_tick_needed()  # type: ignore[attr-defined]
    return int(z_index), True


def apply_volume_visual_params(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Update volume visual properties from the snapshot."""

    if worker.viewport_state.mode is not RenderMode.VOLUME:  # type: ignore[attr-defined]
        return

    visual = worker._ensure_volume_visual()  # type: ignore[attr-defined]

    if snapshot.volume_mode:
        token = str(snapshot.volume_mode).strip().lower()
        visual.method = NapariImageRendering(token).value  # type: ignore[attr-defined]

    if snapshot.volume_colormap is not None:
        cmap = ensure_colormap(snapshot.volume_colormap)
        visual.cmap = _napari_cmap_to_vispy(cmap)  # type: ignore[attr-defined]

    clim = snapshot.volume_clim
    if isinstance(clim, tuple) and len(clim) >= 2:
        lo = float(clim[0])
        hi = float(clim[1])
        if hi < lo:
            lo, hi = hi, lo
        visual.clim = (lo, hi)  # type: ignore[attr-defined]

    if snapshot.volume_opacity is not None:
        visual.opacity = float(max(0.0, min(1.0, float(snapshot.volume_opacity))))  # type: ignore[attr-defined]

    if snapshot.volume_sample_step is not None:
        if hasattr(visual, "relative_step_size"):
            visual.relative_step_size = float(  # type: ignore[attr-defined]
                max(0.1, min(4.0, float(snapshot.volume_sample_step)))
            )
        else:
            logger.debug(
                "volume params: visual %s missing relative_step_size; skipping sample_step update",
                type(visual).__name__,
            )


def apply_layer_updates(worker: Any, updates: Mapping[str, Mapping[str, Any]]) -> bool:
    """Apply layer property updates from the snapshot."""

    layer = getattr(worker, "_napari_layer", None)
    if layer is None or not updates:
        return False

    def _active_visual() -> Any:
        if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
            return worker._ensure_volume_visual()  # type: ignore[attr-defined]
        return worker._ensure_plane_visual()  # type: ignore[attr-defined]

    for props in updates.values():
        for key, value in props.items():
            if key == "visible":
                layer.visible = bool(value)  # type: ignore[assignment]
                visual = _active_visual()
                visual.visible = bool(value)  # type: ignore[attr-defined]
            elif key == "opacity":
                layer.opacity = float(max(0.0, min(1.0, float(value))))  # type: ignore[assignment]
            elif key == "blending":
                layer.blending = str(value)  # type: ignore[assignment]
            elif key == "interpolation":
                layer.interpolation = str(value)  # type: ignore[assignment]
            elif key == "gamma":
                gamma = float(value)
                if not gamma > 0.0:
                    raise ValueError("gamma must be positive")
                layer.gamma = gamma  # type: ignore[assignment]
                visual = _active_visual()
                visual.gamma = gamma  # type: ignore[attr-defined]
            elif key == "contrast_limits":
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    raise ValueError("contrast_limits must be a length-2 sequence")
                lo = float(value[0])
                hi = float(value[1])
                if hi < lo:
                    lo, hi = hi, lo
                layer.contrast_limits = [lo, hi]  # type: ignore[assignment]
                visual = _active_visual()
                visual.clim = (lo, hi)  # type: ignore[attr-defined]
            elif key == "colormap":
                cmap = ensure_colormap(value)
                layer.colormap = cmap  # type: ignore[assignment]
                visual = _active_visual()
                logger.debug("layer updates: updating visual colormap to %s", cmap.name)
                visual.cmap = _napari_cmap_to_vispy(cmap)
            elif key == "depiction":
                if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
                    continue
                layer.depiction = str(value)  # type: ignore[assignment]
            elif key == "rendering":
                if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
                    continue
                layer.rendering = str(value)  # type: ignore[assignment]
            elif key == "attenuation":
                layer.attenuation = float(value)  # type: ignore[assignment]
            elif key == "iso_threshold":
                layer.iso_threshold = float(value)  # type: ignore[assignment]
            elif key == "metadata":
                if not isinstance(value, Mapping):
                    raise ValueError("metadata updates require mapping payloads")
                metadata_map = dict(layer.metadata)
                for m_key, m_val in value.items():
                    metadata_map[str(m_key)] = m_val
                layer.metadata = metadata_map
            else:
                raise KeyError(f"Unsupported layer property '{key}'")

    worker._mark_render_tick_needed()  # type: ignore[attr-defined]
    return True


def apply_camera_overrides(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Sync camera state from the snapshot when provided."""

    view = worker.view  # type: ignore[attr-defined]
    if view is None:
        return
    cam = view.camera
    if cam is None:
        return

    if snapshot.plane_center is not None and not isinstance(cam, TurntableCamera) and hasattr(cam, "center"):
        cam.center = tuple(float(v) for v in snapshot.plane_center)  # type: ignore[attr-defined]
    if snapshot.plane_zoom is not None and not isinstance(cam, TurntableCamera) and hasattr(cam, "zoom"):
        cam.zoom = float(snapshot.plane_zoom)  # type: ignore[attr-defined]

    if isinstance(cam, TurntableCamera):
        if snapshot.volume_center is not None:
            cx, cy, cz = snapshot.volume_center
            cam.center = (float(cx), float(cy), float(cz))
        if snapshot.volume_angles is not None:
            az, el, roll = snapshot.volume_angles
            cam.azimuth = float(az)  # type: ignore[attr-defined]
            cam.elevation = float(el)  # type: ignore[attr-defined]
            cam.roll = float(roll)  # type: ignore[attr-defined]
        if snapshot.volume_distance is not None:
            cam.distance = float(snapshot.volume_distance)
        if snapshot.volume_fov is not None:
            cam.fov = float(snapshot.volume_fov)


def drain_render_state(worker: Any, snapshot: RenderLedgerSnapshot) -> DrainOutcome:
    """Apply viewer, visual, and camera updates from the snapshot."""

    assert worker.view is not None, "drain_render_state requires an active VisPy view"  # type: ignore[attr-defined]

    z_index_update: Optional[int] = None
    render_marked = False

    if worker.viewport_state.mode is not RenderMode.VOLUME and snapshot.current_step is not None:  # type: ignore[attr-defined]
        z_idx, marked = apply_dims_step(worker, snapshot.current_step)
        if z_idx is not None:
            z_index_update = z_idx
        render_marked = render_marked or marked

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        apply_volume_visual_params(worker, snapshot)

    if snapshot.layer_values:
        if apply_layer_updates(worker, snapshot.layer_values):
            render_marked = True

    apply_camera_overrides(worker, snapshot)

    return DrainOutcome(z_index=z_index_update, render_marked=render_marked)


__all__ = [
    "DrainOutcome",
    "apply_camera_overrides",
    "apply_dims_step",
    "apply_layer_updates",
    "apply_volume_visual_params",
    "drain_render_state",
]
