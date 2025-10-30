"""Worker runtime helpers for scene source initialisation and camera resets."""

from __future__ import annotations

import logging
import time
from typing import Any

from vispy.geometry import Rect

from napari_cuda.server.data.roi import plane_scale_for_level, plane_wh_for_level
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.runtime.viewport import RenderMode

logger = logging.getLogger(__name__)

try:
    import dask.array as da  # type: ignore
except Exception as _da_err:  # pragma: no cover - optional dependency
    da = None  # type: ignore[assignment]
    logger.warning(
        "dask.array not available; OME-Zarr features disabled: %s",
        _da_err,
    )


def create_scene_source(worker: Any) -> Optional[ZarrSceneSource]:
    """Create a ``ZarrSceneSource`` based on worker configuration."""

    path = getattr(worker, "_zarr_path", None)
    if not path:
        return None
    if da is None:
        raise RuntimeError("ZarrSceneSource requires dask.array to be available")
    axes_override = getattr(worker, "_zarr_axes", None)
    if isinstance(axes_override, str):
        axes_override = tuple(axes_override)
    source = ZarrSceneSource(
        path,
        preferred_level=getattr(worker, "_zarr_level", None),
        axis_override=axes_override,
    )
    return source


def ensure_scene_source(worker: Any) -> ZarrSceneSource:
    """Return a configured ``ZarrSceneSource`` and synchronise worker metadata."""

    assert worker._zarr_path, "No OME-Zarr path configured for scene source"

    source = worker._scene_source
    if source is None:
        source = create_scene_source(worker)
        assert source is not None, "Failed to create ZarrSceneSource"
        worker._scene_source = source

    target_level = source.current_level
    if worker._zarr_level:
        target_level = source.level_index_for_path(worker._zarr_level)

    if worker._log_layer_debug:
        current = int(source.current_level)
        key = (current, worker._zarr_level)
        if getattr(worker, "_last_ensure_log", None) != key:
            logger.debug(
                "ensure_source: current=%d target=%d path=%s",
                current,
                int(target_level),
                worker._zarr_level,
            )
            worker._last_ensure_log = key
            worker._last_ensure_log_ts = time.perf_counter()

    with worker._state_lock:
        ledger_step = None
        ledger = worker._ledger
        assert ledger is not None, "state ledger must be attached before ensure_scene_source"
        entry = ledger.get("dims", "main", "current_step")
        if entry is not None and isinstance(entry.value, (list, tuple)):
            ledger_step = tuple(int(v) for v in entry.value)
        if ledger_step is None:
            initial_step = source.initial_step(level=target_level)
            ledger_step = tuple(int(v) for v in initial_step)
        step = source.set_current_slice(ledger_step, int(target_level))

    descriptor = source.level_descriptors[source.current_level]
    worker._set_current_level_index(int(source.current_level))  # type: ignore[attr-defined]
    worker._zarr_level = descriptor.path or None  # type: ignore[attr-defined]
    worker._zarr_axes = "".join(source.axes)  # type: ignore[attr-defined]
    worker._zarr_shape = descriptor.shape  # type: ignore[attr-defined]
    worker._zarr_dtype = str(source.dtype)  # type: ignore[attr-defined]

    axes_lower = [str(ax).lower() for ax in source.axes]
    if step:
        z_index_pos = axes_lower.index("z") if "z" in axes_lower else 0
        worker._z_index = int(step[z_index_pos])  # type: ignore[attr-defined]

    return source


def reset_worker_camera(worker: Any, cam: Any) -> None:
    """Reset the VisPy camera to match the current worker dataset extent."""

    assert cam is not None, "VisPy camera expected"
    assert hasattr(cam, "set_range"), "Camera missing set_range handler"

    data_wh = worker._data_wh
    assert data_wh is not None, "Worker missing _data_wh for camera reset"
    width, height = data_wh
    data_depth = worker._data_d

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        extent = worker._volume_world_extents()  # type: ignore[attr-defined]
        if extent is None:
            depth = data_depth or 1
            extent = (float(width), float(height), float(depth))
        world_w, world_h, world_d = extent
        cam.set_range(
            x=(0.0, max(1.0, world_w)),
            y=(0.0, max(1.0, world_h)),
            z=(0.0, max(1.0, world_d)),
        )
        worker._frame_volume_camera(world_w, world_h, world_d)  # type: ignore[attr-defined]
        return

    source = worker._scene_source
    assert source is not None, "scene source must be initialised for 2D reset"
    current_level = int(worker._current_level_index())  # type: ignore[attr-defined]
    sy, sx = plane_scale_for_level(source, current_level)
    full_h, full_w = plane_wh_for_level(source, current_level)
    world_w = float(full_w) * float(max(1e-12, sx))
    world_h = float(full_h) * float(max(1e-12, sy))
    cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
    cam.rect = Rect(0.0, 0.0, max(1.0, world_w), max(1.0, world_h))


__all__ = ["create_scene_source", "ensure_scene_source", "reset_worker_camera"]
