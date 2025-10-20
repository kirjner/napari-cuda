"""Worker runtime helpers for scene sources, ROI selection, and level switches."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Mapping, NamedTuple
import logging
import time
from vispy.geometry import Rect

import numpy as np

from napari_cuda.server.runtime.scene_state_applier import SceneStateApplier
from napari_cuda.server.data.zarr_source import ZarrSceneSource
from napari_cuda.server.data.level_budget import LevelBudgetError
from napari_cuda.server.data.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
    viewport_debug_snapshot,
    resolve_worker_viewport_roi,
)
from napari_cuda.server.runtime.scene_types import SliceROI

logger = logging.getLogger(__name__)


def ensure_scene_source(worker) -> ZarrSceneSource:
    """Return a configured ``ZarrSceneSource`` and synchronise worker metadata."""

    assert worker._zarr_path, "No OME-Zarr path configured for scene source"

    source = worker._scene_source
    if source is None:
        source = worker._create_scene_source()
        assert source is not None, "Failed to create ZarrSceneSource"
        worker._scene_source = source

    target_level = source.current_level
    if worker._zarr_level:
        target_level = source.level_index_for_path(worker._zarr_level)

    if worker._log_layer_debug:
        current = int(source.current_level)
        key = (current, worker._zarr_level)
        if getattr(worker, "_last_ensure_log", None) != key:
            logger.debug("ensure_source: current=%d target=%d path=%s", current, int(target_level), worker._zarr_level)
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
    worker._active_ms_level = int(source.current_level)  # type: ignore[attr-defined]
    worker._zarr_level = descriptor.path or None  # type: ignore[attr-defined]
    worker._zarr_axes = ''.join(source.axes)  # type: ignore[attr-defined]
    worker._zarr_shape = descriptor.shape  # type: ignore[attr-defined]
    worker._zarr_dtype = str(source.dtype)  # type: ignore[attr-defined]

    axes_lower = [str(ax).lower() for ax in source.axes]
    if step:
        z_index_pos = axes_lower.index('z') if 'z' in axes_lower else 0
        worker._z_index = int(step[z_index_pos])  # type: ignore[attr-defined]

    return source


def roi_equal(a: SliceROI, b: SliceROI) -> bool:
    """Return True when two ROIs cover identical bounds."""

    return (
        int(a.y_start) == int(b.y_start)
        and int(a.y_stop) == int(b.y_stop)
        and int(a.x_start) == int(b.x_start)
        and int(a.x_stop) == int(b.x_stop)
    )


def apply_plane_slice_roi(
    worker: Any,
    source: ZarrSceneSource,
    level: int,
    roi: SliceROI,
    *,
    update_contrast: bool,
) -> tuple[int, int]:
    """Load and apply a plane slice for the given ROI."""

    z_idx = int(worker._z_index or 0)
    slab = source.slice(int(level), z_idx, compute=True, roi=roi)

    layer = worker._napari_layer
    if layer is not None:
        view = worker.view
        assert view is not None, "VisPy view required for slice apply"
        ctx = worker._build_scene_state_context(view.camera)
        SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi,
            update_contrast=bool(update_contrast),
        )

    height_px = int(slab.shape[0])
    width_px = int(slab.shape[1])
    worker._data_wh = (int(width_px), int(height_px))
    worker._data_d = None
    worker._last_roi = (int(level), roi)
    worker._mark_render_tick_needed()
    return height_px, width_px


def reset_worker_camera(worker, cam) -> None:
    """Reset the VisPy camera to match the current worker dataset extent."""

    assert cam is not None, "VisPy camera expected"
    assert hasattr(cam, "set_range"), "Camera missing set_range handler"

    data_wh = worker._data_wh
    assert data_wh is not None, "Worker missing _data_wh for camera reset"
    w, h = data_wh
    data_d = worker._data_d

    if worker.use_volume:
        extent = worker._volume_world_extents()  # type: ignore[attr-defined]
        if extent is None:
            depth = data_d or 1
            extent = (float(w), float(h), float(depth))
        world_w, world_h, world_d = extent
        cam.set_range(
            x=(0.0, max(1.0, world_w)),
            y=(0.0, max(1.0, world_h)),
            z=(0.0, max(1.0, world_d)),
        )
        worker._frame_volume_camera(world_w, world_h, world_d)  # type: ignore[attr-defined]
        return

    # Set camera range to world extents in 2D
    source = worker._scene_source
    assert source is not None, "scene source must be initialised for 2D reset"
    sy, sx = plane_scale_for_level(source, int(worker._active_ms_level))
    full_h, full_w = plane_wh_for_level(source, int(worker._active_ms_level))
    world_w = float(full_w) * float(max(1e-12, sx))
    world_h = float(full_h) * float(max(1e-12, sy))
    cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))
    cam.rect = Rect(0.0, 0.0, max(1.0, world_w), max(1.0, world_h))





def format_worker_level_roi(
    worker: object,
    source: ZarrSceneSource,
    level: int,
) -> str:
    """Return the stringified ROI description for logging."""

    if worker.use_volume:
        return "volume"
    roi = viewport_roi_for_level(worker, source, level)
    if roi.is_empty():
        return "full"
    return f"y={roi.y_start}:{roi.y_stop} x={roi.x_start}:{roi.x_stop}"


def viewport_roi_for_level(
    worker: object,
    source: ZarrSceneSource,
    level: int,
    *,
    quiet: bool = False,
    for_policy: bool = False,
) -> SliceROI:
    view = worker.view
    align_chunks = (not for_policy) and bool(worker._roi_align_chunks)
    ensure_contains = (not for_policy) and bool(worker._roi_ensure_contains_viewport)
    edge_threshold = int(worker._roi_edge_threshold)
    chunk_pad = int(worker._roi_pad_chunks)

    roi_log = worker._roi_log_state
    log_state = roi_log if isinstance(roi_log, dict) else None

    data_wh = worker._data_wh
    data_depth = worker._data_d

    def _snapshot() -> Dict[str, Any]:
        return viewport_debug_snapshot(
            view=view,
            canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
            data_wh=data_wh,
            data_depth=data_depth,
        )

    reason = "policy-roi" if for_policy else "roi-request"

    roi_cache = worker._roi_cache

    return resolve_worker_viewport_roi(
        view=view,
        canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
        source=source,
        level=int(level),
        align_chunks=align_chunks,
        chunk_pad=chunk_pad,
        ensure_contains_viewport=ensure_contains,
        edge_threshold=edge_threshold,
        for_policy=for_policy,
        roi_cache=roi_cache,
        roi_log_state=log_state,
        snapshot_cb=_snapshot,
        log_layer_debug=worker._log_layer_debug,
        quiet=quiet,
        data_wh=data_wh,
        reason=reason,
        logger_ref=logger,
    )
__all__ = [
    "ensure_scene_source",
    "reset_worker_camera",
    "format_worker_level_roi",
    "viewport_roi_for_level",
    "apply_plane_slice_roi",
    "roi_equal",
]
