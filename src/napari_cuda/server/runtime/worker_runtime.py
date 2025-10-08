"""Worker runtime helpers for scene sources, ROI selection, and level switches."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple, Mapping
import logging
import time

import numpy as np

import napari_cuda.server.data.lod as lod
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
from napari_cuda.server.data.roi_applier import refresh_slice_for_worker

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

    if worker._last_step is None:
        worker._last_step = source.initial_step(level=target_level)

    with worker._state_lock:
        step = source.set_current_slice(worker._last_step, int(target_level))

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


def notify_scene_refresh(worker, step_hint: Optional[Tuple[int, ...]] = None) -> None:
    """Invoke the worker's scene refresh callback with the best step hint."""

    callback = worker._scene_refresh_cb
    if callback is None:
        return

    if step_hint is not None:
        callback(step_hint)
        return

    assert worker._last_step is not None, "Worker missing last_step for scene refresh"
    callback(worker._last_step)


def refresh_worker_slice_if_needed(worker) -> None:
    """Refresh the napari slice layer when ROI drift warrants it."""

    if getattr(worker, "use_volume", False):
        return

    source = getattr(worker, "_scene_source", None)
    layer = getattr(worker, "_napari_layer", None)
    if source is None or layer is None:
        return

    def _apply_slice(slab: np.ndarray, roi_to_apply: SliceROI) -> None:
        view = getattr(worker, "view", None)
        cam = getattr(view, "camera", None)
        ctx = worker._build_scene_state_context(cam)  # type: ignore[attr-defined]
        SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi_to_apply,
            update_contrast=False,
        )

    def _viewport(src: ZarrSceneSource, lvl: int) -> SliceROI:
        return viewport_roi_for_level(worker, src, lvl)

    refreshed, new_last = refresh_slice_for_worker(
        source=source,
        level=int(getattr(worker, "_active_ms_level", 0)),
        last_roi=getattr(worker, "_last_roi", None),
        z_index=getattr(worker, "_z_index", None),
        edge_threshold=int(getattr(worker, "_roi_edge_threshold", 0)),
        viewport_roi_for_level=_viewport,
        load_slice=worker._load_slice,  # type: ignore[attr-defined]
        apply_slice=_apply_slice,
    )

    if refreshed:
        worker._last_roi = new_last  # type: ignore[attr-defined]


def reset_worker_camera(worker, cam) -> None:
    """Reset the VisPy camera to match the current worker dataset extent."""

    assert cam is not None, "VisPy camera expected"
    assert hasattr(cam, "set_range"), "Camera missing set_range handler"

    data_wh = getattr(worker, "_data_wh", None)
    assert data_wh is not None, "Worker missing _data_wh for camera reset"
    w, h = data_wh
    data_d = getattr(worker, "_data_d", None)

    if getattr(worker, "use_volume", False):
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

    sy = sx = 1.0
    source = getattr(worker, "_scene_source", None)
    if source is not None:
        sy, sx = plane_scale_for_level(source, int(getattr(worker, "_active_ms_level", 0)))
    world_w = float(w) * float(max(1e-12, sx))
    world_h = float(h) * float(max(1e-12, sy))
    cam.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))


def apply_worker_level(
    worker: object,
    source: ZarrSceneSource,
    level: int,
    *,
    prev_level: Optional[int] = None,
    restoring_plane_state: bool = False,
) -> lod.AppliedLevel:
    """Apply ``level`` through the worker's existing helpers and return the snapshot."""

    applied = lod.apply_level(
        source=source,
        target_level=int(level),
        prev_level=prev_level,
        last_step=getattr(worker, "_last_step", None),
        viewer=getattr(worker, "_viewer", None),
        restoring_plane_state=restoring_plane_state,
    )

    descriptor = source.level_descriptors[int(level)]
    worker._update_level_metadata(descriptor, applied)  # type: ignore[attr-defined]

    if getattr(worker, "use_volume", False):
        apply_worker_volume_level(worker, source, applied)
    else:
        apply_worker_slice_level(worker, source, applied)

    # Render loop drain will emit the notify.dims once the new level is in place.
    return applied


def apply_worker_volume_level(
    worker: object,
    source: ZarrSceneSource,
    applied: lod.AppliedLevel,
) -> None:
    """Apply a volume level and emit layer logging via the worker hooks."""

    try:
        scale_vals = [float(s) for s in source.level_scale(applied.level)]
    except Exception:
        scale_vals = []
    while len(scale_vals) < 3:
        scale_vals.insert(0, 1.0)
    scale_tuple = (
        float(scale_vals[-3]),
        float(scale_vals[-2]),
        float(scale_vals[-1]),
    )
    worker._volume_scale = scale_tuple  # type: ignore[attr-defined]
    volume = worker._get_level_volume(source, applied.level)  # type: ignore[attr-defined]
    cam = worker.view.camera if getattr(worker, "view", None) is not None else None
    ctx = worker._build_scene_state_context(cam)  # type: ignore[attr-defined]
    if ctx.volume_scale is None:
        ctx = replace(ctx, volume_scale=scale_tuple)
    data_wh, data_d = SceneStateApplier.apply_volume_layer(
        ctx,
        volume=volume,
        contrast=applied.contrast,
    )
    worker._data_wh = data_wh  # type: ignore[attr-defined]
    worker._data_d = data_d  # type: ignore[attr-defined]
    volume_shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )
    worker._layer_logger.log(  # type: ignore[attr-defined]
        enabled=worker._log_layer_debug,  # type: ignore[attr-defined]
        mode="volume",
        level=applied.level,
        z_index=None,
        shape=volume_shape,
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # type: ignore[attr-defined]
    )


def apply_worker_slice_level(
    worker: object,
    source: ZarrSceneSource,
    applied: lod.AppliedLevel,
) -> None:
    """Apply a slice level and refresh the napari layer state."""

    layer = getattr(worker, "_napari_layer", None)
    sy, sx = applied.scale_yx
    if layer is not None:
        try:
            layer.scale = (sy, sx)
        except Exception:
            logger.debug("apply_level: setting 2D layer scale pre-slab failed", exc_info=True)

    z_idx = int(getattr(worker, "_z_index", 0) or 0)
    slab = worker._load_slice(source, applied.level, z_idx)  # type: ignore[attr-defined]
    roi_for_layer = None
    last_roi = getattr(worker, "_last_roi", None)
    if last_roi is not None and int(last_roi[0]) == int(applied.level):
        roi_for_layer = last_roi[1]

    if layer is not None:
        view = getattr(worker, "view", None)
        assert view is not None
        ctx = worker._build_scene_state_context(view.camera)  # type: ignore[attr-defined]
        SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi_for_layer,
            update_contrast=not getattr(worker, "_sticky_contrast", False),
        )

    h, w = int(slab.shape[0]), int(slab.shape[1])
    worker._data_wh = (w, h)  # type: ignore[attr-defined]
    worker._data_d = None  # type: ignore[attr-defined]

    view = getattr(worker, "view", None)
    if (
        not getattr(worker, "_preserve_view_on_switch", False)
        and view is not None
        and getattr(view, "camera", None) is not None
    ):
        world_w = float(w) * float(max(1e-12, sx))
        world_h = float(h) * float(max(1e-12, sy))
        view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    worker._layer_logger.log(  # type: ignore[attr-defined]
        enabled=worker._log_layer_debug,  # type: ignore[attr-defined]
        mode="slice",
        level=applied.level,
        z_index=getattr(worker, "_z_index", None),
        shape=(int(h), int(w)),
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # type: ignore[attr-defined]
    )


def format_worker_level_roi(
    worker: object,
    source: ZarrSceneSource,
    level: int,
) -> str:
    """Return the stringified ROI description for logging."""

    if getattr(worker, "use_volume", False):
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
    view = getattr(worker, "view", None)
    align_chunks = (not for_policy) and bool(getattr(worker, "_roi_align_chunks", True))
    ensure_contains = (not for_policy) and bool(getattr(worker, "_roi_ensure_contains_viewport", True))
    edge_threshold = int(getattr(worker, "_roi_edge_threshold", 4))
    chunk_pad = int(getattr(worker, "_roi_pad_chunks", 1))

    roi_log = getattr(worker, "_roi_log_state", None)
    log_state = roi_log if isinstance(roi_log, dict) else None

    def _snapshot() -> Dict[str, Any]:
        return viewport_debug_snapshot(
            view=getattr(worker, "view", None),
            canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
            data_wh=getattr(worker, "_data_wh", None),
            data_depth=getattr(worker, "_data_d", None),
        )

    reason = "policy-roi" if for_policy else "roi-request"

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
        roi_cache=getattr(worker, "_roi_cache", None),
        roi_log_state=log_state,
        snapshot_cb=_snapshot,
        log_layer_debug=getattr(worker, "_log_layer_debug", False),
        quiet=quiet,
        data_wh=getattr(worker, "_data_wh", None),
        reason=reason,
        logger_ref=logger,
    )


def set_level_with_budget(
    worker: object,
    desired_level: int,
    *,
    reason: str,
    budget_error: type[Exception],
    restoring_plane_state: bool = False,
) -> None:
    source = worker._ensure_scene_source()  # type: ignore[attr-defined]

    def _budget_check(scene: ZarrSceneSource, level: int) -> None:
        try:
            if getattr(worker, "use_volume", False):
                worker._volume_budget_allows(scene, level)  # type: ignore[attr-defined]
            else:
                worker._slice_budget_allows(scene, level)  # type: ignore[attr-defined]
        except budget_error as exc:
            raise LevelBudgetError(str(exc)) from exc

    def _apply(scene: ZarrSceneSource, level: int, prev_level: Optional[int]) -> lod.AppliedLevel:
        return apply_worker_level(
            worker,
            scene,
            level,
            prev_level=prev_level,
            restoring_plane_state=restoring_plane_state,
        )

    def _on_switch(prev_level: int, applied: int, elapsed_ms: float) -> None:
        roi_desc = format_worker_level_roi(worker, source, applied)
        worker._switch_logger.log(  # type: ignore[attr-defined]
            enabled=getattr(worker, "_log_layer_debug", False),
            previous=prev_level,
            applied=applied,
            roi_desc=roi_desc,
            reason=reason,
            elapsed_ms=elapsed_ms,
        )
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]

    try:
        applied_snapshot, downgraded = lod.apply_level_with_context(
            desired_level=desired_level,
            use_volume=getattr(worker, "use_volume", False),
            source=source,
            current_level=int(getattr(worker, "_active_ms_level", 0)),
            log_layer_debug=getattr(worker, "_log_layer_debug", False),
            budget_check=_budget_check,
            apply_level_fn=_apply,
            on_switch=_on_switch,
            roi_cache=getattr(worker, "_roi_cache", None),
            roi_log_state=getattr(worker, "_roi_log_state", None),
        )
    except LevelBudgetError as exc:
        raise budget_error(str(exc)) from exc


    worker._active_ms_level = int(applied_snapshot.level)  # type: ignore[attr-defined]
    worker._level_downgraded = bool(downgraded)  # type: ignore[attr-defined]


def perform_level_switch(
    worker: object,
    *,
    target_level: int,
    reason: str,
    requested_level: Optional[int],
    selected_level: Optional[int],
    source: Optional[ZarrSceneSource] = None,
    budget_error: type[Exception],
    restoring_plane_state: bool = False,
) -> None:
    if not getattr(worker, "_zarr_path", None):
        return
    if getattr(worker, "_lock_level", None) is not None:
        if getattr(worker, "_log_layer_debug", False) and logger.isEnabledFor(logging.INFO):
            logger.info("perform_level_switch ignored due to lock_level=%s", str(worker._lock_level))  # type: ignore[attr-defined]
        return
    if source is None:
        source = worker._ensure_scene_source()  # type: ignore[attr-defined]
    target_level = int(target_level)
    ctx = worker._build_policy_context(source, requested_level=target_level)  # type: ignore[attr-defined]
    prev = int(getattr(worker, "_active_ms_level", 0))
    set_level_with_budget(
        worker,
        target_level,
        reason=reason,
        budget_error=budget_error,
        restoring_plane_state=restoring_plane_state,
    )
    if reason in {"zoom-in", "zoom-out"}:
        worker._last_zoom_hint_ts = time.perf_counter()  # type: ignore[attr-defined]
    idle_ms = max(0.0, (time.perf_counter() - getattr(worker, "_last_interaction_ts", time.perf_counter())) * 1000.0)
    worker._policy_metrics.record(  # type: ignore[attr-defined]
        policy=getattr(worker, "_policy_name", "oversampling"),
        requested_level=requested_level if requested_level is not None else ctx.requested_level,
        selected_level=selected_level if selected_level is not None else target_level,
        desired_level=target_level,
        applied_level=int(getattr(worker, "_active_ms_level", target_level)),
        reason=reason,
        idle_ms=idle_ms,
        oversampling=ctx.level_oversampling,
        downgraded=bool(getattr(worker, "_level_downgraded", False)),
        from_level=prev,
    )


__all__ = [
    "ensure_scene_source",
    "notify_scene_refresh",
    "refresh_worker_slice_if_needed",
    "reset_worker_camera",
    "apply_worker_level",
    "apply_worker_volume_level",
    "apply_worker_slice_level",
    "format_worker_level_roi",
    "viewport_roi_for_level",
    "set_level_with_budget",
    "perform_level_switch",
]
