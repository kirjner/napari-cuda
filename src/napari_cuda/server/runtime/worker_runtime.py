"""Worker runtime helpers for scene sources, ROI selection, and level switches."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple, Mapping, NamedTuple
import logging
import time
from vispy.geometry import Rect

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




def prepare_worker_level(
    worker: object,
    source: ZarrSceneSource,
    level: int,
    *,
    prev_level: Optional[int] = None,
    ledger_step: Optional[Sequence[int]] = None,
) -> lod.AppliedLevel:
    """Return the ``AppliedLevel`` snapshot without touching the napari layer."""

    step_authoritative = ledger_step is not None
    if ledger_step is not None:
        step_hint: Optional[tuple[int, ...]] = tuple(int(v) for v in ledger_step)
    else:
        recorded_step = worker._ledger_step()
        step_hint = tuple(int(v) for v in recorded_step) if recorded_step is not None else None

    applied = lod.apply_level(
        source=source,
        target_level=int(level),
        prev_level=prev_level,
        last_step=step_hint,
        viewer=getattr(worker, "_viewer", None),
        step_is_authoritative=step_authoritative,
    )

    descriptor = source.level_descriptors[int(level)]
    worker._update_level_metadata(descriptor, applied)  # type: ignore[attr-defined]

    return applied


def apply_worker_level(
    worker: object,
    source: ZarrSceneSource,
    level: int,
    *,
    prev_level: Optional[int] = None,
    ledger_step: Optional[Sequence[int]] = None,
) -> lod.AppliedLevel:
    """Apply ``level`` and update the napari layer, returning the snapshot."""

    applied = prepare_worker_level(
        worker,
        source,
        level,
        prev_level=prev_level,
        ledger_step=ledger_step,
    )

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

    layer = worker._napari_layer
    sy, sx = applied.scale_yx
    if layer is not None:
        try:
            layer.scale = (sy, sx)
        except Exception:
            logger.debug("apply_level: setting 2D layer scale pre-slab failed", exc_info=True)

    view = worker.view
    assert view is not None, "VisPy view must be initialised for 2D apply"

    roi_for_layer = viewport_roi_for_level(worker, source, int(applied.level))
    worker._emit_current_camera_pose("slice-apply")  # noqa: SLF001
    height_px, width_px = apply_plane_slice_roi(
        worker,
        source,
        int(applied.level),
        roi_for_layer,
        update_contrast=not worker._sticky_contrast,
    )

    if (
        not worker._preserve_view_on_switch
        and view is not None
    ):
        full_h, full_w = plane_wh_for_level(source, int(applied.level))
        world_w = float(full_w) * float(max(1e-12, sx))
        world_h = float(full_h) * float(max(1e-12, sy))
        view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    worker._layer_logger.log(  # type: ignore[attr-defined]
        enabled=worker._log_layer_debug,  # type: ignore[attr-defined]
        mode="slice",
        level=applied.level,
        z_index=worker._z_index,
        shape=(int(height_px), int(width_px)),
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # type: ignore[attr-defined]
    )



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
    ledger_step: Optional[Sequence[int]] = None,
    stage_only: bool = False,
) -> lod.AppliedLevel:
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
        if stage_only:
            return prepare_worker_level(
                worker,
                scene,
                level,
                prev_level=prev_level,
                ledger_step=ledger_step,
            )
        return apply_worker_level(
            worker,
            scene,
            level,
            prev_level=prev_level,
            ledger_step=ledger_step,
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
    worker._level_update_cb(applied_snapshot, bool(downgraded))  # type: ignore[operator]
    return applied_snapshot


def perform_level_switch(
    worker: object,
    *,
    target_level: int,
    reason: str,
    requested_level: Optional[int],
    selected_level: Optional[int],
    source: Optional[ZarrSceneSource] = None,
    budget_error: type[Exception],
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
    )
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

    # (policy intent moved to EGL worker after switch; no synthesis here)


__all__ = [
    "ensure_scene_source",
    "reset_worker_camera",
    "prepare_worker_level",
    "apply_worker_level",
    "apply_worker_volume_level",
    "apply_worker_slice_level",
    "format_worker_level_roi",
    "viewport_roi_for_level",
    "apply_plane_slice_roi",
    "roi_equal",
    "set_level_with_budget",
    "perform_level_switch",
]
