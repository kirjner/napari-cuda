"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Dict
import logging
import time

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera, TurntableCamera

from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.scene_state_applier import SceneStateApplier
from dataclasses import replace
import napari_cuda.server.data.lod as lod
from napari_cuda.server.data.level_budget import select_volume_level
from napari_cuda.server.data.roi import (
    plane_scale_for_level,
    plane_wh_for_level,
    viewport_debug_snapshot,
    resolve_worker_viewport_roi,
)
from napari_cuda.server.runtime.scene_types import SliceROI

logger = logging.getLogger(__name__)


def viewport_roi_for_level(
    worker: object,
    source: Any,
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

    def _snapshot() -> Dict[str, Any]:
        return viewport_debug_snapshot(
            view=view,
            canvas_size=(int(worker.width), int(worker.height)),  # type: ignore[attr-defined]
            data_wh=data_wh,
            data_depth=worker._data_d,
        )

    reason = "policy-roi" if for_policy else "roi-request"
    roi_cache = worker._roi_cache

    roi = resolve_worker_viewport_roi(
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

    if log_state is not None:
        log_state["roi"] = roi
        log_state["level"] = int(level)
        log_state["requested"] = (align_chunks, ensure_contains, chunk_pad)

    return roi


def apply_plane_slice_roi(
    worker: Any,
    source: Any,
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


def apply_render_snapshot(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = worker._viewer  # noqa: SLF001
    assert viewer is not None, "RenderTxn requires an active viewer"

    # Suppress ONLY napari's fit_to_view during dims apply so that layer
    # adapters still observe ndisplay/order events and can rebuild visuals.
    # We temporarily disconnect the specific callback, apply dims, then
    # reconnect.
    import logging as _logging
    _l = _logging.getLogger(__name__)

    nd = viewer.dims.events.ndisplay
    od = viewer.dims.events.order

    signature = worker._dims_signature(snapshot)  # noqa: SLF001
    dims_changed = signature != getattr(worker, "_last_dims_signature", None)

    if dims_changed:
        nd.disconnect(viewer.fit_to_view)
        od.disconnect(viewer.fit_to_view)
        if _l.isEnabledFor(_logging.INFO):
            _l.info("snapshot.apply.begin: suppress fit; applying dims")
        worker._apply_dims_from_snapshot(snapshot, signature=signature)  # noqa: SLF001
        _apply_snapshot_multiscale(worker, snapshot)
        if _l.isEnabledFor(_logging.INFO):
            _l.info("snapshot.apply.end: dims applied; resuming fit callbacks")
        nd.connect(viewer.fit_to_view)
        od.connect(viewer.fit_to_view)
    else:
        _apply_snapshot_multiscale(worker, snapshot)


__all__ = [
    "apply_render_snapshot",
    "apply_plane_slice_roi",
    "viewport_roi_for_level",
    "apply_volume_level",
    "apply_slice_level",
]


def _apply_snapshot_multiscale(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply multiscale state reflected in a controller-authored snapshot."""

    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    target_volume = nd >= 3

    source = worker._ensure_scene_source()  # noqa: SLF001
    prev_level = int(worker._active_ms_level)
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    level_changed = target_level != prev_level

    ledger_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    if target_volume:
        entering_volume = not worker.use_volume
        if entering_volume:
            worker.use_volume = True
            worker._last_dims_signature = None  # noqa: SLF001

        requested_level = int(target_level)
        effective_level = _resolve_volume_level(worker, source, requested_level)
        worker._level_downgraded = bool(effective_level != requested_level)
        load_needed = entering_volume or (int(effective_level) != prev_level)

        if load_needed:
            applied_context = _build_level_context(
                worker,
                source,
                int(effective_level),
                prev_level=prev_level,
                ledger_step=ledger_step,
            )
            apply_volume_level(worker, source, applied_context)
            target_level = int(applied_context.level)
            level_changed = target_level != prev_level
            worker._configure_camera_for_mode()
        _apply_volume_camera_pose(worker, snapshot)
        return

    stage_prev_level = prev_level
    if worker.use_volume and not target_volume:
        stage_prev_level = target_level

    applied_context = _build_level_context(
        worker,
        source,
        int(target_level),
        prev_level=stage_prev_level,
        ledger_step=ledger_step,
    )

    if worker.use_volume:
        worker.use_volume = False
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None  # noqa: SLF001
    worker._level_downgraded = False
    _apply_plane_camera_pose(worker, snapshot)
    apply_slice_level(worker, source, applied_context)


def _apply_volume_camera_pose(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, TurntableCamera):
        return

    center = snapshot.center
    if center is not None and len(center) >= 3:
        cam.center = (
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )

    angles = snapshot.angles
    if angles is not None and len(angles) >= 2:
        cam.azimuth = float(angles[0])
        cam.elevation = float(angles[1])
        if len(angles) >= 3:
            cam.roll = float(angles[2])  # type: ignore[attr-defined]

    if snapshot.distance is not None:
        cam.distance = float(snapshot.distance)
    if snapshot.fov is not None:
        cam.fov = float(snapshot.fov)


def _apply_plane_camera_pose(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    view = worker.view
    if view is None:
        return
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    rect = snapshot.rect
    if rect is not None and len(rect) >= 4:
        cam.rect = Rect(
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )

    center = snapshot.center
    if center is not None and len(center) >= 2:
        cam.center = (
            float(center[0]),
            float(center[1]),
        )


def _resolve_volume_level(worker: Any, source: Any, requested_level: int) -> int:
    max_voxels_cfg = worker._volume_max_voxels or worker._hw_limits.volume_max_voxels
    max_bytes_cfg = worker._volume_max_bytes or worker._hw_limits.volume_max_bytes
    max_voxels = int(max_voxels_cfg) if max_voxels_cfg else None
    max_bytes = int(max_bytes_cfg) if max_bytes_cfg else None
    level, _ = select_volume_level(
        source,
        int(requested_level),
        max_voxels=max_voxels,
        max_bytes=max_bytes,
        error_cls=worker._budget_error_cls,
    )
    return int(level)


def _build_level_context(
    worker: Any,
    source: Any,
    level: int,
    *,
    prev_level: Optional[int],
    ledger_step: Optional[Sequence[int]],
) -> lod.LevelContext:
    same_level = prev_level is None or int(prev_level) == int(level)
    step_authoritative = bool(ledger_step is not None and same_level)

    if ledger_step is not None:
        step_hint: Optional[tuple[int, ...]] = tuple(int(v) for v in ledger_step)
    else:
        recorded_step = worker._ledger_step()
        step_hint = (
            tuple(int(v) for v in recorded_step)
            if recorded_step is not None
            else None
        )

    decision = lod.LevelDecision(
        desired_level=int(level),
        selected_level=int(level),
        reason="direct",
        timestamp=time.perf_counter(),
        oversampling={},
        downgraded=False,
    )

    context = lod.build_level_context(
        decision,
        source=source,
        prev_level=prev_level,
        last_step=step_hint,
        step_authoritative=step_authoritative,
    )

    _stage_level_context(worker, source, context)
    return context


def _stage_level_context(worker: Any, source: Any, context: lod.LevelContext) -> None:
    viewer = getattr(worker, "_viewer", None)
    if viewer is not None:
        set_range = getattr(worker, "_set_dims_range_for_level", None)
        if callable(set_range):
            set_range(source, int(context.level))
        viewer.dims.current_step = tuple(int(v) for v in context.step)

    descriptor = source.level_descriptors[int(context.level)]
    worker._update_level_metadata(descriptor, context)  # noqa: SLF001


def apply_volume_level(worker: Any, source: Any, applied: lod.LevelContext) -> None:
    scale_vals = list(source.level_scale(applied.level)) if hasattr(source, "level_scale") else []
    scales = [float(s) for s in scale_vals[-3:]] if scale_vals else []
    while len(scales) < 3:
        scales.insert(0, 1.0)
    scale_tuple = (float(scales[-3]), float(scales[-2]), float(scales[-1]))
    worker._volume_scale = scale_tuple  # noqa: SLF001

    volume = worker._get_level_volume(source, applied.level)  # noqa: SLF001
    cam = worker.view.camera if getattr(worker, "view", None) is not None else None
    ctx = worker._build_scene_state_context(cam)  # noqa: SLF001
    if ctx.volume_scale is None:
        ctx = replace(ctx, volume_scale=scale_tuple)

    data_wh, data_d = SceneStateApplier.apply_volume_layer(
        ctx,
        volume=volume,
        contrast=applied.contrast,
    )
    worker._data_wh = data_wh  # noqa: SLF001
    worker._data_d = data_d  # noqa: SLF001

    shape = (
        (int(data_d), int(data_wh[1]), int(data_wh[0]))
        if data_d is not None
        else (int(data_wh[1]), int(data_wh[0]))
    )

    worker._layer_logger.log(  # noqa: SLF001
        enabled=worker._log_layer_debug,  # noqa: SLF001
        mode="volume",
        level=applied.level,
        z_index=None,
        shape=shape,
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # noqa: SLF001
    )


def apply_slice_level(worker: Any, source: Any, applied: lod.LevelContext) -> None:
    layer = getattr(worker, "_napari_layer", None)
    sy, sx = applied.scale_yx

    if layer is not None:
        layer.scale = (float(sy), float(sx))

    view = worker.view
    assert view is not None, "VisPy view must be initialised for 2D apply"

    roi = viewport_roi_for_level(worker, source, int(applied.level))
    worker._emit_current_camera_pose("slice-apply")  # noqa: SLF001
    height_px, width_px = apply_plane_slice_roi(
        worker,
        source,
        int(applied.level),
        roi,
        update_contrast=not worker._sticky_contrast,  # noqa: SLF001
    )

    if not worker._preserve_view_on_switch:  # noqa: SLF001
        full_h, full_w = plane_wh_for_level(source, int(applied.level))
        world_w = float(full_w) * float(max(1e-12, sx))
        world_h = float(full_h) * float(max(1e-12, sy))
        view.camera.set_range(x=(0.0, max(1.0, world_w)), y=(0.0, max(1.0, world_h)))

    worker._layer_logger.log(  # noqa: SLF001
        enabled=worker._log_layer_debug,  # noqa: SLF001
        mode="slice",
        level=applied.level,
        z_index=worker._z_index,  # noqa: SLF001
        shape=(int(height_px), int(width_px)),
        contrast=applied.contrast,
        downgraded=worker._level_downgraded,  # noqa: SLF001
    )
