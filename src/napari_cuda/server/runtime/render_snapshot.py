"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.state_structs import RenderMode

from .plane_loader import (
    apply_plane_slice_roi as _apply_plane_slice_roi,
    viewport_roi_for_level as _viewport_roi_for_level,
)
from .plane_snapshot import apply_plane_camera_pose, apply_slice_level
from .volume_snapshot import apply_volume_camera_pose, apply_volume_level
from .viewer_stage import apply_plane_metadata, apply_volume_metadata

logger = logging.getLogger(__name__)

apply_plane_slice_roi = _apply_plane_slice_roi
viewport_roi_for_level = _viewport_roi_for_level


@contextmanager
def _suspend_fit_callbacks(viewer: Any):
    """Disconnect fit callbacks while applying dims, then restore them."""

    nd_event = viewer.dims.events.ndisplay
    order_event = viewer.dims.events.order

    nd_event.disconnect(viewer.fit_to_view)
    order_event.disconnect(viewer.fit_to_view)
    try:
        yield
    finally:
        nd_event.connect(viewer.fit_to_view)
        order_event.connect(viewer.fit_to_view)


def stage_level_context(worker: Any, source: Any, context: lod.LevelContext) -> None:
    """Legacy shim that now delegates to viewer_stage helpers."""

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        apply_volume_metadata(worker, source, context)
    else:
        apply_plane_metadata(worker, source, context)


def apply_render_snapshot(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = worker._viewer
    assert viewer is not None, "RenderTxn requires an active viewer"

    signature = worker._dims_signature(snapshot)
    dims_changed = signature != getattr(worker, "_last_dims_signature", None)

    if dims_changed:
        with _suspend_fit_callbacks(viewer):
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.begin: suppress fit; applying dims")
            worker._apply_dims_from_snapshot(snapshot, signature=signature)
            _apply_snapshot_multiscale(worker, snapshot)
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")
    else:
        _apply_snapshot_multiscale(worker, snapshot)


__all__ = [
    "apply_plane_slice_roi",
    "apply_render_snapshot",
    "apply_slice_level",
    "apply_volume_level",
    "stage_level_context",
    "viewport_roi_for_level",
]


def _apply_snapshot_multiscale(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply multiscale state reflected in a controller-authored snapshot."""

    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    target_volume = nd >= 3

    source = worker._ensure_scene_source()
    prev_level = int(worker._current_level_index())  # type: ignore[attr-defined]
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    level_changed = target_level != prev_level

    ledger_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    was_volume = worker.viewport_state.mode is RenderMode.VOLUME  # type: ignore[attr-defined]

    if target_volume:
        entering_volume = not was_volume
        if entering_volume:
            worker.viewport_state.mode = RenderMode.VOLUME  # type: ignore[attr-defined]
            worker._last_dims_signature = None

        runner = worker._viewport_runner
        if runner is not None and entering_volume:
            state = runner.state
            state.level_reload_required = False
            state.awaiting_level_confirm = False
            state.roi_reload_required = False
            state.pending_roi = None
            state.pending_roi_signature = None
            state.applied_roi = None
            state.applied_roi_signature = None
            state.pose_reason = None
            state.camera_pose_dirty = False

        requested_level = int(target_level)
        selected_level, downgraded = worker._resolve_volume_intent_level(source, requested_level)
        effective_level = int(selected_level)
        worker.viewport_state.volume.downgraded = bool(downgraded)  # type: ignore[attr-defined]
        load_needed = entering_volume or (int(effective_level) != prev_level)

        if load_needed:
            if ledger_step is not None:
                step_hint = tuple(int(v) for v in ledger_step)
            else:
                recorded_step = worker._ledger_step()
                step_hint = (
                    tuple(int(v) for v in recorded_step)
                    if recorded_step is not None
                    else None
                )

            decision = lod.LevelDecision(
                desired_level=int(effective_level),
                selected_level=int(effective_level),
                reason="direct",
                timestamp=time.perf_counter(),
                oversampling={},
                downgraded=False,
            )
            applied_context = lod.build_level_context(
                decision,
                source=source,
                prev_level=prev_level,
                last_step=step_hint,
            )
            apply_volume_metadata(worker, source, applied_context)
            apply_volume_level(
                worker,
                source,
                applied_context,
                downgraded=bool(downgraded),
            )
            if runner is not None:
                runner.mark_level_applied(int(applied_context.level))
                rect = worker._current_panzoom_rect()
                runner.update_camera_rect(rect)
            target_level = int(applied_context.level)
            level_changed = target_level != prev_level
            worker._configure_camera_for_mode()
        apply_volume_camera_pose(worker, snapshot)
        return

    stage_prev_level = prev_level
    if was_volume and not target_volume:
        stage_prev_level = target_level

    if ledger_step is not None:
        step_hint = tuple(int(v) for v in ledger_step)
    else:
        recorded_step = worker._ledger_step()
        step_hint = (
            tuple(int(v) for v in recorded_step)
            if recorded_step is not None
            else None
        )

    decision = lod.LevelDecision(
        desired_level=int(target_level),
        selected_level=int(target_level),
        reason="direct",
        timestamp=time.perf_counter(),
        oversampling={},
        downgraded=False,
    )
    applied_context = lod.build_level_context(
        decision,
        source=source,
        prev_level=stage_prev_level,
        last_step=step_hint,
    )
    if was_volume:
        worker.viewport_state.mode = RenderMode.PLANE  # type: ignore[attr-defined]
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None
    apply_plane_metadata(worker, source, applied_context)
    worker.viewport_state.volume.downgraded = False  # type: ignore[attr-defined]
    apply_plane_camera_pose(worker, snapshot)
    apply_slice_level(worker, source, applied_context)
