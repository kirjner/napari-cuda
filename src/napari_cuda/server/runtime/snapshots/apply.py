"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Tuple, TYPE_CHECKING

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.viewport.state import RenderMode

from .build import RenderLedgerSnapshot
from .plane import (
    apply_slice_camera_pose,
    apply_slice_level,
    apply_slice_roi as _apply_slice_roi,
)
from .volume import apply_volume_camera_pose, apply_volume_level
from .viewer import apply_plane_metadata, apply_volume_metadata
from napari_cuda.server.runtime.viewport.roi import viewport_roi_for_level

logger = logging.getLogger(__name__)

apply_slice_roi = _apply_slice_roi
apply_plane_slice_roi = apply_slice_roi


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


def apply_render_snapshot(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = worker._viewer
    assert viewer is not None, "RenderTxn requires an active viewer"

    snapshot_ops = _resolve_snapshot_ops(worker, snapshot)
    ops_signature = snapshot_ops["signature"]
    if ops_signature == getattr(worker, "_last_snapshot_signature", None):
        return

    dims_signature = worker._dims_signature(snapshot)
    dims_changed = dims_signature != getattr(worker, "_last_dims_signature", None)

    if dims_changed:
        with _suspend_fit_callbacks(viewer):
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.begin: suppress fit; applying dims")
            worker._apply_dims_from_snapshot(snapshot, signature=dims_signature)
            _apply_snapshot_ops(worker, snapshot, snapshot_ops)
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")
    else:
        _apply_snapshot_ops(worker, snapshot, snapshot_ops)

    worker._last_snapshot_signature = ops_signature


__all__ = [
    "apply_plane_slice_roi",
    "apply_slice_roi",
    "apply_render_snapshot",
    "apply_slice_level",
    "apply_volume_level",
    "viewport_roi_for_level",
]


def _resolve_snapshot_ops(
    worker: Any, snapshot: RenderLedgerSnapshot
) -> Dict[str, Any]:
    """Compute the metadata and slice ops before mutating worker state."""

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
    ledger_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    ops: Dict[str, Any] = {
        "source": source,
        "target_volume": target_volume,
        "was_volume": was_volume,
        "signature": None,
        "plane": None,
        "volume": None,
    }

    if target_volume:
        requested_level = int(target_level)
        selected_level, downgraded = worker._resolve_volume_intent_level(source, requested_level)
        effective_level = int(selected_level)
        load_needed = (effective_level != prev_level) or (not was_volume)
        if ledger_step is not None:
            step_hint = ledger_step
        else:
            recorded_step = worker._ledger_step()
            step_hint = (
                tuple(int(v) for v in recorded_step)
                if recorded_step is not None
                else None
            )

        applied_context = None
        if load_needed:
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

        signature_token: Tuple[Any, ...] = ("volume", int(effective_level), step_hint)
        ops["signature"] = (
            snapshot.dims_version,
            snapshot.view_version,
            snapshot.multiscale_level_version,
            signature_token,
        )
        ops["volume"] = {
            "entering_volume": not was_volume,
            "downgraded": bool(downgraded),
            "load_needed": load_needed,
            "applied_context": applied_context,
            "effective_level": int(effective_level),
            "step_hint": step_hint,
            "prev_level": prev_level,
        }
        return ops

    stage_prev_level = prev_level
    if was_volume and not target_volume:
        stage_prev_level = target_level

    if ledger_step is not None:
        step_hint = ledger_step
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
    step_tuple = tuple(int(v) for v in applied_context.step)
    level_int = int(applied_context.level)
    roi_current = viewport_roi_for_level(worker, source, level_int)
    aligned_roi, chunk_shape, roi_signature = worker._aligned_roi_signature(source, level_int, roi_current)
    signature_token = (level_int, step_tuple, roi_signature)
    last_slice_signature = getattr(worker, "_last_slice_signature", None)
    level_changed = target_level != prev_level
    skip_slice = not level_changed and last_slice_signature == signature_token
    snapshot_signature = (
        snapshot.dims_version,
        snapshot.view_version,
        snapshot.multiscale_level_version,
        signature_token,
    )
    chunk_tuple = (
        (int(chunk_shape[0]), int(chunk_shape[1]))
        if chunk_shape is not None
        else (0, 0)
    )
    ops["signature"] = snapshot_signature
    ops["plane"] = {
        "applied_context": applied_context,
        "aligned_roi": aligned_roi,
        "chunk_shape": chunk_shape,
        "slice_payload": {
            "level": level_int,
            "step": step_tuple,
            "roi": aligned_roi,
            "chunk_shape": chunk_tuple,
            "signature": roi_signature,
        },
        "skip_slice": skip_slice,
        "level_int": level_int,
        "level_changed": level_changed,
        "was_volume": was_volume,
    }
    return ops


def _apply_snapshot_ops(
    worker: Any,
    snapshot: RenderLedgerSnapshot,
    ops: Dict[str, Any],
) -> None:
    """Apply the precomputed snapshot plan."""

    source = ops["source"]

    if ops["target_volume"]:
        volume_ops = ops["volume"]
        assert volume_ops is not None
        entering_volume = volume_ops["entering_volume"]
        if entering_volume:
            worker.viewport_state.mode = RenderMode.VOLUME  # type: ignore[attr-defined]
            worker._last_dims_signature = None

        runner = worker._viewport_runner
        if runner is not None and entering_volume:
            runner.reset_for_volume()

        worker.viewport_state.volume.downgraded = bool(volume_ops["downgraded"])  # type: ignore[attr-defined]
        if volume_ops["load_needed"]:
            applied_context = volume_ops["applied_context"]
            assert applied_context is not None
            apply_volume_metadata(worker, source, applied_context)
            apply_volume_level(
                worker,
                source,
                applied_context,
                downgraded=bool(volume_ops["downgraded"]),
            )
            if runner is not None:
                runner.mark_level_applied(int(applied_context.level))
                if worker.viewport_state.mode is RenderMode.PLANE:  # type: ignore[attr-defined]
                    rect = worker._current_panzoom_rect()
                    if rect is not None:
                        runner.update_camera_rect(rect)
            worker._configure_camera_for_mode()
        elif entering_volume:
            worker._configure_camera_for_mode()

        apply_volume_camera_pose(worker, snapshot)
        return

    plane_ops = ops["plane"]
    assert plane_ops is not None

    if plane_ops["was_volume"]:
        worker.viewport_state.mode = RenderMode.PLANE  # type: ignore[attr-defined]
        worker._configure_camera_for_mode()
        worker._last_dims_signature = None

    apply_plane_metadata(worker, source, plane_ops["applied_context"])
    worker.viewport_state.volume.downgraded = False  # type: ignore[attr-defined]
    apply_slice_camera_pose(worker, snapshot)

    if plane_ops["skip_slice"]:
        runner = worker._viewport_runner
        if runner is not None:
            from napari_cuda.server.runtime.viewport.runner import SliceTask

            slice_task = SliceTask(**plane_ops["slice_payload"])
            runner.mark_level_applied(slice_task.level)
            runner.mark_slice_applied(slice_task)
        return

    apply_slice_level(worker, source, plane_ops["applied_context"])
    if TYPE_CHECKING:
        from napari_cuda.server.runtime.viewport.runner import SliceTask
