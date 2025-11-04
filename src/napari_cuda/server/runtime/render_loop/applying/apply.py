"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.lod.context import build_level_context
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import RenderLedgerSnapshot

from .plane import (
    aligned_roi_signature,
    apply_dims_from_snapshot,
    apply_slice_camera_pose,
    apply_slice_level,
    apply_slice_roi as _apply_slice_roi,
    dims_signature,
    update_z_index_from_snapshot,
)
from .viewer_metadata import apply_plane_metadata, apply_volume_metadata
from .volume import apply_volume_camera_pose, apply_volume_level

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


def apply_render_snapshot(snapshot_iface: RenderApplyInterface, snapshot: RenderLedgerSnapshot) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = snapshot_iface.viewer
    assert viewer is not None, "RenderTxn requires an active viewer"

    snapshot_ops = _resolve_snapshot_ops(snapshot_iface, snapshot)
    ops_signature = snapshot_ops["signature"]
    if ops_signature == snapshot_iface.last_snapshot_signature():
        return

    signature = dims_signature(snapshot)
    dims_changed = signature != snapshot_iface.last_dims_signature()

    if dims_changed:
        with _suspend_fit_callbacks(viewer):
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.begin: suppress fit; applying dims")
            apply_dims_from_snapshot(snapshot_iface, snapshot, signature=signature)
            _apply_snapshot_ops(snapshot_iface, snapshot, snapshot_ops)
            if logger.isEnabledFor(logging.INFO):
                logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")
    else:
        _apply_snapshot_ops(snapshot_iface, snapshot, snapshot_ops)

    snapshot_iface.set_last_snapshot_signature(ops_signature)


__all__ = [
    "apply_plane_slice_roi",
    "apply_render_snapshot",
    "apply_slice_level",
    "apply_slice_roi",
    "apply_volume_level",
]


def _resolve_snapshot_ops(
    snapshot_iface: RenderApplyInterface, snapshot: RenderLedgerSnapshot
) -> dict[str, Any]:
    """Compute the metadata and slice ops before mutating worker state."""

    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    target_volume = nd >= 3

    source = snapshot_iface.ensure_scene_source()
    prev_level = snapshot_iface.current_level_index()
    target_level = int(snapshot.current_level) if snapshot.current_level is not None else prev_level
    level_changed = target_level != prev_level

    snapshot_step = (
        tuple(int(v) for v in snapshot.current_step)
        if snapshot.current_step is not None
        else None
    )

    was_volume = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    ledger_snapshot_step = snapshot_iface.ledger_step()

    ops: dict[str, Any] = {
        "source": source,
        "target_volume": target_volume,
        "was_volume": was_volume,
        "signature": None,
        "plane": None,
        "volume": None,
    }

    version_prefix = (
        None if snapshot.dims_version is None else int(snapshot.dims_version),
        None if snapshot.view_version is None else int(snapshot.view_version),
        None if snapshot.multiscale_level_version is None else int(snapshot.multiscale_level_version),
    )

    if target_volume:
        requested_level = int(target_level)
        selected_level, downgraded = snapshot_iface.resolve_volume_intent_level(
            source,
            requested_level,
        )
        effective_level = int(selected_level)
        load_needed = (effective_level != prev_level) or (not was_volume)
        if snapshot_step is not None:
            step_hint = snapshot_step
        else:
            step_hint = (
                tuple(int(v) for v in ledger_snapshot_step)
                if ledger_snapshot_step is not None
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
            applied_context = build_level_context(
                decision,
                source=source,
                prev_level=prev_level,
                last_step=step_hint,
            )

        signature_token: tuple[Any, ...] = ("volume", int(effective_level), step_hint)
        ops["signature"] = (
            version_prefix[0],
            version_prefix[1],
            version_prefix[2],
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

    if snapshot_step is not None:
        step_hint = snapshot_step
    else:
        step_hint = (
            tuple(int(v) for v in ledger_snapshot_step)
            if ledger_snapshot_step is not None
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
    applied_context = build_level_context(
        decision,
        source=source,
        prev_level=stage_prev_level,
        last_step=step_hint,
    )
    step_tuple = tuple(int(v) for v in applied_context.step)
    level_int = int(applied_context.level)
    roi_current = snapshot_iface.viewport_roi_for_level(source, level_int)
    aligned_roi, chunk_shape, roi_signature = aligned_roi_signature(
        snapshot_iface,
        source,
        level_int,
        roi_current,
    )
    signature_token = (level_int, step_tuple, roi_signature)
    last_slice_signature = snapshot_iface.last_slice_signature()
    level_changed = target_level != prev_level
    skip_slice = not level_changed and last_slice_signature == signature_token
    snapshot_signature = (
        version_prefix[0],
        version_prefix[1],
        version_prefix[2],
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
    snapshot_iface: RenderApplyInterface,
    snapshot: RenderLedgerSnapshot,
    ops: dict[str, Any],
) -> None:
    """Apply the precomputed snapshot plan."""

    source = ops["source"]

    if ops["target_volume"]:
        volume_ops = ops["volume"]
        assert volume_ops is not None
        entering_volume = volume_ops["entering_volume"]
        if entering_volume:
            snapshot_iface.viewport_state.mode = RenderMode.VOLUME
            snapshot_iface.ensure_volume_visual()
            snapshot_iface.set_last_dims_signature(None)

        runner = snapshot_iface.viewport_runner
        if runner is not None and entering_volume:
            runner.reset_for_volume()

        snapshot_iface.viewport_state.volume.downgraded = bool(volume_ops["downgraded"])
        if volume_ops["load_needed"]:
            applied_context = volume_ops["applied_context"]
            assert applied_context is not None
            apply_volume_metadata(snapshot_iface, source, applied_context)
            apply_volume_level(
                snapshot_iface,
                source,
                applied_context,
                downgraded=bool(volume_ops["downgraded"]),
            )
            snapshot_iface.mark_level_applied(int(applied_context.level))
            if (
                runner is not None
                and snapshot_iface.viewport_state.mode is RenderMode.PLANE
            ):
                rect = snapshot_iface.current_panzoom_rect()
                if rect is not None:
                    runner.update_camera_rect(rect)
            snapshot_iface.configure_camera_for_mode()
        elif entering_volume:
            snapshot_iface.configure_camera_for_mode()

        apply_volume_camera_pose(snapshot_iface, snapshot)
        return

    plane_ops = ops["plane"]
    assert plane_ops is not None

    if plane_ops["was_volume"]:
        snapshot_iface.viewport_state.mode = RenderMode.PLANE
        snapshot_iface.ensure_plane_visual()
        snapshot_iface.configure_camera_for_mode()
        snapshot_iface.set_last_dims_signature(None)

    apply_plane_metadata(snapshot_iface, source, plane_ops["applied_context"])
    snapshot_iface.viewport_state.volume.downgraded = False
    apply_slice_camera_pose(snapshot_iface, snapshot)

    if plane_ops["skip_slice"]:
        runner = snapshot_iface.viewport_runner
        if runner is not None:
            from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask

            slice_task = SliceTask(**plane_ops["slice_payload"])
            runner.mark_level_applied(slice_task.level)
            runner.mark_slice_applied(slice_task)
        update_z_index_from_snapshot(snapshot_iface, snapshot)
        return

    apply_slice_level(snapshot_iface, source, plane_ops["applied_context"])
    update_z_index_from_snapshot(snapshot_iface, snapshot)
    if TYPE_CHECKING:
        from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask
