"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from napari_cuda.server.runtime.lod.context import build_level_context
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.scene.models import SceneBlockSnapshot
from napari_cuda.server.utils.signatures import SignatureToken
from napari_cuda.shared.dims_spec import dims_spec_remap_step_for_level

from .plane import (
    aligned_roi_signature,
    apply_dims_from_snapshot,
    apply_slice_camera_pose,
    apply_slice_level,
    apply_slice_roi as _apply_slice_roi,
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


def apply_render_snapshot(
    snapshot_iface: RenderApplyInterface,
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None = None,
) -> None:
    """Apply the snapshot atomically, suppressing napari auto-fit during dims.

    This ensures that napari's fit_to_view callback does not run against a
    transiently inconsistent (order/displayed/ndim) state while we are applying
    the toggle back to 2D or 3D. Camera and level application are handled by
    the worker helpers invoked from within the dims application.
    """
    viewer = snapshot_iface.viewer
    assert viewer is not None, "RenderTxn requires an active viewer"

    snapshot_ops = _resolve_snapshot_ops(snapshot_iface, snapshot, blocks)

    with _suspend_fit_callbacks(viewer):
        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.begin: suppress fit; applying dims")
        apply_dims_from_snapshot(snapshot_iface, snapshot)
        _apply_snapshot_ops(snapshot_iface, snapshot, snapshot_ops)
        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")


__all__ = [
    "apply_plane_slice_roi",
    "apply_render_snapshot",
    "apply_slice_level",
    "apply_slice_roi",
    "apply_volume_level",
]


def _resolve_snapshot_ops(
    snapshot_iface: RenderApplyInterface,
    snapshot: RenderLedgerSnapshot,
    blocks: SceneBlockSnapshot | None = None,
) -> dict[str, Any]:
    """Compute the metadata and slice ops before mutating worker state."""

    if snapshot.ndisplay is not None:
        nd = int(snapshot.ndisplay)
    elif blocks is not None:
        nd = int(blocks.view.ndim)
    else:
        nd = 2
    target_volume = nd >= 3

    source = snapshot_iface.ensure_scene_source()
    current_level = snapshot_iface.current_level_index()
    if snapshot.current_level is not None:
        snapshot_level = int(snapshot.current_level)
    elif blocks is not None and blocks.lod is not None:
        snapshot_level = int(blocks.lod.level)
    else:
        snapshot_level = int(current_level)

    dims_spec = snapshot.dims_spec
    assert dims_spec is not None, "render snapshot missing dims_spec"

    snapshot_step = dims_spec.current_step
    if snapshot_step is None and blocks is not None and blocks.index is not None:
        snapshot_step = tuple(int(v) for v in blocks.index.value)

    was_volume = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    ledger_snapshot_step = snapshot_iface.ledger_step()

    ops: dict[str, Any] = {
        "source": source,
        "target_volume": target_volume,
        "was_volume": was_volume,
        "plane": None,
        "volume": None,
    }

    if target_volume:
        # Apply path is read-only for level: use the level carried by the snapshot
        load_needed = (snapshot_level != current_level) or (not was_volume)
        step_hint = snapshot_step
        if step_hint is None and ledger_snapshot_step is not None:
            step_hint = tuple(int(v) for v in ledger_snapshot_step)

        spec = snapshot.dims_spec
        applied_context = None
        if load_needed:
            assert step_hint is not None, "volume load requires explicit step"
            base_step = tuple(int(v) for v in step_hint)
            if spec is not None and spec.current_level is not None:
                curr_level = int(spec.current_level)
                if curr_level != int(snapshot_level):
                    base_step = dims_spec_remap_step_for_level(
                        spec,
                        step=base_step,
                        prev_level=curr_level,
                        next_level=int(snapshot_level),
                    )
            applied_context = build_level_context(
                source=source,
                level=snapshot_level,
                step=base_step,
            )

        ops["volume"] = {
            "entering_volume": not was_volume,
            "load_needed": load_needed,
            "applied_context": applied_context,
            "snapshot_level": int(snapshot_level),
            "step_hint": step_hint,
            "current_level": current_level,
        }
        return ops

    step_hint = snapshot_step
    if step_hint is None and ledger_snapshot_step is not None:
        step_hint = tuple(int(v) for v in ledger_snapshot_step)
    assert step_hint is not None, "plane apply requires explicit step"
    spec = snapshot.dims_spec
    base_step = tuple(int(v) for v in step_hint)
    if spec is not None and spec.current_level is not None:
        curr_level = int(spec.current_level)
        if curr_level != int(snapshot_level):
            base_step = dims_spec_remap_step_for_level(
                spec,
                step=base_step,
                prev_level=curr_level,
                next_level=int(snapshot_level),
            )
    applied_context = build_level_context(
        source=source,
        level=int(snapshot_level),
        step=base_step,
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
    slice_token = SignatureToken((level_int, step_tuple, roi_signature))
    last_slice_token = snapshot_iface.last_slice_signature()
    level_changed = snapshot_level != current_level
    skip_slice = not level_changed and last_slice_token is not None and last_slice_token.value == slice_token.value
    chunk_tuple = (
        (int(chunk_shape[0]), int(chunk_shape[1]))
        if chunk_shape is not None
        else (0, 0)
    )
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
        "slice_token": slice_token,
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

        runner = snapshot_iface.viewport_runner
        if runner is not None and entering_volume:
            runner.reset_for_volume()

        if volume_ops["load_needed"]:
            applied_context = volume_ops["applied_context"]
            assert applied_context is not None
            apply_volume_metadata(snapshot_iface, source, applied_context)
            apply_volume_level(
                snapshot_iface,
                source,
                applied_context,
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
    slice_token: Optional[SignatureToken] = plane_ops.get("slice_token")

    if plane_ops["was_volume"]:
        snapshot_iface.viewport_state.mode = RenderMode.PLANE
        snapshot_iface.ensure_plane_visual()
        snapshot_iface.configure_camera_for_mode()

    apply_plane_metadata(snapshot_iface, source, plane_ops["applied_context"])
    apply_slice_camera_pose(snapshot_iface, snapshot)

    if plane_ops["skip_slice"]:
        runner = snapshot_iface.viewport_runner
        if runner is not None:
            from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask

            slice_task = SliceTask(**plane_ops["slice_payload"])
            runner.mark_level_applied(slice_task.level)
            runner.mark_slice_applied(slice_task)
        if slice_token is not None:
            snapshot_iface.set_last_slice_signature(slice_token)
        update_z_index_from_snapshot(snapshot_iface, snapshot)
        return

    apply_slice_level(snapshot_iface, source, plane_ops["applied_context"])
    update_z_index_from_snapshot(snapshot_iface, snapshot)
    if TYPE_CHECKING:
        from napari_cuda.server.runtime.render_loop.planning.viewport_planner import SliceTask
