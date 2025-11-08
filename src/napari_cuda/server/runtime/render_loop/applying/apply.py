"""Render snapshot application helpers.

These helpers consume a controller-authored render snapshot and apply it to the
napari viewer model while temporarily suppressing ``fit_to_view`` so the viewer
never observes a partially-updated dims state.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

import napari_cuda.server.data.lod as lod

from napari_cuda.server.control.state_reducers import reduce_level_update
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.runtime.render_loop.planning.viewport_planner import (
    SliceTask,
    ViewportOps,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.utils.signatures import SignatureToken

from napari_cuda.server.runtime.lod.context import build_level_context

from .plane import apply_dims_from_snapshot, apply_slice_camera_pose, apply_slice_level
from .plane import apply_slice_roi as _apply_slice_roi
from .plane import update_z_index_from_snapshot
from .viewer_metadata import apply_plane_metadata, apply_volume_metadata
from .volume import apply_volume_camera_pose, apply_volume_level

logger = logging.getLogger(__name__)

apply_slice_roi = _apply_slice_roi
apply_plane_slice_roi = apply_slice_roi


def plane_level_context_from_store(
    snapshot_iface: RenderApplyInterface,
    source: Any,
) -> lod.LevelContext:
    """Rebuild a plane level context from cached worker state."""

    plane_state = snapshot_iface.viewport_state.plane
    level = int(plane_state.applied.level)
    step = _resolve_plane_step(plane_state, source)
    return build_level_context(
        source=source,
        level=level,
        step=step,
    )


def volume_level_context_from_store(
    snapshot_iface: RenderApplyInterface,
    source: Any,
) -> lod.LevelContext:
    """Rebuild a volume level context from cached worker state."""

    volume_state = snapshot_iface.viewport_state.volume
    level = int(volume_state.level)
    plane_state = snapshot_iface.viewport_state.plane
    step = _resolve_plane_step(plane_state, source)
    context = build_level_context(
        source=source,
        level=level,
        step=step,
    )
    contrast = source.ensure_contrast(level=level)
    scale = volume_state.scale
    if scale is not None and len(scale) >= 2:
        scale_yx = (float(scale[-2]), float(scale[-1]))
    else:
        scale_yx = context.scale_yx
    return replace(
        context,
        contrast=(float(contrast[0]), float(contrast[1])),
        scale_yx=scale_yx,
    )


def _resolve_plane_step(plane_state, source: Any) -> tuple[int, ...]:
    step = plane_state.applied.step or plane_state.request.step
    if step is None:
        descriptors = getattr(source, "level_descriptors", None)
        level_idx = int(getattr(plane_state.applied, "level", 0))
        ndim = 0
        if descriptors and 0 <= level_idx < len(descriptors):
            ndim = len(getattr(descriptors[level_idx], "shape", ()))
        if ndim == 0:
            axes = getattr(source, "axes", None)
            ndim = len(axes) if axes is not None else 0
        step = tuple(0 for _ in range(ndim))
    return tuple(int(v) for v in step)


# No hydration: planner emits full replay plans on mode changes.


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

    runner = snapshot_iface.viewport_runner
    assert runner is not None, "RenderTxn requires an active viewport planner"

    source = snapshot_iface.ensure_scene_source()

    def _roi_resolver(level: int, _rect: tuple[float, float, float, float]) -> Any:
        return snapshot_iface.viewport_roi_for_level(source, int(level), quiet=True)

    def _volume_level_resolver(_source: Any, level: int) -> int:
        return snapshot_iface.resolve_volume_intent_level(source, int(level))

    with _suspend_fit_callbacks(viewer):
        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.begin: suppress fit; applying dims")
        apply_dims_from_snapshot(snapshot_iface, snapshot)
        # Ensure z-index is updated before any slice/ROI apply so the render
        # uses the step reflected by the dims slider.
        update_z_index_from_snapshot(snapshot_iface, snapshot)

        plan = runner.plan_from_snapshot(
            snapshot,
            source=source,
            roi_resolver=_roi_resolver,
            volume_level_resolver=_volume_level_resolver,
        )

        # Planner should emit complete replay plans on mode change; no hydration needed.

        apply_viewport_plan(snapshot_iface, snapshot, plan, source=source)

        if logger.isEnabledFor(logging.INFO):
            logger.info("snapshot.apply.end: dims applied; resuming fit callbacks")


__all__ = [
    "apply_plane_slice_roi",
    "apply_render_snapshot",
    "apply_slice_level",
    "apply_slice_roi",
    "apply_volume_level",
    "apply_viewport_plan",
]


def apply_viewport_plan(
    snapshot_iface: RenderApplyInterface,
    snapshot: Optional[RenderLedgerSnapshot],
    plan: ViewportOps,
    *,
    source: Optional[Any] = None,
) -> None:
    """Apply a planner-authored viewport plan."""

    source_obj = source if source is not None else snapshot_iface.ensure_scene_source()
    runner = snapshot_iface.viewport_runner

    if plan.mode is RenderMode.VOLUME:
        entering_volume = snapshot_iface.viewport_state.mode is not RenderMode.VOLUME
        snapshot_iface.viewport_state.mode = RenderMode.VOLUME
        snapshot_iface.ensure_volume_visual()
        if entering_volume:
            snapshot_iface.configure_camera_for_mode()
            if runner is not None:
                runner.reset_for_volume()
        print(
            f"[apply] volume plan entering={entering_volume} level_change={plan.level_change}"
            f" has_context={plan.level_context is not None}",
            flush=True,
        )

        if plan.level_context is not None:
            apply_volume_metadata(snapshot_iface, source_obj, plan.level_context)
            apply_volume_level(snapshot_iface, source_obj, plan.level_context)
            # Do not mutate plane runner/applied level when in VOLUME; ledger update
            # records volume level, and plane cache remains intact for replay.
            _update_level_ledger(snapshot_iface, plan.level_context, RenderMode.VOLUME)
        elif plan.metadata_replay:
            raise AssertionError("volume metadata replay requires level context")

        if snapshot is not None:
            apply_volume_camera_pose(snapshot_iface, snapshot)
        if plan.pose_event is not None:
            snapshot_iface.emit_current_camera_pose(plan.pose_event.value)
        return

    entering_plane = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    snapshot_iface.viewport_state.mode = RenderMode.PLANE
    if entering_plane:
        snapshot_iface.ensure_plane_visual()
        snapshot_iface.configure_camera_for_mode()

    ledger_context = plan.level_context if (plan.metadata_replay or plan.level_change) else None

    context = plan.level_context

    if plan.metadata_replay:
        assert context is not None, "plane metadata replay requires level context"
        apply_plane_metadata(snapshot_iface, source_obj, context)
        if runner is not None:
            runner.mark_plane_metadata_applied()

    # On replay entry (mode toggle), prefer cached pose rather than snapshot pose
    entering_plane = snapshot_iface.viewport_state.mode is RenderMode.VOLUME
    if snapshot is not None and not (entering_plane and plan.metadata_replay):
        apply_slice_camera_pose(snapshot_iface, snapshot)
    elif entering_plane and plan.metadata_replay:
        pose = snapshot_iface.viewport_state.plane.pose
        print("[apply] plane replay: applying cached pose and skipping snapshot pose", flush=True)
        if pose.rect is not None and pose.center is not None and pose.zoom is not None:
            from .plane_ops import apply_pose_to_camera
            view = snapshot_iface.view
            if view is not None:
                apply_pose_to_camera(view.camera, rect=pose.rect, center=pose.center, zoom=pose.zoom)

    if context is not None:
        if plan.slice_task is not None and entering_plane and plan.metadata_replay:
            print("[apply] plane replay: applying cached slice task (no ROI recompute)", flush=True)
            _apply_slice_task(snapshot_iface, source_obj, plan.slice_task)
            _update_level_ledger(snapshot_iface, context, RenderMode.PLANE)
        else:
            apply_slice_level(snapshot_iface, source_obj, context)
            _update_level_ledger(snapshot_iface, context, RenderMode.PLANE)
    elif plan.slice_task is not None:
        _apply_slice_task(snapshot_iface, source_obj, plan.slice_task)
    elif plan.slice_signature is not None:
        snapshot_iface.set_last_slice_signature(SignatureToken(plan.slice_signature))
    else:
        # Enforce dims step after 3D→2D if planner emitted no work.
        # If snapshot step differs from applied step at the same level, apply the slice directly.
        if snapshot is not None and snapshot.ndisplay is not None and int(snapshot.ndisplay) < 3:
            dims_spec = snapshot.dims_spec
            if dims_spec is not None and dims_spec.current_level is not None and dims_spec.current_step is not None:
                level_idx = int(dims_spec.current_level)
                step_tuple = tuple(int(v) for v in dims_spec.current_step)
                plane_applied = snapshot_iface.viewport_state.plane.applied
                if int(plane_applied.level) == level_idx and (plane_applied.step is None or tuple(int(v) for v in plane_applied.step) != step_tuple):
                    ctx = build_level_context(source=source_obj, level=level_idx, step=step_tuple)
                    print(
                        f"[apply] enforcing dims step: level={level_idx} step={step_tuple} (applied={plane_applied.step})",
                        flush=True,
                    )
                    apply_slice_level(snapshot_iface, source_obj, ctx)
                    _update_level_ledger(snapshot_iface, ctx, RenderMode.PLANE)

    if snapshot is not None:
        update_z_index_from_snapshot(snapshot_iface, snapshot)
    if plan.pose_event is not None:
        snapshot_iface.emit_current_camera_pose(plan.pose_event.value)


def _apply_slice_task(
    snapshot_iface: RenderApplyInterface,
    source: Any,
    slice_task: SliceTask,
) -> None:
    runner = snapshot_iface.viewport_runner
    step_tuple: Optional[tuple[int, ...]] = (
        tuple(int(v) for v in slice_task.step) if slice_task.step is not None else None
    )
    apply_slice_roi(
        snapshot_iface,
        source,
        slice_task.level,
        slice_task.roi,
        update_contrast=False,
        step=step_tuple,
    )
    if runner is not None:
        runner.mark_slice_applied(slice_task)
    signature_token = SignatureToken(
        (
            int(slice_task.level),
            tuple(int(v) for v in slice_task.step)
            if slice_task.step is not None
            else None,
            slice_task.signature,
        )
    )
    snapshot_iface.set_last_slice_signature(signature_token)


def _update_level_ledger(
    snapshot_iface: RenderApplyInterface,
    context: lod.LevelContext,
    mode: RenderMode,
) -> None:
    worker = snapshot_iface.worker
    ledger = getattr(worker, "_ledger", None)
    if ledger is None:
        return
    step_tuple = tuple(int(v) for v in context.step)
    reduce_level_update(
        ledger,
        level=int(context.level),
        step=step_tuple,
        mode=mode,
        plane_state=snapshot_iface.viewport_state.plane,
        volume_state=snapshot_iface.viewport_state.volume,
        origin="runtime.apply",
    )
