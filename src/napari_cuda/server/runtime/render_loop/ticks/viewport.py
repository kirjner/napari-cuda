"""Helpers to apply ViewportRunner plans on behalf of the worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.server.runtime.viewport import RenderMode
from napari_cuda.server.runtime.lod import level_policy
from napari_cuda.server.runtime.snapshots.plane import (
    apply_slice_level,
    apply_slice_roi,
)
from napari_cuda.server.runtime.snapshots.viewer_metadata import (
    apply_plane_metadata,
    apply_volume_metadata,
)
from napari_cuda.server.runtime.snapshots.volume import apply_volume_level
from ..tick_interface import RenderTickInterface
from napari_cuda.server.runtime.snapshots.interface import SnapshotInterface

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker

logger = logging.getLogger(__name__)


def run(worker: "EGLRendererWorker") -> None:
    """Drive a single viewport tick using the runner's current plan."""

    tick_iface = RenderTickInterface(worker)
    runner = tick_iface.viewport_runner
    if runner is None:
        return

    if tick_iface.level_policy_suppressed:
        runner.state.zoom_hint = None
        return

    if tick_iface.viewport_state.mode is RenderMode.VOLUME:
        runner.state.zoom_hint = None
        return

    source = tick_iface.get_scene_source() or tick_iface.ensure_scene_source()
    snapshot_iface = SnapshotInterface(worker)
    def _resolve(level: int, _rect):
        return snapshot_iface.viewport_roi_for_level(source, int(level), quiet=True)

    plan = runner.plan_tick(source=source, roi_resolver=_resolve)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "viewport.tick plan level_change=%s slice=%s pose_event=%s target_level=%d applied_level=%d awaiting=%s",
            plan.level_change,
            plan.slice_task.signature if plan.slice_task is not None else None,
            plan.pose_event.value if plan.pose_event is not None else None,
            int(runner.state.target_level),
            int(runner.state.applied_level),
            runner.state.awaiting_level_confirm,
        )

    if plan.zoom_hint is not None:
        tick_iface.render_mailbox_record_zoom_hint(float(plan.zoom_hint))

    level_applied = False

    if plan.level_change:
        target_level = int(runner.state.request.level)
        prev_level = tick_iface.current_level_index()
        applied_context = level_policy.build_level_context(
            worker,
            source,
            level=target_level,
            prev_level=prev_level,
        )

        snapshot_iface.set_current_level_index(int(applied_context.level))
        if snapshot_iface.viewport_state.mode is RenderMode.VOLUME:
            apply_volume_metadata(worker, source, applied_context)
            apply_volume_level(
                worker,
                source,
                applied_context,
                downgraded=bool(snapshot_iface.viewport_state.volume.downgraded),
            )
        else:
            apply_plane_metadata(worker, source, applied_context)
            apply_slice_level(snapshot_iface, source, applied_context)
        level_applied = True
        runner.mark_level_applied(int(applied_context.level))

    if (
        plan.slice_task is not None
        and snapshot_iface.viewport_state.mode is not RenderMode.VOLUME
        and not level_applied
    ):
        slice_task = plan.slice_task
        level_int = int(slice_task.level)
        step_tuple = tuple(int(v) for v in slice_task.step) if slice_task.step is not None else None
        signature_token = (level_int, step_tuple, slice_task.signature)
        if snapshot_iface.last_slice_signature() == signature_token:
            runner.mark_slice_applied(slice_task)
        else:
            apply_slice_roi(
                snapshot_iface,
                source,
                level_int,
                slice_task.roi,
                update_contrast=False,
                step=step_tuple,
            )
            runner.mark_slice_applied(slice_task)

    rect = snapshot_iface.current_panzoom_rect()
    if rect is not None:
        runner.update_camera_rect(rect)
    if plan.pose_event is not None:
        snapshot_iface.emit_current_camera_pose(plan.pose_event.value)
