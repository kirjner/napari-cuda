"""Helpers to apply ViewportRunner plans on behalf of the worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.server.runtime.viewport import RenderMode
from napari_cuda.server.runtime.viewport.roi import viewport_roi_for_level
from napari_cuda.server.runtime.worker import level_policy
from napari_cuda.server.runtime.worker.snapshots import (
    apply_plane_metadata,
    apply_slice_level,
    apply_slice_roi,
    apply_volume_level,
    apply_volume_metadata,
)

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker

logger = logging.getLogger(__name__)


def run(worker: "EGLRendererWorker") -> None:
    """Drive a single viewport tick using the runner's current plan."""

    runner = worker._viewport_runner
    if runner is None:
        return

    if worker._level_policy_suppressed:
        runner.state.zoom_hint = None
        return

    if worker._viewport_state.mode is RenderMode.VOLUME:
        runner.state.zoom_hint = None
        return

    source = worker._scene_source or worker._ensure_scene_source()

    def _resolve(level: int, _rect):
        return viewport_roi_for_level(worker, source, int(level), quiet=True)

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
        worker._render_mailbox.record_zoom_hint(float(plan.zoom_hint))

    level_applied = False

    if plan.level_change:
        target_level = int(runner.state.request.level)
        prev_level = int(worker._current_level_index())
        applied_context = level_policy.build_level_context(
            worker,
            source,
            level=target_level,
            prev_level=prev_level,
        )

        worker._set_current_level_index(int(applied_context.level))
        if worker._viewport_state.mode is RenderMode.VOLUME:
            apply_volume_metadata(worker, source, applied_context)
            apply_volume_level(
                worker,
                source,
                applied_context,
                downgraded=bool(worker._viewport_state.volume.downgraded),
            )
        else:
            apply_plane_metadata(worker, source, applied_context)
            apply_slice_level(worker, source, applied_context)
        level_applied = True
        runner.mark_level_applied(int(applied_context.level))

    if (
        plan.slice_task is not None
        and worker._viewport_state.mode is not RenderMode.VOLUME
        and not level_applied
    ):
        slice_task = plan.slice_task
        level_int = int(slice_task.level)
        step_tuple = tuple(int(v) for v in slice_task.step) if slice_task.step is not None else None
        signature_token = (level_int, step_tuple, slice_task.signature)
        if worker._last_slice_signature == signature_token:
            runner.mark_slice_applied(slice_task)
        else:
            apply_slice_roi(
                worker,
                source,
                level_int,
                slice_task.roi,
                update_contrast=False,
                step=step_tuple,
            )
            runner.mark_slice_applied(slice_task)

    rect = worker._current_panzoom_rect()
    if rect is not None:
        runner.update_camera_rect(rect)
    if plan.pose_event is not None:
        worker._emit_current_camera_pose(plan.pose_event.value)
