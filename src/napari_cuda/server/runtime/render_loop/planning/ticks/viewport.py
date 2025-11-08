"""Helpers to apply ViewportPlanner plans on behalf of the worker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari_cuda.server.runtime.lod import level_policy
from napari_cuda.server.runtime.render_loop.applying.apply import apply_viewport_plan
from napari_cuda.server.runtime.render_loop.applying.interface import (
    RenderApplyInterface,
)
from napari_cuda.server.scene.viewport import RenderMode

from ..interface import RenderPlanInterface

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker

logger = logging.getLogger(__name__)


def run(worker: EGLRendererWorker) -> None:
    """Drive a single viewport tick using the runner's current plan."""

    tick_iface = RenderPlanInterface(worker)
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
    apply_iface = RenderApplyInterface(worker)
    def _resolve(level: int, _rect):
        return apply_iface.viewport_roi_for_level(source, int(level), quiet=True)

    level_resolver = lambda src, level: apply_iface.resolve_volume_intent_level(src, int(level))
    plan = runner.plan_tick(
        source=source,
        roi_resolver=_resolve,
        volume_level_resolver=level_resolver,
    )
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

    apply_viewport_plan(apply_iface, None, plan, source=source)

    rect = apply_iface.current_panzoom_rect()
    if rect is not None:
        runner.update_camera_rect(rect)
