"""Camera delta handling helpers for the EGL render worker."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from napari_cuda.server.runtime.camera.controller import (
    process_camera_deltas as _process_camera_deltas,
)
from napari_cuda.server.runtime.viewport import RenderMode
from napari_cuda.server.scene import CameraDeltaCommand

from ..apply import updates as render_updates
from ..plan_interface import RenderPlanInterface
from . import viewport

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraCommandResult:
    policy_triggered: bool
    camera_changed: bool
    last_command_seq: Optional[int]


def _apply_commands(
    worker: EGLRendererWorker,
    commands: Sequence[CameraDeltaCommand],
    *,
    reset_policy_suppression: bool,
) -> CameraCommandResult:
    tick_iface = RenderPlanInterface(worker)
    outcome = _process_camera_deltas(tick_iface, commands)

    last_seq: Optional[int]
    try:
        last_seq = int(outcome.last_command_seq)
    except (TypeError, ValueError):
        last_seq = None

    if last_seq is not None:
        tick_iface.bump_camera_sequences(last_seq)

    runner = tick_iface.viewport_runner
    if runner is not None:
        runner.ingest_camera_deltas(commands)
        if tick_iface.viewport_state.mode is RenderMode.PLANE:
            rect = tick_iface.current_panzoom_rect()
            if rect is not None:
                runner.update_camera_rect(rect)

    tick_iface.mark_render_tick_needed()
    tick_iface.record_user_interaction()

    if reset_policy_suppression:
        tick_iface.level_policy_suppressed = False

    if outcome.camera_changed and tick_iface.viewport_state.mode is RenderMode.VOLUME:
        tick_iface.emit_camera_pose("camera-delta")

    record_zoom_hint(tick_iface, commands)

    return CameraCommandResult(
        policy_triggered=bool(outcome.policy_triggered),
        camera_changed=bool(outcome.camera_changed),
        last_command_seq=last_seq,
    )


def process_commands(
    worker: EGLRendererWorker,
    commands: Sequence[CameraDeltaCommand],
) -> CameraCommandResult:
    """Apply camera commands immediately, mirroring the old worker method."""

    if not commands:
        return CameraCommandResult(False, False, None)

    tick_iface = RenderPlanInterface(worker)
    tick_iface.record_user_interaction()
    result = _apply_commands(worker, commands, reset_policy_suppression=False)
    viewport.run(worker)

    if (
        result.policy_triggered
        and tick_iface.viewport_state.mode is not RenderMode.VOLUME
        and not tick_iface.level_policy_suppressed
    ):
        tick_iface.evaluate_level_policy()

    return result


def drain(worker: EGLRendererWorker) -> None:
    """Drain queued camera deltas, then scene updates, for a render tick."""

    tick_iface = RenderPlanInterface(worker)
    commands = tick_iface.camera_queue_pop_all()
    result: Optional[CameraCommandResult] = None

    if commands:
        result = _apply_commands(worker, commands, reset_policy_suppression=True)

    render_updates.drain_scene_updates(worker)
    viewport.run(worker)

    if (
        result is not None
        and result.policy_triggered
        and tick_iface.viewport_state.mode is not RenderMode.VOLUME
        and not tick_iface.level_policy_suppressed
    ):
        tick_iface.evaluate_level_policy()


def record_zoom_hint(
    tick_iface: RenderPlanInterface,
    commands: Sequence[CameraDeltaCommand],
) -> None:
    """Capture the most recent zoom factor for policy evaluation."""

    if not commands:
        return

    mailbox = tick_iface.render_mailbox

    for command in reversed(commands):
        if getattr(command, "kind", None) != "zoom":
            continue
        factor = getattr(command, "factor", None)
        if factor is None:
            continue
        zoom_factor = float(factor)
        if zoom_factor > 0.0:
            mailbox.record_zoom_hint(zoom_factor)
            break


__all__ = [
    "CameraCommandResult",
    "drain",
    "process_commands",
    "record_zoom_hint",
]
