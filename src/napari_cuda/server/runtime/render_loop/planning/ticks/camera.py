"""Camera delta handling helpers for the EGL render worker."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from napari_cuda.server.runtime.camera.controller import (
    process_camera_deltas as _process_camera_deltas,
)
from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import CameraDeltaCommand

from ...render_interface import RenderInterface
from ..staging import drain_scene_updates
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
    render_iface = RenderInterface(worker)
    outcome = _process_camera_deltas(render_iface, commands)

    last_seq: Optional[int]
    try:
        last_seq = int(outcome.last_command_seq)
    except (TypeError, ValueError):
        last_seq = None

    if last_seq is not None:
        render_iface.bump_camera_sequences(last_seq)

    runner = render_iface.viewport_runner
    if runner is not None:
        runner.ingest_camera_deltas(commands)
        if render_iface.viewport_state.mode is RenderMode.PLANE:
            rect = render_iface.current_panzoom_rect()
            if rect is not None:
                runner.update_camera_rect(rect)

    render_iface.mark_render_tick_needed()
    render_iface.record_user_interaction()

    if reset_policy_suppression:
        render_iface.level_policy_suppressed = False

    if outcome.camera_changed and render_iface.viewport_state.mode is RenderMode.VOLUME:
        render_iface.emit_current_camera_pose("camera-delta")

    record_zoom_hint(render_iface, commands)

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

    render_iface = RenderInterface(worker)
    render_iface.record_user_interaction()
    result = _apply_commands(worker, commands, reset_policy_suppression=False)
    viewport.run(worker)

    if (
        result.policy_triggered
        and render_iface.viewport_state.mode is not RenderMode.VOLUME
        and not render_iface.level_policy_suppressed
    ):
        render_iface.evaluate_level_policy()

    return result


def drain(worker: EGLRendererWorker) -> None:
    """Drain queued camera deltas, then scene updates, for a render tick."""

    render_iface = RenderInterface(worker)
    commands = render_iface.camera_queue_pop_all()
    result: Optional[CameraCommandResult] = None

    if commands:
        result = _apply_commands(worker, commands, reset_policy_suppression=True)

    drain_scene_updates(worker)
    viewport.run(worker)

    if (
        result is not None
        and result.policy_triggered
        and render_iface.viewport_state.mode is not RenderMode.VOLUME
        and not render_iface.level_policy_suppressed
    ):
        render_iface.evaluate_level_policy()


def record_zoom_hint(
    tick_iface: RenderInterface,
    commands: Sequence[CameraDeltaCommand],
) -> None:
    """Capture the most recent zoom factor for policy evaluation."""

    if not commands:
        return

    for command in reversed(commands):
        if getattr(command, "kind", None) != "zoom":
            continue
        factor = getattr(command, "factor", None)
        if factor is None:
            continue
        zoom_factor = float(factor)
        if zoom_factor > 0.0:
            tick_iface.record_zoom_hint(zoom_factor)
            break


__all__ = [
    "CameraCommandResult",
    "drain",
    "process_commands",
    "record_zoom_hint",
]
