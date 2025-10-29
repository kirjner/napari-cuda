"""Camera delta handling helpers for the EGL render worker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from napari_cuda.server.runtime.camera.controller import (
    process_camera_deltas as _process_camera_deltas,
)
from napari_cuda.server.runtime.viewport import RenderMode
from napari_cuda.server.scene import CameraDeltaCommand

from .. import render_updates
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
    worker: "EGLRendererWorker",
    commands: Sequence[CameraDeltaCommand],
    *,
    reset_policy_suppression: bool,
) -> CameraCommandResult:
    outcome = _process_camera_deltas(worker, commands)

    last_seq: Optional[int]
    try:
        last_seq = int(outcome.last_command_seq)
    except (TypeError, ValueError):
        last_seq = None

    if last_seq is not None:
        worker._max_camera_command_seq = max(  # noqa: SLF001
            int(worker._max_camera_command_seq),
            last_seq,
        )
        worker._pose_seq = max(  # noqa: SLF001
            int(worker._pose_seq),
            int(worker._max_camera_command_seq),
        )

    runner = worker._viewport_runner  # noqa: SLF001
    if runner is not None:
        runner.ingest_camera_deltas(commands)
        if worker._viewport_state.mode is RenderMode.PLANE:  # noqa: SLF001
            rect = worker._current_panzoom_rect()  # noqa: SLF001
            if rect is not None:
                runner.update_camera_rect(rect)

    worker._mark_render_tick_needed()  # noqa: SLF001
    worker._user_interaction_seen = True  # noqa: SLF001
    worker._last_interaction_ts = time.perf_counter()  # noqa: SLF001

    if reset_policy_suppression:
        worker._level_policy_suppressed = False  # noqa: SLF001

    if outcome.camera_changed and worker._viewport_state.mode is RenderMode.VOLUME:  # noqa: SLF001
        worker._emit_current_camera_pose("camera-delta")  # noqa: SLF001

    record_zoom_hint(worker, commands)

    return CameraCommandResult(
        policy_triggered=bool(outcome.policy_triggered),
        camera_changed=bool(outcome.camera_changed),
        last_command_seq=last_seq,
    )


def process_commands(
    worker: "EGLRendererWorker",
    commands: Sequence[CameraDeltaCommand],
) -> CameraCommandResult:
    """Apply camera commands immediately, mirroring the old worker method."""

    if not commands:
        return CameraCommandResult(False, False, None)

    result = _apply_commands(worker, commands, reset_policy_suppression=False)
    viewport.run(worker)

    if (
        result.policy_triggered
        and worker._viewport_state.mode is not RenderMode.VOLUME  # noqa: SLF001
        and not worker._level_policy_suppressed  # noqa: SLF001
    ):
        worker._evaluate_level_policy()  # noqa: SLF001

    return result


def drain(worker: "EGLRendererWorker") -> None:
    """Drain queued camera deltas, then scene updates, for a render tick."""

    commands = worker._camera_queue.pop_all()  # noqa: SLF001
    result: Optional[CameraCommandResult] = None

    if commands:
        result = _apply_commands(worker, commands, reset_policy_suppression=True)

    render_updates.drain_scene_updates(worker)
    viewport.run(worker)

    if (
        result is not None
        and result.policy_triggered
        and worker._viewport_state.mode is not RenderMode.VOLUME  # noqa: SLF001
        and not worker._level_policy_suppressed  # noqa: SLF001
    ):
        worker._evaluate_level_policy()  # noqa: SLF001


def record_zoom_hint(
    worker: "EGLRendererWorker",
    commands: Sequence[CameraDeltaCommand],
) -> None:
    """Capture the most recent zoom factor for policy evaluation."""

    if not commands:
        return

    mailbox = worker._render_mailbox  # noqa: SLF001

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
    "process_commands",
    "drain",
    "record_zoom_hint",
]
