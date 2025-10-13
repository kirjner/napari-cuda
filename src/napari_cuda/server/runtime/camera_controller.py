"""Camera command execution helpers.

These helpers transform high-level camera commands into concrete VisPy camera
operations, returning metadata so the worker can orchestrate policy triggers
and zoom hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import logging

from vispy import scene  # type: ignore

from napari_cuda.server.runtime import camera_ops as camops
from napari_cuda.server.scene import CameraDeltaCommand


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDebugFlags:
    zoom: bool = False
    pan: bool = False
    orbit: bool = False
    reset: bool = False


@dataclass(frozen=True)
class CameraDeltaOutcome:
    camera_changed: bool
    policy_triggered: bool
    last_command_seq: int
    last_target: str


def apply_camera_deltas(
    commands: Sequence[CameraDeltaCommand],
    *,
    camera,
    view,
    canvas_size: Tuple[int, int],
    reset_camera: Callable[[object], None],
    debug_flags: CameraDebugFlags,
    mark_render_tick_needed: Optional[Callable[[], None]] = None,
    trigger_policy_refresh: Optional[Callable[[], None]] = None,
) -> CameraDeltaOutcome:
    """Apply camera deltas and report whether policy evaluation is needed."""

    if not commands:
        return CameraDeltaOutcome(
            camera_changed=False,
            policy_triggered=False,
            last_command_seq=0,
            last_target="main",
        )

    if camera is None:
        return CameraDeltaOutcome(
            camera_changed=False,
            policy_triggered=False,
            last_command_seq=int(commands[-1].command_seq),
            last_target=str(commands[-1].target),
        )

    width, height = canvas_size
    camera_changed = False
    policy_touch = False
    last_seq = int(commands[-1].command_seq)
    last_target = str(commands[-1].target)

    for command in commands:
        if command.kind == "zoom":
            factor = command.factor if command.factor is not None else 0.0
            if factor <= 0.0:
                raise ValueError("zoom factor must be positive")
            anchor = command.anchor_px or (width * 0.5, height * 0.5)
            if isinstance(camera, scene.cameras.TurntableCamera):
                camops.apply_zoom_3d(camera, factor)
            else:
                camops.apply_zoom_2d(camera, factor, (float(anchor[0]), float(anchor[1])), (width, height), view)
            camera_changed = True
            policy_touch = True
            if debug_flags.zoom and logger.isEnabledFor(logging.INFO):
                ax, ay = float(anchor[0]), float(anchor[1])
                logger.info("command zoom factor=%.4f anchor=(%.1f,%.1f)", float(factor), ax, ay)
            continue

        if command.kind == "pan":
            dx = float(command.dx_px)
            dy = float(command.dy_px)
            if dx == 0.0 and dy == 0.0:
                continue
            if isinstance(camera, scene.cameras.TurntableCamera):
                camops.apply_pan_3d(camera, dx, dy, (width, height))
            else:
                camops.apply_pan_2d(camera, dx, dy, (width, height), view)
            camera_changed = True
            policy_touch = True
            if debug_flags.pan and logger.isEnabledFor(logging.INFO):
                logger.info("command pan dx=%.2f dy=%.2f", dx, dy)
            continue

        if command.kind == "orbit":
            if not isinstance(camera, scene.cameras.TurntableCamera):
                raise ValueError("orbit command requires a TurntableCamera")
            daz = float(command.d_az_deg)
            delv = float(command.d_el_deg)
            if daz == 0.0 and delv == 0.0:
                continue
            camops.apply_orbit(camera, daz, delv)
            camera_changed = True
            policy_touch = True
            if debug_flags.orbit and logger.isEnabledFor(logging.INFO):
                logger.info("command orbit daz=%.2f del=%.2f", daz, delv)
            continue

        if command.kind == "reset":
            reset_camera(camera)
            camera_changed = True
            if debug_flags.reset and logger.isEnabledFor(logging.INFO):
                logger.info("command reset view")
            continue

        raise ValueError(f"unsupported camera command kind: {command.kind}")

    if camera_changed and mark_render_tick_needed is not None:
        mark_render_tick_needed()
    if policy_touch and trigger_policy_refresh is not None:
        trigger_policy_refresh()

    return CameraDeltaOutcome(
        camera_changed=camera_changed,
        policy_triggered=policy_touch,
        last_command_seq=last_seq,
        last_target=last_target,
    )


def process_camera_deltas(worker, commands: Sequence[CameraDeltaCommand]) -> CameraDeltaOutcome:
    """Process camera commands on the worker, updating its state as needed."""

    worker._user_interaction_seen = True
    logger.debug("worker processing %d camera command(s)", len(commands))

    view = worker.view
    assert view is not None, "process_camera_deltas requires an active VisPy view"
    camera = view.camera

    if worker.canvas is not None:
        canvas_wh = (int(worker.canvas.size[0]), int(worker.canvas.size[1]))
    else:
        canvas_wh = (worker.width, worker.height)

    debug_flags = CameraDebugFlags(
        zoom=worker._debug_zoom_drift,
        pan=worker._debug_pan,
        orbit=worker._debug_orbit,
        reset=worker._debug_reset,
    )

    def _mark_render() -> None:
        worker._mark_render_tick_needed()

    def _trigger_policy() -> None:
        worker._level_policy_refresh_needed = True

    return apply_camera_deltas(
        commands,
        camera=camera,
        view=view,
        canvas_size=canvas_wh,
        reset_camera=worker._apply_camera_reset,
        debug_flags=debug_flags,
        mark_render_tick_needed=_mark_render,
        trigger_policy_refresh=_trigger_policy,
    )


__all__ = [
    "CameraDeltaOutcome",
    "CameraDebugFlags",
    "apply_camera_deltas",
    "process_camera_deltas",
]
