"""Camera command execution helpers.

These helpers transform high-level camera commands into concrete VisPy camera
operations, returning intent metadata so the worker can orchestrate policy
triggers and zoom hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import logging

from vispy import scene  # type: ignore

from napari_cuda.server import camera_ops as camops
from napari_cuda.server.state_machine import CameraCommand


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDebugFlags:
    zoom: bool = False
    pan: bool = False
    orbit: bool = False
    reset: bool = False


@dataclass(frozen=True)
class CameraCommandOutcome:
    camera_changed: bool
    policy_triggered: bool
    zoom_intent: Optional[float]


def apply_camera_commands(
    commands: Sequence[CameraCommand],
    *,
    camera,
    view,
    canvas_size: Tuple[int, int],
    reset_camera: Callable[[object], None],
    debug_flags: CameraDebugFlags,
) -> CameraCommandOutcome:
    """Apply camera commands and report whether policy evaluation is needed."""

    if camera is None:
        raise ValueError("camera must be initialised before applying commands")

    cw, ch = canvas_size
    camera_changed = False
    policy_touch = False
    zoom_intent: Optional[float] = None

    for command in commands:
        kind = command.kind
        if kind == "zoom":
            if command.factor is None:
                raise ValueError("zoom command requires a factor")
            factor = float(command.factor)
            if factor <= 0.0:
                raise ValueError("zoom factor must be positive")
            anchor = command.anchor_px or (cw * 0.5, ch * 0.5)
            if isinstance(camera, scene.cameras.TurntableCamera):
                camops.apply_zoom_3d(camera, factor)
            else:
                camops.apply_zoom_2d(camera, factor, (float(anchor[0]), float(anchor[1])), (cw, ch), view)
            camera_changed = True
            policy_touch = True
            zoom_intent = factor
            if debug_flags.zoom and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "command zoom factor=%.4f anchor=(%.1f,%.1f)",
                    factor,
                    float(anchor[0]),
                    float(anchor[1]),
                )
        elif kind == "pan":
            dx = float(command.dx_px)
            dy = float(command.dy_px)
            if dx == 0.0 and dy == 0.0:
                continue
            if isinstance(camera, scene.cameras.TurntableCamera):
                camops.apply_pan_3d(camera, dx, dy, (cw, ch))
            else:
                camops.apply_pan_2d(camera, dx, dy, (cw, ch), view)
            camera_changed = True
            policy_touch = True
            if debug_flags.pan and logger.isEnabledFor(logging.INFO):
                logger.info("command pan dx=%.2f dy=%.2f", dx, dy)
        elif kind == "orbit":
            if not isinstance(camera, scene.cameras.TurntableCamera):
                raise RuntimeError("orbit command requires a TurntableCamera")
            daz = float(command.d_az_deg)
            delv = float(command.d_el_deg)
            if daz == 0.0 and delv == 0.0:
                continue
            camops.apply_orbit(camera, daz, delv)
            camera_changed = True
            policy_touch = True
            if debug_flags.orbit and logger.isEnabledFor(logging.INFO):
                logger.info("command orbit daz=%.2f del=%.2f", daz, delv)
        elif kind == "reset":
            reset_camera(camera)
            camera_changed = True
            if debug_flags.reset and logger.isEnabledFor(logging.INFO):
                logger.info("command reset view")
        else:
            raise ValueError(f"unsupported camera command kind: {kind}")

    return CameraCommandOutcome(camera_changed, policy_touch, zoom_intent)


__all__ = [
    "CameraCommandOutcome",
    "CameraDebugFlags",
    "apply_camera_commands",
]
