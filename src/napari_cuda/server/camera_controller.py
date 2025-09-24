"""Camera command execution helpers.

These helpers transform high-level camera commands into concrete VisPy camera
operations, returning intent metadata so the worker can orchestrate policy
triggers and zoom hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import logging
import time

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
    last_zoom_hint_ts: Optional[float]
    interaction_ts: Optional[float]


def apply_camera_commands(
    commands: Sequence[CameraCommand],
    *,
    camera,
    view,
    canvas_size: Tuple[int, int],
    reset_camera: Callable[[object], None],
    debug_flags: CameraDebugFlags,
    mark_render_tick_needed: Optional[Callable[[], None]] = None,
    trigger_policy_refresh: Optional[Callable[[], None]] = None,
    record_zoom_intent: Optional[Callable[[float], None]] = None,
    last_zoom_hint_ts: Optional[float] = None,
    zoom_hint_hold_s: float = 0.0,
    now_fn: Callable[[], float] = time.perf_counter,
) -> CameraCommandOutcome:
    """Apply camera commands and report whether policy evaluation is needed."""

    if camera is None:
        # Camera not ready yet: record zoom intents for policy but skip execution.
        zoom_intent: Optional[float] = None
        now_val = now_fn() if commands else None
        if record_zoom_intent is not None:
            for command in commands:
                if command.kind == "zoom" and command.factor is not None and command.factor > 0.0:
                    ratio = float(command.factor)
                    record_zoom_intent(ratio)
                    zoom_intent = ratio
        interaction_ts = now_val
        return CameraCommandOutcome(
            camera_changed=False,
            policy_triggered=False,
            zoom_intent=zoom_intent,
            last_zoom_hint_ts=now_val if zoom_intent is not None else last_zoom_hint_ts,
            interaction_ts=interaction_ts,
        )

    cw, ch = canvas_size
    camera_changed = False
    policy_touch = False
    zoom_intent: Optional[float] = None
    zoom_hint_ts = last_zoom_hint_ts
    interaction_ts: Optional[float] = None

    for command in commands:
        interaction_ts = now_fn()
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
            recorded = False
            ratio_recorded: Optional[float] = None
            if record_zoom_intent is not None:
                prev_ts = float(zoom_hint_ts) if zoom_hint_ts is not None else float('-inf')
                if (interaction_ts - prev_ts) >= float(zoom_hint_hold_s):
                    ratio = float(factor)
                    if ratio > 1.0:
                        ratio = 1.0 / ratio
                    record_zoom_intent(ratio)
                    ratio_recorded = ratio
                    zoom_hint_ts = interaction_ts
                    recorded = True
            if recorded:
                zoom_intent = ratio_recorded
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

    if camera_changed and mark_render_tick_needed is not None:
        mark_render_tick_needed()
    if policy_touch and trigger_policy_refresh is not None:
        trigger_policy_refresh()

    return CameraCommandOutcome(
        camera_changed=camera_changed,
        policy_triggered=policy_touch,
        zoom_intent=zoom_intent,
        last_zoom_hint_ts=zoom_hint_ts if zoom_intent is not None else last_zoom_hint_ts,
        interaction_ts=interaction_ts,
    )


__all__ = [
    "CameraCommandOutcome",
    "CameraDebugFlags",
    "apply_camera_commands",
]
