"""Camera command execution helpers.

These helpers transform high-level camera commands into concrete VisPy camera
operations, returning metadata so the worker can orchestrate policy triggers
and zoom hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import logging
import time

from vispy import scene  # type: ignore

from napari_cuda.server.state import camera_ops as camops
from napari_cuda.server.state.server_scene import ServerSceneCommand


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
    zoom_hint: Optional[float]
    last_zoom_hint_ts: Optional[float]
    interaction_ts: Optional[float]


def apply_camera_commands(
    commands: Sequence[ServerSceneCommand],
    *,
    camera,
    view,
    canvas_size: Tuple[int, int],
    reset_camera: Callable[[object], None],
    debug_flags: CameraDebugFlags,
    mark_render_tick_needed: Optional[Callable[[], None]] = None,
    trigger_policy_refresh: Optional[Callable[[], None]] = None,
    record_zoom_hint: Optional[Callable[[float], None]] = None,
    last_zoom_hint_ts: Optional[float] = None,
    zoom_hint_hold_s: float = 0.0,
    now_fn: Callable[[], float] = time.perf_counter,
) -> CameraCommandOutcome:
    """Apply camera commands and report whether policy evaluation is needed."""

    if camera is None:
        # Camera not ready yet: record zoom hints for policy but skip execution.
        zoom_hint: Optional[float] = None
        now_val = now_fn() if commands else None
        if record_zoom_hint is not None:
            for command in commands:
                if command.kind == "zoom" and command.factor is not None and command.factor > 0.0:
                    ratio = float(command.factor)
                    record_zoom_hint(ratio)
                    zoom_hint = ratio
        interaction_ts = now_val
        return CameraCommandOutcome(
            camera_changed=False,
            policy_triggered=False,
            zoom_hint=zoom_hint,
            last_zoom_hint_ts=now_val if zoom_hint is not None else last_zoom_hint_ts,
            interaction_ts=interaction_ts,
        )

    cw, ch = canvas_size
    camera_changed = False
    policy_touch = False
    zoom_hint_ratio: Optional[float] = None
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
            if record_zoom_hint is not None:
                prev_ts = float(zoom_hint_ts) if zoom_hint_ts is not None else float('-inf')
                if (interaction_ts - prev_ts) >= float(zoom_hint_hold_s):
                    ratio = float(factor)
                    if ratio > 1.0:
                        ratio = 1.0 / ratio
                    record_zoom_hint(ratio)
                    ratio_recorded = ratio
                    zoom_hint_ts = interaction_ts
                    recorded = True
            if recorded:
                zoom_hint_ratio = ratio_recorded
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
        zoom_hint=zoom_hint_ratio,
        last_zoom_hint_ts=zoom_hint_ts if zoom_hint_ratio is not None else last_zoom_hint_ts,
        interaction_ts=interaction_ts,
    )


def process_commands(worker, commands: Sequence[ServerSceneCommand]) -> None:
    """Process camera commands on the worker, updating its state as needed."""

    if not commands:
        return

    worker._user_interaction_seen = True
    logger.debug("worker processing %d camera command(s)", len(commands))

    view = worker.view
    assert view is not None, "process_camera_commands requires an active VisPy view"
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

    def _record_zoom_hint(ratio: float) -> None:
        worker._render_mailbox.record_zoom_hint(float(ratio))

    outcome = apply_camera_commands(
        commands,
        camera=camera,
        view=view,
        canvas_size=canvas_wh,
        reset_camera=worker._apply_camera_reset,
        debug_flags=debug_flags,
        mark_render_tick_needed=_mark_render,
        trigger_policy_refresh=_trigger_policy,
        record_zoom_hint=_record_zoom_hint,
        last_zoom_hint_ts=worker._last_zoom_hint_ts,
        zoom_hint_hold_s=worker._zoom_hint_hold_s,
    )

    if outcome.last_zoom_hint_ts is not None:
        worker._last_zoom_hint_ts = float(outcome.last_zoom_hint_ts)
    if outcome.interaction_ts is not None:
        worker._last_interaction_ts = float(outcome.interaction_ts)


def log_zoom_drift(view, zoom_factor: float, anchor_px: tuple[float, float], center_world: tuple[float, float], canvas_size: tuple[int, int]) -> None:
    """Instrument anchored zoom to quantify pixel-space drift."""

    try:
        cam = view.camera
        tr = view.transform * view.scene.transform
        cw, ch = canvas_size
        ax_px = float(anchor_px[0])
        ay_tl = float(ch) - float(anchor_px[1])
        pre_map = tr.map([float(center_world[0]), float(center_world[1]), 0, 1])
        pre_x = float(pre_map[0]); pre_y = float(pre_map[1])
        pre_dx = pre_x - ax_px; pre_dy = pre_y - ay_tl
        cam.zoom(float(zoom_factor), center=center_world)  # type: ignore[call-arg]
        tr2 = view.transform * view.scene.transform
        post_map = tr2.map([float(center_world[0]), float(center_world[1]), 0, 1])
        post_x = float(post_map[0]); post_y = float(post_map[1])
        post_dx = post_x - ax_px; post_dy = post_y - ay_tl
        rect = getattr(cam, 'rect', None)
        rect_tuple = None
        if rect is not None:
            rect_tuple = (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))
        logger.info(
            "zoom_drift: f=%.4f ancTL=(%.1f,%.1f) world=(%.3f,%.3f) preTL=(%.1f,%.1f) err_pre=(%.2f,%.2f) postTL=(%.1f,%.1f) err_post=(%.2f,%.2f) cam.rect=%s canvas=%dx%d",
            float(zoom_factor), float(ax_px), float(ay_tl), float(center_world[0]), float(center_world[1]),
            pre_x, pre_y, pre_dx, pre_dy, post_x, post_y, post_dx, post_dy, str(rect_tuple), int(cw), int(ch)
        )
    except Exception:
        logger.debug("zoom_drift instrumentation failed", exc_info=True)


def log_pan_mapping(view, dx_px: float, dy_px: float, canvas_size: tuple[int, int]) -> None:
    """Instrument pixel-space pan mapping to world delta."""

    try:
        cam = view.camera
        tr = view.transform * view.scene.transform
        cw, ch = canvas_size
        cx_px = float(cw) * 0.5
        cy_px = float(ch) * 0.5
        p0 = tr.imap((cx_px, cy_px))
        p1 = tr.imap((cx_px + dx_px, cy_px + dy_px))
        dwx = float(p1[0] - p0[0])
        dwy = float(p1[1] - p0[1])
        cam.center = (float(cam.center[0] - dwx), float(cam.center[1] - dwy))  # type: ignore[attr-defined]
        logger.info(
            "pan_map: dx=%.2f dy=%.2f world_dx=%.3f world_dy=%.3f center=%s canvas=%dx%d",
            dx_px,
            dy_px,
            dwx,
            dwy,
            getattr(cam, 'center', None),
            cw,
            ch,
        )
    except Exception:
        logger.debug("pan mapping instrumentation failed", exc_info=True)


__all__ = [
    "CameraCommandOutcome",
    "CameraDebugFlags",
    "apply_camera_commands",
    "process_commands",
    "log_zoom_drift",
    "log_pan_mapping",
]
