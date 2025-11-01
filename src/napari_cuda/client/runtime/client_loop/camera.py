"""Camera helpers for the streaming client loop."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from napari_cuda.client.control.emitters import NapariCameraIntentEmitter

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


@dataclass
class CameraState:
    """Track camera interaction state for the streaming client."""

    zoom_base: float
    cam_min_dt: float
    orbit_deg_per_px_x: float
    orbit_deg_per_px_y: float
    last_cam_send: float = 0.0
    dragging: bool = False
    orbit_dragging: bool = False
    last_wx: float = 0.0
    last_wy: float = 0.0
    pan_dx_accum: float = 0.0
    pan_dy_accum: float = 0.0
    orbit_daz_accum: float = 0.0
    orbit_del_accum: float = 0.0
    cursor_wx: float = 0.0
    cursor_wy: float = 0.0
    last_zoom_factor: Optional[float] = None
    last_zoom_widget_px: Optional[tuple[float, float]] = None
    last_zoom_video_px: Optional[tuple[float, float]] = None
    last_zoom_anchor_px: Optional[tuple[float, float]] = None
    last_pan_dx_sent: float = 0.0
    last_pan_dy_sent: float = 0.0

    @classmethod
    def from_env(cls, env_cfg: object) -> CameraState:
        rate = getattr(env_cfg, 'camera_rate_hz', 60.0) or 60.0
        deg_x = getattr(env_cfg, 'orbit_deg_per_px_x', 0.3) or 0.3
        deg_y = getattr(env_cfg, 'orbit_deg_per_px_y', 0.3) or 0.3
        zoom_base = float(getattr(env_cfg, 'zoom_base', 1.2) or 1.2)
        cam_min_dt = 1.0 / max(1.0, float(rate))
        return cls(
            zoom_base=zoom_base,
            cam_min_dt=cam_min_dt,
            orbit_deg_per_px_x=float(deg_x),
            orbit_deg_per_px_y=float(deg_y),
        )


def handle_wheel_zoom(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    data: dict,
    *,
    widget_to_video: Callable[[float, float], tuple[float, float]],
    server_anchor_from_video: Callable[[float, float], tuple[float, float]],
    log_dims_info: bool,
) -> None:
    ay = float(data.get('angle_y') or 0.0)
    py = float(data.get('pixel_y') or 0.0)
    xw = float(data.get('x_px') or 0.0)
    yw = float(data.get('y_px') or 0.0)
    base = float(state.zoom_base)
    if ay != 0.0:
        s = 1.0 if ay > 0 else -1.0
        factor = base ** s
    elif py != 0.0:
        factor = base ** (py / 30.0)
    else:
        return
    xv, yv = widget_to_video(xw, yw)
    ax, ay_server = server_anchor_from_video(xv, yv)
    state.last_zoom_factor = float(factor)
    state.last_zoom_widget_px = (float(xw), float(yw))
    state.last_zoom_video_px = (float(xv), float(yv))
    state.last_zoom_anchor_px = (float(ax), float(ay_server))
    ok = camera_emitter.zoom(
        factor=float(factor),
        anchor_px=(float(ax), float(ay_server)),
        origin='wheel',
    )
    if ok:
        state.last_cam_send = time.perf_counter()
    if log_dims_info:
        logger.info(
            "wheel+mod->camera.zoom f=%.4f at(%.1f,%.1f) sent=%s",
            float(factor),
            float(ax),
            float(ay_server),
            bool(ok),
        )


def handle_pointer(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    data: dict,
    *,
    widget_to_video: Callable[[float, float], tuple[float, float]],
    video_delta_to_canvas: Callable[[float, float], tuple[float, float]],
    log_dims_info: bool,
    in_vol3d: bool,
    alt_mask: int,
) -> None:
    phase = (data.get('phase') or '').lower()
    xw_raw = data.get('x_px')
    yw_raw = data.get('y_px')
    xw = float(xw_raw) if xw_raw is not None else 0.0
    yw = float(yw_raw) if yw_raw is not None else 0.0
    state.cursor_wx = xw
    state.cursor_wy = yw
    mods = int(data.get('mods') or 0)
    alt = (mods & int(alt_mask)) != 0

    if phase == 'down':
        state.dragging = True
        state.last_wx = xw
        state.last_wy = yw
        state.pan_dx_accum = 0.0
        state.pan_dy_accum = 0.0
        if alt and in_vol3d:
            state.orbit_dragging = True
            state.orbit_daz_accum = 0.0
            state.orbit_del_accum = 0.0
        return

    if phase == 'move' and state.dragging:
        xv0, yv0 = widget_to_video(state.last_wx, state.last_wy)
        xv1, yv1 = widget_to_video(xw, yw)
        dx_v = (xv1 - xv0)
        dy_v = (yv1 - yv0)
        dx_c, dy_c = video_delta_to_canvas(dx_v, dy_v)
        if log_dims_info:
            logger.info(
                "pointer move: mods=%d alt=%s vol3d=%s dx_c=%.2f dy_c=%.2f",
                int(mods), bool(alt), bool(in_vol3d), float(dx_c), float(dy_c),
            )
        if alt and in_vol3d:
            if not state.orbit_dragging:
                state.orbit_dragging = True
                state.orbit_daz_accum = 0.0
                state.orbit_del_accum = 0.0
            state.orbit_daz_accum += float(dx_c) * float(state.orbit_deg_per_px_x)
            state.orbit_del_accum += float(-dy_c) * float(state.orbit_deg_per_px_y)
        else:
            if state.orbit_dragging:
                flush_orbit(
                    camera_emitter,
                    state,
                    force=True,
                    origin='pointer.move',
                    log_dims_info=log_dims_info,
                )
                state.orbit_dragging = False
            state.pan_dx_accum += float(dx_c)
            state.pan_dy_accum += float(dy_c)
        state.last_wx = xw
        state.last_wy = yw
        if state.orbit_dragging:
            flush_orbit_if_due(
                camera_emitter,
                state,
                log_dims_info=log_dims_info,
            )
        else:
            flush_pan_if_due(
                camera_emitter,
                state,
                log_dims_info=log_dims_info,
            )
        return

    if phase == 'up':
        state.dragging = False
        if state.orbit_dragging:
            flush_orbit(
                camera_emitter,
                state,
                force=True,
                origin='pointer.up',
                log_dims_info=log_dims_info,
            )
            state.orbit_dragging = False
        else:
            flush_pan(
                camera_emitter,
                state,
                force=True,
                origin='pointer.up',
                log_dims_info=log_dims_info,
            )


def flush_pan_if_due(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    *,
    log_dims_info: bool,
) -> None:
    now = time.perf_counter()
    if (now - float(state.last_cam_send or 0.0)) >= state.cam_min_dt:
        flush_pan(
            camera_emitter,
            state,
            force=False,
            origin='pointer.drag',
            log_dims_info=log_dims_info,
        )


def flush_pan(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    *,
    force: bool,
    origin: str,
    log_dims_info: bool,
) -> None:
    dx = float(state.pan_dx_accum or 0.0)
    dy = float(state.pan_dy_accum or 0.0)
    if not force and abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return
    ok = camera_emitter.pan(dx_px=float(dx), dy_px=float(dy), origin=origin)
    if ok:
        state.last_pan_dx_sent = float(dx)
        state.last_pan_dy_sent = float(dy)
        state.last_cam_send = time.perf_counter()
    state.pan_dx_accum = 0.0
    state.pan_dy_accum = 0.0
    if log_dims_info:
        logger.info(
            "drag->camera.pan dx=%.1f dy=%.1f sent=%s",
            float(dx),
            float(dy),
            bool(ok),
        )


def flush_orbit_if_due(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    *,
    log_dims_info: bool,
) -> None:
    now = time.perf_counter()
    if (now - float(state.last_cam_send or 0.0)) >= state.cam_min_dt:
        flush_orbit(
            camera_emitter,
            state,
            force=False,
            origin='pointer.drag',
            log_dims_info=log_dims_info,
        )


def flush_orbit(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    *,
    force: bool,
    origin: str,
    log_dims_info: bool,
) -> None:
    daz = float(state.orbit_daz_accum or 0.0)
    delv = float(state.orbit_del_accum or 0.0)
    if not force and abs(daz) < 1e-2 and abs(delv) < 1e-2:
        return
    ok = camera_emitter.orbit(d_az_deg=float(daz), d_el_deg=float(delv), origin=origin)
    if ok:
        state.last_cam_send = time.perf_counter()
    state.orbit_daz_accum = 0.0
    state.orbit_del_accum = 0.0
    if log_dims_info:
        logger.info(
            "alt-drag->camera.orbit daz=%.2f del=%.2f sent=%s",
            float(daz),
            float(delv),
            bool(ok),
        )


def zoom_steps_at_center(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    steps: int,
    *,
    widget_to_video: Callable[[float, float], tuple[float, float]],
    server_anchor_from_video: Callable[[float, float], tuple[float, float]],
    log_dims_info: bool,
    vid_size: tuple[Optional[int], Optional[int]],
) -> None:
    base = float(state.zoom_base)
    s = int(steps)
    if s == 0:
        return
    factor = base ** (-s)
    cursor_xw = state.cursor_wx
    cursor_yw = state.cursor_wy
    vw, vh = vid_size
    if math.isfinite(cursor_xw) and math.isfinite(cursor_yw):
        xv, yv = widget_to_video(float(cursor_xw), float(cursor_yw))
    else:
        xv = float(vw or 0) / 2.0
        yv = float(vh or 0) / 2.0
    ax_s, ay_s = server_anchor_from_video(xv, yv)
    ok = camera_emitter.zoom(
        factor=float(factor),
        anchor_px=(float(ax_s), float(ay_s)),
        origin='keys',
    )
    if ok:
        state.last_cam_send = time.perf_counter()
    if log_dims_info:
        logger.info(
            "key->camera.zoom_at f=%.4f at(%.1f,%.1f) sent=%s",
            float(factor),
            float(ax_s),
            float(ay_s),
            bool(ok),
        )


def reset_camera(camera_emitter: NapariCameraIntentEmitter, *, origin: str) -> bool:
    logger.info("%s->camera.reset (sending)", origin)
    ok = camera_emitter.reset(reason=origin, origin=origin)
    logger.info("%s->camera.reset sent=%s", origin, bool(ok))
    return bool(ok)


def set_camera(
    camera_emitter: NapariCameraIntentEmitter,
    *,
    center: Optional[Sequence[float]] = None,
    zoom: Optional[float] = None,
    angles: Optional[Sequence[float]] = None,
    origin: str,
) -> bool:
    payload_preview = {}
    if center is not None:
        payload_preview['center'] = list(center)
    if zoom is not None:
        payload_preview['zoom'] = float(zoom)
    if angles is not None:
        payload_preview['angles'] = list(angles)
    if not payload_preview:
        return False
    logger.info("%s->camera.set %s", origin, payload_preview)
    ok = camera_emitter.set(
        center=center,
        zoom=zoom,
        angles=angles,
        origin=origin,
    )
    return bool(ok)
