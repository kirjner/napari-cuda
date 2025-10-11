"""Camera helpers for the streaming client loop."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from napari_cuda.client.control.emitters import NapariCameraIntentEmitter

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


CameraSnapshot = Tuple[Optional[Tuple[float, ...]], Optional[float], Optional[Tuple[float, ...]]]


@dataclass
class CameraState:
    """Track camera interaction state for the streaming client."""

    zoom_base: float
    cam_min_dt: float
    last_cam_send: float = 0.0
    dragging: bool = False
    last_wx: float = 0.0
    last_wy: float = 0.0
    cursor_wx: float = 0.0
    cursor_wy: float = 0.0
    last_zoom_factor: Optional[float] = None
    last_zoom_widget_px: Optional[tuple[float, float]] = None
    last_zoom_video_px: Optional[tuple[float, float]] = None
    last_zoom_anchor_px: Optional[tuple[float, float]] = None
    last_payload: Optional[CameraSnapshot] = None

    @classmethod
    def from_env(cls, env_cfg: object) -> "CameraState":
        rate = getattr(env_cfg, 'camera_rate_hz', 60.0) or 60.0
        zoom_base = float(getattr(env_cfg, 'zoom_base', 1.2) or 1.2)
        cam_min_dt = 1.0 / max(1.0, float(rate))
        return cls(
            zoom_base=zoom_base,
            cam_min_dt=cam_min_dt,
        )


def _to_float_tuple(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    try:
        seq = tuple(float(v) for v in value)
    except Exception:
        return None
    if not seq:
        return None
    return seq


def _snapshot_viewer_camera(viewer: Any) -> CameraSnapshot:
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return (None, None, None)
    center = _to_float_tuple(getattr(camera, "center", None))
    zoom_value = getattr(camera, "zoom", None)
    try:
        zoom = float(zoom_value) if zoom_value is not None else None
    except Exception:
        zoom = None
    angles_raw = getattr(camera, "angles", None)
    angles = _to_float_tuple(angles_raw)
    return (center, zoom, angles)


def emit_camera_set_from_viewer(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    viewer: Any | None,
    *,
    origin: str,
    force: bool = False,
) -> bool:
    """Read the viewer camera and emit a `camera.set` intent if needed."""

    if viewer is None:
        return False

    center, zoom, angles = _snapshot_viewer_camera(viewer)
    if center is None and zoom is None and angles is None:
        return False

    payload: CameraSnapshot = (center, zoom, angles)
    now = time.perf_counter()

    if not force:
        if state.cam_min_dt > 0.0 and (now - float(state.last_cam_send or 0.0)) < state.cam_min_dt:
            return False
        if payload == state.last_payload:
            return False

    kwargs: dict[str, Any] = {}
    if center is not None:
        kwargs["center"] = center
    if zoom is not None:
        kwargs["zoom"] = zoom
    if angles is not None:
        kwargs["angles"] = angles
    if not kwargs:
        return False

    ok = camera_emitter.set(origin=origin, **kwargs)
    if ok:
        state.last_cam_send = now
        state.last_payload = payload
    return bool(ok)


def handle_wheel_zoom(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    data: dict,
    *,
    widget_to_video: Callable[[float, float], tuple[float, float]],
    server_anchor_from_video: Callable[[float, float], tuple[float, float]],
    log_dims_info: bool,
    viewer: Any | None,
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
        factor = 1.0
    xv, yv = widget_to_video(xw, yw)
    ax, ay_server = server_anchor_from_video(xv, yv)
    state.last_zoom_factor = float(factor)
    state.last_zoom_widget_px = (float(xw), float(yw))
    state.last_zoom_video_px = (float(xv), float(yv))
    state.last_zoom_anchor_px = (float(ax), float(ay_server))

    sent = emit_camera_set_from_viewer(
        camera_emitter,
        state,
        viewer,
        origin='wheel',
        force=True,
    )
    if log_dims_info:
        logger.info(
            "wheel+mod camera.set factor=%.4f at(%.1f,%.1f) sent=%s",
            float(factor),
            float(ax),
            float(ay_server),
            bool(sent),
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
    viewer: Any | None,
) -> None:
    del in_vol3d, alt_mask  # kept for signature compatibility; no-op in absolute mode

    phase = (data.get('phase') or '').lower()
    xw_raw = data.get('x_px')
    yw_raw = data.get('y_px')
    xw = float(xw_raw) if xw_raw is not None else 0.0
    yw = float(yw_raw) if yw_raw is not None else 0.0
    state.cursor_wx = xw
    state.cursor_wy = yw

    if phase == 'down':
        state.dragging = True
        state.last_wx = xw
        state.last_wy = yw
        emit_camera_set_from_viewer(
            camera_emitter,
            state,
            viewer,
            origin='pointer.down',
            force=True,
        )
        return

    if phase == 'move' and state.dragging:
        xv0, yv0 = widget_to_video(state.last_wx, state.last_wy)
        xv1, yv1 = widget_to_video(xw, yw)
        dx_v = (xv1 - xv0)
        dy_v = (yv1 - yv0)
        dx_c, dy_c = video_delta_to_canvas(dx_v, dy_v)
        if log_dims_info:
            logger.info(
                "pointer move drag dx_c=%.2f dy_c=%.2f",
                float(dx_c),
                float(dy_c),
            )
        sent = emit_camera_set_from_viewer(
            camera_emitter,
            state,
            viewer,
            origin='pointer.drag',
            force=False,
        )
        if sent:
            state.last_wx = xw
            state.last_wy = yw
        return

    if phase == 'up':
        if state.dragging:
            emit_camera_set_from_viewer(
                camera_emitter,
                state,
                viewer,
                origin='pointer.up',
                force=True,
            )
        state.dragging = False
        return

    if phase == 'cancel':
        state.dragging = False


def zoom_steps_at_center(
    camera_emitter: NapariCameraIntentEmitter,
    state: CameraState,
    steps: int,
    *,
    widget_to_video: Callable[[float, float], tuple[float, float]],
    server_anchor_from_video: Callable[[float, float], tuple[float, float]],
    log_dims_info: bool,
    vid_size: tuple[Optional[int], Optional[int]],
    viewer: Any | None,
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
    state.last_zoom_factor = float(factor)
    state.last_zoom_widget_px = (float(cursor_xw), float(cursor_yw))
    state.last_zoom_video_px = (float(xv), float(yv))
    state.last_zoom_anchor_px = (float(ax_s), float(ay_s))

    if viewer is None:
        return
    viewer_camera = getattr(viewer, "camera", None)
    if viewer_camera is None:
        return
    try:
        current_zoom = float(getattr(viewer_camera, "zoom"))
    except Exception:
        current_zoom = None
    if current_zoom is not None:
        try:
            viewer_camera.zoom = current_zoom * float(factor)
        except Exception:
            logger.debug("zoom_steps_at_center failed to adjust viewer zoom", exc_info=True)

    sent = emit_camera_set_from_viewer(
        camera_emitter,
        state,
        viewer,
        origin='keys',
        force=True,
    )
    if log_dims_info:
        logger.info(
            "keys camera.set factor=%.4f at(%.1f,%.1f) sent=%s",
            float(factor),
            float(ax_s),
            float(ay_s),
            bool(sent),
        )


def reset_camera(
    camera_emitter: NapariCameraIntentEmitter,
    *,
    origin: str,
    viewer: Any | None,
    state: CameraState,
) -> bool:
    logger.info("%s->camera.reset (absolute snapshot)", origin)
    sent = emit_camera_set_from_viewer(
        camera_emitter,
        state,
        viewer,
        origin=origin,
        force=True,
    )
    logger.info("%s->camera.set sent=%s", origin, bool(sent))
    return bool(sent)


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
