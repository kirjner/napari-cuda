"""Camera operations (free functions) extracted from the worker.

These functions mirror the behavior of the internal `_CameraOps` helper inside
``render_worker``. ``render_worker`` delegates to these helpers to keep behavior
unchanged while making the math testable and reusable.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from vispy.scene.cameras import PanZoomCamera, TurntableCamera


def anchor_to_world(ax_px: float, ay_px: float, canvas_wh: tuple[int, int], view) -> tuple[float, float]:
    """Map an anchor pixel to world coordinates using the view transform."""
    width, height = canvas_wh
    mapped = view.transform * view.scene.transform
    ay_top = float(height) - float(ay_px)
    point = mapped.imap([float(ax_px), ay_top, 0.0, 1.0])
    return float(point[0]), float(point[1])


def per_pixel_world_scale_3d(cam: TurntableCamera, canvas_wh: tuple[int, int]) -> tuple[float, float]:
    """Estimate world-units per pixel for a 3D camera from FOV and distance."""
    width, height = canvas_wh
    fov_rad = math.radians(float(cam.fov))
    distance = float(cam.distance)
    denom = max(1e-6, float(height))
    sy = 2.0 * distance * math.tan(0.5 * fov_rad) / denom
    sx = sy * (float(width) / denom)
    return sx, sy


def apply_orbit(cam: TurntableCamera, d_az_deg: float, d_el_deg: float) -> None:
    """Apply azimuth/elevation deltas to a turntable camera."""
    if d_az_deg == 0.0 and d_el_deg == 0.0:
        return
    cam.azimuth = float(cam.azimuth) + float(d_az_deg)
    cam.elevation = float(cam.elevation) + float(d_el_deg)


def apply_zoom_3d(cam: TurntableCamera, factor: float) -> None:
    """Apply multiplicative zoom in 3D by scaling distance."""
    if factor <= 0.0:
        return
    cam.distance = max(1e-6, float(cam.distance) * float(factor))


def apply_zoom_2d(cam: PanZoomCamera, factor: float, anchor_px: tuple[float, float], canvas_wh: tuple[int, int], view) -> None:
    """Zoom around an anchor in 2D using view transforms for correct centering."""
    if factor <= 0.0:
        return
    anchor_world = anchor_to_world(anchor_px[0], anchor_px[1], canvas_wh, view)
    cam_zoom = cam.zoom
    if callable(cam_zoom):
        cam_zoom(float(factor), center=anchor_world)
        return
    type(cam).zoom(cam, float(factor), center=anchor_world)


def apply_pan_3d(cam: TurntableCamera, dx_px: float, dy_px: float, canvas_wh: tuple[int, int]) -> None:
    """Pan/dolly in 3D using pixel deltas mapped to world units."""
    if dx_px == 0.0 and dy_px == 0.0:
        return
    scale_x, scale_y = per_pixel_world_scale_3d(cam, canvas_wh)
    center_x, center_y = cam.center[:2]
    cam.center = (float(center_x) - float(dx_px) * scale_x, float(center_y))
    if dy_px != 0.0:
        cam.distance = max(1e-6, float(cam.distance) - float(dy_px) * scale_y)


def apply_pan_2d(cam: PanZoomCamera, dx_px: float, dy_px: float, canvas_wh: tuple[int, int], view) -> None:
    """Pan in 2D using pixel deltas mapped through view transforms."""
    if dx_px == 0.0 and dy_px == 0.0:
        return
    width, height = canvas_wh
    transform = view.transform * view.scene.transform
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    p0 = transform.imap((cx, cy))
    p1 = transform.imap((cx + float(dx_px), cy + float(dy_px)))
    dwx = float(p1[0] - p0[0])
    dwy = float(p1[1] - p0[1])
    center_x, center_y = cam.center[:2]
    cam.center = (float(center_x) - dwx, float(center_y) - dwy)


def animate_camera(
    *,
    camera: Any,
    width: int,
    height: int,
    animate_dps: float,
    anim_start: float,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Advance the camera animation for both 2D and 3D modes."""

    if isinstance(camera, TurntableCamera):
        elapsed = clock() - float(anim_start)
        camera.azimuth = (float(animate_dps) * elapsed) % 360.0
        return

    if isinstance(camera, PanZoomCamera):
        elapsed = clock() - float(anim_start)
        mid_x = float(width) * 0.5
        mid_y = float(height) * 0.5
        pan_x = float(width) * 0.05
        pan_y = float(height) * 0.05
        ox = pan_x * math.sin(0.6 * elapsed)
        oy = pan_y * math.cos(0.4 * elapsed)
        scale = 1.0 + 0.08 * math.sin(0.8 * elapsed)
        half_w = (float(width) * 0.5) / max(1e-6, scale)
        half_h = (float(height) * 0.5) / max(1e-6, scale)
        x_range = (mid_x + ox - half_w, mid_x + ox + half_w)
        y_range = (mid_y + oy - half_h, mid_y + oy + half_h)
        camera.set_range(x=x_range, y=y_range)
