"""Camera operations (free functions) extracted from the worker.

These functions mirror the behavior of the internal `_CameraOps` helper inside
``egl_worker``. ``egl_worker`` delegates to these helpers to keep behavior
unchanged while making the math testable and reusable.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


def anchor_to_world(ax_px: float, ay_px: float, canvas_wh: Tuple[int, int], view) -> Tuple[float, float]:
    """Map an anchor pixel to world coords using the view transform."""
    cw, ch = canvas_wh
    if not hasattr(view, 'transform') or not hasattr(view, 'scene') or not hasattr(view.scene, 'transform'):
        return float(ax_px), float(ay_px)
    ay_tl = float(ch) - float(ay_px)
    tr = view.transform * view.scene.transform
    try:
        mapped = tr.imap([float(ax_px), ay_tl, 0, 1])
    except Exception:
        logger.debug("anchor_to_world: transform mapping failed", exc_info=True)
        return float(ax_px), float(ay_px)
    return float(mapped[0]), float(mapped[1])


def per_pixel_world_scale_3d(cam, canvas_wh: Tuple[int, int]) -> Tuple[float, float]:
    """Estimate world-units per pixel for a 3D camera from FOV and distance."""
    cw, ch = canvas_wh
    fov_deg = getattr(cam, 'fov', 60.0)
    dist = getattr(cam, 'distance', 1.0)
    try:
        fov_rad = math.radians(max(1e-3, min(179.0, float(fov_deg))))
        denom = max(1e-6, float(ch))
        sy = 2.0 * float(dist) * math.tan(0.5 * fov_rad) / denom
        sx = sy * (float(cw) / denom)
    except (TypeError, ValueError) as e:
        logger.debug("per_pixel_world_scale_3d: bad params fov=%r dist=%r: %s", fov_deg, dist, e)
        return 0.01, 0.01
    return sx, sy


def apply_orbit(cam, d_az_deg: float, d_el_deg: float) -> None:
    """Apply azimuth/elevation deltas to a turntable camera."""
    if (d_az_deg == 0.0 and d_el_deg == 0.0) or not hasattr(cam, 'azimuth'):
        return
    cur_az = float(getattr(cam, 'azimuth', 0.0) or 0.0)
    cur_el = float(getattr(cam, 'elevation', 0.0) or 0.0)
    cam.azimuth = cur_az + float(d_az_deg)  # type: ignore[attr-defined]
    cam.elevation = cur_el + float(d_el_deg)  # type: ignore[attr-defined]


def apply_zoom_3d(cam, factor: float) -> None:
    """Apply multiplicative zoom in 3D by scaling distance."""
    if factor <= 0.0 or not hasattr(cam, 'distance'):
        return
    cur_d = float(getattr(cam, 'distance', 1.0) or 1.0)
    cam.distance = max(1e-6, cur_d * float(factor))  # type: ignore[attr-defined]


def apply_zoom_2d(cam, factor: float, anchor_px: Tuple[float, float], canvas_wh: Tuple[int, int], view) -> None:
    """Zoom around an anchor in 2D using view transforms for correct centering."""
    if factor <= 0.0:
        return
    wx, wy = anchor_to_world(anchor_px[0], anchor_px[1], canvas_wh, view)
    try:
        cam.zoom(float(factor), center=(wx, wy))  # type: ignore[call-arg]
    except Exception:
        logger.debug("apply_zoom_2d: cam.zoom failed", exc_info=True)


def apply_pan_3d(cam, dx_px: float, dy_px: float, canvas_wh: Tuple[int, int]) -> None:
    """Pan/dolly in 3D using pixel deltas mapped to world units."""
    if (dx_px == 0.0 and dy_px == 0.0):
        return
    sx, sy = per_pixel_world_scale_3d(cam, canvas_wh)
    dwx = dx_px * sx
    if dy_px != 0.0 and hasattr(cam, 'distance'):
        dist = float(getattr(cam, 'distance', 1.0) or 1.0)
        cam.distance = float(max(1e-6, dist - (dy_px * sy)))  # type: ignore[attr-defined]
    c = getattr(cam, 'center', None)
    if isinstance(c, (tuple, list)) and len(c) >= 2:
        cam.center = (float(c[0]) - dwx, float(c[1]))  # type: ignore[attr-defined]


def apply_pan_2d(cam, dx_px: float, dy_px: float, canvas_wh: Tuple[int, int], view) -> None:
    """Pan in 2D using pixel deltas mapped through view transforms when available."""
    if (dx_px == 0.0 and dy_px == 0.0):
        return
    cw, ch = canvas_wh
    if hasattr(view, 'transform') and hasattr(view, 'scene') and hasattr(view.scene, 'transform'):
        try:
            cx_px = float(cw) * 0.5
            cy_px = float(ch) * 0.5
            tr = view.transform * view.scene.transform
            p0 = tr.imap((cx_px, cy_px))
            p1 = tr.imap((cx_px + dx_px, cy_px + dy_px))
            dwx = float(p1[0] - p0[0])
            dwy = float(p1[1] - p0[1])
        except Exception:
            logger.debug("apply_pan_2d: transform mapping failed", exc_info=True)
            z = float(getattr(cam, 'zoom', 1.0) or 1.0)
            inv = 1.0 / max(1e-6, z)
            dwx = dx_px * inv
            dwy = dy_px * inv
    else:
        z = float(getattr(cam, 'zoom', 1.0) or 1.0)
        inv = 1.0 / max(1e-6, z)
        dwx = dx_px * inv
        dwy = dy_px * inv
    c = getattr(cam, 'center', None)
    if isinstance(c, (tuple, list)) and len(c) >= 2:
        cam.center = (float(c[0]) - dwx, float(c[1]) - dwy)  # type: ignore[attr-defined]


__all__ = [
    "anchor_to_world",
    "per_pixel_world_scale_3d",
    "apply_orbit",
    "apply_zoom_3d",
    "apply_zoom_2d",
    "apply_pan_3d",
    "apply_pan_2d",
]
