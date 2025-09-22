"""Viewport region-of-interest helpers shared by the EGL worker.

Centralises the math that converts the current vispy view into a multiscale
slice ROI so both the render loop and policy code paths stay in sync while the
worker sheds its bespoke implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple
import math

from vispy import scene

from napari_cuda.server.scene_types import SliceROI
from napari_cuda.server.zarr_source import ZarrSceneSource


@dataclass(frozen=True)
class ViewportROIResult:
    """Return value from :func:`compute_viewport_roi`."""

    roi: SliceROI
    transform_signature: Optional[tuple[float, ...]]

def _axis_index(axes_lower: Sequence[str], axis: str, fallback: int) -> int:
    return axes_lower.index(axis) if axis in axes_lower else fallback


def plane_wh_for_level(source: ZarrSceneSource, level: int) -> Tuple[int, int]:
    """Return the plane height/width for the requested multiscale level."""

    descriptor = source.level_descriptors[level]
    axes = source.axes
    axes_lower = [str(ax).lower() for ax in axes]
    y_pos = _axis_index(axes_lower, "y", max(0, len(descriptor.shape) - 2))
    x_pos = _axis_index(axes_lower, "x", max(0, len(descriptor.shape) - 1))
    h = int(descriptor.shape[y_pos]) if 0 <= y_pos < len(descriptor.shape) else int(descriptor.shape[-2])
    w = int(descriptor.shape[x_pos]) if 0 <= x_pos < len(descriptor.shape) else int(descriptor.shape[-1])
    return h, w


def plane_scale_for_level(source: ZarrSceneSource, level: int) -> Tuple[float, float]:
    """Return the physical Y/X scale for the requested multiscale level."""

    axes = source.axes
    axes_lower = [str(ax).lower() for ax in axes]
    scale = source.level_scale(level)
    y_pos = _axis_index(axes_lower, "y", max(0, len(scale) - 2))
    x_pos = _axis_index(axes_lower, "x", max(0, len(scale) - 1))
    sy = float(scale[y_pos]) if 0 <= y_pos < len(scale) else float(scale[-2])
    sx = float(scale[x_pos]) if 0 <= x_pos < len(scale) else float(scale[-1])
    return sy, sx


def _transform_signature(view: Any) -> Optional[tuple[float, ...]]:
    """Compute a stable signature for caching against the current transform."""

    if view is None or not hasattr(view, "scene"):
        return None
    scene_graph = getattr(view, "scene", None)
    transform = getattr(scene_graph, "transform", None)
    if transform is None or not hasattr(transform, "matrix"):
        return None
    mat = transform.matrix
    values = mat.ravel() if hasattr(mat, "ravel") else mat
    return tuple(float(v) for v in values)


def compute_viewport_roi(
    *,
    view: Any,
    canvas_size: Tuple[int, int],
    source: ZarrSceneSource,
    level: int,
    align_chunks: bool,
    chunk_pad: int,
    ensure_contains_viewport: bool,
    edge_threshold: int,
    prev_roi: Optional[SliceROI],
    for_policy: bool,
    transform_signature: Optional[tuple[float, ...]] = None,
) -> ViewportROIResult:
    """Return the SliceROI covering the current viewport at ``level``.

    Parameters mirror the legacy worker implementation so the render loop can
    call this helper without additional glue code. Hysteresis and viewport
    containment are skipped when ``for_policy`` is True so the policy path can
    observe the raw viewport footprint.
    """

    h, w = plane_wh_for_level(source, level)

    if view is None or not hasattr(view, "camera"):
        raise RuntimeError("view or view.camera missing for ROI compute")

    cam = view.camera
    if not isinstance(cam, scene.cameras.PanZoomCamera):
        raise RuntimeError("PanZoomCamera required for ROI compute")

    scene_graph = getattr(view, "scene", None)
    transform = getattr(scene_graph, "transform", None)
    if transform is None or not hasattr(transform, "imap"):
        raise RuntimeError("view.scene.transform.imap unavailable")

    if transform_signature is None:
        transform_signature = _transform_signature(view)

    width = int(canvas_size[0])
    height = int(canvas_size[1])

    corners = (
        (0.0, 0.0),
        (float(width), 0.0),
        (0.0, float(height)),
        (float(width), float(height)),
    )
    try:
        world_pts = [transform.imap((float(x), float(y), 0.0)) for x, y in corners]
    except Exception as exc:
        raise RuntimeError("transform.imap failed for ROI compute") from exc
    xs = [float(pt[0]) for pt in world_pts]
    ys = [float(pt[1]) for pt in world_pts]

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    sy, sx = plane_scale_for_level(source, level)
    sy = max(1e-12, float(sy))
    sx = max(1e-12, float(sx))

    x_start = int(math.floor(min(x0, x1) / sx))
    x_stop = int(math.ceil(max(x0, x1) / sx))
    y_start = int(math.floor(min(y0, y1) / sy))
    y_stop = int(math.ceil(max(y0, y1) / sy))

    viewport_bounds = (y_start, y_stop, x_start, x_stop)
    roi = SliceROI(y_start, y_stop, x_start, x_stop).clamp(h, w)

    if not for_policy and align_chunks:
        arr = source.get_level(level)
        chunks = getattr(arr, "chunks", None)
        if chunks is not None:
            axes_lower = [str(ax).lower() for ax in source.axes]
            y_pos = _axis_index(axes_lower, "y", max(0, len(chunks) - 2))
            x_pos = _axis_index(axes_lower, "x", max(0, len(chunks) - 1))
            cy = int(chunks[y_pos]) if 0 <= y_pos < len(chunks) else 1
            cx = int(chunks[x_pos]) if 0 <= x_pos < len(chunks) else 1
            cy = max(1, cy)
            cx = max(1, cx)
            ys = (roi.y_start // cy) * cy
            ye = ((roi.y_stop + cy - 1) // cy) * cy
            xs = (roi.x_start // cx) * cx
            xe = ((roi.x_stop + cx - 1) // cx) * cx
            pad = max(0, int(chunk_pad))
            if pad:
                ys = max(0, ys - pad * cy)
                ye = min(h, ye + pad * cy)
                xs = max(0, xs - pad * cx)
                xe = min(w, xe + pad * cx)
            roi = SliceROI(ys, ye, xs, xe).clamp(h, w)

    if not for_policy and prev_roi is not None and edge_threshold > 0:
        thr = int(edge_threshold)
        prev_covers_view = (
            int(prev_roi.y_start) <= int(viewport_bounds[0])
            and int(prev_roi.y_stop) >= int(viewport_bounds[1])
            and int(prev_roi.x_start) <= int(viewport_bounds[2])
            and int(prev_roi.x_stop) >= int(viewport_bounds[3])
        )
        if prev_covers_view:
            if (
                abs(roi.y_start - prev_roi.y_start) < thr
                and abs(roi.y_stop - prev_roi.y_stop) < thr
                and abs(roi.x_start - prev_roi.x_start) < thr
                and abs(roi.x_stop - prev_roi.x_stop) < thr
            ):
                roi = prev_roi

    if not for_policy and ensure_contains_viewport:
        ys = min(int(roi.y_start), int(viewport_bounds[0]))
        ye = max(int(roi.y_stop), int(viewport_bounds[1]))
        xs = min(int(roi.x_start), int(viewport_bounds[2]))
        xe = max(int(roi.x_stop), int(viewport_bounds[3]))
        if align_chunks:
            arr = source.get_level(level)
            chunks = getattr(arr, "chunks", None)
            if chunks is not None:
                axes_lower = [str(ax).lower() for ax in source.axes]
                y_pos = _axis_index(axes_lower, "y", max(0, len(chunks) - 2))
                x_pos = _axis_index(axes_lower, "x", max(0, len(chunks) - 1))
                cy = int(chunks[y_pos]) if 0 <= y_pos < len(chunks) else 1
                cx = int(chunks[x_pos]) if 0 <= x_pos < len(chunks) else 1
                cy = max(1, cy)
                cx = max(1, cx)
                ys = (ys // cy) * cy
                ye = ((ye + cy - 1) // cy) * cy
                xs = (xs // cx) * cx
                xe = ((xe + cx - 1) // cx) * cx
        roi = SliceROI(ys, ye, xs, xe).clamp(h, w)

    if roi.is_empty():
        return ViewportROIResult(SliceROI(0, h, 0, w), transform_signature)

    return ViewportROIResult(roi, transform_signature)


__all__ = [
    "ViewportROIResult",
    "compute_viewport_roi",
    "plane_scale_for_level",
    "plane_wh_for_level",
]
