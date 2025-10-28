"""Viewport region-of-interest helpers shared by the EGL worker.

Centralises the math that converts the current vispy view into a multiscale
slice ROI so both the render loop and policy code paths stay in sync while the
worker sheds its bespoke implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple
import logging
import math

from vispy import scene

from napari_cuda.server.runtime.scene_types import SliceROI
from napari_cuda.server.runtime.roi_math import (
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
)
from napari_cuda.server.data.zarr_source import ZarrSceneSource


logger = logging.getLogger(__name__)


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

    chunk_shape: Optional[Tuple[int, int]] = None
    if align_chunks:
        chunk_shape = chunk_shape_for_level(source, level)

    if not for_policy and chunk_shape is not None:
        roi = align_roi_to_chunk_grid(
            roi,
            chunk_shape,
            chunk_pad,
            height=h,
            width=w,
        )

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
        if chunk_shape is not None:
            roi = align_roi_to_chunk_grid(
                SliceROI(ys, ye, xs, xe),
                chunk_shape,
                0,
                height=h,
                width=w,
            )
        else:
            roi = SliceROI(ys, ye, xs, xe).clamp(h, w)

    if roi.is_empty():
        return ViewportROIResult(SliceROI(0, h, 0, w), transform_signature)

    return ViewportROIResult(roi, transform_signature)


def viewport_debug_snapshot(
    *,
    view: Any,
    canvas_size: Tuple[int, int],
    data_wh: Optional[Tuple[int, int]],
    data_depth: Optional[int],
) -> dict[str, Any]:
    width, height = canvas_size
    info: dict[str, Any] = {
        "canvas_size": (int(width), int(height)),
        "data_wh": tuple(int(v) for v in data_wh) if data_wh else None,
        "data_depth": int(data_depth) if data_depth is not None else None,
    }

    if view is None:
        info["view"] = None
        return info

    info["view_class"] = view.__class__.__name__
    try:
        cam = getattr(view, "camera", None)
        if cam is not None:
            cam_info: dict[str, Any] = {"type": cam.__class__.__name__}
            rect = getattr(cam, "rect", None)
            if rect is not None:
                try:
                    cam_info["rect"] = tuple(float(v) for v in rect)
                except Exception:
                    cam_info["rect"] = str(rect)
            center = getattr(cam, "center", None)
            if center is not None:
                try:
                    cam_info["center"] = tuple(float(v) for v in center)
                except Exception:
                    cam_info["center"] = str(center)
            if hasattr(cam, "zoom"):
                try:
                    cam_info["zoom"] = float(cam.zoom)  # type: ignore[arg-type]
                except Exception:
                    cam_info["zoom"] = str(cam.zoom)
            if hasattr(cam, "scale"):
                try:
                    cam_info["scale"] = tuple(float(v) for v in cam.scale)
                except Exception:
                    cam_info["scale"] = str(cam.scale)
            if hasattr(cam, "_viewbox"):
                try:
                    vb = cam._viewbox
                    cam_info["viewbox_size"] = tuple(float(v) for v in getattr(vb, "size", ()) or ())
                except Exception:
                    cam_info["viewbox_size"] = "unavailable"
            info["camera"] = cam_info
    except Exception:
        info["camera"] = "error"

    try:
        scene_graph = getattr(view, "scene", None)
        transform = getattr(scene_graph, "transform", None)
        if transform is not None and hasattr(transform, "matrix"):
            mat = getattr(transform, "matrix")
            try:
                info["transform_matrix"] = tuple(float(v) for v in mat.ravel())
            except Exception:
                info["transform_matrix"] = str(mat)
    except Exception:
        info["transform"] = "error"

    return info


def resolve_viewport_roi(
    *,
    view: Any,
    canvas_size: Tuple[int, int],
    source: ZarrSceneSource,
    level: int,
    align_chunks: bool,
    chunk_pad: int,
    ensure_contains_viewport: bool,
    edge_threshold: int,
    for_policy: bool,
    prev_roi: Optional[SliceROI],
    snapshot_cb: Callable[[], dict[str, Any]],
    log_layer_debug: bool,
    quiet: bool,
    logger_ref: logging.Logger = logger,
) -> SliceROI:
    h, w = plane_wh_for_level(source, level)

    if view is None or not hasattr(view, "camera"):
        if not quiet and log_layer_debug and logger_ref.isEnabledFor(logging.INFO):
            logger_ref.info(
                "viewport ROI boundary: inactive view; returning full frame (level=%d dims=%dx%d snapshot=%s)",
                level,
                h,
                w,
                snapshot_cb(),
            )
        return SliceROI(0, h, 0, w)

    cam = getattr(view, "camera", None)
    if not isinstance(cam, scene.cameras.PanZoomCamera):
        raise RuntimeError("PanZoomCamera required for ROI compute")

    signature = _transform_signature(view)

    result = compute_viewport_roi(
        view=view,
        canvas_size=canvas_size,
        source=source,
        level=level,
        align_chunks=align_chunks,
        chunk_pad=chunk_pad,
        ensure_contains_viewport=ensure_contains_viewport,
        edge_threshold=edge_threshold,
        prev_roi=prev_roi,
        for_policy=for_policy,
        transform_signature=signature,
    )

    roi = result.roi
    if roi.is_empty():
        raise RuntimeError("resolve_viewport_roi produced an empty ROI")

    return roi


def ensure_panzoom_camera(
    *,
    view: Any,
    width: int,
    height: int,
    data_wh: Optional[tuple[int, int]],
    log_layer_debug: bool,
    reason: str,
    logger_ref: logging.Logger = logger,
) -> Optional[scene.cameras.PanZoomCamera]:
    """Ensure a :class:`~vispy.scene.cameras.PanZoomCamera` is active for ROI work."""

    if view is None:
        return None

    cam = getattr(view, "camera", None)
    if isinstance(cam, scene.cameras.PanZoomCamera):
        return cam

    if cam is not None and log_layer_debug and logger_ref.isEnabledFor(logging.INFO):
        logger_ref.info(
            "ensure panzoom camera: reason=%s current=%s",
            reason,
            cam.__class__.__name__,
        )

    try:
        panzoom = scene.cameras.PanZoomCamera(aspect=1.0)
        view.camera = panzoom
        if data_wh:
            w, h = data_wh
            panzoom.set_range(x=(0, float(w)), y=(0, float(h)))
        else:
            panzoom.set_range(x=(0, float(width)), y=(0, float(height)))
        return panzoom
    except Exception:
        logger_ref.debug("ensure_panzoom_camera failed", exc_info=True)
        return None


def resolve_worker_viewport_roi(
    *,
    view: Any,
    canvas_size: Tuple[int, int],
    source: ZarrSceneSource,
    level: int,
    align_chunks: bool,
    chunk_pad: int,
    ensure_contains_viewport: bool,
    edge_threshold: int,
    for_policy: bool,
    prev_roi: Optional[SliceROI],
    snapshot_cb: Callable[[], dict[str, Any]],
    log_layer_debug: bool,
    quiet: bool,
    data_wh: Optional[tuple[int, int]],
    reason: str,
    logger_ref: logging.Logger = logger,
) -> SliceROI:
    """Resolve the viewport ROI using worker settings, ensuring camera readiness."""

    if view is not None:
        cam = getattr(view, "camera", None)
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            ensured = ensure_panzoom_camera(
                view=view,
                width=int(canvas_size[0]),
                height=int(canvas_size[1]),
                data_wh=data_wh,
                log_layer_debug=log_layer_debug,
                reason=reason,
                logger_ref=logger_ref,
            )
            cam = ensured or getattr(view, "camera", None)
        if not isinstance(cam, scene.cameras.PanZoomCamera):
            raise RuntimeError("PanZoomCamera required for ROI compute")

    return resolve_viewport_roi(
        view=view,
        canvas_size=canvas_size,
        source=source,
        level=level,
        align_chunks=align_chunks,
        chunk_pad=chunk_pad,
        ensure_contains_viewport=ensure_contains_viewport,
        edge_threshold=edge_threshold,
        for_policy=for_policy,
        prev_roi=prev_roi,
        snapshot_cb=snapshot_cb,
        log_layer_debug=log_layer_debug,
        quiet=quiet,
        logger_ref=logger_ref,
    )


__all__ = [
    "ViewportROIResult",
    "compute_viewport_roi",
    "plane_scale_for_level",
    "plane_wh_for_level",
    "viewport_debug_snapshot",
    "resolve_viewport_roi",
    "ensure_panzoom_camera",
    "resolve_worker_viewport_roi",
]
logger = logging.getLogger(__name__)
