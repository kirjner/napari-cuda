"""Viewer metadata helpers for render snapshot application."""

from __future__ import annotations

from typing import Any

import logging

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod


def _apply_dims_and_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    viewer = getattr(worker, "_viewer", None)
    if viewer is not None:
        set_range = getattr(worker, "_set_dims_range_for_level", None)
        if callable(set_range):
            set_range(source, int(context.level))
        viewer.dims.current_step = tuple(int(v) for v in context.step)

    descriptor = source.level_descriptors[int(context.level)]
    worker._update_level_metadata(descriptor, context)


def apply_plane_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Update viewer metadata for plane rendering."""

    _apply_dims_and_metadata(worker, source, context)

    plane_state = getattr(getattr(worker, "viewport_state", None), "plane", None)
    view = getattr(worker, "view", None)
    cam = getattr(view, "camera", None)
    if plane_state is None or not isinstance(cam, PanZoomCamera):
        return

    rect = getattr(plane_state, "camera_rect", None)
    center = getattr(plane_state, "camera_center", None)
    zoom = getattr(plane_state, "camera_zoom", None)

    restored = False
    if rect is not None and len(rect) >= 4:
        cam.rect = Rect(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
        restored = True
    if center is not None and len(center) >= 2:
        cam.center = (float(center[0]), float(center[1]))
        restored = True
    if zoom is not None:
        cam.zoom = float(zoom)
        restored = True

    if restored and logging.getLogger(__name__).isEnabledFor(logging.INFO):
        logging.getLogger(__name__).info(
            "plane.metadata restored camera rect=%s center=%s zoom=%s",
            rect,
            center,
            zoom,
        )


def apply_volume_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Update viewer metadata for volume rendering."""

    _apply_dims_and_metadata(worker, source, context)


__all__ = ["apply_plane_metadata", "apply_volume_metadata"]
