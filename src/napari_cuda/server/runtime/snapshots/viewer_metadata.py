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

    viewport_state = worker.viewport_state
    plane_state = viewport_state.plane
    view = worker.view
    cam = view.camera
    if not isinstance(cam, PanZoomCamera):
        return

    pose = plane_state.pose
    assert pose.rect is not None, "plane pose missing rect"
    assert pose.center is not None, "plane pose missing center"
    assert pose.zoom is not None, "plane pose missing zoom"

    rect = pose.rect
    center = pose.center
    zoom = pose.zoom

    cam.rect = Rect(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    cam.center = (float(center[0]), float(center[1]))
    cam.zoom = float(zoom)

    if logging.getLogger(__name__).isEnabledFor(logging.INFO):
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
