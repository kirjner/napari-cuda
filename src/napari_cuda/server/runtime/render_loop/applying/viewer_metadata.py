"""Viewer metadata helpers for render snapshot application."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vispy.geometry import Rect
from vispy.scene.cameras import PanZoomCamera

import napari_cuda.server.data.lod as lod

if TYPE_CHECKING:
    from napari_cuda.server.runtime.render_loop.render_interface import RenderInterface


def _apply_dims_and_metadata(
    snapshot_iface: "RenderInterface",
    source: Any,
    context: lod.LevelContext,
) -> None:
    viewer = snapshot_iface.viewer
    if viewer is not None:
        snapshot_iface.set_dims_range_for_level(source, int(context.level))
        viewer.dims.current_step = tuple(int(v) for v in context.step)

    descriptor = source.level_descriptors[int(context.level)]
    snapshot_iface.update_level_metadata(descriptor, context)


def apply_plane_metadata(
    snapshot_iface: "RenderInterface",
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Update viewer metadata for plane rendering."""

    _apply_dims_and_metadata(snapshot_iface, source, context)

    viewport_state = snapshot_iface.viewport_state
    plane_state = viewport_state.plane
    view = snapshot_iface.view
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
    snapshot_iface: "RenderInterface",
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Update viewer metadata for volume rendering."""

    _apply_dims_and_metadata(snapshot_iface, source, context)


def apply_plane_metadata_for_worker(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Bootstrap helper that routes through the apply façade internally."""

    from napari_cuda.server.runtime.render_loop.render_interface import RenderInterface

    snapshot_iface = RenderInterface(worker)
    apply_plane_metadata(snapshot_iface, source, context)


def apply_volume_metadata_for_worker(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Bootstrap helper that routes through the apply façade internally."""

    from napari_cuda.server.runtime.render_loop.render_interface import RenderInterface

    snapshot_iface = RenderInterface(worker)
    apply_volume_metadata(snapshot_iface, source, context)


__all__ = ["apply_plane_metadata", "apply_volume_metadata"]
