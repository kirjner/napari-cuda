"""Viewer metadata helpers for render snapshot application."""

from __future__ import annotations

from typing import Any

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


def apply_volume_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Update viewer metadata for volume rendering."""

    _apply_dims_and_metadata(worker, source, context)


__all__ = ["apply_plane_metadata", "apply_volume_metadata"]
