"""Bootstrap-specific helpers that forward to render-loop snapshot logic."""

from __future__ import annotations

from typing import Any

import napari_cuda.server.data.lod as lod

from napari_cuda.server.runtime.render_loop.apply_interface import (
    RenderApplyInterface,
)
from napari_cuda.server.runtime.render_loop.apply.render_state.viewer_metadata import (
    apply_plane_metadata as _apply_plane_metadata,
    apply_volume_metadata as _apply_volume_metadata,
)
from napari_cuda.server.runtime.render_loop.apply.render_state.volume import (
    VolumeApplyResult,
    apply_volume_level as _apply_volume_level,
)


def apply_plane_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Apply plane metadata during bootstrap without exporting the apply façade."""

    snapshot_iface = RenderApplyInterface(worker)
    _apply_plane_metadata(snapshot_iface, source, context)


def apply_volume_metadata(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
) -> None:
    """Apply volume metadata during bootstrap without exporting the apply façade."""

    snapshot_iface = RenderApplyInterface(worker)
    _apply_volume_metadata(snapshot_iface, source, context)


def apply_volume_level(
    worker: Any,
    source: Any,
    context: lod.LevelContext,
    *,
    downgraded: bool,
) -> VolumeApplyResult:
    """Load + apply volume level mutations via the render-loop façade."""

    snapshot_iface = RenderApplyInterface(worker)
    return _apply_volume_level(
        snapshot_iface,
        source,
        context,
        downgraded=downgraded,
    )


__all__ = [
    "VolumeApplyResult",
    "apply_plane_metadata",
    "apply_volume_metadata",
    "apply_volume_level",
]
