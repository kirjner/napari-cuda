"""Orchestrators for applying render snapshots to the viewer/visuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.server.scene.viewport import RenderMode
from napari_cuda.server.scene import RenderLedgerSnapshot

from .dims import apply_dims_step
from .layers import apply_layer_visual_updates
from .volume import apply_volume_visual_params


@dataclass(frozen=True)
class DrainOutcome:
    """Result of applying viewer/camera updates."""

    z_index: Optional[int] = None
    data_wh: Optional[tuple[int, int]] = None
    render_marked: bool = False


def drain_render_state(worker: Any, snapshot: RenderLedgerSnapshot) -> DrainOutcome:
    """Apply viewer, visual, and camera updates from the snapshot."""

    assert worker.view is not None, "drain_render_state requires an active VisPy view"  # type: ignore[attr-defined]

    z_index_update: Optional[int] = None
    render_marked = False

    if worker.viewport_state.mode is not RenderMode.VOLUME and snapshot.current_step is not None:  # type: ignore[attr-defined]
        z_idx, marked = apply_dims_step(worker, snapshot.current_step)
        if z_idx is not None:
            z_index_update = z_idx
        render_marked = render_marked or marked

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        apply_volume_visual_params(worker, snapshot)

    if snapshot.layer_values and apply_layer_visual_updates(worker, snapshot.layer_values):
        render_marked = True

    return DrainOutcome(z_index=z_index_update, render_marked=render_marked)


__all__ = ["DrainOutcome", "drain_render_state"]
