"""Shared viewer state data transfer objects for control/runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Literal


@dataclass(frozen=True)
class BootstrapSceneMetadata:
    """Snapshot of worker bootstrap state for seeding reducer intents."""

    step: tuple[int, ...]
    axis_labels: tuple[str, ...]
    order: tuple[int, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    levels: tuple[dict[str, Any], ...]
    current_level: int
    ndisplay: int
    plane_rect: Optional[tuple[float, float, float, float]] = None
    plane_center: Optional[tuple[float, float]] = None
    plane_zoom: Optional[float] = None


@dataclass(frozen=True)
class CameraDeltaCommand:
    """Queued camera delta consumed by the render thread."""

    kind: Literal["zoom", "pan", "orbit", "reset"]
    target: str = "main"
    command_seq: int = 0
    factor: Optional[float] = None
    anchor_px: Optional[tuple[float, float]] = None
    dx_px: float = 0.0
    dy_px: float = 0.0
    d_az_deg: float = 0.0
    d_el_deg: float = 0.0


@dataclass(frozen=True)
class RenderLedgerSnapshot:
    """Authoritative scene state consumed by the render thread."""

    plane_center: Optional[tuple[float, float]] = None
    plane_zoom: Optional[float] = None
    plane_rect: Optional[tuple[float, float, float, float]] = None
    volume_center: Optional[tuple[float, float, float]] = None
    volume_angles: Optional[tuple[float, float, float]] = None
    volume_distance: Optional[float] = None
    volume_fov: Optional[float] = None
    current_step: Optional[tuple[int, ...]] = None
    dims_version: Optional[int] = None
    ndisplay: Optional[int] = None
    view_version: Optional[int] = None
    displayed: Optional[tuple[int, ...]] = None
    order: Optional[tuple[int, ...]] = None
    axis_labels: Optional[tuple[str, ...]] = None
    dims_labels: Optional[tuple[str, ...]] = None
    level_shapes: Optional[tuple[tuple[int, ...], ...]] = None
    current_level: Optional[int] = None
    multiscale_level_version: Optional[int] = None
    dims_mode: Optional[str] = None
    volume_mode: Optional[str] = None
    volume_colormap: Optional[str] = None
    volume_clim: Optional[tuple[float, float]] = None
    volume_opacity: Optional[float] = None
    volume_sample_step: Optional[float] = None
    layer_values: Optional[dict[str, dict[str, Any]]] = None
    layer_versions: Optional[dict[str, dict[str, int]]] = None
    camera_versions: Optional[dict[str, int]] = None
    op_seq: int = 0


__all__ = [
    "BootstrapSceneMetadata",
    "CameraDeltaCommand",
    "RenderLedgerSnapshot",
]
