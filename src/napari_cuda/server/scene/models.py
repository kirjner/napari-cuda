"""Shared viewer state data transfer objects for control/runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Literal, Optional


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
    layer_values: Optional[dict[str, "LayerVisualState"]] = None
    camera_versions: Optional[dict[str, int]] = None
    op_seq: int = 0


@dataclass(frozen=True)
class LayerVisualState:
    """Per-layer appearance snapshot staged for the render worker."""

    layer_id: str
    visible: Optional[bool] = None
    opacity: Optional[float] = None
    blending: Optional[str] = None
    interpolation: Optional[str] = None
    colormap: Optional[str] = None
    gamma: Optional[float] = None
    contrast_limits: Optional[tuple[float, float]] = None
    depiction: Optional[str] = None
    rendering: Optional[str] = None
    attenuation: Optional[float] = None
    iso_threshold: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    thumbnail: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    versions: Mapping[str, int] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:  # legacy helper used by existing call sites
        mapping: dict[str, Any] = {}
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                mapping[key] = value
        return mapping

    def version_dict(self) -> dict[str, int]:  # legacy helper used by existing call sites
        return {str(key): int(value) for key, value in self.versions.items()}

    def keys(self) -> tuple[str, ...]:
        base_keys = (
            "visible",
            "opacity",
            "blending",
            "interpolation",
            "colormap",
            "gamma",
            "contrast_limits",
            "depiction",
            "rendering",
            "attenuation",
            "iso_threshold",
            "metadata",
            "thumbnail",
        )
        keys = [
            key
            for key in base_keys
            if key in self.versions or self.get(key) is not None
        ]
        for key, value in self.extra.items():
            str_key = str(key)
            if str_key in self.versions or value is not None:
                keys.append(str_key)
        return tuple(keys)

    def get(self, key: str) -> Any:
        if key == "visible":
            return None if self.visible is None else bool(self.visible)
        if key == "opacity":
            return None if self.opacity is None else float(self.opacity)
        if key == "blending":
            return None if self.blending is None else str(self.blending)
        if key == "interpolation":
            return None if self.interpolation is None else str(self.interpolation)
        if key == "colormap":
            return None if self.colormap is None else str(self.colormap)
        if key == "gamma":
            return None if self.gamma is None else float(self.gamma)
        if key == "contrast_limits":
            return None if self.contrast_limits is None else tuple(float(v) for v in self.contrast_limits)
        if key == "depiction":
            return None if self.depiction is None else str(self.depiction)
        if key == "rendering":
            return None if self.rendering is None else str(self.rendering)
        if key == "attenuation":
            return None if self.attenuation is None else float(self.attenuation)
        if key == "iso_threshold":
            return None if self.iso_threshold is None else float(self.iso_threshold)
        if key == "metadata":
            return dict(self.metadata) if self.metadata else None
        if key == "thumbnail":
            return dict(self.thumbnail) if self.thumbnail is not None else None
        return self.extra.get(key)


__all__ = [
    "BootstrapSceneMetadata",
    "CameraDeltaCommand",
    "RenderLedgerSnapshot",
]
