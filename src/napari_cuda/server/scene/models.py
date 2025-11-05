"""Shared viewer state data transfer objects for control/runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Literal, Optional

from napari_cuda.shared.dims_spec import DimsSpec as AxesSpec


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
    axes_spec: Optional[AxesSpec] = None
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

    def version_dict(self) -> dict[str, int]:
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

    def subset(self, props: Iterable[str]) -> "LayerVisualState":
        prop_set = {str(prop) for prop in props}
        versions_subset = {
            str(key): int(value)
            for key, value in self.versions.items()
            if str(key) in prop_set
        }

        kwargs: dict[str, Any] = {"layer_id": self.layer_id}

        for field_name in (
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
        ):
            if field_name in prop_set:
                value = getattr(self, field_name)
                if value is not None:
                    kwargs[field_name] = value

        if "metadata" in prop_set:
            kwargs["metadata"] = dict(self.metadata) if self.metadata else {}
        if "thumbnail" in prop_set:
            kwargs["thumbnail"] = (
                dict(self.thumbnail) if self.thumbnail is not None else None
            )

        extra_subset = {
            str(key): value
            for key, value in self.extra.items()
            if str(key) in prop_set
        }
        if extra_subset:
            kwargs["extra"] = extra_subset
        if versions_subset:
            kwargs["versions"] = versions_subset

        return LayerVisualState(**kwargs)

    def with_updates(
        self,
        *,
        updates: Mapping[str, Any],
        versions: Mapping[str, int | None] | None = None,
    ) -> "LayerVisualState":
        state = self
        if updates:
            base_kwargs: dict[str, Any] = {}
            extra_updates: dict[str, Any] | None = None
            for key, value in updates.items():
                skey = str(key)
                if skey == "visible":
                    base_kwargs["visible"] = None if value is None else bool(value)
                elif skey == "opacity":
                    base_kwargs["opacity"] = None if value is None else float(value)
                elif skey == "blending":
                    base_kwargs["blending"] = None if value is None else str(value)
                elif skey == "interpolation":
                    base_kwargs["interpolation"] = None if value is None else str(value)
                elif skey == "colormap":
                    base_kwargs["colormap"] = None if value is None else str(value)
                elif skey == "gamma":
                    base_kwargs["gamma"] = None if value is None else float(value)
                elif skey == "contrast_limits":
                    base_kwargs["contrast_limits"] = (
                        None
                        if value is None
                        else tuple(float(component) for component in value)  # type: ignore[arg-type]
                    )
                elif skey == "depiction":
                    base_kwargs["depiction"] = None if value is None else str(value)
                elif skey == "rendering":
                    base_kwargs["rendering"] = None if value is None else str(value)
                elif skey == "attenuation":
                    base_kwargs["attenuation"] = (
                        None if value is None else float(value)
                    )
                elif skey == "iso_threshold":
                    base_kwargs["iso_threshold"] = (
                        None if value is None else float(value)
                    )
                elif skey == "metadata":
                    base_kwargs["metadata"] = (
                        dict(value) if isinstance(value, Mapping) else {}
                    )
                elif skey == "thumbnail":
                    base_kwargs["thumbnail"] = (
                        dict(value) if isinstance(value, Mapping) else None
                    )
                else:
                    if extra_updates is None:
                        extra_updates = dict(self.extra)
                    if value is None:
                        extra_updates.pop(skey, None)
                    else:
                        extra_updates[skey] = value
            if extra_updates is not None:
                base_kwargs["extra"] = extra_updates
            state = replace(state, **base_kwargs)

        if versions:
            current_versions = dict(state.versions)
            for key, value in versions.items():
                skey = str(key)
                if value is None:
                    current_versions.pop(skey, None)
                else:
                    current_versions[skey] = int(value)
            state = replace(state, versions=current_versions)

        return state


__all__ = [
    "BootstrapSceneMetadata",
    "CameraDeltaCommand",
    "RenderLedgerSnapshot",
]
