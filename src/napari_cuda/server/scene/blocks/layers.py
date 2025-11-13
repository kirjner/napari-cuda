from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableSequence, Sequence, TypedDict

LAYER_BLOCK_SCOPE = "scene_layers"


@dataclass(frozen=True)
class LayerControlsBlock:
    """Layer appearance knobs applied by the runtime worker."""

    visible: bool
    opacity: float
    blending: str
    interpolation: str
    colormap: str
    gamma: float
    contrast_limits: tuple[float, float] | None = None
    depiction: str | None = None
    rendering: str | None = None
    attenuation: float | None = None
    iso_threshold: float | None = None
    projection_mode: str | None = None
    plane_thickness: float | None = None


class LayerControlsPayload(TypedDict, total=False):
    visible: bool
    opacity: float
    blending: str
    interpolation: str
    colormap: str
    gamma: float
    contrast_limits: Sequence[float] | None
    depiction: str | None
    rendering: str | None
    attenuation: float | None
    iso_threshold: float | None
    projection_mode: str | None
    plane_thickness: float | None


def layer_controls_block_to_payload(block: LayerControlsBlock) -> LayerControlsPayload:
    payload: LayerControlsPayload = {
        "visible": bool(block.visible),
        "opacity": float(block.opacity),
        "blending": str(block.blending),
        "interpolation": str(block.interpolation),
        "colormap": str(block.colormap),
        "gamma": float(block.gamma),
    }
    if block.contrast_limits is not None:
        payload["contrast_limits"] = [float(block.contrast_limits[0]), float(block.contrast_limits[1])]
    if block.depiction is not None:
        payload["depiction"] = str(block.depiction)
    if block.rendering is not None:
        payload["rendering"] = str(block.rendering)
    if block.attenuation is not None:
        payload["attenuation"] = float(block.attenuation)
    if block.iso_threshold is not None:
        payload["iso_threshold"] = float(block.iso_threshold)
    if block.projection_mode is not None:
        payload["projection_mode"] = str(block.projection_mode)
    if block.plane_thickness is not None:
        payload["plane_thickness"] = float(block.plane_thickness)
    return payload


def layer_controls_block_from_payload(payload: LayerControlsPayload) -> LayerControlsBlock:
    contrast = payload.get("contrast_limits")
    contrast_tuple = (float(contrast[0]), float(contrast[1])) if contrast is not None else None
    return LayerControlsBlock(
        visible=bool(payload["visible"]),
        opacity=float(payload["opacity"]),
        blending=str(payload["blending"]),
        interpolation=str(payload["interpolation"]),
        colormap=str(payload["colormap"]),
        gamma=float(payload["gamma"]),
        contrast_limits=contrast_tuple,
        depiction=payload.get("depiction"),
        rendering=payload.get("rendering"),
        attenuation=payload.get("attenuation"),
        iso_threshold=payload.get("iso_threshold"),
        projection_mode=payload.get("projection_mode"),
        plane_thickness=payload.get("plane_thickness"),
    )


@dataclass(frozen=True)
class LayerLevelDescriptor:
    """Single level entry for layer multiscale metadata."""

    shape: tuple[int, ...]
    downsample: tuple[float, ...]


class LayerLevelDescriptorPayload(TypedDict):
    shape: Sequence[int]
    downsample: Sequence[float]


def layer_level_descriptor_to_payload(descriptor: LayerLevelDescriptor) -> LayerLevelDescriptorPayload:
    return {
        "shape": [int(dim) for dim in descriptor.shape],
        "downsample": [float(value) for value in descriptor.downsample],
    }


def layer_level_descriptor_from_payload(payload: LayerLevelDescriptorPayload) -> LayerLevelDescriptor:
    return LayerLevelDescriptor(
        shape=tuple(int(dim) for dim in payload["shape"]),
        downsample=tuple(float(value) for value in payload["downsample"]),
    )


@dataclass(frozen=True)
class LayerMultiscaleBlock:
    """Multiscale ladder info for remote layers + HUDs."""

    current_level: int
    levels: tuple[LayerLevelDescriptor, ...]
    policy: str | None = None
    index_space: str | None = None


class LayerMultiscalePayload(TypedDict, total=False):
    current_level: int
    levels: Sequence[LayerLevelDescriptorPayload]
    policy: str | None
    index_space: str | None


def layer_multiscale_block_to_payload(block: LayerMultiscaleBlock) -> LayerMultiscalePayload:
    payload: LayerMultiscalePayload = {
        "current_level": int(block.current_level),
        "levels": [layer_level_descriptor_to_payload(level) for level in block.levels],
    }
    if block.policy is not None:
        payload["policy"] = str(block.policy)
    if block.index_space is not None:
        payload["index_space"] = str(block.index_space)
    return payload


def layer_multiscale_block_from_payload(payload: LayerMultiscalePayload) -> LayerMultiscaleBlock:
    levels = tuple(layer_level_descriptor_from_payload(entry) for entry in payload.get("levels", ()))
    return LayerMultiscaleBlock(
        current_level=int(payload["current_level"]),
        levels=levels,
        policy=payload.get("policy"),
        index_space=payload.get("index_space"),
    )


@dataclass(frozen=True)
class LayerThumbnail:
    """Worker-captured layer thumbnail payload."""

    array: tuple[tuple[tuple[int, ...], ...], ...]
    dtype: str
    shape: tuple[int, ...]
    generated_at: float | None = None


class LayerThumbnailPayload(TypedDict):
    array: Sequence[Sequence[Sequence[int]]]
    dtype: str
    shape: Sequence[int]
    generated_at: float | None


def _tuple3d(value: Sequence[Sequence[Sequence[int]]]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(int(component) for component in row) for row in plane) for plane in value)


def layer_thumbnail_to_payload(thumb: LayerThumbnail) -> LayerThumbnailPayload:
    array_payload: MutableSequence[MutableSequence[MutableSequence[int]]] = []
    for plane in thumb.array:
        plane_list: MutableSequence[MutableSequence[int]] = []
        for row in plane:
            plane_list.append([int(component) for component in row])
        array_payload.append(plane_list)
    return {
        "array": array_payload,
        "dtype": str(thumb.dtype),
        "shape": [int(dim) for dim in thumb.shape],
        "generated_at": float(thumb.generated_at) if thumb.generated_at is not None else None,
    }


def layer_thumbnail_from_payload(payload: LayerThumbnailPayload) -> LayerThumbnail:
    return LayerThumbnail(
        array=_tuple3d(payload["array"]),
        dtype=str(payload["dtype"]),
        shape=tuple(int(dim) for dim in payload["shape"]),
        generated_at=float(payload["generated_at"]) if payload.get("generated_at") is not None else None,
    )


@dataclass(frozen=True)
class LayerBlock:
    """Lean per-layer bundle carried inside SceneBlockSnapshot."""

    layer_id: str
    layer_type: str
    controls: LayerControlsBlock
    metadata: Mapping[str, Any] = field(default_factory=dict)
    thumbnail: LayerThumbnail | None = None
    multiscale: LayerMultiscaleBlock | None = None
    versions: Mapping[str, int] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class LayerBlockPayload(TypedDict, total=False):
    layer_id: str
    layer_type: str
    controls: LayerControlsPayload
    metadata: Mapping[str, Any]
    thumbnail: LayerThumbnailPayload | None
    multiscale: LayerMultiscalePayload | None
    versions: Mapping[str, int]
    extras: Mapping[str, Any]


def layer_block_to_payload(block: LayerBlock) -> LayerBlockPayload:
    payload: LayerBlockPayload = {
        "layer_id": str(block.layer_id),
        "layer_type": str(block.layer_type),
        "controls": layer_controls_block_to_payload(block.controls),
    }
    if block.metadata:
        payload["metadata"] = dict(block.metadata)
    if block.thumbnail is not None:
        payload["thumbnail"] = layer_thumbnail_to_payload(block.thumbnail)
    if block.multiscale is not None:
        payload["multiscale"] = layer_multiscale_block_to_payload(block.multiscale)
    if block.versions:
        payload["versions"] = {str(key): int(value) for key, value in block.versions.items()}
    if block.extras:
        payload["extras"] = dict(block.extras)
    return payload


def layer_block_from_payload(payload: LayerBlockPayload) -> LayerBlock:
    multiscale_value = payload.get("multiscale")
    thumbnail_value = payload.get("thumbnail")
    return LayerBlock(
        layer_id=str(payload["layer_id"]),
        layer_type=str(payload["layer_type"]),
        controls=layer_controls_block_from_payload(payload["controls"]),
        metadata=dict(payload.get("metadata") or {}),
        thumbnail=layer_thumbnail_from_payload(thumbnail_value) if isinstance(thumbnail_value, Mapping) else None,
        multiscale=layer_multiscale_block_from_payload(multiscale_value) if isinstance(multiscale_value, Mapping) else None,
        versions={str(key): int(value) for key, value in (payload.get("versions") or {}).items()} or None,
        extras=dict(payload.get("extras") or {}),
    )


__all__ = [
    "LAYER_BLOCK_SCOPE",
    "LayerBlock",
    "LayerBlockPayload",
    "LayerControlsBlock",
    "LayerControlsPayload",
    "LayerLevelDescriptor",
    "LayerLevelDescriptorPayload",
    "LayerMultiscaleBlock",
    "LayerMultiscalePayload",
    "LayerThumbnail",
    "LayerThumbnailPayload",
    "layer_block_from_payload",
    "layer_block_to_payload",
    "layer_controls_block_from_payload",
    "layer_controls_block_to_payload",
    "layer_level_descriptor_from_payload",
    "layer_level_descriptor_to_payload",
    "layer_multiscale_block_from_payload",
    "layer_multiscale_block_to_payload",
    "layer_thumbnail_from_payload",
    "layer_thumbnail_to_payload",
]
