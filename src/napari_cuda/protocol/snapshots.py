"""Typed snapshots feeding control protocol notify payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from napari_cuda.protocol.messages import (
    NotifyLayersPayload,
    NotifyScenePayload,
)


@dataclass(slots=True)
class ViewerSnapshot:
    """Minimal viewer block used by ``notify.scene``."""

    settings: Dict[str, Any]
    dims: Dict[str, Any]
    camera: Dict[str, Any]

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "dims": dict(self.dims),
            "camera": dict(self.camera),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ViewerSnapshot":
        settings = mapping.get("settings")
        dims = mapping.get("dims")
        camera = mapping.get("camera")
        return cls(
            settings=dict(settings) if isinstance(settings, Mapping) else {},
            dims=dict(dims) if isinstance(dims, Mapping) else {},
            camera=dict(camera) if isinstance(camera, Mapping) else {},
        )


@dataclass(slots=True)
class LayerSnapshot:
    """Layer entry mirrored in ``notify.scene.layers``."""

    layer_id: str
    block: Dict[str, Any]

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.block)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LayerSnapshot":
        layer_id = mapping.get("layer_id") or mapping.get("id")
        if not layer_id:
            layer_id = "layer-0"
        return cls(layer_id=str(layer_id), block=dict(mapping))


@dataclass(slots=True)
class SceneSnapshot:
    """Authoritative scene snapshot emitted on baseline."""

    viewer: ViewerSnapshot
    layers: Tuple[LayerSnapshot, ...]
    policies: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_notify_scene_payload(self) -> NotifyScenePayload:
        return NotifyScenePayload(
            viewer=self.viewer.to_mapping(),
            layers=tuple(layer.to_mapping() for layer in self.layers),
            metadata=dict(self.metadata) if self.metadata else None,
            policies=dict(self.policies) if self.policies else None,
        )

    @classmethod
    def from_payload(cls, payload: NotifyScenePayload) -> "SceneSnapshot":
        viewer = ViewerSnapshot.from_mapping(payload.viewer)
        layers = tuple(LayerSnapshot.from_mapping(block) for block in payload.layers)
        policies = dict(payload.policies) if payload.policies else {}
        metadata = dict(payload.metadata) if payload.metadata else {}
        return cls(viewer=viewer, layers=layers, policies=policies, metadata=metadata)


@dataclass(slots=True)
class LayerDelta:
    """Incremental layer changes for ``notify.layers`` (structured)."""

    layer_id: str
    controls: Dict[str, Any] | None = None
    metadata: Dict[str, Any] | None = None
    data: Dict[str, Any] | None = None
    thumbnail: Dict[str, Any] | None = None
    removed: bool | None = None

    def to_payload(self) -> NotifyLayersPayload:
        return NotifyLayersPayload(
            layer_id=self.layer_id,
            controls=dict(self.controls) if self.controls else None,
            metadata=dict(self.metadata) if self.metadata else None,
            data=dict(self.data) if self.data else None,
            thumbnail=dict(self.thumbnail) if self.thumbnail else None,
            removed=bool(self.removed) if self.removed else None,
        )

    @classmethod
    def controls_only(cls, layer_id: str, controls: Mapping[str, Any]) -> "LayerDelta":
        return cls(layer_id=layer_id, controls=dict(controls))

    @classmethod
    def removal(cls, layer_id: str) -> "LayerDelta":
        return cls(layer_id=layer_id, removed=True)

    @classmethod
    def from_state_update(cls, layer_id: str, key: str, value: Any) -> "LayerDelta":
        # Fallback: treat unknown singletons as data section
        return cls(layer_id=layer_id, data={str(key): value})

    @classmethod
    def from_payload(cls, payload: NotifyLayersPayload) -> "LayerDelta":
        return cls(
            layer_id=str(payload.layer_id),
            controls=dict(payload.controls) if payload.controls else None,
            metadata=dict(payload.metadata) if payload.metadata else None,
            data=dict(payload.data) if payload.data else None,
            thumbnail=dict(payload.thumbnail) if payload.thumbnail else None,
            removed=bool(payload.removed) if payload.removed else None,
        )


def viewer_snapshot_from_blocks(*, settings: Mapping[str, Any], dims: Mapping[str, Any], camera: Mapping[str, Any]) -> ViewerSnapshot:
    return ViewerSnapshot(settings=dict(settings), dims=dict(dims), camera=dict(camera))


def scene_snapshot(
    *,
    viewer: ViewerSnapshot,
    layers: Sequence[LayerSnapshot],
    policies: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> SceneSnapshot:
    return SceneSnapshot(
        viewer=viewer,
        layers=tuple(layers),
        policies=dict(policies),
        metadata=dict(metadata),
    )


def scene_snapshot_from_payload(payload: NotifyScenePayload) -> SceneSnapshot:
    return SceneSnapshot.from_payload(payload)


def layer_delta_from_payload(payload: NotifyLayersPayload) -> LayerDelta:
    return LayerDelta.from_payload(payload)
