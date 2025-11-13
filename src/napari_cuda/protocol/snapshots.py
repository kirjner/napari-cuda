"""Typed snapshots feeding control protocol notify payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from napari_cuda.protocol.messages import (
    NotifyLayerBlockPayload,
    NotifyLayersPayload,
    NotifyScenePayload,
    layer_block_payload_from_mapping,
)


@dataclass(slots=True)
class ViewerSnapshot:
    """Minimal viewer block used by ``notify.scene``."""

    settings: dict[str, Any]
    dims: dict[str, Any]
    camera: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "dims": dict(self.dims),
            "camera": dict(self.camera),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ViewerSnapshot:
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
    block: NotifyLayerBlockPayload

    def to_mapping(self) -> NotifyLayerBlockPayload:
        return dict(self.block)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> LayerSnapshot:
        payload = layer_block_payload_from_mapping(mapping, context="scene.snapshot.layer")
        layer_id = str(payload["layer_id"])
        return cls(layer_id=layer_id, block=payload)


@dataclass(slots=True)
class SceneSnapshot:
    """Authoritative scene snapshot emitted on baseline."""

    viewer: ViewerSnapshot
    layers: tuple[LayerSnapshot, ...]
    policies: dict[str, Any]
    metadata: dict[str, Any]

    def to_notify_scene_payload(self) -> NotifyScenePayload:
        return NotifyScenePayload(
            viewer=self.viewer.to_mapping(),
            layers=tuple(layer.to_mapping() for layer in self.layers),
            metadata=dict(self.metadata) if self.metadata else None,
            policies=dict(self.policies) if self.policies else None,
        )

    @classmethod
    def from_payload(cls, payload: NotifyScenePayload) -> SceneSnapshot:
        viewer = ViewerSnapshot.from_mapping(payload.viewer)
        layers = tuple(LayerSnapshot.from_mapping(block) for block in payload.layers)
        policies = dict(payload.policies) if payload.policies else {}
        metadata = dict(payload.metadata) if payload.metadata else {}
        return cls(viewer=viewer, layers=layers, policies=policies, metadata=metadata)


@dataclass(slots=True)
class LayerDelta:
    """Incremental layer changes for ``notify.layers`` (structured)."""

    layer_id: str
    layer_type: str = "image"
    controls: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    thumbnail: dict[str, Any] | None = None
    removed: bool | None = None

    def to_payload(self) -> NotifyLayersPayload:
        layer_payload = _layer_block_payload_from_sections(
            layer_id=self.layer_id,
            layer_type=self.layer_type or "image",
            controls=self.controls,
            metadata=self.metadata,
            data=self.data,
            thumbnail=self.thumbnail,
        )
        removed_flag = bool(self.removed) if self.removed else None
        if layer_payload is None and not removed_flag:
            raise ValueError("layer delta payload requires layer data or removal flag")
        return NotifyLayersPayload(
            layer_id=self.layer_id,
            layer=layer_payload,
            removed=removed_flag,
        )

    @classmethod
    def controls_only(cls, layer_id: str, controls: Mapping[str, Any], *, layer_type: str = "image") -> LayerDelta:
        return cls(layer_id=layer_id, layer_type=layer_type, controls=dict(controls))

    @classmethod
    def removal(cls, layer_id: str) -> LayerDelta:
        return cls(layer_id=layer_id, removed=True)

    @classmethod
    def from_state_update(cls, layer_id: str, key: str, value: Any) -> LayerDelta:
        # Fallback: treat unknown singletons as data section
        return cls(layer_id=layer_id, data={str(key): value})

    @classmethod
    def from_payload(cls, payload: NotifyLayersPayload) -> LayerDelta:
        controls = None
        metadata = None
        data_payload = None
        thumbnail_payload = None
        layer_type = "image"
        if payload.layer is not None:
            layer_payload = layer_block_payload_from_mapping(
                payload.layer,
                context="notify.layers payload.layer",
            )
            layer_type = str(layer_payload.get("layer_type", "image"))
            controls = dict(layer_payload["controls"])
            metadata = dict(layer_payload.get("metadata") or {})
            thumbnail = layer_payload.get("thumbnail")
            if isinstance(thumbnail, Mapping):
                thumbnail_payload = dict(thumbnail)
            extras = dict(layer_payload.get("extras") or {})
            if extras:
                data_payload = extras
        return cls(
            layer_id=str(payload.layer_id),
            layer_type=layer_type,
            controls=controls,
            metadata=metadata if metadata else None,
            data=data_payload,
            thumbnail=thumbnail_payload,
            removed=bool(payload.removed) if payload.removed else None,
        )


def _layer_block_payload_from_sections(
    *,
    layer_id: str,
    layer_type: str,
    controls: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    thumbnail: Mapping[str, Any] | None,
) -> NotifyLayerBlockPayload | None:
    if controls is None and metadata is None and data is None and thumbnail is None:
        return None
    payload: NotifyLayerBlockPayload = {
        "layer_id": str(layer_id),
        "layer_type": str(layer_type),
        "controls": {str(key): value for key, value in (controls or {}).items()},
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    if data:
        payload["extras"] = dict(data)
    if thumbnail:
        payload["thumbnail"] = dict(thumbnail)
    return payload


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
