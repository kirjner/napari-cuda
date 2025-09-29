"""Greenfield control protocol message shapes.

These dataclasses encode the authoritative schema described in
``docs/protocol_greenfield.md``. They intentionally avoid coupling to the
legacy helpers so both stacks can evolve independently during the migration
window.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

STATE_UPDATE_TYPE = "state.update"
SCENE_SPEC_TYPE = "scene.spec"
LAYER_UPDATE_TYPE = "layer.update"
LAYER_REMOVE_TYPE = "layer.remove"
SPEC_VERSION = 1


def _as_list(value: Optional[Iterable[Any]]) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    return list(value)


def _coerce_list(value: Optional[Iterable[Any]], cast) -> Optional[List[Any]]:
    if value is None:
        return None
    converted: List[Any] = []
    for item in value:
        try:
            converted.append(cast(item))
        except Exception:
            converted.append(item)
    return converted


def _strip_none(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = {field.name: getattr(obj, field.name) for field in obj.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    if isinstance(obj, dict):
        return {key: _strip_none(val) for key, val in obj.items() if val is not None}
    if isinstance(obj, list):
        return [_strip_none(val) for val in obj if val is not None]
    return obj


@dataclass(slots=True)
class MultiscaleLevelSpec:
    shape: Sequence[int]
    downsample: Sequence[float]
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(
            {
                "shape": [int(x) for x in self.shape],
                "downsample": [float(x) for x in self.downsample],
                "path": self.path,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultiscaleLevelSpec":
        return cls(
            shape=[int(x) for x in data.get("shape", [])],
            downsample=[float(x) for x in data.get("downsample", [])],
            path=data.get("path"),
        )


@dataclass(slots=True)
class MultiscaleSpec:
    levels: List[MultiscaleLevelSpec]
    current_level: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "levels": [level.to_dict() for level in self.levels],
            "current_level": int(self.current_level),
            "metadata": self.metadata,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultiscaleSpec":
        return cls(
            levels=[MultiscaleLevelSpec.from_dict(entry) for entry in data.get("levels", [])],
            current_level=int(data.get("current_level", 0)),
            metadata=dict(data["metadata"]) if isinstance(data.get("metadata"), MutableMapping) else data.get("metadata"),
        )


@dataclass(slots=True)
class LayerRenderHints:
    mode: Optional[str] = None
    colormap: Optional[str] = None
    opacity: Optional[float] = None
    visibility: Optional[bool] = None
    iso_threshold: Optional[float] = None
    attenuation: Optional[float] = None
    gamma: Optional[float] = None
    shading: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "mode": self.mode,
            "colormap": self.colormap,
            "opacity": self.opacity,
            "visibility": self.visibility,
            "iso_threshold": self.iso_threshold,
            "attenuation": self.attenuation,
            "gamma": self.gamma,
            "shading": self.shading,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerRenderHints":
        return cls(
            mode=data.get("mode"),
            colormap=data.get("colormap"),
            opacity=data.get("opacity"),
            visibility=data.get("visibility"),
            iso_threshold=data.get("iso_threshold"),
            attenuation=data.get("attenuation"),
            gamma=data.get("gamma"),
            shading=data.get("shading"),
        )


@dataclass(slots=True)
class LayerSpec:
    layer_id: str
    layer_type: str
    name: str
    ndim: int
    shape: List[int]
    dtype: Optional[str] = None
    axis_order: Optional[List[str]] = None
    axis_labels: Optional[List[str]] = None
    scale: Optional[List[float]] = None
    translate: Optional[List[float]] = None
    channel_axis: Optional[int] = None
    channel_names: Optional[List[str]] = None
    contrast_limits: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    render: Optional[LayerRenderHints] = None
    multiscale: Optional[MultiscaleSpec] = None
    extras: Optional[Dict[str, Any]] = None
    controls: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.layer_id = str(self.layer_id)
        self.layer_type = str(self.layer_type)
        self.name = str(self.name)
        self.ndim = int(self.ndim)
        self.shape = [int(x) for x in self.shape]
        if self.axis_order is not None:
            self.axis_order = [str(x) for x in _as_list(self.axis_order) or []]
        if self.axis_labels is not None:
            self.axis_labels = [str(x) for x in _as_list(self.axis_labels) or []]
        if self.scale is not None:
            self.scale = [float(x) for x in _as_list(self.scale) or []]
        if self.translate is not None:
            self.translate = [float(x) for x in _as_list(self.translate) or []]
        if self.channel_axis is not None:
            try:
                self.channel_axis = int(self.channel_axis)
            except Exception:
                self.channel_axis = None
        if self.channel_names is not None:
            self.channel_names = [str(x) for x in _as_list(self.channel_names) or []]
        if self.contrast_limits is not None:
            self.contrast_limits = [float(x) for x in _as_list(self.contrast_limits) or []]
        if self.metadata is not None:
            self.metadata = dict(self.metadata)
        if isinstance(self.render, Mapping):
            self.render = LayerRenderHints.from_dict(self.render)
        if isinstance(self.multiscale, Mapping):
            self.multiscale = MultiscaleSpec.from_dict(self.multiscale)
        if self.extras is not None:
            self.extras = dict(self.extras)
        if self.controls is not None:
            self.controls = dict(self.controls)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "name": self.name,
            "ndim": self.ndim,
            "shape": [int(x) for x in self.shape],
            "dtype": self.dtype,
            "axis_order": self.axis_order,
            "axis_labels": self.axis_labels,
            "scale": self.scale,
            "translate": self.translate,
            "channel_axis": self.channel_axis,
            "channel_names": self.channel_names,
            "contrast_limits": self.contrast_limits,
            "metadata": self.metadata,
            "render": self.render.to_dict() if isinstance(self.render, LayerRenderHints) else self.render,
            "multiscale": self.multiscale.to_dict() if isinstance(self.multiscale, MultiscaleSpec) else self.multiscale,
            "extras": self.extras,
            "controls": self.controls,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerSpec":
        render = data.get("render")
        multiscale = data.get("multiscale")
        return cls(
            layer_id=data.get("layer_id") or data.get("id") or "",
            layer_type=data.get("layer_type") or data.get("type") or "image",
            name=data.get("name") or data.get("layer_id") or data.get("id") or "",
            ndim=data.get("ndim", len(data.get("shape", []) or [])),
            shape=[int(x) for x in data.get("shape", [])],
            dtype=data.get("dtype"),
            axis_order=_as_list(data.get("axis_order")),
            axis_labels=_as_list(data.get("axis_labels")),
            scale=_coerce_list(data.get("scale"), float),
            translate=_coerce_list(data.get("translate"), float),
            channel_axis=data.get("channel_axis"),
            channel_names=_as_list(data.get("channel_names")),
            contrast_limits=_coerce_list(data.get("contrast_limits"), float),
            metadata=dict(data["metadata"]) if isinstance(data.get("metadata"), MutableMapping) else data.get("metadata"),
            render=LayerRenderHints.from_dict(render) if isinstance(render, Mapping) else render,
            multiscale=MultiscaleSpec.from_dict(multiscale) if isinstance(multiscale, Mapping) else multiscale,
            extras=dict(data["extras"]) if isinstance(data.get("extras"), MutableMapping) else data.get("extras"),
            controls=dict(data["controls"]) if isinstance(data.get("controls"), MutableMapping) else data.get("controls"),
        )


@dataclass(slots=True)
class DimsSpec:
    ndim: int
    axis_labels: List[str]
    order: List[str]
    sizes: List[int]
    range: Optional[List[List[int]]] = None
    current_step: Optional[List[int]] = None
    displayed: Optional[List[int]] = None
    ndisplay: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "ndim": int(self.ndim),
            "axis_labels": list(self.axis_labels),
            "order": list(self.order),
            "sizes": [int(x) for x in self.sizes],
            "range": [[int(v) for v in pair] for pair in self.range] if self.range else None,
            "current_step": [int(x) for x in self.current_step] if self.current_step else None,
            "displayed": [int(x) for x in self.displayed] if self.displayed else None,
            "ndisplay": self.ndisplay,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DimsSpec":
        return cls(
            ndim=int(data.get("ndim", 0)),
            axis_labels=[str(x) for x in data.get("axis_labels", [])],
            order=[str(x) for x in data.get("order", [])],
            sizes=[int(x) for x in data.get("sizes", [])],
            range=[[int(v) for v in pair] for pair in data.get("range", [])] or None,
            current_step=[int(x) for x in data.get("current_step", [])] or None,
            displayed=[int(x) for x in data.get("displayed", [])] or None,
            ndisplay=data.get("ndisplay"),
        )


@dataclass(slots=True)
class CameraSpec:
    center: Optional[List[float]] = None
    zoom: Optional[float] = None
    angles: Optional[List[float]] = None
    distance: Optional[float] = None
    fov: Optional[float] = None
    perspective: Optional[float] = None
    ndisplay: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "center": [float(x) for x in self.center] if self.center is not None else None,
            "zoom": self.zoom,
            "angles": [float(x) for x in self.angles] if self.angles is not None else None,
            "distance": self.distance,
            "fov": self.fov,
            "perspective": self.perspective,
            "ndisplay": self.ndisplay,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraSpec":
        return cls(
            center=[float(x) for x in data.get("center", [])] if data.get("center") is not None else None,
            zoom=data.get("zoom"),
            angles=[float(x) for x in data.get("angles", [])] if data.get("angles") is not None else None,
            distance=data.get("distance"),
            fov=data.get("fov"),
            perspective=data.get("perspective"),
            ndisplay=data.get("ndisplay"),
        )


@dataclass(slots=True)
class SceneSpec:
    layers: List[LayerSpec] = field(default_factory=list)
    dims: Optional[DimsSpec] = None
    camera: Optional[CameraSpec] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "layers": [layer.to_dict() for layer in self.layers],
            "dims": self.dims.to_dict() if self.dims else None,
            "camera": self.camera.to_dict() if self.camera else None,
            "capabilities": list(self.capabilities) if self.capabilities else None,
            "metadata": self.metadata,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneSpec":
        layers = [LayerSpec.from_dict(entry) for entry in data.get("layers", [])]
        dims = DimsSpec.from_dict(data["dims"]) if data.get("dims") else None
        camera = CameraSpec.from_dict(data["camera"]) if data.get("camera") else None
        capabilities = [str(x) for x in data.get("capabilities", [])]
        metadata = dict(data["metadata"]) if isinstance(data.get("metadata"), MutableMapping) else data.get("metadata")
        return cls(layers=layers, dims=dims, camera=camera, capabilities=capabilities, metadata=metadata)


@dataclass(slots=True)
class SceneSpecMessage:
    scene: SceneSpec
    version: int = SPEC_VERSION
    type: str = SCENE_SPEC_TYPE
    timestamp: Optional[float] = None
    capabilities: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "scene": self.scene.to_dict(),
            "capabilities": list(self.capabilities) if self.capabilities else None,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneSpecMessage":
        return cls(
            scene=SceneSpec.from_dict(data.get("scene", {})),
            version=int(data.get("version", SPEC_VERSION)),
            type=str(data.get("type", SCENE_SPEC_TYPE)),
            timestamp=data.get("timestamp"),
            capabilities=[str(x) for x in data.get("capabilities", [])] or None,
        )


@dataclass(slots=True)
class LayerUpdateMessage:
    layer: LayerSpec
    partial: bool = False
    version: int = SPEC_VERSION
    type: str = LAYER_UPDATE_TYPE
    timestamp: Optional[float] = None
    ack: Optional[bool] = None
    intent_seq: Optional[int] = None
    controls: Optional[Dict[str, Any]] = None
    server_seq: Optional[int] = None
    source_client_id: Optional[str] = None
    source_client_seq: Optional[int] = None
    interaction_id: Optional[str] = None
    phase: Optional[str] = None
    control_versions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "layer": self.layer.to_dict(),
            "partial": bool(self.partial),
            "ack": self.ack,
            "intent_seq": self.intent_seq,
            "controls": self.controls,
            "server_seq": self.server_seq,
            "source_client_id": self.source_client_id,
            "source_client_seq": self.source_client_seq,
            "interaction_id": self.interaction_id,
            "phase": self.phase,
            "control_versions": self.control_versions,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerUpdateMessage":
        layer_payload = data.get("layer")
        if not isinstance(layer_payload, Mapping):
            raise ValueError("layer.update payload requires 'layer'")
        return cls(
            layer=LayerSpec.from_dict(layer_payload),
            partial=bool(data.get("partial", False)),
            version=int(data.get("version", SPEC_VERSION)),
            type=str(data.get("type", LAYER_UPDATE_TYPE)),
            timestamp=data.get("timestamp"),
            ack=data.get("ack"),
            intent_seq=int(data["intent_seq"]) if data.get("intent_seq") is not None else None,
            controls=dict(data["controls"]) if isinstance(data.get("controls"), MutableMapping) else data.get("controls"),
            server_seq=int(data["server_seq"]) if data.get("server_seq") is not None else None,
            source_client_id=data.get("source_client_id"),
            source_client_seq=int(data["source_client_seq"]) if data.get("source_client_seq") is not None else None,
            interaction_id=data.get("interaction_id"),
            phase=data.get("phase"),
            control_versions=dict(data["control_versions"]) if isinstance(data.get("control_versions"), MutableMapping) else data.get("control_versions"),
        )


@dataclass(slots=True)
class LayerRemoveMessage:
    layer_id: str
    version: int = SPEC_VERSION
    type: str = LAYER_REMOVE_TYPE
    reason: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "layer_id": self.layer_id,
            "reason": self.reason,
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerRemoveMessage":
        return cls(
            layer_id=str(data.get("layer_id", "")),
            version=int(data.get("version", SPEC_VERSION)),
            type=str(data.get("type", LAYER_REMOVE_TYPE)),
            reason=data.get("reason"),
            timestamp=data.get("timestamp"),
        )


@dataclass(slots=True)
class StateUpdateMessage:
    scope: str
    target: str
    key: str
    value: Any
    type: str = STATE_UPDATE_TYPE
    phase: Optional[str] = None
    timestamp: Optional[float] = None
    client_id: Optional[str] = None
    client_seq: Optional[int] = None
    interaction_id: Optional[str] = None
    server_seq: Optional[int] = None
    axis_index: Optional[int] = None
    current_step: Optional[List[int]] = None
    meta: Optional[Dict[str, Any]] = None
    ack: Optional[bool] = None
    intent_seq: Optional[int] = None
    last_client_id: Optional[str] = None
    last_client_seq: Optional[int] = None
    extras: Optional[Dict[str, Any]] = None
    controls: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.type,
            "scope": self.scope,
            "target": self.target,
            "key": self.key,
            "value": self._normalise_value(self.value),
            "phase": self.phase,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "client_seq": self._int_or_none(self.client_seq),
            "interaction_id": self.interaction_id,
            "server_seq": self._int_or_none(self.server_seq),
            "axis_index": self._int_or_none(self.axis_index),
            "current_step": self._normalise_step(self.current_step),
            "meta": dict(self.meta) if isinstance(self.meta, MutableMapping) else self.meta,
            "ack": self.ack,
            "intent_seq": self._int_or_none(self.intent_seq),
            "last_client_id": self.last_client_id,
            "last_client_seq": self._int_or_none(self.last_client_seq),
            "extras": dict(self.extras) if isinstance(self.extras, MutableMapping) else self.extras,
            "controls": dict(self.controls) if isinstance(self.controls, MutableMapping) else self.controls,
        }
        return _strip_none(payload)

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateUpdateMessage":
        current_step = data.get("current_step")
        meta_payload = data.get("meta")
        return cls(
            type=str(data.get("type", STATE_UPDATE_TYPE)),
            scope=str(data["scope"]),
            target=str(data["target"]),
            key=str(data["key"]),
            value=data.get("value"),
            phase=data.get("phase"),
            timestamp=data.get("timestamp"),
            client_id=data.get("client_id"),
            client_seq=cls._parse_optional_int(data.get("client_seq")),
            interaction_id=data.get("interaction_id"),
            server_seq=cls._parse_optional_int(data.get("server_seq")),
            axis_index=cls._parse_optional_int(data.get("axis_index")),
            current_step=cls._parse_step(current_step),
            meta=dict(meta_payload) if isinstance(meta_payload, MutableMapping) else meta_payload,
            ack=data.get("ack"),
            intent_seq=cls._parse_optional_int(data.get("intent_seq")),
            last_client_id=data.get("last_client_id"),
            last_client_seq=cls._parse_optional_int(data.get("last_client_seq")),
            extras=dict(data.get("extras", {})) if isinstance(data.get("extras"), MutableMapping) else data.get("extras"),
            controls=dict(data.get("controls", {})) if isinstance(data.get("controls"), MutableMapping) else data.get("controls"),
        )

    @staticmethod
    def _normalise_value(value: Any) -> Any:
        if isinstance(value, tuple):
            return [StateUpdateMessage._normalise_value(v) for v in value]
        if isinstance(value, list):
            return [StateUpdateMessage._normalise_value(v) for v in value]
        return value

    @staticmethod
    def _normalise_step(step: Optional[Sequence[object]]) -> Optional[List[int]]:
        if step is None:
            return None
        return [int(x) for x in step]

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_step(value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parsed: List[int] = []
            for entry in value:
                try:
                    parsed.append(int(entry))
                except (TypeError, ValueError):
                    parsed.append(int(float(entry)))
            return parsed
        return None


__all__ = [
    "STATE_UPDATE_TYPE",
    "SCENE_SPEC_TYPE",
    "LAYER_UPDATE_TYPE",
    "LAYER_REMOVE_TYPE",
    "SPEC_VERSION",
    "MultiscaleLevelSpec",
    "MultiscaleSpec",
    "LayerRenderHints",
    "LayerSpec",
    "DimsSpec",
    "CameraSpec",
    "SceneSpec",
    "SceneSpecMessage",
    "LayerUpdateMessage",
    "LayerRemoveMessage",
    "StateUpdateMessage",
]
