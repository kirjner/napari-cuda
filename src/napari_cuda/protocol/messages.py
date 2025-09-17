"""
Message definitions for napari-cuda protocol.

Uses dataclasses for type safety and easy serialization.
"""

import enum
import json
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Union


class MessageType(enum.Enum):
    """Types of messages in the legacy protocol (kept for compatibility helpers).

    Note: the streaming client uses intents (dims.intent.*) and expects
    server-authoritative dims.update; it does not emit dims.set.
    """
    # Client -> Server
    SET_CAMERA = "set_camera"
    REQUEST_FRAME = "request_frame"
    PING = "ping"

    # Server -> Client
    CAMERA_UPDATE = "camera_update"
    FRAME_READY = "frame_ready"
    PONG = "pong"
    ERROR = "error"


@dataclass
class StateMessage:
    """Base class for state synchronization messages."""
    type: str
    timestamp: float = None
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str):
        """Deserialize from JSON string."""
        return cls(**json.loads(data))


@dataclass
class CameraUpdate(StateMessage):
    """Camera state update message."""
    type: str = MessageType.CAMERA_UPDATE.value
    center: List[float] = None
    zoom: float = None
    angles: List[float] = None  # For 3D
    perspective: float = None
    
    def __post_init__(self):
        if self.center:
            self.center = list(self.center)
        if self.angles:
            self.angles = list(self.angles)


@dataclass
class DimsUpdate(StateMessage):  # Deprecated container used by helpers only
    type: str = "dims.update"
    current_step: List[int] = None
    ndisplay: int = None
    order: List[int] = None
    def __post_init__(self):
        if self.current_step:
            self.current_step = list(self.current_step)
        if self.order:
            self.order = list(self.order)


@dataclass
class Command(StateMessage):
    """Generic command from client to server."""
    type: str
    payload: dict = None


@dataclass
class Response(StateMessage):
    """Generic response from server to client."""
    type: str
    success: bool = True
    payload: dict = None
    error: str = None


@dataclass
class FrameMessage:
    """
    Frame data message for pixel stream.
    
    This is typically sent as binary data, not JSON.
    """
    frame_number: int
    timestamp: float
    width: int
    height: int
    encoding: str  # "h264", "jpeg", "raw"
    data: bytes
    
    def to_bytes(self) -> bytes:
        """Pack into binary format for transmission."""
        # Simple format: 
        # [4 bytes frame_num][8 bytes timestamp][4 bytes w][4 bytes h][1 byte encoding][data]
        import struct
        
        encoding_byte = {
            "h264": 0,
            "jpeg": 1, 
            "raw": 2
        }.get(self.encoding, 255)
        
        header = struct.pack(
            "!Id2IB",  # Network byte order: int, double, 2 ints, byte
            self.frame_number,
            self.timestamp,
            self.width,
            self.height,
            encoding_byte
        )
        
        return header + self.data
    
    @classmethod
    def from_bytes(cls, data: bytes):
        """Unpack from binary format."""
        import struct
        
        header_size = 4 + 8 + 4 + 4 + 1  # 21 bytes
        header = data[:header_size]
        frame_data = data[header_size:]
        
        frame_num, timestamp, width, height, encoding_byte = struct.unpack(
            "!Id2IB", header
        )
        
        encoding = {
            0: "h264",
            1: "jpeg",
            2: "raw"
        }.get(encoding_byte, "unknown")
        
        return cls(
            frame_number=frame_num,
            timestamp=timestamp,
            width=width,
            height=height,
            encoding=encoding,
            data=frame_data
        )


# -----------------------------------------------------------------------------
# Layer/scene specification (shared schema between server and client)

SPEC_VERSION = 1

SCENE_SPEC_TYPE = "scene.spec"
LAYER_UPDATE_TYPE = "layer.update"
LAYER_REMOVE_TYPE = "layer.remove"


def _ensure_list(value: Optional[Iterable[Any]]) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _coerce(value: Optional[Iterable[Any]], cast) -> Optional[List[Any]]:
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
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {key: _strip_none(val) for key, val in obj.items() if val is not None}
    if isinstance(obj, list):
        return [_strip_none(val) for val in obj if val is not None]
    return obj


@dataclass
class MultiscaleLevelSpec:
    """Description of one multiscale level."""

    shape: List[int]
    downsample: List[float]
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
    def from_dict(cls, data: Dict[str, Any]) -> "MultiscaleLevelSpec":
        return cls(
            shape=[int(x) for x in (data.get("shape") or [])],
            downsample=[float(x) for x in (data.get("downsample") or [])],
            path=data.get("path"),
        )


@dataclass
class MultiscaleSpec:
    """Collection of multiscale levels for a layer."""

    levels: List[MultiscaleLevelSpec]
    current_level: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "levels": [level.to_dict() for level in self.levels],
            "current_level": int(self.current_level),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiscaleSpec":
        levels = [MultiscaleLevelSpec.from_dict(entry) for entry in data.get("levels", [])]
        return cls(
            levels=levels,
            current_level=int(data.get("current_level", 0)),
            metadata=data.get("metadata"),
        )


@dataclass
class LayerRenderHints:
    """Optional rendering hints for volume/image layers."""

    mode: Optional[str] = None
    colormap: Optional[str] = None
    opacity: Optional[float] = None
    visibility: Optional[bool] = None
    iso_threshold: Optional[float] = None
    attenuation: Optional[float] = None
    gamma: Optional[float] = None
    shading: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerRenderHints":
        return cls(**data)


@dataclass
class LayerSpec:
    """Serialized description of a napari Layer."""

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

    def __post_init__(self) -> None:
        self.layer_id = str(self.layer_id)
        self.layer_type = str(self.layer_type)
        self.name = str(self.name)
        try:
            self.ndim = int(self.ndim)
        except Exception:
            self.ndim = len(self.shape)
        self.shape = [int(x) for x in self.shape]
        if self.axis_order is not None:
            self.axis_order = [str(x) for x in _ensure_list(self.axis_order) or []]
        if self.axis_labels is not None:
            self.axis_labels = [str(x) for x in _ensure_list(self.axis_labels) or []]
        if self.scale is not None:
            self.scale = [float(x) for x in _ensure_list(self.scale) or []]
        if self.translate is not None:
            self.translate = [float(x) for x in _ensure_list(self.translate) or []]
        if self.channel_axis is not None:
            try:
                self.channel_axis = int(self.channel_axis)
            except Exception:
                pass
        if self.channel_names is not None:
            self.channel_names = [str(x) for x in _ensure_list(self.channel_names) or []]
        if self.contrast_limits is not None:
            self.contrast_limits = [float(x) for x in _ensure_list(self.contrast_limits) or []]
        if self.metadata is not None:
            self.metadata = dict(self.metadata)
        if isinstance(self.render, dict):
            self.render = LayerRenderHints.from_dict(self.render)
        if isinstance(self.multiscale, dict):
            self.multiscale = MultiscaleSpec.from_dict(self.multiscale)
        if self.extras is not None:
            self.extras = dict(self.extras)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "name": self.name,
            "ndim": int(self.ndim),
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
        }
        return _strip_none(payload)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerSpec":
        render = data.get("render")
        multiscale = data.get("multiscale")
        return cls(
            layer_id=data.get("layer_id") or data.get("id") or "",
            layer_type=str(data.get("layer_type") or data.get("type") or "image"),
            name=str(data.get("name") or data.get("layer_id") or data.get("id") or ""),
            ndim=int(data.get("ndim", len(data.get("shape", []) or []))),
            shape=[int(x) for x in (data.get("shape") or [])],
            dtype=data.get("dtype"),
            axis_order=_ensure_list(data.get("axis_order")),
            axis_labels=_ensure_list(data.get("axis_labels")),
            scale=_coerce(data.get("scale"), float),
            translate=_coerce(data.get("translate"), float),
            channel_axis=data.get("channel_axis"),
            channel_names=_ensure_list(data.get("channel_names")),
            contrast_limits=_coerce(data.get("contrast_limits"), float),
            metadata=data.get("metadata"),
            render=LayerRenderHints.from_dict(render) if isinstance(render, dict) else render,
            multiscale=MultiscaleSpec.from_dict(multiscale) if isinstance(multiscale, dict) else multiscale,
            extras=data.get("extras"),
        )


@dataclass
class DimsSpec:
    """Canonical description of the dims state."""

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
    def from_dict(cls, data: Dict[str, Any]) -> "DimsSpec":
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


@dataclass
class CameraSpec:
    """Snapshot of the viewer camera."""

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
    def from_dict(cls, data: Dict[str, Any]) -> "CameraSpec":
        return cls(
            center=[float(x) for x in data.get("center", [])] if data.get("center") is not None else None,
            zoom=data.get("zoom"),
            angles=[float(x) for x in data.get("angles", [])] if data.get("angles") is not None else None,
            distance=data.get("distance"),
            fov=data.get("fov"),
            perspective=data.get("perspective"),
            ndisplay=data.get("ndisplay"),
        )


@dataclass
class SceneSpec:
    """Authoritative scene state hosted by the server."""

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
    def from_dict(cls, data: Dict[str, Any]) -> "SceneSpec":
        layers = [LayerSpec.from_dict(entry) for entry in data.get("layers", [])]
        dims = DimsSpec.from_dict(data["dims"]) if data.get("dims") else None
        camera = CameraSpec.from_dict(data["camera"]) if data.get("camera") else None
        capabilities = [str(x) for x in data.get("capabilities", [])]
        metadata = data.get("metadata")
        return cls(layers=layers, dims=dims, camera=camera, capabilities=capabilities, metadata=metadata)


@dataclass
class SceneSpecMessage(StateMessage):
    """Message containing a full scene specification."""

    type: str = SCENE_SPEC_TYPE
    version: int = SPEC_VERSION
    scene: SceneSpec = field(default_factory=SceneSpec)
    capabilities: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "scene": self.scene.to_dict(),
            "capabilities": self.capabilities,
        }
        return _strip_none(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneSpecMessage":
        scene = SceneSpec.from_dict(data.get("scene", {}))
        return cls(
            type=data.get("type", SCENE_SPEC_TYPE),
            version=int(data.get("version", SPEC_VERSION)),
            timestamp=data.get("timestamp"),
            scene=scene,
            capabilities=_ensure_list(data.get("capabilities")),
        )


@dataclass
class LayerUpdateMessage(StateMessage):
    """Message describing a layer update."""

    type: str = LAYER_UPDATE_TYPE
    version: int = SPEC_VERSION
    layer: Optional[LayerSpec] = None
    partial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        if self.layer is None:
            raise ValueError("LayerUpdateMessage.layer must be set before serialization")
        payload = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "layer": self.layer.to_dict(),
            "partial": bool(self.partial),
        }
        return _strip_none(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerUpdateMessage":
        layer_data = data.get("layer")
        layer = LayerSpec.from_dict(layer_data) if isinstance(layer_data, dict) else None
        return cls(
            type=data.get("type", LAYER_UPDATE_TYPE),
            version=int(data.get("version", SPEC_VERSION)),
            timestamp=data.get("timestamp"),
            layer=layer,
            partial=bool(data.get("partial", False)),
        )


@dataclass
class LayerRemoveMessage(StateMessage):
    """Message indicating a layer should be removed."""

    type: str = LAYER_REMOVE_TYPE
    version: int = SPEC_VERSION
    layer_id: str = ""
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "version": int(self.version),
            "timestamp": self.timestamp,
            "layer_id": self.layer_id,
            "reason": self.reason,
        }
        return _strip_none(payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerRemoveMessage":
        return cls(
            type=data.get("type", LAYER_REMOVE_TYPE),
            version=int(data.get("version", SPEC_VERSION)),
            timestamp=data.get("timestamp"),
            layer_id=str(data.get("layer_id", "")),
            reason=data.get("reason"),
        )


class StreamProtocol:
    """Helper utilities for napari-cuda protocol messages."""

    @staticmethod
    def parse_message(data: Union[str, bytes]) -> Union[StateMessage, FrameMessage]:
        """Parse incoming data into a protocol object."""
        if isinstance(data, bytes):
            return FrameMessage.from_bytes(data)

        msg_dict = json.loads(data)
        msg_type = msg_dict.get("type")

        if msg_type == MessageType.CAMERA_UPDATE.value:
            return CameraUpdate(**msg_dict)
        if msg_type in ("dims_update", "dims.update"):
            return DimsUpdate(**msg_dict)
        if msg_type == SCENE_SPEC_TYPE:
            return SceneSpecMessage.from_dict(msg_dict)
        if msg_type == LAYER_UPDATE_TYPE:
            return LayerUpdateMessage.from_dict(msg_dict)
        if msg_type == LAYER_REMOVE_TYPE:
            return LayerRemoveMessage.from_dict(msg_dict)
        if msg_type in [t.value for t in MessageType]:
            return StateMessage(**msg_dict)
        return Command(**msg_dict)

    @staticmethod
    def create_camera_command(center=None, zoom=None, angles=None) -> str:
        """Create camera update command."""
        cmd = CameraUpdate(
            type=MessageType.SET_CAMERA.value,
            center=center,
            zoom=zoom,
            angles=angles,
        )
        return cmd.to_json()

    @staticmethod
    def create_dims_command(*args, **kwargs) -> str:  # pragma: no cover
        """Deprecated: the client no longer emits dims.set. Use intents."""
        raise RuntimeError("create_dims_command is deprecated; use dims intents")

    @staticmethod
    def create_error_response(error_msg: str) -> str:
        """Create error response."""
        resp = Response(
            type=MessageType.ERROR.value,
            success=False,
            error=error_msg,
        )
        return resp.to_json()
