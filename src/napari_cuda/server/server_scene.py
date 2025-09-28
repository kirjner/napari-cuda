"""Server control-channel data bag and helpers.

This module centralises the mutable scene metadata tracked by the
headless server so state-channel handlers can operate on a single bag of
data and emit immutable snapshots to the worker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Literal, Optional

from napari_cuda.server.scene_state import ServerSceneState


CONTROL_KEYS: tuple[str, ...] = (
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "gamma",
    "colormap",
    "contrast_limits",
    "depiction",
    "rendering",
    "attenuation",
    "iso_threshold",
)


def layer_controls_to_dict(control: LayerControlState) -> dict[str, Any]:
    """Serialise a LayerControlState into JSON-friendly primitives."""

    payload: dict[str, Any] = {}
    for key in CONTROL_KEYS:
        value = getattr(control, key)
        if value is None:
            continue
        if isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


__all__ = [
    "CONTROL_KEYS",
    "LayerControlMeta",
    "LayerControlState",
    "ServerSceneCommand",
    "ServerSceneData",
    "create_server_scene_data",
    "default_layer_controls",
    "default_multiscale_state",
    "default_volume_state",
    "increment_dims_sequence",
    "layer_controls_to_dict",
]


def default_volume_state() -> Dict[str, Any]:
    """Return the canonical defaults for volume render hints."""

    return {
        "mode": "mip",
        "colormap": "gray",
        "clim": [0.0, 1.0],
        "opacity": 1.0,
        "sample_step": 1.0,
    }


def default_multiscale_state() -> Dict[str, Any]:
    """Return the canonical defaults for multiscale metadata."""

    return {
        "levels": [],
        "current_level": 0,
        "policy": "oversampling",
        "index_space": "base",
    }


@dataclass(frozen=True)
class ServerSceneCommand:
    """Queued camera command consumed by the render thread."""

    kind: Literal["zoom", "pan", "orbit", "reset"]
    factor: Optional[float] = None
    anchor_px: Optional[tuple[float, float]] = None
    dx_px: float = 0.0
    dy_px: float = 0.0
    d_az_deg: float = 0.0
    d_el_deg: float = 0.0


@dataclass
class LayerControlState:
    """Canonical per-layer controls owned by the control thread."""

    visible: bool = True
    opacity: float = 1.0
    blending: str = "opaque"
    interpolation: str = "bilinear"
    gamma: float = 1.0
    colormap: Optional[str] = None
    contrast_limits: Optional[tuple[float, float]] = None
    depiction: Optional[str] = None
    rendering: Optional[str] = None
    attenuation: Optional[float] = None
    iso_threshold: Optional[float] = None


@dataclass
class LayerControlMeta:
    """Sequencing metadata for a specific control property."""

    last_server_seq: int = 0
    last_client_id: Optional[str] = None
    last_client_seq: Optional[int] = None
    last_interaction_id: Optional[str] = None
    last_phase: Optional[str] = None
    client_seq_by_id: Dict[str, int] = field(default_factory=dict)


def default_layer_controls() -> LayerControlState:
    """Return the default LayerControlState for new layers."""

    return LayerControlState(colormap="gray")


@dataclass
class ServerSceneData:
    """Mutable scene metadata owned by the headless server."""

    latest_state: ServerSceneState = field(default_factory=ServerSceneState)
    use_volume: bool = False
    camera_commands: Deque[ServerSceneCommand] = field(default_factory=deque)
    dims_seq: int = 0
    last_dims_client_id: Optional[str] = None
    volume_state: Dict[str, Any] = field(default_factory=default_volume_state)
    multiscale_state: Dict[str, Any] = field(default_factory=default_multiscale_state)
    policy_metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_written_decision_seq: int = 0
    policy_event_path: Path = field(default_factory=Path)
    last_scene_spec: Optional[Dict[str, Any]] = None
    pending_scene_spec: Optional[Dict[str, Any]] = None
    last_dims_payload: Optional[Dict[str, Any]] = None
    last_scene_spec_json: Optional[str] = None
    layer_controls: Dict[str, LayerControlState] = field(default_factory=dict)
    layer_control_meta: Dict[str, Dict[str, LayerControlMeta]] = field(default_factory=dict)
    dims_control_meta: Dict[str, LayerControlMeta] = field(default_factory=dict)
    next_layer_server_seq: int = 0


def create_server_scene_data(*, policy_event_path: Optional[str | Path] = None) -> ServerSceneData:
    """Instantiate :class:`ServerSceneData` with an optional policy log path."""

    data = ServerSceneData()
    if policy_event_path is not None:
        data.policy_event_path = Path(policy_event_path)
    return data


def increment_dims_sequence(scene: ServerSceneData, client_id: Optional[str]) -> int:
    """Advance the authoritative dims sequence and record the request origin."""

    seq = int(scene.dims_seq) & 0x7FFFFFFF
    scene.dims_seq = (int(scene.dims_seq) + 1) & 0x7FFFFFFF
    scene.last_dims_client_id = client_id
    return seq
