"""Server control-channel data bag and helpers.

This module centralises the mutable scene metadata tracked by the
headless server so state-channel handlers can operate on a single bag of
data and emit immutable snapshots to the worker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Literal, Mapping, Optional, Sequence

from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.plane_restore_state import PlaneRestoreState


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
    "increment_server_sequence",
    "get_control_meta",
    "clear_control_meta",
    "prune_control_metadata",
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
    last_timestamp: Optional[float] = None


def default_layer_controls() -> LayerControlState:
    """Return the default LayerControlState for new layers."""

    return LayerControlState(colormap="gray")


@dataclass
class ServerSceneData:
    """Mutable scene metadata owned by the headless server."""

    latest_state: ServerSceneState = field(default_factory=ServerSceneState)
    use_volume: bool = False
    camera_commands: Deque[ServerSceneCommand] = field(default_factory=deque)
    next_server_seq: int = 0
    volume_state: Dict[str, Any] = field(default_factory=default_volume_state)
    multiscale_state: Dict[str, Any] = field(default_factory=default_multiscale_state)
    policy_metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_written_decision_seq: int = 0
    policy_event_path: Path = field(default_factory=Path)
    last_scene_snapshot: Optional[Dict[str, Any]] = None
    last_dims_payload: Optional[Dict[str, Any]] = None
    last_scene_seq: int = 0
    layer_controls: Dict[str, LayerControlState] = field(default_factory=dict)
    control_meta: Dict[tuple[str, str, str], LayerControlMeta] = field(default_factory=dict)
    plane_restore_state: Optional[PlaneRestoreState] = None


def create_server_scene_data(*, policy_event_path: Optional[str | Path] = None) -> ServerSceneData:
    """Instantiate :class:`ServerSceneData` with an optional policy log path."""

    data = ServerSceneData()
    if policy_event_path is not None:
        data.policy_event_path = Path(policy_event_path)
    return data


def _meta_key(scope: str, target: str, key: str) -> tuple[str, str, str]:
    """Compute the canonical metadata key for a control property."""

    return (str(scope), str(target), str(key))


def get_control_meta(
    scene: ServerSceneData,
    scope: str,
    target: str,
    key: str,
) -> LayerControlMeta:
    """Fetch (and create if needed) control metadata for ``scope/target/key``."""

    meta_key = _meta_key(scope, target, key)
    meta = scene.control_meta.get(meta_key)
    if meta is None:
        meta = LayerControlMeta()
        scene.control_meta[meta_key] = meta
    return meta


def clear_control_meta(scene: ServerSceneData, scope: str, target: str, key: str) -> None:
    """Remove stored metadata for ``scope/target/key`` if present."""

    meta_key = _meta_key(scope, target, key)
    scene.control_meta.pop(meta_key, None)


def increment_server_sequence(scene: ServerSceneData) -> int:
    """Advance and return the global server sequence counter."""

    scene.next_server_seq = (int(scene.next_server_seq) + 1) & 0x7FFFFFFF
    return scene.next_server_seq


def prune_control_metadata(
    scene: ServerSceneData,
    *,
    layer_ids: Iterable[str] | None = None,
    dims_meta: Optional[Mapping[str, Any]] = None,
    current_step: Optional[Sequence[int]] = None,
) -> None:
    """Drop metadata for layers/axes that no longer exist.

    ``layer_ids`` should contain the canonical identifiers for active layers.
    ``dims_meta`` is the latest dims metadata snapshot (matching
    ``ViewerSceneManager.dims_metadata``.  ``current_step``
    provides a fallback axis count if metadata is incomplete.
    """

    layer_ids_specified = layer_ids is not None

    active_layers: set[str] = set()
    if layer_ids is not None:
        for layer_id in layer_ids:
            if layer_id is None:
                continue
            text = str(layer_id).strip()
            if text:
                active_layers.add(text)
    else:
        active_layers.update(str(key) for key in scene.layer_controls.keys())

    dims_known = dims_meta is not None
    dims_targets: set[str] = set()
    if dims_meta is not None:
        dims_targets = _dims_targets_from_meta(dims_meta, current_step=current_step)

    for meta_key in list(scene.control_meta.keys()):
        scope, target, prop_key = meta_key
        if scope == "layer" and layer_ids_specified and target not in active_layers:
            clear_control_meta(scene, scope, target, prop_key)
        elif scope == "dims" and dims_known and target not in dims_targets:
            clear_control_meta(scene, scope, target, prop_key)

    if layer_ids_specified:
        for stored_layer in list(scene.layer_controls.keys()):
            if stored_layer not in active_layers:
                scene.layer_controls.pop(stored_layer, None)


def _dims_targets_from_meta(
    meta: Optional[Mapping[str, Any]],
    *,
    current_step: Optional[Sequence[int]] = None,
) -> set[str]:
    if not meta:
        if current_step is None:
            return set()
        axis_count = len(list(current_step))
        return {str(idx) for idx in range(axis_count)}

    targets: set[str] = set()

    def _ensure_sequence(value: object) -> list[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        if isinstance(value, str):
            return list(value)
        return []

    order_seq = _ensure_sequence(meta.get("order"))
    axis_seq = _ensure_sequence(meta.get("axis_labels"))
    sizes_seq = _ensure_sequence(meta.get("sizes"))
    range_seq = _ensure_sequence(meta.get("range"))

    axis_count = 0
    try:
        axis_count = int(meta.get("ndim") or 0)
    except Exception:
        axis_count = 0
    axis_count = max(axis_count, len(order_seq), len(axis_seq), len(sizes_seq), len(range_seq))
    if axis_count <= 0 and current_step is not None:
        axis_count = len(list(current_step))

    for idx in range(max(0, axis_count)):
        targets.add(str(idx))

        label: Optional[str] = None
        if idx < len(order_seq):
            candidate = order_seq[idx]
            text = str(candidate).strip()
            if text:
                label = text
        if label is None and idx < len(axis_seq):
            candidate = axis_seq[idx]
            text = str(candidate).strip()
            if text:
                label = text
        if label:
            targets.add(label)

    return targets
