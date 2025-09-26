"""Server control-channel data bag and helpers.

This module centralises the mutable scene metadata tracked by the
headless server so state-channel handlers can operate on a single bag of
data and emit immutable snapshots to the worker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene_queue import ServerSceneCommand


__all__ = [
    "ServerSceneData",
    "create_server_scene_data",
    "default_multiscale_state",
    "default_volume_state",
    "increment_dims_sequence",
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


@dataclass
class ServerSceneData:
    """Mutable scene metadata owned by the headless server."""

    latest_state: ServerSceneState = field(default_factory=ServerSceneState)
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
    pending_worker_step: Optional[dict[str, Any]] = None


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
