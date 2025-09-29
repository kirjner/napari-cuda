"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from dataclasses import asdict

from napari_cuda.protocol.messages import SceneSpecMessage, StateUpdateMessage
from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import (
    CONTROL_KEYS,
    ServerSceneData,
    layer_controls_to_dict,
)
from napari_cuda.server.server_state_updates import StateUpdateResult


# ---------------------------------------------------------------------------
# Scene spec helpers


def build_scene_spec_message(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    timestamp: Optional[float] = None,
) -> SceneSpecMessage:
    """Build a `SceneSpecMessage` and cache it on the scene bag."""

    message = manager.scene_message(timestamp)
    json_payload = message.to_json()
    scene.last_scene_spec = message.to_dict()
    scene.last_scene_spec_json = json_payload
    return message


def build_scene_spec_json(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    timestamp: Optional[float] = None,
) -> str:
    """Return a JSON-encoded scene spec, caching it on the scene bag."""

    message = build_scene_spec_message(scene, manager, timestamp=timestamp)
    return scene.last_scene_spec_json or message.to_json()


# ---------------------------------------------------------------------------
# Unified state.update payload helper


def build_state_update_payload(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    result: StateUpdateResult,
    include_control_versions: bool = True,
) -> Dict[str, Any]:
    """Serialise a :class:`StateUpdateResult` into a transport payload."""

    value = result.value
    if isinstance(value, tuple):
        value = list(value)

    message = StateUpdateMessage(
        scope=result.scope,
        target=result.target,
        key=result.key,
        value=value,
        phase=result.phase,
        timestamp=result.timestamp,
        client_id=result.client_id,
        client_seq=result.client_seq,
        interaction_id=result.interaction_id,
        server_seq=result.server_seq,
        axis_index=result.axis_index,
        current_step=list(result.current_step) if result.current_step is not None else None,
        meta=dict(result.meta) if result.meta is not None else None,
        ack=result.ack,
        intent_seq=result.intent_seq,
        last_client_id=result.last_client_id or result.client_id,
        last_client_seq=result.last_client_seq,
    )
    payload = message.to_dict()

    if include_control_versions:
        versions = _control_versions(scene, result.scope, result.target, result.key)
        if versions:
            payload["control_versions"] = versions

    if result.scope == "layer":
        _inject_layer_context(scene, manager, payload, result)
    elif result.scope == "dims":
        _inject_dims_context(result, payload)

    return payload


def _control_versions(
    scene: ServerSceneData, scope: str, target: str, key: str
) -> Optional[Dict[str, Dict[str, Any]]]:
    meta = scene.control_meta.get((str(scope), str(target), str(key)))
    if meta is None:
        return None
    entry = {
        "server_seq": meta.last_server_seq or None,
        "source_client_id": meta.last_client_id,
        "source_client_seq": meta.last_client_seq,
        "interaction_id": meta.last_interaction_id,
        "phase": meta.last_phase,
    }
    return {str(key): entry}


def _inject_layer_context(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    payload: Dict[str, Any],
    result: StateUpdateResult,
) -> None:
    """Attach layer metadata to the payload for observability."""

    layer_id = result.target
    control_state = scene.layer_controls.get(layer_id)
    if control_state is None:
        return

    controls = layer_controls_to_dict(control_state)
    if controls:
        payload.setdefault("controls", controls)

    spec = manager.scene_spec()
    if spec is None:
        return
    match = next((layer for layer in spec.layers or [] if layer.layer_id == layer_id), None)
    if match is None:
        return
    patch_dict = match.to_dict()
    extras = patch_dict.get("extras") or {}
    extras = {
        key: value
        for key, value in extras.items()
        if key not in CONTROL_KEYS
    }
    if extras:
        payload.setdefault("extras", extras)


def _inject_dims_context(result: StateUpdateResult, payload: Dict[str, Any]) -> None:
    """Attach dims extras to the payload if supplied."""

    if result.axis_index is not None:
        payload.setdefault("axis_index", int(result.axis_index))
    if result.current_step is not None:
        payload.setdefault("current_step", [int(x) for x in result.current_step])
    if result.meta is not None:
        payload.setdefault("meta", dict(result.meta))
    if result.last_client_id is not None:
        payload.setdefault("last_client_id", result.last_client_id)
    if result.last_client_seq is not None:
        payload.setdefault("last_client_seq", int(result.last_client_seq))
    if result.ack is not None:
        payload.setdefault("ack", bool(result.ack))
    if result.intent_seq is not None:
        payload.setdefault("intent_seq", int(result.intent_seq))
