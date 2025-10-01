"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from napari_cuda.protocol.messages import (
    NotifyScenePayload,
    NotifyDimsPayload,
    NotifyLayersPayload,
    SceneSpecMessage,
)
from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import ServerSceneData, layer_controls_to_dict
from napari_cuda.server.control.state_update_engine import StateUpdateResult


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
# Greenfield payload adapters


def build_notify_scene_payload(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    timestamp: Optional[float] = None,
    viewer_settings: Optional[Mapping[str, Any]] = None,
    ancillary: Optional[Mapping[str, Any]] = None,
) -> NotifyScenePayload:
    """Build a ``notify.scene`` payload aligned with the greenfield schema."""

    message = build_scene_spec_message(scene, manager, timestamp=timestamp)
    spec = message.scene

    scene_dict = spec.to_dict()

    viewer_block = _build_viewer_block(scene_dict, viewer_settings)
    layers_block = tuple(layer.to_dict() for layer in spec.layers)
    policies_block = _build_policies_block(scene)
    ancillary_block = _build_ancillary_block(scene_dict, scene, ancillary)

    return NotifyScenePayload(
        viewer=viewer_block,
        layers=layers_block,
        policies=policies_block,
        ancillary=ancillary_block,
    )


def build_notify_layers_delta_payload(result: StateUpdateResult) -> NotifyLayersPayload:
    """Convert a single layer result into a `notify.layers` payload."""

    return build_notify_layers_payload(
        layer_id=result.target,
        changes={result.key: _normalize_value(result.value)},
    )


def build_notify_layers_payload(
    *, layer_id: str, changes: Mapping[str, Any]
) -> NotifyLayersPayload:
    return NotifyLayersPayload(
        layer_id=str(layer_id),
        changes={key: _normalize_value(value) for key, value in changes.items()},
    )


def build_layer_controls_payload(layer_id: str, state: ServerSceneData) -> NotifyLayersPayload | None:
    control_state = state.layer_controls.get(layer_id)
    if control_state is None:
        return None
    controls = layer_controls_to_dict(control_state)
    if not controls:
        return None
    return build_notify_layers_payload(layer_id=layer_id, changes=controls)


def build_notify_dims_payload(
    *,
    current_step: Sequence[int],
    ndisplay: int,
    mode: str,
    source: str,
) -> NotifyDimsPayload:
    return NotifyDimsPayload(
        current_step=tuple(int(x) for x in current_step),
        ndisplay=int(ndisplay),
        mode=str(mode),
        source=str(source),
    )


def build_notify_dims_from_result(
    result: StateUpdateResult,
    *,
    ndisplay: int,
    mode: str,
    source: str,
) -> NotifyDimsPayload:
    step = result.current_step or tuple()
    return build_notify_dims_payload(
        current_step=step,
        ndisplay=ndisplay,
        mode=mode,
        source=source,
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    return value


def _build_viewer_block(
    scene_dict: Mapping[str, Any],
    viewer_settings: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    viewer: Dict[str, Any] = {}

    dims_block = scene_dict.get("dims")
    viewer["dims"] = dict(dims_block) if isinstance(dims_block, Mapping) else {}

    camera_block = scene_dict.get("camera")
    viewer["camera"] = dict(camera_block) if isinstance(camera_block, Mapping) else {}

    settings: Dict[str, Any] = {}
    if isinstance(viewer_settings, Mapping):
        for key, value in viewer_settings.items():
            normalized = _normalize_value(value)
            if normalized is not None:
                settings[str(key)] = normalized
    if not settings:
        settings = {}
    viewer["settings"] = settings
    return viewer


def _build_policies_block(scene: ServerSceneData) -> Optional[Dict[str, Any]]:
    policies: Dict[str, Any] = {}
    multiscale = scene.multiscale_state or {}
    if isinstance(multiscale, Mapping):
        policy_payload: Dict[str, Any] = {}
        policy = multiscale.get("policy")
        if policy is not None:
            policy_payload["policy"] = str(policy)
        current_level = multiscale.get("current_level")
        if current_level is not None:
            try:
                policy_payload["active_level"] = int(current_level)
            except Exception:
                policy_payload["active_level"] = current_level
        downgraded = multiscale.get("downgraded")
        if downgraded is not None:
            policy_payload["downgraded"] = bool(downgraded)
        index_space = multiscale.get("index_space")
        if index_space is not None:
            policy_payload["index_space"] = str(index_space)
        levels = multiscale.get("levels")
        if isinstance(levels, Sequence):
            level_payload = []
            for entry in levels:
                if isinstance(entry, Mapping):
                    level_payload.append({str(k): _normalize_value(v) for k, v in entry.items()})
            if level_payload:
                policy_payload["levels"] = level_payload
        if policy_payload:
            policies["multiscale"] = policy_payload

    return policies or None


def _build_ancillary_block(
    scene_dict: Mapping[str, Any],
    scene: ServerSceneData,
    ancillary: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {}

    metadata = scene_dict.get("metadata")
    if isinstance(metadata, Mapping) and metadata:
        payload["metadata"] = {str(k): _normalize_value(v) for k, v in metadata.items() if v is not None}

    capabilities = scene_dict.get("capabilities")
    if isinstance(capabilities, Sequence):
        caps = [str(item) for item in capabilities if item is not None]
        if caps:
            payload["capabilities"] = caps

    if scene.volume_state:
        payload["volume_state"] = {str(k): _normalize_value(v) for k, v in scene.volume_state.items() if v is not None}

    if scene.policy_metrics_snapshot:
        payload["policy_metrics"] = {
            str(k): _normalize_value(v)
            for k, v in scene.policy_metrics_snapshot.items()
            if v is not None
        }

    if isinstance(ancillary, Mapping):
        for key, value in ancillary.items():
            payload[str(key)] = _normalize_value(value)

    return payload or None
