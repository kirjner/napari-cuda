"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from napari_cuda.protocol.messages import (
    NotifyScenePayload,
    NotifySceneLevelPayload,
    NotifyDimsPayload,
    NotifyLayersPayload,
)
from napari_cuda.server.state.layer_manager import ViewerSceneManager
from napari_cuda.server.state.server_scene import ServerSceneData, layer_controls_to_dict
from napari_cuda.server.control.state_update_engine import StateUpdateResult


# ---------------------------------------------------------------------------
# Greenfield payload adapters


def build_notify_scene_payload(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    timestamp: Optional[float] = None,
    viewer_settings: Optional[Mapping[str, Any]] = None,
) -> NotifyScenePayload:
    """Build a ``notify.scene`` payload aligned with the greenfield schema."""

    snapshot = manager.scene_snapshot()
    assert snapshot is not None, "viewer scene manager not initialised"

    payload = snapshot.to_notify_scene_payload()

    if viewer_settings:
        settings = payload.viewer.setdefault("settings", {})
        settings.update({str(key): _normalize_value(value) for key, value in viewer_settings.items()})

    payload.policies = _build_policies_block(scene)
    payload.metadata = _merge_scene_metadata(snapshot.metadata, scene)

    scene.last_scene_snapshot = payload.to_dict()
    return payload


def build_notify_scene_level_payload(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
) -> NotifySceneLevelPayload:
    """Build a ``notify.scene.level`` payload representing active LOD metadata."""

    multiscale = scene.multiscale_state or {}
    current_level_raw = multiscale.get("current_level", multiscale.get("level", 0))
    current_level = int(current_level_raw)

    downgraded_val = multiscale.get("downgraded")
    downgraded = None if downgraded_val is None else bool(downgraded_val)

    levels_payload = multiscale.get("levels")
    levels: list[dict[str, Any]] = []
    if isinstance(levels_payload, Sequence):
        for entry in levels_payload:
            if isinstance(entry, Mapping):
                levels.append({str(k): _normalize_value(v) for k, v in entry.items()})

    return NotifySceneLevelPayload(
        current_level=current_level,
        downgraded=downgraded,
        levels=tuple(levels) if levels else None,
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


def _merge_scene_metadata(
    snapshot_metadata: Mapping[str, Any],
    scene: ServerSceneData,
) -> Optional[Dict[str, Any]]:
    metadata: Dict[str, Any] = dict(snapshot_metadata)

    if scene.volume_state:
        metadata.setdefault("volume_state", {})
        metadata["volume_state"].update({str(k): _normalize_value(v) for k, v in scene.volume_state.items()})

    if scene.policy_metrics_snapshot:
        metadata["policy_metrics"] = {
            str(k): _normalize_value(v)
            for k, v in scene.policy_metrics_snapshot.items()
            if v is not None
        }

    return metadata or None
