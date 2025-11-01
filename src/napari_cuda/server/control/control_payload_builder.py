"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from napari_cuda.protocol.messages import NotifyScenePayload, NotifyLayersPayload
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.scene import snapshot_volume_state, snapshot_viewport_state
from napari_cuda.server.state_ledger import LedgerEntry
from napari_cuda.server.control.state_models import ServerLedgerUpdate


# ---------------------------------------------------------------------------
# Greenfield payload adapters


def build_notify_scene_payload(
    *,
    scene_snapshot: SceneSnapshot,
    ledger_snapshot: Mapping[tuple[str, str, str], LedgerEntry],
    viewer_settings: Optional[Mapping[str, Any]] = None,
) -> NotifyScenePayload:
    """Build a ``notify.scene`` payload aligned with the greenfield schema."""

    payload = scene_snapshot.to_notify_scene_payload()

    if viewer_settings:
        settings = payload.viewer.setdefault("settings", {})
        settings.update({str(key): _normalize_value(value) for key, value in viewer_settings.items()})

    payload.policies = _build_policies_block(ledger_snapshot)
    payload.metadata = _merge_scene_metadata(scene_snapshot.metadata, ledger_snapshot)

    viewport_state = snapshot_viewport_state(ledger_snapshot)
    if viewport_state:
        metadata_block = payload.metadata or {}
        metadata_copy = dict(metadata_block)
        metadata_copy["viewport_state"] = viewport_state
        payload.metadata = metadata_copy

    return payload


def build_notify_layers_delta_payload(result: ServerLedgerUpdate) -> NotifyLayersPayload:
    """Convert a single layer ledger update into a structured `notify.layers` payload."""

    key = str(result.key)
    value = _normalize_value(result.value)
    layer_id = str(result.target)

    control_keys = {
        "opacity",
        "visible",
        "blending",
        "interpolation",
        "colormap",
        "rendering",
        "gamma",
        "contrast_limits",
        "iso_threshold",
        "attenuation",
    }

    if key in control_keys:
        return NotifyLayersPayload(layer_id=layer_id, controls={key: value})
    if key == "metadata":
        md = dict(value) if isinstance(value, dict) else {"value": value}
        return NotifyLayersPayload(layer_id=layer_id, metadata=md)
    if key == "thumbnail":
        th = dict(value) if isinstance(value, dict) else {"array": value}
        return NotifyLayersPayload(layer_id=layer_id, thumbnail=th)
    if key == "removed" and bool(value):
        return NotifyLayersPayload(layer_id=layer_id, removed=True)
    # Fallback: data section for structural/other keys
    return NotifyLayersPayload(layer_id=layer_id, data={key: value})


def build_notify_layers_payload(
    *,
    layer_id: str,
    controls: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    thumbnail: Mapping[str, Any] | None = None,
    removed: bool | None = None,
) -> NotifyLayersPayload:
    return NotifyLayersPayload(
        layer_id=str(layer_id),
        controls={k: _normalize_value(v) for k, v in controls.items()} if controls else None,
        metadata={k: _normalize_value(v) for k, v in metadata.items()} if metadata else None,
        data={k: _normalize_value(v) for k, v in data.items()} if data else None,
        thumbnail={k: _normalize_value(v) for k, v in thumbnail.items()} if thumbnail else None,
        removed=bool(removed) if removed else None,
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    return value


def _build_policies_block(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Optional[Dict[str, Any]]:
    policy_payload: Dict[str, Any] = {}

    policy_entry = snapshot.get(("multiscale", "main", "policy"))
    if policy_entry is not None and policy_entry.value is not None:
        policy_payload["policy"] = str(policy_entry.value)

    level_entry = snapshot.get(("multiscale", "main", "level"))
    if level_entry is not None and level_entry.value is not None:
        level_value = level_entry.value
        if isinstance(level_value, (int, float)):
            policy_payload["active_level"] = int(level_value)
        else:
            policy_payload["active_level"] = level_value

    downgraded_entry = snapshot.get(("multiscale", "main", "downgraded"))
    if downgraded_entry is not None and downgraded_entry.value is not None:
        policy_payload["downgraded"] = bool(downgraded_entry.value)

    index_space_entry = snapshot.get(("multiscale", "main", "index_space"))
    if index_space_entry is not None and index_space_entry.value is not None:
        policy_payload["index_space"] = str(index_space_entry.value)

    levels_entry = snapshot.get(("multiscale", "main", "levels"))
    if levels_entry is not None and isinstance(levels_entry.value, Sequence):
        level_payload = []
        for entry in levels_entry.value:
            if isinstance(entry, Mapping):
                level_payload.append({str(k): _normalize_value(v) for k, v in entry.items()})
        if level_payload:
            policy_payload["levels"] = level_payload

    if not policy_payload:
        return None

    return {"multiscale": policy_payload}


def _merge_scene_metadata(
    snapshot_metadata: Mapping[str, Any],
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
) -> Optional[Dict[str, Any]]:
    metadata: Dict[str, Any] = dict(snapshot_metadata)

    volume_state = snapshot_volume_state(snapshot)
    if volume_state:
        metadata.setdefault("volume_state", {})
        metadata["volume_state"].update({str(k): _normalize_value(v) for k, v in volume_state.items()})

    return metadata or None
