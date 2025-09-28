"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from dataclasses import asdict

from napari_cuda.protocol.messages import LayerSpec, LayerUpdateMessage, SceneSpecMessage
from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import (
    CONTROL_KEYS,
    ServerSceneData,
    increment_dims_sequence,
    layer_controls_to_dict,
)


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
# Dims payload helpers


def build_dims_payload(
    scene: ServerSceneData,
    *,
    step_list: Sequence[int],
    last_client_id: Optional[str],
    meta: Mapping[str, Any],
    use_volume: bool,
    ack: bool = False,
    intent_seq: Optional[int] = None,
    server_seq: Optional[int] = None,
    source_client_id: Optional[str] = None,
    source_client_seq: Optional[int] = None,
    interaction_id: Optional[str] = None,
    phase: Optional[str] = None,
    control_versions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct a dims.update payload and cache it on the scene bag."""

    ndim = int(meta.get("ndim", len(step_list) if step_list else 1))
    assert ndim >= 1
    assert len(step_list) >= 1
    assert len(step_list) <= ndim

    ndisplay = 3 if (ndim >= 3 and bool(meta.get("volume"))) else 2

    order = tuple(meta.get("order", ()))
    ranges = meta.get("range")
    assert order is None or isinstance(order, Iterable)

    full = [0 for _ in range(ndim)]
    for idx in range(min(len(step_list), ndim)):
        full[idx] = int(step_list[idx])

    if len(step_list) == 1 and len(step_list) < ndim:
        val = int(step_list[0])
        if order:
            lower = [str(ax).lower() for ax in order]
            if "z" in lower:
                z_index = lower.index("z")
                if z_index < ndim:
                    full[z_index] = val
                else:
                    full[0] = val
            else:
                full[0] = val
        else:
            full[0] = val

    if ranges:
        assert isinstance(ranges, Sequence)
        for i in range(min(ndim, len(ranges))):
            lo_hi = ranges[i]
            if isinstance(lo_hi, Sequence) and len(lo_hi) >= 2:
                lo, hi = int(lo_hi[0]), int(lo_hi[1])
                if hi < lo:
                    lo, hi = hi, lo
                full[i] = max(lo, min(hi, full[i]))

    if server_seq is None:
        seq_val = increment_dims_sequence(scene, last_client_id)
    else:
        seq_val = int(server_seq)
        if last_client_id is not None:
            scene.last_dims_client_id = last_client_id
    payload: Dict[str, Any] = {
        "type": "dims.update",
        "seq": seq_val,
        "last_client_id": last_client_id,
        "current_step": full,
        "meta": {**meta, "ndisplay": ndisplay},
    }

    meta_out = payload["meta"]
    assert isinstance(meta_out, dict)

    multiscale_state = scene.multiscale_state or {}
    levels_state = multiscale_state.get("levels")
    levels: list[dict[str, Any]] = []
    if isinstance(levels_state, Sequence):
        for entry in levels_state:
            if isinstance(entry, Mapping):
                shape_seq = entry.get("shape")
                shape_vals = (
                    [int(s) for s in shape_seq]
                    if isinstance(shape_seq, Sequence)
                    else []
                )
                downsample_seq = entry.get("downsample")
                downsample_vals = (
                    list(downsample_seq)
                    if isinstance(downsample_seq, Sequence)
                    else []
                )
                levels.append(
                    {
                        "shape": shape_vals,
                        "downsample": downsample_vals,
                        "path": entry.get("path"),
                    }
                )

    if levels:
        level = int(multiscale_state.get("current_level", 0))
        level = max(0, min(level, len(levels) - 1))
        meta_out["level"] = level

        shape = levels[level].get("shape")
        if isinstance(shape, Sequence) and shape:
            meta_out["level_shape"] = [int(s) for s in shape]

        meta_out["multiscale"] = {
            "levels": levels,
            "current_level": level,
            "policy": multiscale_state.get("policy", "auto"),
            "index_space": multiscale_state.get("index_space", "base"),
        }

    if meta_out.get("range"):
        meta_out.setdefault("ranges", meta_out["range"])

    if ack:
        payload["ack"] = True
        if intent_seq is not None:
            payload["intent_seq"] = int(intent_seq)

    payload["server_seq"] = seq_val
    if source_client_id is not None:
        payload["source_client_id"] = source_client_id
    if source_client_seq is not None:
        payload["source_client_seq"] = int(source_client_seq)
    if interaction_id is not None:
        payload["interaction_id"] = interaction_id
    if phase is not None:
        payload["phase"] = phase
    if control_versions:
        sanitized = {
            str(prop): {
                key: value
                for key, value in entry.items()
                if value is not None
            }
            for prop, entry in control_versions.items()
        }
        if sanitized:
            payload["control_versions"] = sanitized

    meta_out["normalized"] = not bool(use_volume)
    if meta_out.get("multiscale"):
        levels_meta = meta_out["multiscale"]["levels"]
        level = int(meta_out["multiscale"].get("current_level", 0))
        if 0 <= level < len(levels_meta):
            shape = levels_meta[level].get("shape")
            if isinstance(shape, Sequence):
                sizes = [max(1, int(s)) for s in shape]
                meta_out["sizes"] = sizes
                meta_out["range"] = [[0, max(0, s - 1)] for s in sizes]

    scene.last_dims_payload = payload
    return payload


def build_layer_update_payload(
    scene: ServerSceneData,
    manager: ViewerSceneManager,
    *,
    layer_id: str,
    changes: Mapping[str, Any],
    intent_seq: Optional[int] = None,
    server_seq: Optional[int] = None,
    source_client_id: Optional[str] = None,
    source_client_seq: Optional[int] = None,
    control_versions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    interaction_id: Optional[str] = None,
    phase: Optional[str] = None,
    ack: bool = True,
) -> Dict[str, Any]:
    """Construct a layer.update payload reflecting *changes* for *layer_id*."""

    spec = manager.scene_spec()
    if spec is None or not spec.layers:
        raise RuntimeError("no scene spec available for layer update")

    layer_spec = None
    for layer in spec.layers:
        if layer.layer_id == layer_id:
            layer_spec = layer
            break
    if layer_spec is None:
        raise KeyError(f"unknown layer_id '{layer_id}'")

    # Merge canonical server-side state with delta to keep extras authoritative.
    base_dict = layer_spec.to_dict()
    extras = {
        key: value
        for key, value in dict(base_dict.get("extras") or {}).items()
        if key not in CONTROL_KEYS
    }
    control_state = scene.layer_controls.get(layer_id)
    control_map = (
        layer_controls_to_dict(control_state)
        if control_state is not None
        else {}
    )
    base_dict["extras"] = extras or None

    if "contrast_limits" in changes:
        pair = changes["contrast_limits"]
        base_dict["contrast_limits"] = [float(pair[0]), float(pair[1])]

    patch = LayerSpec.from_dict(base_dict)
    sanitized_versions: Optional[Dict[str, Dict[str, Any]]] = None
    if control_versions:
        sanitized_versions = {
            str(prop): {
                key: value
                for key, value in meta.items()
                if value is not None
            }
            for prop, meta in control_versions.items()
        }

    message = LayerUpdateMessage(
        layer=patch,
        partial=True,
        ack=ack,
        intent_seq=intent_seq,
        controls=control_map or None,
        server_seq=server_seq,
        source_client_id=source_client_id,
        source_client_seq=source_client_seq,
        interaction_id=interaction_id,
        phase=phase,
        control_versions=sanitized_versions,
    )
    return message.to_dict()
