"""Pure helpers for building scene specs and dims payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from dataclasses import asdict

from napari_cuda.protocol.messages import LayerSpec, LayerUpdateMessage, SceneSpecMessage
from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import ServerSceneData, increment_dims_sequence


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
    worker_scene_source: Optional[Any],
    use_volume: bool,
    ack: bool = False,
    intent_seq: Optional[int] = None,
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

    seq_val = increment_dims_sequence(scene, last_client_id)
    payload: Dict[str, Any] = {
        "type": "dims.update",
        "seq": seq_val,
        "last_client_id": last_client_id,
        "current_step": full,
        "meta": {**meta, "ndisplay": ndisplay},
    }

    meta_out = payload["meta"]
    assert isinstance(meta_out, dict)

    src = worker_scene_source
    if src is not None:
        level = int(getattr(src, "current_level", 0))
        meta_out["level"] = level

        descs = getattr(src, "level_descriptors", [])
        if isinstance(descs, Sequence) and descs:
            level = max(0, min(level, len(descs) - 1))
            shape = getattr(descs[level], "shape", None)
            if shape is not None:
                meta_out["level_shape"] = [int(s) for s in shape]
            levels_meta = []
            for desc in descs:
                levels_meta.append(
                    {
                        "shape": [int(s) for s in getattr(desc, "shape", [])],
                        "downsample": list(getattr(desc, "downsample", [])),
                        "path": getattr(desc, "path", None),
                    }
                )
            meta_out["multiscale"] = {
                "levels": levels_meta,
                "current_level": level,
                "policy": scene.multiscale_state.get("policy", "auto"),
                "index_space": "base",
            }

    if meta_out.get("range"):
        meta_out.setdefault("ranges", meta_out["range"])

    if ack:
        payload["ack"] = True
        if intent_seq is not None:
            payload["intent_seq"] = int(intent_seq)

    meta_out["normalized"] = not bool(use_volume)
    if src is not None and meta_out.get("multiscale"):
        levels = meta_out["multiscale"]["levels"]
        level = int(meta_out["multiscale"].get("current_level", 0))
        if 0 <= level < len(levels):
            shape = levels[level].get("shape")
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
    extras = dict(base_dict.get("extras") or {})
    control_state = scene.layer_controls.get(layer_id)
    control_map: dict[str, Any] = asdict(control_state) if control_state is not None else {}
    if control_map:
        extras.update({k: v for k, v in control_map.items() if v is not None})
    extras.update(changes)
    base_dict["extras"] = extras

    if "contrast_limits" in changes:
        pair = changes["contrast_limits"]
        base_dict["contrast_limits"] = [float(pair[0]), float(pair[1])]

    patch = LayerSpec.from_dict(base_dict)
    message = LayerUpdateMessage(layer=patch, partial=True)
    payload = message.to_dict()
    payload["ack"] = True
    if intent_seq is not None:
        payload["intent_seq"] = int(intent_seq)
    if control_map:
        payload["controls"] = control_map
    return payload
