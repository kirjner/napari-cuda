"""Shared helpers for server-side state update handling.

These utilities keep the state-channel dispatcher and future MCP surfaces
consistent by encapsulating all mutations of ``ServerSceneData`` associated
with control updates. They operate purely on the data bag plus simple
parameters so the websocket loop, worker bridge, or tests can invoke them
without reaching into the ``EGLHeadlessServer`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Mapping, Optional, Sequence
import logging
import time

from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.colormaps.colormap_utils import ensure_colormap

from napari.layers.image._image_constants import Interpolation as NapariInterpolation

from napari_cuda.server.state.scene_state import ServerSceneState
from napari_cuda.server.state.server_scene import (
    LayerControlState,
    ServerSceneData,
    default_layer_controls,
    get_control_meta,
    increment_server_sequence,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateUpdateResult:
    """Authoritative result from applying a state update."""

    scope: str
    target: str
    key: str
    value: Any
    server_seq: int
    intent_id: Optional[str] = None
    timestamp: Optional[float] = None
    axis_index: Optional[int] = None
    current_step: Optional[tuple[int, ...]] = None


def _update_latest_state(scene: ServerSceneData, lock: Lock, **updates: Any) -> ServerSceneState:
    """Apply ``updates`` to ``scene.latest_state`` under the given lock."""

    with lock:
        scene.latest_state = replace(scene.latest_state, **updates)
        if logging.getLogger(__name__).isEnabledFor(logging.INFO) and "current_step" in updates:
            logging.getLogger(__name__).info(
                "latest_state updated: current_step=%s",
                updates["current_step"],
            )
        return scene.latest_state


def resolve_axis_index(axis: object, meta: Mapping[str, Any], cur_len: int) -> Optional[int]:
    """Resolve *axis* (index or label) to an integer position."""

    try:
        if isinstance(axis, int):
            idx = int(axis)
            return idx if 0 <= idx < max(0, cur_len) else None
        if isinstance(axis, str):
            stripped = axis.strip()
            if stripped.isdigit():
                idx2 = int(stripped)
                return idx2 if 0 <= idx2 < max(0, cur_len) else None
            lowered = stripped.lower()
            order = meta.get("order") or []
            if isinstance(order, Sequence):
                lowered_order = [str(x).lower() for x in order]
                if lowered in lowered_order:
                    pos = lowered_order.index(lowered)
                    return pos if 0 <= pos < max(0, cur_len) else None
            labels = meta.get("axis_labels") or []
            if isinstance(labels, Sequence):
                lowered_labels = [str(x).lower() for x in labels]
                if lowered in lowered_labels:
                    pos = lowered_labels.index(lowered)
                    return pos if 0 <= pos < max(0, cur_len) else None
    except Exception:
        return None
    return None


def apply_dims_delta(
    scene: ServerSceneData,
    lock: Lock,
    meta: Mapping[str, Any],
    *,
    axis: object,
    step_delta: Optional[int],
    set_value: Optional[int],
) -> Optional[list[int]]:
    """Apply a dims step delta and return the updated step list (or ``None``)."""

    with lock:
        current = scene.latest_state.current_step
    try:
        ndim = int(meta.get("ndim") or (len(current) if current is not None else 0))
    except Exception:
        ndim = len(current) if current is not None else 0
    if ndim <= 0:
        ndim = len(current) if current is not None else 1

    step = list(int(x) for x in (current or (0,) * int(ndim)))
    if len(step) < int(ndim):
        step.extend([0] * (int(ndim) - len(step)))

    idx = resolve_axis_index(axis, meta, len(step))
    if idx is None:
        idx = 0 if step else None
    if idx is None:
        return None

    target = int(step[idx])
    if step_delta is not None:
        try:
            target = target + int(step_delta)
        except Exception:
            pass
    if set_value is not None:
        try:
            target = int(set_value)
        except Exception:
            pass

    shapes_raw = meta["level_shapes"]
    assert isinstance(shapes_raw, Sequence) and shapes_raw, "dims metadata missing level_shapes"
    level_idx_raw = meta.get("current_level", 0)
    level_idx = int(level_idx_raw)
    assert 0 <= level_idx < len(shapes_raw), "current_level out of bounds for level_shapes"
    shape_raw = shapes_raw[level_idx]
    assert isinstance(shape_raw, Sequence), "level_shapes entry must be sequence"
    assert idx < len(shape_raw), "axis index outside level_shapes entry"
    size_val = int(shape_raw[idx])
    assert size_val > 0, "level_shapes entries must be positive"
    target = max(0, min(size_val - 1, target))

    step[idx] = int(target)
    _update_latest_state(scene, lock, current_step=tuple(step))
    return step


def is_valid_render_mode(mode: str, allowed_modes: Sequence[str]) -> bool:
    return str(mode or "").lower() in {str(m).lower() for m in allowed_modes}


def normalize_clim(lo: object, hi: object) -> Optional[tuple[float, float]]:
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except Exception:
        return None
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    return (lo_f, hi_f)


def clamp_opacity(alpha: object) -> Optional[float]:
    try:
        val = float(alpha)
    except Exception:
        return None
    return max(0.0, min(1.0, val))


def clamp_sample_step(rel: object) -> Optional[float]:
    try:
        val = float(rel)
    except Exception:
        return None
    return max(0.1, min(4.0, val))


def clamp_level(level: object, levels: Sequence[Mapping[str, Any]]) -> Optional[int]:
    try:
        idx = int(level)
    except Exception:
        return None
    count = len(levels)
    if count > 0:
        return max(0, min(count - 1, idx))
    return max(0, idx)


def update_volume_mode(scene: ServerSceneData, lock: Lock, mode: str) -> None:
    scene.volume_state["mode"] = mode
    _update_latest_state(scene, lock, volume_mode=str(mode))


def update_volume_clim(scene: ServerSceneData, lock: Lock, lo: float, hi: float) -> None:
    scene.volume_state["clim"] = [float(lo), float(hi)]
    _update_latest_state(scene, lock, volume_clim=(float(lo), float(hi)))


def update_volume_colormap(scene: ServerSceneData, lock: Lock, name: str) -> None:
    scene.volume_state["colormap"] = name
    _update_latest_state(scene, lock, volume_colormap=str(name))


def update_volume_opacity(scene: ServerSceneData, lock: Lock, alpha: float) -> None:
    scene.volume_state["opacity"] = float(alpha)
    _update_latest_state(scene, lock, volume_opacity=float(alpha))


def update_volume_sample_step(scene: ServerSceneData, lock: Lock, rel: float) -> None:
    scene.volume_state["sample_step"] = float(rel)
    _update_latest_state(scene, lock, volume_sample_step=float(rel))


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}

_ALLOWED_INTERPOLATIONS = {mode.value for mode in NapariInterpolation}

_ALLOWED_BLENDING = {
    "opaque",
    "translucent",
    "additive",
    "minimum",
    "maximum",
    "average",
}

_ALLOWED_DEPICTION = {"volume", "plane"}

_ALLOWED_RENDERING = {
    "mip",
    "attenuated_mip",
    "translucent",
    "iso",
    "additive",
    "average",
}


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    raise ValueError("invalid boolean value")


def _normalize_opacity(value: object) -> float:
    val = float(value)
    return max(0.0, min(1.0, val))


def _normalize_gamma(value: object) -> float:
    val = float(value)
    if not val > 0.0:
        raise ValueError("gamma must be positive")
    return val


def _normalize_string(value: object, *, allowed: Optional[set[str]] = None) -> str:
    if isinstance(value, str):
        lowered = value.strip()
    elif isinstance(value, Mapping):
        name = value.get("name")  # type: ignore[arg-type]
        if not isinstance(name, str):
            raise ValueError("mapping requires a 'name' field")
        lowered = name.strip()
    else:
        lowered = str(value).strip()
    if lowered == "":
        raise ValueError("string update value may not be empty")
    normalized = lowered.lower()
    if allowed is not None and normalized not in allowed:
        raise ValueError(f"value '{normalized}' not in allowed set {sorted(allowed)}")
    return normalized


def _normalize_colormap(value: object) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in AVAILABLE_COLORMAPS:
            return ensure_colormap(stripped).name

        lowered = stripped.lower()
        for key in AVAILABLE_COLORMAPS:
            if key.lower() == lowered:
                return ensure_colormap(key).name

    cmap = ensure_colormap(value)
    return cmap.name


def _normalize_contrast_limits(value: object) -> list[float]:
    if isinstance(value, Mapping):
        low = value.get("lo")
        high = value.get("hi")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            raise ValueError("contrast_limits requires two values")
        low, high = value[0], value[1]
    else:
        raise ValueError("contrast_limits requires a sequence or mapping")

    pair = normalize_clim(low, high)
    if pair is None:
        raise ValueError("contrast_limits failed normalization")
    return [float(pair[0]), float(pair[1])]


def _normalize_layer_property(prop: str, value: object) -> Any:
    if prop == "visible":
        return _normalize_bool(value)
    if prop == "opacity":
        return _normalize_opacity(value)
    if prop == "blending":
        return _normalize_string(value, allowed=_ALLOWED_BLENDING)
    if prop == "interpolation":
        assert isinstance(value, str), "interpolation update must be a string"
        token = value.strip().lower()
        assert token in _ALLOWED_INTERPOLATIONS, f"invalid interpolation mode: {token}"
        return token
    if prop == "gamma":
        return _normalize_gamma(value)
    if prop == "contrast_limits":
        pair = _normalize_contrast_limits(value)
        return (float(pair[0]), float(pair[1]))
    if prop == "colormap":
        return _normalize_colormap(value)
    if prop == "depiction":
        return _normalize_string(value, allowed=_ALLOWED_DEPICTION)
    if prop == "rendering":
        return _normalize_string(value, allowed=_ALLOWED_RENDERING)
    if prop == "attenuation":
        return max(0.0, float(value))
    if prop == "iso_threshold":
        return float(value)
    raise KeyError(f"unsupported layer property '{prop}'")


def apply_layer_state_update(
    scene: ServerSceneData,
    lock: Lock,
    *,
    layer_id: str,
    prop: str,
    value: object,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> StateUpdateResult:
    """Normalize a layer update and persist it in the scene snapshot."""

    canonical = _normalize_layer_property(prop, value)

    with lock:
        control = scene.layer_controls.setdefault(layer_id, default_layer_controls())

        setattr(control, prop, canonical)

        latest = scene.latest_state
        pending = dict(latest.layer_updates or {})
        layer_updates = dict(pending.get(layer_id, {}))
        layer_updates[prop] = canonical
        pending[layer_id] = layer_updates
        scene.latest_state = replace(latest, layer_updates=pending)

        server_seq = increment_server_sequence(scene)

        meta = get_control_meta(scene, "layer", layer_id, prop)
        meta.last_server_seq = server_seq
        meta.last_timestamp = float(timestamp) if timestamp is not None else time.time()

    ts = float(timestamp) if timestamp is not None else time.time()

    return StateUpdateResult(
        scope="layer",
        target=layer_id,
        key=prop,
        value=canonical,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
    )

def apply_dims_state_update(
    scene: ServerSceneData,
    lock: Lock,
    meta: Mapping[str, Any],
    *,
    axis: object,
    prop: str,
    value: object,
    step_delta: Optional[int] = None,
    set_value: Optional[int] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Optional[StateUpdateResult]:
    """Apply a dims state update returning the updated step list."""

    with lock:
        current = scene.latest_state.current_step
    try:
        ndim = int(meta.get("ndim") or (len(current) if current is not None else 0))
    except Exception:
        ndim = len(current) if current is not None else 0
    if ndim <= 0:
        ndim = len(current) if current is not None else 1

    step = list(int(x) for x in (current or (0,) * int(ndim)))
    if len(step) < int(ndim):
        step.extend([0] * (int(ndim) - len(step)))

    idx = resolve_axis_index(axis, meta, len(step))
    if idx is None:
        idx = 0 if step else None
    if idx is None:
        return None

    axis_label = axis_label_from_meta(meta, idx)
    control_target = axis_label or str(idx)

    with lock:
        meta_entry = get_control_meta(scene, "dims", control_target, prop)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims apply incoming: axis=%s target=%s prop=%s value=%r step_delta=%r set_value=%r current_step=%s",
                idx,
                control_target,
                prop,
                value,
                step_delta,
                set_value,
                current,
            )

        target = int(step[idx])
        if value is not None:
            try:
                target = int(value)
            except Exception:
                pass
        if step_delta is not None:
            try:
                target = target + int(step_delta)
            except Exception:
                pass
        if set_value is not None:
            try:
                target = int(set_value)
            except Exception:
                pass

        shapes_raw = meta["level_shapes"]
        assert isinstance(shapes_raw, Sequence) and shapes_raw, "dims metadata missing level_shapes"
        level_idx_raw = meta.get("current_level", 0)
        level_idx = int(level_idx_raw)
        assert 0 <= level_idx < len(shapes_raw), "current_level out of bounds for level_shapes"
        shape_raw = shapes_raw[level_idx]
        assert isinstance(shape_raw, Sequence), "level_shapes entry must be sequence"
        assert idx < len(shape_raw), "axis index outside level_shapes entry"
        size_val = int(shape_raw[idx])
        assert size_val > 0, "level_shapes entries must be positive"
        target = max(0, min(size_val - 1, target))

        step[idx] = int(target)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dims apply result: axis=%s target=%s prop=%s stored=%s step=%s",
                idx,
                control_target,
                prop,
                step[idx],
                tuple(step),
            )

        latest = replace(scene.latest_state, current_step=tuple(step))
        scene.latest_state = latest

        server_seq = increment_server_sequence(scene)
        meta_entry.last_server_seq = server_seq
        meta_entry.last_timestamp = float(timestamp) if timestamp is not None else time.time()

    ts = float(timestamp) if timestamp is not None else time.time()

    return StateUpdateResult(
        scope="dims",
        target=control_target,
        key=prop,
        value=int(step[idx]),
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        axis_index=idx,
        current_step=tuple(step),
    )


def axis_label_from_meta(meta: Mapping[str, Any], idx: int) -> Optional[str]:
    """Resolve the preferred axis label for *idx*, if available."""

    try:
        order = meta.get("order")
        if isinstance(order, Sequence) and idx < len(order):
            label = order[idx]
            if isinstance(label, str) and label.strip():
                return str(label)
        labels = meta.get("axis_labels")
        if isinstance(labels, Sequence) and idx < len(labels):
            label = labels[idx]
            if isinstance(label, str) and label.strip():
                return str(label)
    except Exception:
        return None
    return None
