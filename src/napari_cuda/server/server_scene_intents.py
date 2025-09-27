"""Shared helpers for server-side intent handling.

These utilities keep the state-channel dispatcher and future MCP surfaces
consistent by encapsulating all mutations of ``ServerSceneData`` associated
with user intents.  They operate purely on the data bag plus simple
parameters so the websocket loop, worker bridge, or tests can invoke them
without reaching into the ``EGLHeadlessServer`` implementation.
"""

from __future__ import annotations

from dataclasses import replace
from threading import Lock
from typing import Any, Mapping, Optional, Sequence

from .scene_state import ServerSceneState
from .server_scene import LayerControlState, ServerSceneData, default_layer_controls


def _update_latest_state(scene: ServerSceneData, lock: Lock, **updates: Any) -> ServerSceneState:
    """Apply ``updates`` to ``scene.latest_state`` under the given lock."""

    with lock:
        scene.latest_state = replace(scene.latest_state, **updates)
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


def apply_dims_intent(
    scene: ServerSceneData,
    lock: Lock,
    meta: Mapping[str, Any],
    *,
    axis: object,
    step_delta: Optional[int],
    set_value: Optional[int],
) -> Optional[list[int]]:
    """Apply a dims intent and return the updated step list (or ``None``)."""

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

    try:
        rng = meta.get("range")
        if isinstance(rng, Sequence) and idx < len(rng):
            bounds = rng[idx]
            if isinstance(bounds, Sequence) and len(bounds) >= 2:
                lo, hi = int(bounds[0]), int(bounds[1])
                if hi < lo:
                    lo, hi = hi, lo
                target = max(lo, min(hi, target))
    except Exception:
        pass

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

_ALLOWED_BLENDING = {
    "opaque",
    "translucent",
    "additive",
    "minimum",
    "maximum",
    "average",
}

_INTERPOLATION_MODES = {
    "nearest": "nearest",
    "linear": "linear",
    "bilinear": "linear",
    "bicubic": "cubic",
    "cubic": "cubic",
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
        raise ValueError("string intent value may not be empty")
    normalized = lowered.lower()
    if allowed is not None and normalized not in allowed:
        raise ValueError(f"value '{normalized}' not in allowed set {sorted(allowed)}")
    return normalized


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
        normal = _normalize_string(value, allowed=set(_INTERPOLATION_MODES.keys()))
        return _INTERPOLATION_MODES[normal]
    if prop == "gamma":
        return _normalize_gamma(value)
    if prop == "contrast_limits":
        pair = _normalize_contrast_limits(value)
        return (float(pair[0]), float(pair[1]))
    if prop == "colormap":
        return _normalize_string(value)
    if prop == "depiction":
        return _normalize_string(value, allowed=_ALLOWED_DEPICTION)
    if prop == "rendering":
        return _normalize_string(value, allowed=_ALLOWED_RENDERING)
    if prop == "attenuation":
        return max(0.0, float(value))
    if prop == "iso_threshold":
        return float(value)
    raise KeyError(f"unsupported layer property '{prop}'")


def apply_layer_intent(
    scene: ServerSceneData,
    lock: Lock,
    *,
    layer_id: str,
    prop: str,
    value: object,
) -> dict[str, Any]:
    """Normalize and stash a layer intent, returning applied updates."""

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

    return {prop: canonical}
