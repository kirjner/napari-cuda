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
from .server_scene import ServerSceneData


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

