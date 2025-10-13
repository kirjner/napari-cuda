"""Ledger-backed reducers for server control updates."""

from __future__ import annotations

import logging
import uuid
import time
from dataclasses import replace
from threading import Lock
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from napari.layers.image._image_constants import Interpolation as NapariInterpolation
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.colormaps.colormap_utils import ensure_colormap

# ruff: noqa: TID252 - absolute imports enforced project-wide
from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.control.latest_intent import set_intent as latest_set_intent
from napari_cuda.server.control.state_models import ServerLedgerUpdate
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.scene import (
    LayerControlState,
    ServerSceneData,
    build_render_scene_state,
    default_layer_controls,
    get_control_meta,
    increment_server_sequence,
)
logger = logging.getLogger(__name__)


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


def _now(timestamp: Optional[float]) -> float:
    return float(timestamp) if timestamp is not None else time.time()


def is_valid_render_mode(mode: str, allowed_modes: Sequence[str]) -> bool:
    return str(mode or "").lower() in {str(m).lower() for m in allowed_modes}


def normalize_clim(lo: object, hi: object) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    return (lo_f, hi_f)


def clamp_opacity(alpha: object) -> float:
    val = float(alpha)
    return max(0.0, min(1.0, val))


def clamp_sample_step(rel: object) -> float:
    val = float(rel)
    return max(0.1, min(4.0, val))


def clamp_level(level: object, levels: Sequence[Mapping[str, Any]]) -> int:
    if isinstance(level, int):
        idx = level
    elif isinstance(level, str):
        stripped = level.strip()
        if not stripped:
            raise ValueError("level string may not be empty")
        if not stripped.lstrip("-").isdigit():
            raise ValueError(f"level '{level}' is not an integer")
        idx = int(stripped)
    else:
        raise TypeError(f"level must be int or str, got {type(level)!r}")
    count = len(levels)
    if count > 0:
        return max(0, min(count - 1, idx))
    return max(0, idx)


def resolve_axis_index(
    axis: object,
    *,
    order: Sequence[Any] | None,
    axis_labels: Sequence[Any] | None,
    ndim: int,
) -> Optional[int]:
    if isinstance(axis, int):
        return axis if 0 <= axis < max(0, ndim) else None
    if isinstance(axis, str):
        stripped = axis.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            idx = int(stripped)
            return idx if 0 <= idx < max(0, ndim) else None
        lowered = stripped.lower()
        if order is not None:
            lowered_order = [str(x).lower() for x in order]
            if lowered in lowered_order:
                position = lowered_order.index(lowered)
                return position if 0 <= position < max(0, ndim) else None
        if axis_labels is not None:
            lowered_labels = [str(x).lower() for x in axis_labels]
            if lowered in lowered_labels:
                position = lowered_labels.index(lowered)
                return position if 0 <= position < max(0, ndim) else None
    return None


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
    if val <= 0.0:
        raise ValueError("gamma must be positive")
    return val


def _normalize_string(value: object, *, allowed: Optional[set[str]] = None) -> str:
    if isinstance(value, str):
        lowered = value.strip()
    elif isinstance(value, Mapping):
        name = value.get("name")
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


def _normalize_contrast_limits(value: object) -> tuple[float, float]:
    if isinstance(value, Mapping):
        low = value.get("lo")
        high = value.get("hi")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            raise ValueError("contrast_limits requires two values")
        low, high = value[0], value[1]
    else:
        raise ValueError("contrast_limits requires a sequence or mapping")
    lo, hi = normalize_clim(low, high)
    return (float(lo), float(hi))


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


def reduce_layer_property(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    *,
    layer_id: str,
    prop: str,
    value: object | None = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.layer",
) -> ServerLedgerUpdate:
    canonical = _normalize_layer_property(prop, value)
    ts = _now(timestamp)

    with lock:
        control = store.layer_controls.setdefault(layer_id, default_layer_controls())
        setattr(control, prop, canonical)

        pending = store.pending_layer_updates.setdefault(layer_id, {})
        pending[prop] = canonical

        server_seq = increment_server_sequence(store)

        meta = get_control_meta(store, "layer", layer_id, prop)
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "layer",
        layer_id,
        prop,
        canonical,
        origin=origin,
        timestamp=ts,
    )

    volume_sync: list[tuple[str, Any]] = []
    if prop == "colormap":
        volume_state = store.volume_state
        if volume_state.get("colormap") != canonical:
            volume_state["colormap"] = canonical
            volume_sync.append(("colormap", canonical))
    elif prop == "contrast_limits":
        lo, hi = float(canonical[0]), float(canonical[1])
        volume_state = store.volume_state
        if volume_state.get("clim") != [lo, hi]:
            volume_state["clim"] = [lo, hi]
            volume_sync.append(("contrast_limits", (lo, hi)))
    elif prop == "opacity":
        alpha = float(canonical)
        volume_state = store.volume_state
        if volume_state.get("opacity") != alpha:
            volume_state["opacity"] = alpha
            volume_sync.append(("opacity", alpha))

    for volume_key, volume_value in volume_sync:
        ledger.record_confirmed(
            "volume",
            "main",
            volume_key,
            volume_value,
            origin=f"{origin}.volume",
            timestamp=ts,
        )

    return ServerLedgerUpdate(
        scope="layer",
        target=layer_id,
        key=prop,
        value=canonical,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_volume_render_mode(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    mode: str,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    normalized = str(mode)

    with lock:
        store.volume_state["mode"] = normalized
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "volume", "main", "render_mode")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "volume",
        "main",
        "render_mode",
        normalized,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="render_mode",
        value=normalized,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_volume_contrast_limits(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    lo: float,
    hi: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    pair = (float(lo), float(hi))

    with lock:
        store.volume_state["clim"] = [pair[0], pair[1]]
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "volume", "main", "contrast_limits")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "volume",
        "main",
        "contrast_limits",
        pair,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="contrast_limits",
        value=pair,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_volume_colormap(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    name: str,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    normalized = str(name)

    with lock:
        store.volume_state["colormap"] = normalized
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "volume", "main", "colormap")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "volume",
        "main",
        "colormap",
        normalized,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="colormap",
        value=normalized,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_volume_opacity(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    alpha: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    value = float(alpha)

    with lock:
        store.volume_state["opacity"] = value
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "volume", "main", "opacity")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "volume",
        "main",
        "opacity",
        value,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="opacity",
        value=value,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_volume_sample_step(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    sample_step: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    value = float(sample_step)

    with lock:
        store.volume_state["sample_step"] = value
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "volume", "main", "sample_step")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "volume",
        "main",
        "sample_step",
        value,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="sample_step",
        value=value,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def _axis_label_from_axes(
    order: Sequence[Any] | None,
    axis_labels: Sequence[Any] | None,
    idx: int,
) -> Optional[str]:
    if isinstance(order, Sequence) and idx < len(order):
        candidate = order[idx]
        if isinstance(candidate, str) and candidate.strip():
            return str(candidate)

    if isinstance(axis_labels, Sequence) and idx < len(axis_labels):
        candidate = axis_labels[idx]
        if isinstance(candidate, str) and candidate.strip():
            return str(candidate)

    return None


def _dims_entries_from_payload(
    payload: NotifyDimsPayload,
    *,
    axis_index: int,
    axis_target: str,
) -> list[tuple[Any, ...]]:
    entries: list[tuple[Any, ...]] = []

    entries.append(("view", "main", "ndisplay", int(payload.ndisplay)))
    if payload.displayed is not None:
        entries.append(("view", "main", "displayed", tuple(int(idx) for idx in payload.displayed)))

    current_step = tuple(int(v) for v in payload.current_step)
    entries.append(
        (
            "dims",
            "main",
            "current_step",
            current_step,
            {"axis_index": axis_index, "axis_target": axis_target},
        )
    )

    entries.append(("dims", "main", "mode", str(payload.mode)))
    if payload.order is not None:
        entries.append(("dims", "main", "order", tuple(int(idx) for idx in payload.order)))
    if payload.axis_labels is not None:
        entries.append(("dims", "main", "axis_labels", tuple(str(label) for label in payload.axis_labels)))
    if getattr(payload, "labels", None) is not None:
        entries.append(("dims", "main", "labels", tuple(str(label) for label in payload.labels)))

    entries.append(("multiscale", "main", "level", int(payload.current_level)))
    entries.append(
        (
            "multiscale",
            "main",
            "levels",
            tuple(dict(level) for level in payload.levels),
        )
    )
    entries.append(
        (
            "multiscale",
            "main",
            "level_shapes",
            tuple(tuple(int(dim) for dim in shape) for shape in payload.level_shapes),
        )
    )
    entries.append(("multiscale", "main", "downgraded", payload.downgraded))

    return entries


def _ledger_dims_payload(ledger: ServerStateLedger) -> NotifyDimsPayload:
    snapshot = ledger.snapshot()

    def require(scope: str, key: str) -> Any:
        entry = snapshot.get((scope, "main", key))
        if entry is None:
            raise AssertionError(f"ledger missing {scope}/{key}")
        return entry.value

    def optional(scope: str, key: str) -> Any:
        entry = snapshot.get((scope, "main", key))
        return None if entry is None else entry.value

    current_step_raw = require("dims", "current_step")
    current_step = tuple(int(v) for v in current_step_raw)
    level_shapes_raw = require("multiscale", "level_shapes")
    level_shapes = tuple(
        tuple(int(dim) for dim in shape) for shape in level_shapes_raw
    )
    levels_raw = require("multiscale", "levels")
    levels = tuple(dict(level) for level in levels_raw)
    current_level = int(require("multiscale", "level"))
    downgraded_raw = optional("multiscale", "downgraded")
    downgraded = None if downgraded_raw is None else bool(downgraded_raw)
    mode = str(require("dims", "mode"))
    ndisplay = int(require("view", "ndisplay"))

    axis_labels_raw = optional("dims", "axis_labels")
    axis_labels = (
        tuple(str(label) for label in axis_labels_raw)
        if axis_labels_raw is not None
        else None
    )

    order_raw = optional("dims", "order")
    order = (
        tuple(int(idx) for idx in order_raw)
        if order_raw is not None
        else None
    )

    displayed_raw = optional("view", "displayed")
    displayed = (
        tuple(int(idx) for idx in displayed_raw)
        if displayed_raw is not None
        else None
    )

    labels_raw = optional("dims", "labels")
    labels = (
        tuple(str(label) for label in labels_raw)
        if labels_raw is not None
        else None
    )

    return NotifyDimsPayload(
        current_step=current_step,
        level_shapes=level_shapes,
        levels=levels,
        current_level=current_level,
        downgraded=downgraded,
        mode=mode,
        ndisplay=ndisplay,
        axis_labels=axis_labels,
        order=order,
        displayed=displayed,
        labels=labels,
    )


def reduce_bootstrap_state(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    *,
    step: Sequence[int],
    axis_labels: Sequence[str],
    order: Sequence[int],
    level_shapes: Sequence[Sequence[int]],
    levels: Sequence[Mapping[str, Any]],
    current_level: int,
    ndisplay: int,
    origin: str = "server.bootstrap",
    timestamp: Optional[float] = None,
) -> list[ServerLedgerUpdate]:
    """Seed reducer intents from a bootstrap scene snapshot."""

    ts = _now(timestamp)
    resolved_step = tuple(int(v) for v in step)
    resolved_axis_labels = tuple(str(label) for label in axis_labels)
    resolved_order = tuple(int(v) for v in order)
    resolved_level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in level_shapes)
    resolved_levels = tuple(dict(level) for level in levels)
    resolved_current_level = int(current_level)
    resolved_ndisplay = 3 if int(ndisplay) >= 3 else 2
    mode_value = "volume" if resolved_ndisplay >= 3 else "plane"

    ndim = max(len(resolved_axis_labels), len(resolved_step), len(resolved_order), len(resolved_level_shapes[resolved_current_level]) if resolved_level_shapes else 0)
    if ndim == 0:
        ndim = 1

    if not resolved_order:
        resolved_order = tuple(range(ndim))

    axis_index = 0 if resolved_step else 0
    axis_target = (
        resolved_axis_labels[axis_index]
        if resolved_axis_labels and axis_index < len(resolved_axis_labels)
        else str(axis_index)
    )

    displayed = resolved_order[-resolved_ndisplay:] if resolved_order else tuple(range(max(ndim - resolved_ndisplay, 0), ndim))
    displayed_tuple = tuple(int(v) for v in displayed)

    with lock:
        store.last_scene_seq = increment_server_sequence(store)
        dims_seq = store.last_scene_seq
        dims_meta = get_control_meta(store, "dims", axis_target, "index")
        dims_meta.last_server_seq = dims_seq
        dims_meta.last_timestamp = ts

        store.multiscale_state["current_level"] = resolved_current_level
        store.multiscale_state["levels"] = [dict(level) for level in levels]
        store.multiscale_state["level_shapes"] = [list(shape) for shape in resolved_level_shapes]
        if "downgraded" in store.multiscale_state:
            store.multiscale_state["downgraded"] = False

        store.last_scene_seq = increment_server_sequence(store)
        view_seq = store.last_scene_seq
        view_meta = get_control_meta(store, "view", "main", "ndisplay")
        view_meta.last_server_seq = view_seq
        view_meta.last_timestamp = ts
        store.use_volume = bool(mode_value == "volume")

        dims_payload = NotifyDimsPayload(
            current_step=resolved_step,
            level_shapes=resolved_level_shapes,
            levels=resolved_levels,
            current_level=resolved_current_level,
            downgraded=False,
            mode=mode_value,
            ndisplay=resolved_ndisplay,
            axis_labels=resolved_axis_labels if resolved_axis_labels else None,
            order=resolved_order if resolved_order else None,
            displayed=displayed_tuple if displayed_tuple else None,
            labels=None,
        )

        entries = _dims_entries_from_payload(
            dims_payload,
            axis_index=axis_index,
            axis_target=axis_target,
        )
        axis_index_metadata = {
            "axis_index": axis_index,
            "axis_target": axis_target,
            "server_seq": dims_seq,
        }
        axis_index_value = int(resolved_step[axis_index]) if resolved_step else 0
        entries.append(
            (
                "dims",
                axis_target,
                "index",
                axis_index_value,
                axis_index_metadata,
            )
        )

    ledger.batch_record_confirmed(
        entries,
        origin=origin,
        timestamp=ts,
        dedupe=False,
    )

    with lock:
        store.latest_state = build_render_scene_state(
            ledger,
            store,
        )
    logger.debug(
        "bootstrap ledger seeded step=%s ndisplay=%d level=%d mode=%s",
        resolved_step,
        resolved_ndisplay,
        resolved_current_level,
        mode_value,
        )

    latest_set_intent("dims", axis_target, resolved_step, int(dims_seq))
    latest_set_intent("view", "ndisplay", resolved_ndisplay, int(view_seq))
    latest_set_intent("multiscale", "level", resolved_current_level, int(view_seq))
    logger.debug(
        "bootstrap intents seeded axis=%s step=%s ndisplay=%d level=%d seqs(dims=%d,view=%d)",
        axis_target,
        resolved_step,
        resolved_ndisplay,
        resolved_current_level,
        dims_seq,
        view_seq,
    )

    dims_intent_id = f"dims-bootstrap-{uuid.uuid4().hex}"
    view_intent_id = f"view-bootstrap-{uuid.uuid4().hex}"

    dims_update = ServerLedgerUpdate(
        scope="dims",
        target=axis_target,
        key="index",
        value=int(resolved_step[axis_index]) if resolved_step else 0,
        server_seq=dims_seq,
        intent_id=dims_intent_id,
        timestamp=ts,
        axis_index=axis_index,
        current_step=resolved_step,
        origin=origin,
    )
    view_update = ServerLedgerUpdate(
        scope="view",
        target="main",
        key="ndisplay",
        value=resolved_ndisplay,
        server_seq=view_seq,
        intent_id=view_intent_id,
        timestamp=ts,
        origin=origin,
    )

    return [dims_update, view_update]


def reduce_dims_update(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    *,
    axis: object,
    prop: str,
    value: object | None = None,
    step_delta: Optional[int] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.dims",
) -> ServerLedgerUpdate:
    if value is None and step_delta is None:
        raise ValueError("dims update requires value or step_delta")

    ts = _now(timestamp)

    payload = _ledger_dims_payload(ledger)

    step = [int(v) for v in payload.current_step]
    ndim = len(step)
    assert ndim > 0, "ledger dims metadata missing dimensions"

    if len(step) < ndim:
        step.extend([0] * (ndim - len(step)))

    idx = resolve_axis_index(
        axis,
        order=payload.order,
        axis_labels=payload.axis_labels,
        ndim=len(step),
    )
    if idx is None:
        raise ValueError("unable to resolve axis index for dims update")

    axis_label = _axis_label_from_axes(payload.order, payload.axis_labels, idx)
    control_target = axis_label or str(idx)

    target = int(step[idx])
    if value is not None:
        target = int(value)
    if step_delta is not None:
        target += int(step_delta)

    shapes_raw = payload.level_shapes
    assert shapes_raw, "ledger dims metadata missing level_shapes"
    level_idx = int(payload.current_level)
    assert 0 <= level_idx < len(shapes_raw), "current_level out of bounds for level_shapes"
    shape_raw = shapes_raw[level_idx]
    assert isinstance(shape_raw, Sequence), "level_shapes entry must be sequence"
    assert idx < len(shape_raw), "axis index outside level_shapes entry"
    size_val = int(shape_raw[idx])
    assert size_val > 0, "level_shapes entries must be positive"
    target = max(0, min(size_val - 1, target))

    step[idx] = int(target)
    requested_step = tuple(int(v) for v in step)
    resolved_intent_id = intent_id or f"dims-{uuid.uuid4().hex}"

    step_metadata = {
        "axis_index": int(idx),
        "axis_target": control_target,
    }

    with lock:
        store.last_scene_seq = increment_server_sequence(store)
        meta_entry = get_control_meta(store, "dims", control_target, prop)
        meta_entry.last_server_seq = store.last_scene_seq
        meta_entry.last_timestamp = ts
        server_seq = store.last_scene_seq
        snapshot = store.latest_state
        store.latest_state = replace(snapshot, current_step=requested_step)

    ledger.record_confirmed(
        "dims",
        "main",
        "current_step",
        requested_step,
        origin=origin,
        timestamp=ts,
        metadata=step_metadata,
    )

    latest_set_intent("dims", control_target, requested_step, int(server_seq))
    logger.debug(
        "dims intent updated axis=%s prop=%s step=%s seq=%d origin=%s",
        control_target,
        prop,
        requested_step,
        server_seq,
        origin,
    )

    return ServerLedgerUpdate(
        scope="dims",
        target=control_target,
        key=prop,
        value=int(step[idx]),
        server_seq=server_seq,
        intent_id=resolved_intent_id,
        timestamp=ts,
        axis_index=idx,
        current_step=requested_step,
        origin=origin,
    )


def reduce_view_update(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    *,
    ndisplay: Optional[int] = None,
    order: Optional[Sequence[int]] = None,
    displayed: Optional[Sequence[int]] = None,
    mode: Optional[str] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.view",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    dims_payload = _ledger_dims_payload(ledger)

    resolved_ndisplay: Optional[int] = None
    if ndisplay is not None:
        resolved_ndisplay = 3 if int(ndisplay) >= 3 else 2
    else:
        entry = ledger.get("view", "main", "ndisplay")
        if entry is not None and isinstance(entry.value, int):
            resolved_ndisplay = int(entry.value)
    if resolved_ndisplay is None:
        resolved_ndisplay = 2

    target_ndisplay = resolved_ndisplay
    resolved_intent_id = intent_id or f"view-{uuid.uuid4().hex}"

    target_name = "main"

    with lock:
        previous_use_volume = bool(store.use_volume)
        store.last_scene_seq = increment_server_sequence(store)
        server_seq = store.last_scene_seq
        meta_entry = get_control_meta(store, "view", "main", "ndisplay")
        meta_entry.last_server_seq = server_seq
        meta_entry.last_timestamp = ts
        store.use_volume = bool(target_ndisplay == 3)

    latest_set_intent("view", "ndisplay", target_ndisplay, int(server_seq))
    logger.debug(
        "view intent updated ndisplay=%d seq=%d origin=%s",
        target_ndisplay,
        server_seq,
        origin,
    )

    ledger.record_confirmed(
        "view",
        "main",
        "ndisplay",
        int(target_ndisplay),
        origin=origin,
        timestamp=ts,
    )

    existing_order = tuple(int(idx) for idx in order) if order is not None else (
        tuple(int(idx) for idx in dims_payload.order) if dims_payload.order is not None else tuple()
    )
    ndim = len(dims_payload.current_step) if dims_payload.current_step else 0
    if ndim <= 0:
        ndim = len(existing_order)
    if ndim <= 0:
        ndim = max(int(target_ndisplay), 1)

    normalized_order: list[int] = []
    seen_axes: set[int] = set()
    for axis in existing_order:
        axis_int = int(axis)
        if 0 <= axis_int < ndim and axis_int not in seen_axes:
            normalized_order.append(axis_int)
            seen_axes.add(axis_int)
    for axis_int in range(ndim):
        if axis_int not in seen_axes:
            normalized_order.append(axis_int)
            seen_axes.add(axis_int)

    if not normalized_order:
        normalized_order = list(range(ndim))

    order_value = tuple(normalized_order)
    if displayed is not None:
        displayed_value = tuple(int(idx) for idx in displayed)
    else:
        displayed_count = min(len(order_value), max(1, int(target_ndisplay)))
        displayed_value = tuple(order_value[-displayed_count:]) if displayed_count > 0 else tuple()

    if mode is not None:
        mode_value = str(mode)
    else:
        mode_value = "volume" if target_ndisplay == 3 else "plane"

    ledger.record_confirmed(
        "dims",
        "main",
        "order",
        order_value,
        origin=origin,
        timestamp=ts,
    )
    ledger.record_confirmed(
        "dims",
        "main",
        "mode",
        mode_value,
        origin=origin,
        timestamp=ts,
    )
    ledger.record_confirmed(
        "view",
        "main",
        "displayed",
        displayed_value,
        origin=origin,
        timestamp=ts,
    )

    with lock:
        snapshot = store.latest_state
        store.latest_state = replace(
            snapshot,
            ndisplay=int(target_ndisplay),
            order=order_value,
            displayed=displayed_value,
            dims_mode=mode_value,
        )
    return ServerLedgerUpdate(
        scope="view",
        target="main",
        key="ndisplay",
        value=int(target_ndisplay),
        server_seq=server_seq,
        intent_id=resolved_intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_multiscale_policy(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    policy: str,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.multiscale",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    normalized = str(policy)

    with lock:
        store.multiscale_state["policy"] = normalized
        server_seq = increment_server_sequence(store)
        meta = get_control_meta(store, "multiscale", "main", "policy")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

    ledger.record_confirmed(
        "multiscale",
        "main",
        "policy",
        normalized,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="multiscale",
        target="main",
        key="policy",
        value=normalized,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
    )


def reduce_level_update(
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    *,
    applied: Mapping[str, Any] | object,
    downgraded: Optional[bool] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.multiscale",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)

    def _applied_value(attr: str, default: Any = None) -> Any:
        if hasattr(applied, attr):
            return getattr(applied, attr)
        if isinstance(applied, Mapping):
            return applied.get(attr, default)
        return default

    level_raw = _applied_value("level")
    if level_raw is None:
        entry = ledger.get("multiscale", "main", "level")
        if entry is not None and isinstance(entry.value, int):
            level_raw = entry.value
    if level_raw is None:
        raise ValueError("level update requires a level index")
    level = int(level_raw)

    step_raw = _applied_value("step")
    step_tuple: Optional[tuple[int, ...]] = None
    if step_raw is not None:
        step_tuple = tuple(int(v) for v in step_raw)

    shape_raw = _applied_value("shape")
    shape_tuple: Optional[tuple[int, ...]] = None
    if shape_raw is not None:
        shape_tuple = tuple(int(v) for v in shape_raw)

    dims_payload = _ledger_dims_payload(ledger)
    current_step = tuple(int(v) for v in dims_payload.current_step)
    if step_tuple is None:
        step_tuple = current_step

    level_shapes_payload: list[tuple[int, ...]] = []
    if dims_payload.level_shapes:
        level_shapes_payload = [tuple(int(v) for v in shape) for shape in dims_payload.level_shapes]
    if shape_tuple is not None:
        while len(level_shapes_payload) <= level:
            level_shapes_payload.append(shape_tuple)
        level_shapes_payload[level] = shape_tuple
    updated_level_shapes = tuple(level_shapes_payload) if level_shapes_payload else tuple()

    with lock:
        store.last_scene_seq = increment_server_sequence(store)
        server_seq = store.last_scene_seq
        meta = get_control_meta(store, "multiscale", "main", "level")
        meta.last_server_seq = server_seq
        meta.last_timestamp = ts

        snapshot = store.latest_state
        store.latest_state = replace(
            snapshot,
            current_level=level,
            current_step=step_tuple,
            level_shapes=updated_level_shapes or snapshot.level_shapes,
        )

        store.multiscale_state["current_level"] = level
        if updated_level_shapes:
            store.multiscale_state["level_shapes"] = [
                [int(dim) for dim in shape]
                for shape in updated_level_shapes
            ]
        levels_state = store.multiscale_state.get("levels")
        if isinstance(levels_state, list) and 0 <= level < len(levels_state):
            entry = dict(levels_state[level])
            if shape_tuple is not None:
                entry["shape"] = [int(v) for v in shape_tuple]
            levels_state[level] = entry
        if downgraded is not None:
            store.multiscale_state["downgraded"] = bool(downgraded)

    latest_set_intent("multiscale", "level", level, int(server_seq))
    logger.debug(
        "multiscale intent updated level=%d seq=%d origin=%s",
        level,
        server_seq,
        origin,
    )

    batch_entries: list[tuple[Any, ...]] = [
        ("multiscale", "main", "level", level),
    ]
    if updated_level_shapes:
        batch_entries.append(
            (
                "multiscale",
                "main",
                "level_shapes",
                updated_level_shapes,
            )
        )
    if downgraded is not None:
        batch_entries.append(
            (
                "multiscale",
                "main",
                "downgraded",
                bool(downgraded),
            )
        )

    step_metadata = {"source": "worker.level_update", "level": level}
    batch_entries.append(
        (
            "dims",
            "main",
            "current_step",
            step_tuple,
            step_metadata,
        )
    )

    ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="multiscale",
        target="main",
        key="level",
        value=level,
        server_seq=server_seq,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        current_step=step_tuple,
    )


# camera reducers -------------------------------------------------------------

def reduce_camera_update(
    ledger: ServerStateLedger,
    *,
    center: Optional[Sequence[float]] = None,
    zoom: Optional[float] = None,
    angles: Optional[Sequence[float]] = None,
    distance: Optional[float] = None,
    fov: Optional[float] = None,
    rect: Optional[Sequence[float]] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.camera",
) -> Dict[str, Any]:
    if (
        center is None
        and zoom is None
        and angles is None
        and distance is None
        and fov is None
        and rect is None
    ):
        raise ValueError("camera reducer requires at least one property")

    ts = _now(timestamp)
    ack: Dict[str, Any] = {}

    pending: list[tuple[str, str, str, Any]] = []

    if center is not None:
        normalized_center = tuple(float(c) for c in center)
        pending.append(("camera", "main", "center", normalized_center))
        ack["center"] = [float(c) for c in normalized_center]

    if zoom is not None:
        zoom_value = float(zoom)
        pending.append(("camera", "main", "zoom", zoom_value))
        ack["zoom"] = zoom_value

    if angles is not None:
        if len(angles) < 3:
            raise ValueError("camera angles require three components")
        normalized_angles = (
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
        )
        pending.append(("camera", "main", "angles", normalized_angles))
        ack["angles"] = [float(a) for a in normalized_angles]

    if distance is not None:
        distance_val = float(distance)
        pending.append(("camera", "main", "distance", distance_val))
        ack["distance"] = distance_val

    if fov is not None:
        fov_val = float(fov)
        pending.append(("camera", "main", "fov", fov_val))
        ack["fov"] = fov_val

    if rect is not None:
        if len(rect) < 4:
            raise ValueError("camera rect requires four components")
        normalized_rect = (
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )
        pending.append(("camera", "main", "rect", normalized_rect))
        ack["rect"] = [float(v) for v in normalized_rect]

    for scope, target, key, value in pending:
        ledger.record_confirmed(scope, target, key, value, origin=origin, timestamp=ts)

    return ack


StateUpdateResult = ServerLedgerUpdate

__all__ = [
    "ServerLedgerUpdate",
    "StateUpdateResult",
    "clamp_level",
    "clamp_opacity",
    "clamp_sample_step",
    "is_valid_render_mode",
    "normalize_clim",
    "reduce_camera_update",
    "reduce_bootstrap_state",
    "reduce_dims_update",
    "reduce_layer_property",
    "reduce_level_update",
    "reduce_multiscale_policy",
    "reduce_volume_colormap",
    "reduce_volume_contrast_limits",
    "reduce_volume_opacity",
    "reduce_volume_render_mode",
    "reduce_volume_sample_step",
    "reduce_view_update",
    "resolve_axis_index",
]
