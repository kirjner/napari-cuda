"""Ledger-backed reducers for server control updates."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Optional

from napari.layers.image._image_constants import (
    Interpolation as NapariInterpolation,
    ImageProjectionMode as NapariProjectionMode,
)
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.colormaps.colormap_utils import ensure_colormap

# ruff: noqa: TID252 - absolute imports enforced project-wide
from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.control.state_models import ServerLedgerUpdate
from napari_cuda.server.control.transactions import (
    apply_bootstrap_transaction,
    apply_camera_update_transaction,
    apply_dims_step_transaction,
    apply_layer_property_transaction,
    apply_level_switch_transaction,
    apply_plane_restore_transaction,
    apply_view_toggle_transaction,
    apply_volume_restore_transaction,
)
from napari_cuda.server.scene import PlaneState, RenderMode, VolumeState
from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)
from napari_cuda.server.control.transactions.thumbnail import apply_thumbnail_capture
from napari_cuda.shared.axis_spec import (
    AxisExtent,
    AxisRole,
    AxisSpec,
    WorldSpan,
    axis_spec_from_payload,
    axis_spec_to_payload,
    with_updated_margins,
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

_ALLOWED_PROJECTION = {mode.value for mode in NapariProjectionMode}

_ROLE_ALIASES: dict[str, AxisRole] = {
    "x": "x",
    "lon": "x",
    "u": "x",
    "y": "y",
    "lat": "y",
    "v": "y",
    "z": "z",
    "depth": "depth",
    "d": "depth",
    "time": "time",
    "t": "time",
    "channel": "channel",
    "c": "channel",
}


def _snapshot_optional(
    snapshot: Mapping[tuple[str, str, str], LedgerEntry],
    scope: str,
    key: str,
) -> Any:
    entry = snapshot.get((scope, "main", key))
    if entry is None:
        return None
    return entry.value


def _infer_axis_role(label: str, *, fallback: AxisRole = "unknown") -> AxisRole:
    text = str(label or "").strip().lower()
    if not text:
        return fallback
    return _ROLE_ALIASES.get(text, fallback)


def _build_axis_spec_from_components(
    *,
    axis_labels: Sequence[str] | None,
    order: Sequence[int] | None,
    displayed: Sequence[int] | None,
    current_step: Sequence[int] | None,
    level_shapes: Sequence[Sequence[int]],
    current_level: int,
    ndisplay: int,
    margin_left: Sequence[float] | None,
    margin_right: Sequence[float] | None,
    plane_mode: bool,
    prior_spec: AxisSpec | None = None,
) -> AxisSpec:
    prior_axes = {axis.index: axis for axis in prior_spec.axes} if prior_spec is not None else {}

    level_shapes_seq: list[tuple[int, ...]] = [
        tuple(int(dim) for dim in shape) for shape in level_shapes
    ]
    if not level_shapes_seq and prior_spec is not None:
        level_shapes_seq = [tuple(int(dim) for dim in shape) for shape in prior_spec.level_shapes]

    if not level_shapes_seq:
        inferred_ndim = len(current_step) if current_step else len(axis_labels or ()) or len(prior_axes) or 1
        level_shapes_seq = [tuple(0 for _ in range(inferred_ndim))]

    level_index = max(0, min(int(current_level), len(level_shapes_seq) - 1))
    base_shape = level_shapes_seq[level_index]
    ndim = len(base_shape) if base_shape else (
        len(current_step) if current_step else len(axis_labels or ()) or len(prior_axes) or 1
    )

    order_tuple = tuple(int(idx) for idx in order) if order else tuple(range(ndim))
    displayed_tuple = (
        tuple(int(idx) for idx in displayed)
        if displayed
        else tuple(order_tuple[-min(len(order_tuple), max(1, int(ndisplay))):])
    )

    step_values = list(current_step[:ndim]) if current_step else [0] * ndim
    if len(step_values) < ndim:
        step_values.extend([0] * (ndim - len(step_values)))

    def _prior_margin(idx: int, side: str) -> float:
        prior_axis = prior_axes.get(idx)
        if prior_axis is None:
            return 0.0
        return float(prior_axis.margin_left_world if side == "left" else prior_axis.margin_right_world)

    left_values = list(margin_left[:ndim]) if margin_left is not None else []
    right_values = list(margin_right[:ndim]) if margin_right is not None else []
    if len(left_values) < ndim:
        left_values.extend(_prior_margin(idx, "left") for idx in range(len(left_values), ndim))
    if len(right_values) < ndim:
        right_values.extend(_prior_margin(idx, "right") for idx in range(len(right_values), ndim))

    labels: list[str] = []
    for axis_idx in range(ndim):
        if axis_labels and axis_idx < len(axis_labels) and str(axis_labels[axis_idx]).strip():
            labels.append(str(axis_labels[axis_idx]))
        elif axis_idx in prior_axes:
            labels.append(prior_axes[axis_idx].label)
        else:
            labels.append(f"axis-{axis_idx}")

    axes: list[AxisExtent] = []
    for axis_idx in range(ndim):
        prior = prior_axes.get(axis_idx)
        role = prior.role if prior and prior.role != "unknown" else _infer_axis_role(labels[axis_idx])
        order_pos = order_tuple.index(axis_idx) if axis_idx in order_tuple else axis_idx

        per_level_steps: list[int] = []
        per_level_world: list[WorldSpan | None] = []
        for level_idx, shape in enumerate(level_shapes_seq):
            step_count = int(shape[axis_idx]) if axis_idx < len(shape) else 0
            per_level_steps.append(step_count)
            prior_span = None
            if prior and level_idx < len(prior.per_level_world):
                prior_span = prior.per_level_world[level_idx]
            if prior_span is not None:
                per_level_world.append(prior_span)
            elif step_count > 0:
                stop = float(max(0, step_count - 1))
                per_level_world.append(WorldSpan(start=0.0, stop=stop, step=1.0, scale=None))
            else:
                per_level_world.append(None)

        if margin_left is None and prior is not None:
            margin_left_world = float(prior.margin_left_world)
            margin_left_steps = float(prior.margin_left_steps)
        else:
            margin_left_world = float(left_values[axis_idx])
            margin_left_steps = (
                float(prior.margin_left_steps)
                if prior is not None
                else float(left_values[axis_idx])
            )

        if margin_right is None and prior is not None:
            margin_right_world = float(prior.margin_right_world)
            margin_right_steps = float(prior.margin_right_steps)
        else:
            margin_right_world = float(right_values[axis_idx])
            margin_right_steps = (
                float(prior.margin_right_steps)
                if prior is not None
                else float(right_values[axis_idx])
            )

        axes.append(
            AxisExtent(
                index=axis_idx,
                label=labels[axis_idx],
                role=role,
                displayed=axis_idx in displayed_tuple,
                order_pos=order_pos,
                current_step=int(step_values[axis_idx]),
                margin_left_world=margin_left_world,
                margin_right_world=margin_right_world,
                margin_left_steps=float(margin_left_steps),
                margin_right_steps=float(margin_right_steps),
                per_level_steps=tuple(per_level_steps),
                per_level_world=tuple(per_level_world),
            )
        )

    return AxisSpec(
        axes=tuple(axes),
        ndim=ndim,
        ndisplay=max(1, int(ndisplay)),
        displayed=displayed_tuple,
        order=order_tuple,
        current_level=level_index,
        level_shapes=tuple(level_shapes_seq),
        plane_mode=bool(plane_mode),
        version=prior_spec.version if prior_spec else 1,
    )


def _axis_extent_from_target(spec: AxisSpec, axis: object) -> AxisExtent:
    if isinstance(axis, (int, float)):
        idx = int(axis)
        return spec.axis_by_index(idx)
    if isinstance(axis, str):
        text = axis.strip()
        if text == "":
            raise ValueError("axis target must not be empty")
        if text.lstrip("-").isdigit():
            return spec.axis_by_index(int(text))
        return spec.axis_by_label(text)
    raise TypeError(f"unsupported axis target type: {type(axis)!r}")


def _axis_spec_from_snapshot(snapshot: Mapping[tuple[str, str, str], LedgerEntry]) -> AxisSpec:
    axes_entry = snapshot.get(("dims", "main", "axes"))
    if axes_entry is None or not isinstance(axes_entry.value, Mapping):
        raise AssertionError("ledger missing axis spec payload")
    return axis_spec_from_payload(axes_entry.value)


def _record_axis_spec(
    ledger: ServerStateLedger,
    *,
    origin: str,
    timestamp: float,
) -> None:
    snapshot = ledger.snapshot()
    prior_spec: AxisSpec | None = None
    try:
        prior_spec = _axis_spec_from_snapshot(snapshot)
    except AssertionError:
        prior_spec = None

    axis_labels_raw = _snapshot_optional(snapshot, "dims", "axis_labels")
    order_raw = _snapshot_optional(snapshot, "dims", "order")
    displayed_raw = _snapshot_optional(snapshot, "view", "displayed")
    current_step_raw = _snapshot_optional(snapshot, "dims", "current_step")
    level_shapes_raw = _snapshot_optional(snapshot, "multiscale", "level_shapes") or ()
    current_level_raw = _snapshot_optional(snapshot, "multiscale", "level")
    ndisplay_raw = _snapshot_optional(snapshot, "view", "ndisplay")
    level_shapes: list[tuple[int, ...]] = []
    for shape in level_shapes_raw:
        if isinstance(shape, Sequence):
            level_shapes.append(tuple(int(dim) for dim in shape))

    spec = _build_axis_spec_from_components(
        axis_labels=tuple(str(label) for label in axis_labels_raw) if axis_labels_raw is not None else None,
        order=tuple(int(idx) for idx in order_raw) if order_raw is not None else None,
        displayed=tuple(int(idx) for idx in displayed_raw) if displayed_raw is not None else None,
        current_step=tuple(int(v) for v in current_step_raw) if current_step_raw is not None else None,
        level_shapes=level_shapes,
        current_level=int(current_level_raw) if current_level_raw is not None else 0,
        ndisplay=int(ndisplay_raw) if ndisplay_raw is not None else (prior_spec.ndisplay if prior_spec else 2),
        margin_left=None,
        margin_right=None,
        plane_mode=bool(int(ndisplay_raw) < 3) if ndisplay_raw is not None else (prior_spec.plane_mode if prior_spec else True),
        prior_spec=prior_spec,
    )

    payload = axis_spec_to_payload(spec)
    ledger.batch_record_confirmed(
        [
            ("dims", "main", "axes", payload),
        ],
        origin=origin,
        timestamp=timestamp,
    )



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


def _metadata_from_intent(intent_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not intent_id:
        return None
    return {"intent_id": intent_id}


def _current_ndisplay(ledger: ServerStateLedger) -> int:
    entry = ledger.get("view", "main", "ndisplay")
    if entry is not None and isinstance(entry.value, int):
        return int(entry.value)
    return 2


def _load_plane_state(ledger: ServerStateLedger) -> PlaneState:
    entry = ledger.get("viewport", "plane", "state")
    if entry is not None and isinstance(entry.value, Mapping):
        payload = dict(entry.value)
        if "intent" in payload and "request" not in payload:
            payload["request"] = payload.pop("intent")
        return PlaneState(**payload)
    return PlaneState()


def _plane_from_payload(value: PlaneState | Mapping[str, Any] | None) -> PlaneState:
    if value is None:
        return PlaneState()
    if isinstance(value, PlaneState):
        return PlaneState(**dict(value.__dict__))
    if isinstance(value, Mapping):
        payload = dict(value)
        if "intent" in payload and "request" not in payload:
            payload["request"] = payload.pop("intent")
        return PlaneState(**payload)
    raise TypeError(f"unsupported plane state payload: {type(value)!r}")


def _store_plane_state(
    ledger: ServerStateLedger,
    plane_state: PlaneState,
    *,
    origin: str,
    timestamp: float,
    metadata: Optional[Mapping[str, Any]],
) -> None:
    payload = asdict(plane_state)
    entries: list[tuple] = []
    if metadata:
        meta = dict(metadata)
        entries.append(("viewport", "plane", "state", payload, meta))
    else:
        entries.append(("viewport", "plane", "state", payload))

    if plane_state.applied_level is not None:
        level_value = int(plane_state.applied_level)
        if metadata:
            entries.append(("view_cache", "plane", "level", level_value, dict(metadata)))
        else:
            entries.append(("view_cache", "plane", "level", level_value))
    if plane_state.applied_step is not None:
        step_value = tuple(int(v) for v in plane_state.applied_step)
        if metadata:
            entries.append(("view_cache", "plane", "step", step_value, dict(metadata)))
        else:
            entries.append(("view_cache", "plane", "step", step_value))

    ledger.batch_record_confirmed(entries, origin=origin, timestamp=timestamp)


def _metadata_from_intent(intent_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not intent_id:
        return None
    return {"intent_id": intent_id}


def _current_ndisplay(ledger: ServerStateLedger) -> int:
    entry = ledger.get("view", "main", "ndisplay")
    if entry is not None and isinstance(entry.value, int):
        return int(entry.value)
    return 2


def _load_plane_state(ledger: ServerStateLedger) -> PlaneState:
    entry = ledger.get("viewport", "plane", "state")
    if entry is not None and isinstance(entry.value, Mapping):
        payload = dict(entry.value)
        try:
            return PlaneState(**payload)
        except Exception:
            logger.debug("failed to deserialize plane state payload", exc_info=True)
    return PlaneState()


def _load_volume_state(ledger: ServerStateLedger) -> VolumeState:
    entry = ledger.get("viewport", "volume", "state")
    if entry is not None and isinstance(entry.value, Mapping):
        payload = dict(entry.value)
        try:
            return VolumeState(**payload)
        except Exception:
            logger.debug("failed to deserialize volume state payload", exc_info=True)
    return VolumeState()


def _store_plane_state(
    ledger: ServerStateLedger,
    plane_state: PlaneState,
    *,
    origin: str,
    timestamp: float,
    metadata: Optional[Mapping[str, Any]],
) -> None:
    payload = asdict(plane_state)
    entries: list[tuple] = []
    if metadata:
        meta = dict(metadata)
        entries.append(("viewport", "plane", "state", payload, meta))
    else:
        entries.append(("viewport", "plane", "state", payload))

    if plane_state.applied_level is not None:
        level_value = int(plane_state.applied_level)
        if metadata:
            entries.append(("view_cache", "plane", "level", level_value, dict(metadata)))
        else:
            entries.append(("view_cache", "plane", "level", level_value))
    if plane_state.applied_step is not None:
        step_value = tuple(int(v) for v in plane_state.applied_step)
        if metadata:
            entries.append(("view_cache", "plane", "step", step_value, dict(metadata)))
        else:
            entries.append(("view_cache", "plane", "step", step_value))

    ledger.batch_record_confirmed(entries, origin=origin, timestamp=timestamp)


def _store_volume_state(
    ledger: ServerStateLedger,
    volume_state: VolumeState,
    *,
    origin: str,
    timestamp: float,
    metadata: Optional[Mapping[str, Any]],
) -> None:
    payload = asdict(volume_state)
    if metadata:
        ledger.batch_record_confirmed(
            [("viewport", "volume", "state", payload, dict(metadata))],
            origin=origin,
            timestamp=timestamp,
        )
    else:
        ledger.batch_record_confirmed(
            [("viewport", "volume", "state", payload)],
            origin=origin,
            timestamp=timestamp,
        )


def _plain_plane_state(state: PlaneState | Mapping[str, Any] | None) -> Optional[dict[str, Any]]:
    if state is None:
        return None
    if isinstance(state, PlaneState):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(f"unsupported plane state payload: {type(state)!r}")


def _plain_volume_state(state: VolumeState | Mapping[str, Any] | None) -> Optional[dict[str, Any]]:
    if state is None:
        return None
    if isinstance(state, VolumeState):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(f"unsupported volume state payload: {type(state)!r}")


def _record_viewport_state(
    ledger: ServerStateLedger,
    *,
    mode: RenderMode | str | None,
    plane_state: PlaneState | Mapping[str, Any] | None,
    volume_state: VolumeState | Mapping[str, Any] | None,
    origin: str,
    timestamp: float,
    metadata: Optional[Mapping[str, Any]],
) -> None:
    entries: list[tuple] = []

    if mode is not None:
        if isinstance(mode, RenderMode):
            mode_value = mode.name
        else:
            mode_value = str(mode)
        if metadata:
            entries.append(("viewport", "state", "mode", mode_value, dict(metadata)))
        else:
            entries.append(("viewport", "state", "mode", mode_value))

    plane_payload = _plain_plane_state(plane_state)
    if plane_payload is not None:
        if metadata:
            entries.append(("viewport", "plane", "state", plane_payload, dict(metadata)))
        else:
            entries.append(("viewport", "plane", "state", plane_payload))

    volume_payload = _plain_volume_state(volume_state)
    if volume_payload is not None:
        if metadata:
            entries.append(("viewport", "volume", "state", volume_payload, dict(metadata)))
        else:
            entries.append(("viewport", "volume", "state", volume_payload))

    if not entries:
        return

    ledger.batch_record_confirmed(entries, origin=origin, timestamp=timestamp)


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


def _next_scene_op_seq(ledger: ServerStateLedger) -> int:
    entry = ledger.get("scene", "main", "op_seq")
    if entry is not None and isinstance(entry.value, int):
        return int(entry.value) + 1
    return 1


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
    if prop == "projection_mode":
        return _normalize_string(value, allowed=_ALLOWED_PROJECTION)
    if prop == "plane_thickness":
        val = float(value)
        return max(0.0, val)
    if prop == "attenuation":
        return max(0.0, float(value))
    if prop == "iso_threshold":
        return float(value)
    raise KeyError(f"unsupported layer property '{prop}'")


def reduce_layer_property(
    ledger: ServerStateLedger,
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

    metadata = {"intent_id": intent_id} if intent_id else None

    updates: list[tuple[str, str, str, Any]] = [("layer", layer_id, prop, canonical)]

    if prop == "colormap":
        updates.append(("volume", "main", "colormap", canonical))
    elif prop == "contrast_limits":
        lo, hi = float(canonical[0]), float(canonical[1])
        updates.append(("volume", "main", "contrast_limits", (lo, hi)))
    elif prop == "opacity":
        updates.append(("volume", "main", "opacity", float(canonical)))
    elif prop == "rendering":
        # Route layer.rendering to volume when in volume mode; ignore in plane
        if _current_ndisplay(ledger) >= 3:
            updates.append(("volume", "main", "rendering", canonical))

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=updates,
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="layer-update",
    )

    primary_entry = stored_entries.get(("layer", layer_id, prop))
    assert primary_entry is not None, "layer transaction must return primary entry"
    assert primary_entry.version is not None, "layer transaction must yield version"
    version = int(primary_entry.version)

    return ServerLedgerUpdate(
        scope="layer",
        target=layer_id,
        key=prop,
        value=canonical,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_volume_rendering(
    ledger: ServerStateLedger,
    mode: str,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    metadata = {"intent_id": intent_id} if intent_id else None
    normalized = str(mode)

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=[
            ("volume", "main", "rendering", normalized),
            ("layer", "layer-0", "rendering", normalized),
        ],
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="volume-update",
    )

    entry = stored_entries.get(("volume", "main", "rendering"))
    assert entry is not None, "volume transaction must return entry"
    assert entry.version is not None, "volume transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="rendering",
        value=normalized,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_volume_contrast_limits(
    ledger: ServerStateLedger,
    lo: float,
    hi: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    pair = (float(lo), float(hi))

    metadata = {"intent_id": intent_id} if intent_id else None

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=[("volume", "main", "contrast_limits", pair)],
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="volume-update",
    )

    entry = stored_entries.get(("volume", "main", "contrast_limits"))
    assert entry is not None, "volume transaction must return entry"
    assert entry.version is not None, "volume transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="contrast_limits",
        value=pair,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_volume_colormap(
    ledger: ServerStateLedger,
    name: str,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    normalized = str(name)

    metadata = {"intent_id": intent_id} if intent_id else None

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=[("volume", "main", "colormap", normalized)],
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="volume-update",
    )

    entry = stored_entries.get(("volume", "main", "colormap"))
    assert entry is not None, "volume transaction must return entry"
    assert entry.version is not None, "volume transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="colormap",
        value=normalized,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_volume_opacity(
    ledger: ServerStateLedger,
    alpha: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    value = float(alpha)

    metadata = {"intent_id": intent_id} if intent_id else None

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=[("volume", "main", "opacity", value)],
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="volume-update",
    )

    entry = stored_entries.get(("volume", "main", "opacity"))
    assert entry is not None, "volume transaction must return entry"
    assert entry.version is not None, "volume transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="opacity",
        value=value,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_volume_sample_step(
    ledger: ServerStateLedger,
    sample_step: float,
    *,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.volume",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    value = float(sample_step)

    metadata = {"intent_id": intent_id} if intent_id else None

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_layer_property_transaction(
        ledger=ledger,
        updates=[("volume", "main", "sample_step", value)],
        origin=origin,
        timestamp=ts,
        metadata=metadata,
        op_seq=next_op_seq,
        op_kind="volume-update",
    )

    entry = stored_entries.get(("volume", "main", "sample_step"))
    assert entry is not None, "volume transaction must return entry"
    assert entry.version is not None, "volume transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="volume",
        target="main",
        key="sample_step",
        value=value,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def _normalize_view_order(
    *,
    baseline_order: Optional[Sequence[int]],
    requested_order: Optional[Sequence[int]],
    ndim: int,
) -> tuple[int, ...]:
    current: tuple[int, ...] = ()
    if requested_order is not None:
        current = tuple(int(idx) for idx in requested_order)
    elif baseline_order is not None:
        current = tuple(int(idx) for idx in baseline_order)

    normalized: list[int] = []
    seen: set[int] = set()
    for axis in current:
        axis_idx = int(axis)
        if 0 <= axis_idx < max(ndim, 1) and axis_idx not in seen:
            normalized.append(axis_idx)
            seen.add(axis_idx)

    for axis_idx in range(max(ndim, 1)):
        if axis_idx not in seen:
            normalized.append(axis_idx)
            seen.add(axis_idx)

    if not normalized:
        normalized = list(range(max(ndim, 1)))

    return tuple(normalized)


def _normalize_view_displayed(
    *,
    requested_displayed: Optional[Sequence[int]],
    order_value: tuple[int, ...],
    target_ndisplay: int,
) -> tuple[int, ...]:
    if requested_displayed is not None:
        return tuple(int(idx) for idx in requested_displayed)

    count = min(len(order_value), max(1, int(target_ndisplay)))
    if count <= 0:
        return tuple()
    return tuple(order_value[-count:])


def _dims_entries_from_payload(
    payload: NotifyDimsPayload,
    *,
    axis_index: int,
    axis_target: str,
) -> list[tuple[Any, ...]]:
    entries: list[tuple[Any, ...]] = []

    spec = payload.axes_spec

    entries.append(("view", "main", "ndisplay", int(payload.ndisplay)))
    entries.append(("view", "main", "displayed", tuple(int(idx) for idx in spec.displayed)))

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

    # mode is derived from view.ndisplay; do not persist dims.mode
    entries.append(("dims", "main", "order", tuple(int(idx) for idx in spec.order)))
    entries.append(("dims", "main", "axis_labels", tuple(axis.label for axis in spec.axes)))
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

    entries.append(("dims", "main", "axes", axis_spec_to_payload(payload.axes_spec)))

    return entries


def _ledger_dims_payload(ledger: ServerStateLedger) -> NotifyDimsPayload:
    snapshot = ledger.snapshot()
    axis_spec = _axis_spec_from_snapshot(snapshot)

    def require(scope: str, key: str) -> Any:
        entry = snapshot.get((scope, "main", key))
        if entry is None:
            raise AssertionError(f"ledger missing {scope}/{key}")
        return entry.value

    def optional(scope: str, key: str) -> Any:
        entry = snapshot.get((scope, "main", key))
        return None if entry is None else entry.value

    require("dims", "current_step")
    level_shapes_raw = require("multiscale", "level_shapes")
    level_shapes = tuple(
        tuple(int(dim) for dim in shape) for shape in level_shapes_raw
    )
    levels_raw = require("multiscale", "levels")
    levels = tuple(dict(level) for level in levels_raw)
    current_level = axis_spec.current_level
    downgraded_raw = optional("multiscale", "downgraded")
    downgraded = None if downgraded_raw is None else bool(downgraded_raw)

    labels_raw = optional("dims", "labels")
    labels = (
        tuple(str(label) for label in labels_raw)
        if labels_raw is not None
        else None
    )

    ndisplay = axis_spec.ndisplay
    mode = "plane" if axis_spec.plane_mode else "volume"

    return NotifyDimsPayload(
        level_shapes=level_shapes,
        levels=levels,
        current_level=current_level,
        downgraded=downgraded,
        mode=mode,
        ndisplay=ndisplay,
        labels=labels,
        axes_spec=axis_spec,
    )


def reduce_bootstrap_state(
    ledger: ServerStateLedger,
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
    """Seed the ledger and local state from bootstrap metadata."""

    ts = _now(timestamp)

    resolved_step = tuple(int(v) for v in step)
    resolved_axis_labels = tuple(str(label) for label in axis_labels)
    resolved_order = tuple(int(idx) for idx in order)
    resolved_level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in level_shapes)
    resolved_levels = tuple(dict(level) for level in levels)
    resolved_current_level = int(current_level)
    resolved_ndisplay = 3 if int(ndisplay) >= 3 else 2
    mode_value = "volume" if resolved_ndisplay >= 3 else "plane"

    ndim = max(
        len(resolved_axis_labels),
        len(resolved_step),
        len(resolved_order),
        len(resolved_level_shapes[resolved_current_level]) if resolved_level_shapes else 0,
    )
    if ndim == 0:
        ndim = 1

    if not resolved_order:
        resolved_order = tuple(range(ndim))

    axis_index = 0 if resolved_step else 0
    if resolved_axis_labels and axis_index < len(resolved_axis_labels):
        axis_target = str(resolved_axis_labels[axis_index])
    else:
        axis_target = str(axis_index)

    displayed_tuple = (
        tuple(resolved_order[-resolved_ndisplay:])
        if resolved_order
        else tuple(range(max(ndim - resolved_ndisplay, 0), ndim))
    )

    axis_spec = _build_axis_spec_from_components(
        axis_labels=resolved_axis_labels,
        order=resolved_order,
        displayed=displayed_tuple,
        current_step=resolved_step,
        level_shapes=resolved_level_shapes,
        current_level=resolved_current_level,
        ndisplay=resolved_ndisplay,
        margin_left=None,
        margin_right=None,
        plane_mode=resolved_ndisplay < 3,
        prior_spec=None,
    )

    dims_payload = NotifyDimsPayload(
        level_shapes=resolved_level_shapes,
        levels=resolved_levels,
        current_level=resolved_current_level,
        downgraded=False,
        mode=mode_value,
        ndisplay=resolved_ndisplay,
        labels=None,
        axes_spec=axis_spec,
    )

    entries = _dims_entries_from_payload(
        dims_payload,
        axis_index=axis_index,
        axis_target=axis_target,
    )

    axis_index_value = int(resolved_step[axis_index]) if resolved_step else 0
    axis_index_metadata = {
        "axis_index": axis_index,
        "axis_target": axis_target,
    }
    entries.append(
        (
            "dims",
            axis_target,
            "index",
            axis_index_value,
            axis_index_metadata,
        )
    )

    op_entry = ledger.get("scene", "main", "op_seq")
    next_op_seq = int(op_entry.value) + 1 if op_entry is not None and isinstance(op_entry.value, int) else 1

    stored_entries = apply_bootstrap_transaction(
        ledger=ledger,
        op_seq=next_op_seq,
        entries=entries,
        origin=origin,
        timestamp=ts,
    )

    dims_entry = stored_entries.get(("dims", axis_target, "index"))
    assert dims_entry is not None, "bootstrap transaction must return dims index entry"
    assert dims_entry.version is not None, "bootstrap transaction must yield dims version"
    dims_version = int(dims_entry.version)

    view_entry = stored_entries.get(("view", "main", "ndisplay"))
    assert view_entry is not None, "bootstrap transaction must return view entry"
    assert view_entry.version is not None, "bootstrap transaction must yield view version"
    view_version = int(view_entry.version)

    dims_intent_id = f"dims-bootstrap-{uuid.uuid4().hex}"
    view_intent_id = f"view-bootstrap-{uuid.uuid4().hex}"

    dims_update = ServerLedgerUpdate(
        scope="dims",
        target=axis_target,
        key="index",
        value=axis_index_value,
        intent_id=dims_intent_id,
        timestamp=ts,
        axis_index=axis_index,
        current_step=resolved_step,
        origin=origin,
        version=dims_version,
    )
    view_update = ServerLedgerUpdate(
        scope="view",
        target="main",
        key="ndisplay",
        value=resolved_ndisplay,
        intent_id=view_intent_id,
        timestamp=ts,
        origin=origin,
        version=view_version,
    )

    return [dims_update, view_update]


def reduce_dims_update(
    ledger: ServerStateLedger,
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
    axis_spec = payload.axes_spec
    extent = _axis_extent_from_target(axis_spec, axis)

    step = [int(v) for v in payload.current_step]
    ndim = len(step)
    assert ndim > 0, "ledger dims metadata missing dimensions"

    idx = extent.index
    control_target = extent.label or f"axis-{idx}"

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

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_dims_step_transaction(
        ledger=ledger,
        step=requested_step,
        metadata=step_metadata,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="dims-update",
    )

    current_step_entry = stored_entries.get(("dims", "main", "current_step"))
    assert current_step_entry is not None, "dims transaction must return current_step entry"
    assert current_step_entry.version is not None, "dims transaction must yield versioned entry"
    version = int(current_step_entry.version)

    logger.debug(
        "dims intent updated axis=%s prop=%s step=%s version=%d origin=%s",
        control_target,
        prop,
        requested_step,
        version,
        origin,
    )

    if _current_ndisplay(ledger) < 3:
        plane_state = _load_plane_state(ledger)
        level_idx = int(payload.current_level)
        step_tuple = tuple(int(v) for v in requested_step)
        plane_state.target_ndisplay = 2
        plane_state.target_level = level_idx
        plane_state.target_step = step_tuple
        plane_state.applied_level = level_idx
        plane_state.applied_step = step_tuple
        plane_state.awaiting_level_confirm = False
        plane_state.camera_pose_dirty = False
        plane_state.applied_roi = None
        plane_state.applied_roi_signature = None
        metadata = _metadata_from_intent(resolved_intent_id)
        _store_plane_state(
            ledger,
            plane_state,
            origin=origin,
            timestamp=ts,
            metadata=metadata,
        )

    _record_axis_spec(
        ledger,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="dims",
        target=control_target,
        key=prop,
        value=int(step[idx]),
        intent_id=resolved_intent_id,
        timestamp=ts,
        axis_index=idx,
        current_step=requested_step,
        origin=origin,
        version=version,
    )

def reduce_dims_margins_update(
    ledger: ServerStateLedger,
    *,
    axis: object,
    side: str,
    value: float,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.dims",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    payload = _ledger_dims_payload(ledger)
    axis_spec = payload.axes_spec
    extent = _axis_extent_from_target(axis_spec, axis)

    side_key = "margin_left" if side == "margin_left" else "margin_right"

    updated_spec = with_updated_margins(
        axis_spec,
        extent.index,
        margin_left_world=value if side == "margin_left" else None,
        margin_right_world=value if side == "margin_right" else None,
    )

    stored = ledger.batch_record_confirmed(
        [
            ("dims", "main", "axes", axis_spec_to_payload(updated_spec)),
        ],
        origin=origin,
        timestamp=ts,
    )

    _record_axis_spec(
        ledger,
        origin=origin,
        timestamp=ts,
    )

    axes_entry = stored.get(("dims", "main", "axes"))
    assert axes_entry is not None, "margin update must persist axis spec"
    assert axes_entry.version is not None, "axes entry must provide version"

    return ServerLedgerUpdate(
        scope="dims",
        target=extent.label or f"axis-{extent.index}",
        key=side_key,
        value=float(value),
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=int(axes_entry.version),
    )


def reduce_view_update(
    ledger: ServerStateLedger,
    *,
    ndisplay: Optional[int] = None,
    order: Optional[Sequence[int]] = None,
    displayed: Optional[Sequence[int]] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.view",
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    dims_payload = _ledger_dims_payload(ledger)

    if ndisplay is not None:
        target_ndisplay = 3 if int(ndisplay) >= 3 else 2
    else:
        entry = ledger.get("view", "main", "ndisplay")
        target_ndisplay = int(entry.value) if entry is not None and isinstance(entry.value, int) else 2

    resolved_intent_id = intent_id or f"view-{uuid.uuid4().hex}"
    metadata = _metadata_from_intent(resolved_intent_id)

    baseline_step = tuple(int(v) for v in dims_payload.current_step) if dims_payload.current_step else None
    baseline_order = tuple(int(idx) for idx in dims_payload.order) if dims_payload.order is not None else None

    if target_ndisplay >= 3:
        plane_state = _load_plane_state(ledger)
        level_idx = int(dims_payload.current_level)
        if baseline_step is not None:
            step_tuple = tuple(int(v) for v in baseline_step)
            plane_state.target_step = step_tuple
            plane_state.applied_step = step_tuple
        plane_state.target_ndisplay = 2
        plane_state.target_level = level_idx
        plane_state.applied_level = level_idx
        plane_state.awaiting_level_confirm = False
        _store_plane_state(
            ledger,
            plane_state,
            origin=origin,
            timestamp=ts,
            metadata=metadata,
        )

        volume_state = _load_volume_state(ledger)
        _store_volume_state(
            ledger,
            volume_state,
            origin=origin,
            timestamp=ts,
            metadata=metadata,
        )

    ndim = 0
    if baseline_step is not None:
        ndim = len(baseline_step)
    elif baseline_order is not None:
        ndim = len(baseline_order)
    if ndim <= 0:
        ndim = max(int(target_ndisplay), 1)

    order_value = _normalize_view_order(
        baseline_order=baseline_order,
        requested_order=order,
        ndim=ndim,
    )
    displayed_value = _normalize_view_displayed(
        requested_displayed=displayed,
        order_value=order_value,
        target_ndisplay=target_ndisplay,
    )

    op_entry = ledger.get("scene", "main", "op_seq")
    next_op_seq = int(op_entry.value) + 1 if op_entry is not None and isinstance(op_entry.value, int) else 1

    stored_entries = apply_view_toggle_transaction(
        ledger=ledger,
        op_seq=next_op_seq,
        target_ndisplay=int(target_ndisplay),
        order_value=order_value,
        displayed_value=displayed_value,
        origin=origin,
        timestamp=ts,
    )

    ndisplay_entry = stored_entries.get(("view", "main", "ndisplay"))
    assert ndisplay_entry is not None, "view toggle transaction must return ndisplay entry"
    assert ndisplay_entry.version is not None, "view toggle transaction must yield versioned entry"
    version = int(ndisplay_entry.version)

    logger.debug(
        "view intent updated ndisplay=%d order=%s displayed=%s version=%d origin=%s",
        target_ndisplay,
        order_value,
        displayed_value,
        version,
        origin,
    )

    _record_viewport_state(
        ledger,
        mode=RenderMode.VOLUME if target_ndisplay >= 3 else RenderMode.PLANE,
        plane_state=None,
        volume_state=None,
        origin=origin,
        timestamp=ts,
        metadata=metadata,
    )

    _record_axis_spec(
        ledger,
        origin=origin,
        timestamp=ts,
    )

    return ServerLedgerUpdate(
        scope="view",
        target="main",
        key="ndisplay",
        value=int(target_ndisplay),
        intent_id=resolved_intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )


def reduce_plane_restore(
    ledger: ServerStateLedger,
    *,
    level: int,
    step: Sequence[int],
    center: Sequence[float],
    zoom: float,
    rect: Sequence[float],
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.view",
) -> dict[PropertyKey, LedgerEntry]:
    ts = _now(timestamp)
    metadata = {"intent_id": intent_id} if intent_id else None
    level_idx = int(level)
    step_tuple = tuple(int(v) for v in step)
    center_tuple = tuple(float(v) for v in center)
    rect_tuple = tuple(float(v) for v in rect)
    zoom_value = float(zoom)

    plane_entry = ledger.get("viewport", "plane", "state")
    base_plane = PlaneState()
    if plane_entry is not None and isinstance(plane_entry.value, Mapping):
        payload_value = dict(plane_entry.value)
        base_plane = PlaneState(**payload_value)

    base_plane.target_level = level_idx
    base_plane.target_ndisplay = 2
    base_plane.target_step = step_tuple
    base_plane.applied_level = level_idx
    base_plane.applied_step = step_tuple
    base_plane.update_pose(
        rect=rect_tuple,
        center=(center_tuple[0], center_tuple[1]),
        zoom=zoom_value,
    )
    base_plane.applied_roi = None
    base_plane.applied_roi_signature = None
    base_plane.camera_pose_dirty = False

    next_op_seq = _next_scene_op_seq(ledger)

    stored = apply_plane_restore_transaction(
        ledger=ledger,
        level=level_idx,
        step=step_tuple,
        center=center_tuple,
        zoom=zoom_value,
        rect=rect_tuple,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="plane-restore",
    )

    _store_plane_state(
        ledger,
        base_plane,
        origin=origin,
        timestamp=ts,
        metadata=metadata,
    )

    return stored


def reduce_volume_restore(
    ledger: ServerStateLedger,
    *,
    level: int,
    center: Sequence[float],
    angles: Sequence[float],
    distance: float,
    fov: float,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.view",
) -> dict[PropertyKey, LedgerEntry]:
    ts = _now(timestamp)
    metadata = {"intent_id": intent_id} if intent_id else None

    if len(center) < 3:
        raise ValueError("volume restore center requires three components")
    if len(angles) < 2:
        raise ValueError("volume restore angles require at least two components")

    center_tuple = (float(center[0]), float(center[1]), float(center[2]))
    roll_value = float(angles[2]) if len(angles) >= 3 else 0.0
    angles_tuple = (float(angles[0]), float(angles[1]), roll_value)
    distance_value = float(distance)
    fov_value = float(fov)
    level_idx = int(level)

    volume_entry = ledger.get("viewport", "volume", "state")
    base_volume = VolumeState()
    if volume_entry is not None and isinstance(volume_entry.value, Mapping):
        payload_value = dict(volume_entry.value)
        base_volume = VolumeState(**payload_value)

    base_volume.level = level_idx
    base_volume.update_pose(
        center=center_tuple,
        angles=angles_tuple,
        distance=distance_value,
        fov=fov_value,
    )

    next_op_seq = _next_scene_op_seq(ledger)

    stored = apply_volume_restore_transaction(
        ledger=ledger,
        level=level_idx,
        center=center_tuple,
        angles=angles_tuple,
        distance=distance_value,
        fov=fov_value,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="volume-restore",
    )

    _store_volume_state(
        ledger,
        base_volume,
        origin=origin,
        timestamp=ts,
        metadata=metadata,
    )

    return stored


def reduce_level_update(
    ledger: ServerStateLedger,
    *,
    applied: Mapping[str, Any] | object,
    downgraded: Optional[bool] = None,
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "control.multiscale",
    mode: RenderMode | str | None = None,
    plane_state: PlaneState | Mapping[str, Any] | None = None,
    volume_state: VolumeState | Mapping[str, Any] | None = None,
) -> ServerLedgerUpdate:
    ts = _now(timestamp)
    metadata = {"intent_id": intent_id} if intent_id else None

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

    step_metadata = {"source": "worker.level_update", "level": level}
    if intent_id is not None:
        step_metadata["intent_id"] = intent_id
    plane_struct: Optional[PlaneState] = None
    if plane_state is not None:
        plane_struct = _plane_from_payload(plane_state)
    volume_payload = _plain_volume_state(volume_state)
    mode_value: Optional[str] = None
    if mode is not None:
        mode_value = mode.name if isinstance(mode, RenderMode) else str(mode)

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_level_switch_transaction(
        ledger=ledger,
        level=level,
        step=step_tuple,
        level_shapes=updated_level_shapes if updated_level_shapes else None,
        downgraded=bool(downgraded) if downgraded is not None else None,
        step_metadata=step_metadata,
        level_metadata=metadata,
        level_shapes_metadata=metadata if metadata is not None and updated_level_shapes else None,
        downgraded_metadata=metadata if metadata is not None and downgraded is not None else None,
        viewport_mode=mode_value,
        viewport_plane_state=None,
        viewport_volume_state=volume_payload,
        viewport_metadata=metadata,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="level-update",
    )

    _record_axis_spec(
        ledger,
        origin=origin,
        timestamp=ts,
    )

    level_entry = stored_entries.get(("multiscale", "main", "level"))
    assert level_entry is not None, "level switch transaction must return level entry"
    level_version = None if level_entry.version is None else int(level_entry.version)

    logger.debug(
        "multiscale intent updated level=%d version=%s origin=%s",
        level,
        level_version,
        origin,
    )

    if plane_struct is not None and _current_ndisplay(ledger) < 3 and int(plane_struct.target_ndisplay) == 2:
        _store_plane_state(
            ledger,
            plane_struct,
            origin=origin,
            timestamp=ts,
            metadata=_metadata_from_intent(intent_id),
        )

    return ServerLedgerUpdate(
        scope="multiscale",
        target="main",
        key="level",
        value=level,
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        current_step=step_tuple,
        version=level_version,
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
    metadata: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[dict[str, Any], int]:
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
    next_op_seq = _next_scene_op_seq(ledger)

    metadata_dict = dict(metadata)
    include_metadata = bool(metadata_dict)

    ledger_updates: list[tuple] = []
    ack: dict[str, Any] = {}

    def _append_entry(scope: str, key: str, value: Any) -> None:
        if include_metadata:
            ledger_updates.append((scope, "main", key, value, metadata_dict))
        else:
            ledger_updates.append((scope, "main", key, value))

    use_volume_path = False
    if distance is not None or fov is not None or (angles is not None and len(angles) >= 2) or _current_ndisplay(ledger) >= 3:
        use_volume_path = True

    if not use_volume_path:
        plane_state = _load_plane_state(ledger)
        dims_payload = _ledger_dims_payload(ledger)
        plane_state.target_ndisplay = 2
        plane_state.target_level = int(dims_payload.current_level)
        plane_state.applied_level = int(dims_payload.current_level)
        if dims_payload.current_step:
            step_tuple = tuple(int(v) for v in dims_payload.current_step)
            plane_state.target_step = step_tuple
            plane_state.applied_step = step_tuple
        if center is not None:
            if len(center) < 2:
                raise ValueError("plane camera center requires at least two components")
            plane_center = (float(center[0]), float(center[1]))
            plane_state.update_pose(center=plane_center)
            ack["center"] = [plane_center[0], plane_center[1]]
            _append_entry("camera_plane", "center", plane_center)
        if zoom is not None:
            zoom_value = float(zoom)
            plane_state.update_pose(zoom=zoom_value)
            ack["zoom"] = zoom_value
            _append_entry("camera_plane", "zoom", zoom_value)
        if rect is not None:
            if len(rect) < 4:
                raise ValueError("plane camera rect requires four components")
            rect_tuple = (
                float(rect[0]),
                float(rect[1]),
                float(rect[2]),
                float(rect[3]),
            )
            plane_state.update_pose(rect=rect_tuple)
            ack["rect"] = [rect_tuple[0], rect_tuple[1], rect_tuple[2], rect_tuple[3]]
            _append_entry("camera_plane", "rect", rect_tuple)
        plane_state.camera_pose_dirty = False
        plane_state.applied_roi = None
        plane_state.applied_roi_signature = None
        _store_plane_state(
            ledger,
            plane_state,
            origin=origin,
            timestamp=ts,
            metadata=metadata,
        )
    else:
        volume_state = _load_volume_state(ledger)
        pose_updates: dict[str, object] = {}
        if center is not None:
            if len(center) < 3:
                raise ValueError("volume camera center requires three components")
            pose_updates["center"] = (
                float(center[0]),
                float(center[1]),
                float(center[2]),
            )
            ack["center"] = [float(center[0]), float(center[1]), float(center[2])]
            _append_entry("camera_volume", "center", pose_updates["center"])
        if angles is not None:
            if len(angles) < 2:
                raise ValueError("volume camera angles require at least two components")
            roll_val = float(angles[2]) if len(angles) >= 3 else (
                float(volume_state.pose.angles[2]) if volume_state.pose.angles is not None and len(volume_state.pose.angles) >= 3 else 0.0
            )
            pose_updates["angles"] = (float(angles[0]), float(angles[1]), roll_val)
            ack["angles"] = [float(angles[0]), float(angles[1]), float(pose_updates["angles"][2])]
            _append_entry("camera_volume", "angles", pose_updates["angles"])
        if distance is not None:
            pose_updates["distance"] = float(distance)
            ack["distance"] = float(distance)
            _append_entry("camera_volume", "distance", float(distance))
        if fov is not None:
            pose_updates["fov"] = float(fov)
            ack["fov"] = float(fov)
            _append_entry("camera_volume", "fov", float(fov))
        if pose_updates:
            volume_state.update_pose(**pose_updates)
            _store_volume_state(
                ledger,
                volume_state,
                origin=origin,
                timestamp=ts,
                metadata=metadata,
            )

    if not ledger_updates:
        raise ValueError("camera reducer failed to build scoped ledger updates")

    stored_entries = apply_camera_update_transaction(
        ledger=ledger,
        updates=ledger_updates,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="camera-update",
    )

    versions: list[int] = []
    for (scope, _, _), entry in stored_entries.items():
        if scope in {"camera_plane", "camera_volume"} and entry.version is not None:
            versions.append(int(entry.version))
    assert versions, "camera transaction must return versioned entries"
    version = max(versions)

    return ack, version


StateUpdateResult = ServerLedgerUpdate

__all__ = [
    "ServerLedgerUpdate",
    "StateUpdateResult",
    "clamp_level",
    "clamp_opacity",
    "clamp_sample_step",
    "is_valid_render_mode",
    "normalize_clim",
    "reduce_bootstrap_state",
    "reduce_camera_update",
    "reduce_dims_update",
    "reduce_layer_property",
    "reduce_level_update",
    "reduce_plane_restore",
    "reduce_view_update",
    "reduce_volume_colormap",
    "reduce_volume_contrast_limits",
    "reduce_volume_opacity",
    "reduce_volume_rendering",
    "reduce_volume_restore",
    "reduce_volume_sample_step",
]


def reduce_thumbnail_capture(
    ledger: ServerStateLedger,
    *,
    layer_id: str,
    payload: Mapping[str, Any],
    intent_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    origin: str = "server.thumbnail",
) -> ServerLedgerUpdate:
    """Reducer wrapper for thumbnail capture ingestion.

    Records the normalized thumbnail payload and updates per-layer thumbnail
    state (last_signature, seq, attempts=0). Returns the layer thumbnail entry
    version as a ServerLedgerUpdate for observability if needed.
    """
    ts = _now(timestamp)

    stored_entries = apply_thumbnail_capture(
        ledger=ledger,
        layer_id=str(layer_id),
        payload=payload,
        origin=origin,
        timestamp=ts,
    )

    entry = stored_entries.get(("layer", str(layer_id), "thumbnail"))
    assert entry is not None, "thumbnail transaction must return layer thumbnail entry"
    assert entry.version is not None, "thumbnail transaction must yield version"
    version = int(entry.version)

    return ServerLedgerUpdate(
        scope="layer",
        target=str(layer_id),
        key="thumbnail",
        value=dict(payload),
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=version,
    )
