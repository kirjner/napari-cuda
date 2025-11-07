"""Ledger-backed reducers for server control updates."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from types import MappingProxyType
from typing import Any, Optional

from napari.layers.image._image_constants import (
    Interpolation as NapariInterpolation,
    ImageProjectionMode as NapariProjectionMode,
)
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.colormaps.colormap_utils import ensure_colormap

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
from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_axis_labels,
    dims_spec_clamp_step,
    dims_spec_displayed,
    dims_spec_from_payload,
    dims_spec_level_shape,
    dims_spec_order,
    dims_spec_remap_step_for_level,
    dims_spec_to_payload,
    validate_ledger_against_dims_spec,
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



def _now(timestamp: Optional[float]) -> float:
    return float(timestamp) if timestamp is not None else time.time()


def is_valid_render_mode(mode: str, allowed_modes: Sequence[str]) -> bool:
    return str(mode or "").lower() in {str(m).lower() for m in allowed_modes}


def _serialize_dims_spec(spec: DimsSpec) -> dict[str, Any]:
    payload = dims_spec_to_payload(spec)
    assert payload is not None, "dims spec serialization failed"
    return payload


def _axis_extents_for_steps(steps: Sequence[int]) -> tuple[AxisExtent, ...]:
    extents: list[AxisExtent] = []
    for count in steps:
        size = int(count)
        stop = float(max(size - 1, 0))
        extents.append(AxisExtent(start=0.0, stop=stop, step=1.0))
    return tuple(extents)


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
    spec_entry = ledger.get("dims", "main", "dims_spec")
    if spec_entry is not None and isinstance(spec_entry.value, Mapping):
        spec = dims_spec_from_payload(spec_entry.value)
        if spec is not None:
            return int(spec.ndisplay)
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
    entry: tuple
    if metadata:
        entry = ("viewport", "plane", "state", payload, dict(metadata))
    else:
        entry = ("viewport", "plane", "state", payload)
    ledger.batch_record_confirmed([entry], origin=origin, timestamp=timestamp)


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


def _get_dims_spec(ledger: ServerStateLedger) -> DimsSpec:
    entry = ledger.get("dims", "main", "dims_spec")
    assert entry is not None, "ledger missing dims_spec entry"
    spec = dims_spec_from_payload(entry.value)
    assert spec is not None, "dims spec payload missing"
    return spec


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

    displayed = (
        resolved_order[-resolved_ndisplay:]
        if resolved_order
        else tuple(range(max(ndim - resolved_ndisplay, 0), ndim))
    )
    displayed_tuple = tuple(int(v) for v in displayed)

    axes_entries: list[DimsSpecAxis] = []
    order_lookup = {int(idx): pos for pos, idx in enumerate(resolved_order)}
    displayed_set = set(int(idx) for idx in displayed_tuple)
    for axis_idx in range(ndim):
        if resolved_axis_labels and axis_idx < len(resolved_axis_labels):
            label = str(resolved_axis_labels[axis_idx])
        else:
            label = f"axis-{axis_idx}"
        current_step_val = resolved_step[axis_idx] if axis_idx < len(resolved_step) else 0
        per_level_steps = [int(shape[axis_idx]) for shape in resolved_level_shapes] if resolved_level_shapes else [1]
        axes_entries.append(
            DimsSpecAxis(
                index=int(axis_idx),
                label=label,
                role=label.lower(),
                displayed=int(axis_idx) in displayed_set,
                order_position=int(order_lookup.get(axis_idx, axis_idx)),
                current_step=int(current_step_val),
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=tuple(int(v) for v in per_level_steps),
                per_level_world=_axis_extents_for_steps(per_level_steps),
            )
        )

    dims_spec = DimsSpec(
        version=1,
        ndim=int(ndim),
        ndisplay=int(resolved_ndisplay),
        order=tuple(int(v) for v in resolved_order),
        displayed=displayed_tuple,
        current_level=int(resolved_current_level),
        current_step=resolved_step,
        level_shapes=resolved_level_shapes,
        plane_mode=resolved_ndisplay < 3,
        axes=tuple(axes_entries),
        levels=resolved_levels,
        downgraded=False,
        labels=None,
    )

    entries: list[tuple[Any, ...]] = [
        ("dims", "main", "current_step", resolved_step),
    ]
    entries.append(("dims", "main", "dims_spec", _serialize_dims_spec(dims_spec)))

    axis_index_value = int(resolved_step[axis_index]) if resolved_step else 0
    entries.append(
        (
            "dims",
            axis_target,
            "index",
            axis_index_value,
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

    spec_entry = stored_entries.get(("dims", "main", "dims_spec"))
    assert spec_entry is not None, "bootstrap transaction must return dims_spec entry"
    assert spec_entry.version is not None, "bootstrap transaction must yield spec version"
    spec_version = int(spec_entry.version)

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
        dims_spec=dims_spec,
    )
    view_update = ServerLedgerUpdate(
        scope="view",
        target="main",
        key="ndisplay",
        value=resolved_ndisplay,
        intent_id=view_intent_id,
        timestamp=ts,
        origin=origin,
        version=spec_version,
        dims_spec=dims_spec,
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
    current_spec = _get_dims_spec(ledger)

    step = [int(v) for v in current_spec.current_step]
    axis_count = len(current_spec.axes)
    if len(step) < axis_count:
        step.extend([0] * (axis_count - len(step)))
    ndim = max(len(step), axis_count)

    idx = resolve_axis_index(
        axis,
        order=dims_spec_order(current_spec),
        axis_labels=dims_spec_axis_labels(current_spec),
        ndim=len(step),
    )
    if idx is None:
        raise ValueError("unable to resolve axis index for dims update")

    spec_labels = dims_spec_axis_labels(current_spec)
    control_target = spec_labels[idx] if idx < len(spec_labels) else str(idx)

    target = int(step[idx])
    if value is not None:
        target = int(value)
    if step_delta is not None:
        target += int(step_delta)

    level_idx = int(current_spec.current_level)
    level_shape = dims_spec_level_shape(current_spec, level_idx)
    assert idx < len(level_shape), "axis index outside level_shapes entry"
    size_val = int(level_shape[idx])
    assert size_val > 0, "level_shapes entries must be positive"
    target = max(0, min(size_val - 1, target))

    step[idx] = int(target)
    requested_step = tuple(int(v) for v in step)
    resolved_intent_id = intent_id or f"dims-{uuid.uuid4().hex}"

    axes_list = list(current_spec.axes)
    assert idx < len(axes_list), "dims spec missing target axis"
    axes_list[idx] = replace(axes_list[idx], current_step=int(requested_step[idx]))
    new_spec = replace(
        current_spec,
        current_step=requested_step,
        axes=tuple(axes_list),
    )

    next_op_seq = _next_scene_op_seq(ledger)

    stored_entries = apply_dims_step_transaction(
        ledger=ledger,
        step=requested_step,
        dims_spec_payload=_serialize_dims_spec(new_spec),
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="dims-update",
    )

    spec_entry = stored_entries.get(("dims", "main", "dims_spec"))
    assert spec_entry is not None, "dims transaction must return dims_spec entry"
    assert spec_entry.version is not None, "dims transaction must yield versioned spec entry"
    version = int(spec_entry.version)

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
        level_idx = int(current_spec.current_level)
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
        dims_spec=new_spec,
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
    spec = _get_dims_spec(ledger)
    current_step = spec.current_step

    idx = resolve_axis_index(
        axis,
        order=dims_spec_order(spec),
        axis_labels=dims_spec_axis_labels(spec),
        ndim=len(current_step),
    )
    if idx is None:
        raise ValueError("unable to resolve axis index for dims margins update")

    side_key = "margin_left" if side == "margin_left" else "margin_right"
    # Start from existing ledger values if present, else zeros of length ndim
    existing_entry = ledger.get("dims", "main", side_key)
    if existing_entry is not None and isinstance(existing_entry.value, (tuple, list)):
        arr = [float(v) for v in existing_entry.value]
    else:
        arr = [0.0 for _ in range(len(current_step))]
    # Ensure length
    if len(arr) < len(current_step):
        arr.extend([0.0] * (len(current_step) - len(arr)))
    arr[idx] = float(value)

    axes_list = list(spec.axes)
    axis_entry = axes_list[idx]
    if side_key == "margin_left":
        axes_list[idx] = replace(
            axis_entry,
            margin_left_steps=float(arr[idx]),
            margin_left_world=float(arr[idx]),
        )
    else:
        axes_list[idx] = replace(
            axis_entry,
            margin_right_steps=float(arr[idx]),
            margin_right_world=float(arr[idx]),
        )
    new_spec = replace(spec, axes=tuple(axes_list))

    stored_entries = apply_dims_step_transaction(
        ledger=ledger,
        step=current_step,
        dims_spec_payload=_serialize_dims_spec(new_spec),
        origin=origin,
        timestamp=ts,
        op_seq=_next_scene_op_seq(ledger),
        op_kind="dims-update",
    )
    # Additionally store margins arrays atomically
    entries: list[tuple] = [("dims", "main", side_key, tuple(float(v) for v in arr))]
    ledger.batch_record_confirmed(entries, origin=origin, timestamp=ts)

    return ServerLedgerUpdate(
        scope="dims",
        target=str(axis),
        key=side_key,
        value=tuple(float(v) for v in arr),
        intent_id=intent_id,
        timestamp=ts,
        origin=origin,
        version=int(stored_entries[("dims", "main", "dims_spec")].version)
        if stored_entries[("dims", "main", "dims_spec")].version is not None
        else 0,
        dims_spec=new_spec,
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
    spec = _get_dims_spec(ledger)

    if ndisplay is not None:
        target_ndisplay = 3 if int(ndisplay) >= 3 else 2
    else:
        target_ndisplay = 3 if int(spec.ndisplay) >= 3 else 2

    resolved_intent_id = intent_id or f"view-{uuid.uuid4().hex}"
    metadata = _metadata_from_intent(resolved_intent_id)

    baseline_step = tuple(int(v) for v in spec.current_step) if spec.current_step else None

    if target_ndisplay >= 3:
        plane_state = _load_plane_state(ledger)
        level_idx = int(spec.current_level)
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

    if order is not None:
        order_value = tuple(int(idx) for idx in order)
    else:
        order_value = tuple(int(idx) for idx in spec.order)
    if not order_value:
        order_value = tuple(int(axis.index) for axis in spec.axes)

    if displayed is not None:
        displayed_value = tuple(int(idx) for idx in displayed)
    else:
        count = min(len(order_value), max(1, int(target_ndisplay)))
        displayed_value = tuple(order_value[-count:]) if count > 0 else tuple()

    axes_list: list[DimsSpecAxis] = []
    order_lookup = {int(axis_idx): pos for pos, axis_idx in enumerate(order_value)}
    displayed_set = {int(idx) for idx in displayed_value}
    for axis in spec.axes:
        axis_index = int(axis.index)
        axes_list.append(
            replace(
                axis,
                order_position=int(order_lookup.get(axis_index, axis.order_position)),
                displayed=axis_index in displayed_set,
            )
        )

    new_spec = replace(
        spec,
        ndisplay=int(target_ndisplay),
        order=tuple(int(v) for v in order_value),
        displayed=tuple(int(v) for v in displayed_value),
        plane_mode=int(target_ndisplay) < 3,
        axes=tuple(axes_list),
    )

    op_entry = ledger.get("scene", "main", "op_seq")
    next_op_seq = int(op_entry.value) + 1 if op_entry is not None and isinstance(op_entry.value, int) else 1

    stored_entries = apply_view_toggle_transaction(
        ledger=ledger,
        op_seq=next_op_seq,
        target_ndisplay=int(target_ndisplay),
        order_value=order_value,
        displayed_value=displayed_value,
        dims_spec_payload=_serialize_dims_spec(new_spec),
        origin=origin,
        timestamp=ts,
    )

    spec_entry = stored_entries.get(("dims", "main", "dims_spec"))
    assert spec_entry is not None, "view toggle transaction must return dims_spec entry"
    assert spec_entry.version is not None, "view toggle transaction must yield versioned dims_spec entry"
    version = int(spec_entry.version)

    logger.debug(
        "view intent updated ndisplay=%d order=%s displayed=%s version=%d origin=%s",
        target_ndisplay,
        order_value,
        displayed_value,
        version,
        origin,
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
        dims_spec=new_spec,
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

    spec = _get_dims_spec(ledger)

    if level_idx >= len(spec.level_shapes):
        level_shapes_list = [tuple(int(dim) for dim in shape) for shape in spec.level_shapes]
        if level_shapes_list:
            template_shape = level_shapes_list[-1]
        else:
            template_shape = tuple(1 for _ in step_tuple)
        while len(level_shapes_list) <= level_idx:
            level_shapes_list.append(tuple(int(dim) for dim in template_shape))
        spec = replace(spec, level_shapes=tuple(level_shapes_list))

    step_tuple = dims_spec_clamp_step(spec, int(level_idx), step_tuple)

    axes_list = []
    for axis_idx, axis in enumerate(spec.axes):
        current_step_val = step_tuple[axis_idx] if axis_idx < len(step_tuple) else axis.current_step
        axes_list.append(replace(axis, current_step=int(current_step_val)))

    new_spec = replace(
        spec,
        current_level=int(level_idx),
        current_step=step_tuple,
        ndisplay=2,
        plane_mode=True,
        axes=tuple(axes_list),
    )

    apply_dims_step_transaction(
        ledger=ledger,
        step=step_tuple,
        dims_spec_payload=_serialize_dims_spec(new_spec),
        origin=origin,
        timestamp=ts,
        op_seq=_next_scene_op_seq(ledger),
        op_kind="plane-restore",
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
    level: Optional[int] = None,
    step: Optional[Sequence[int]] = None,
    level_shape: Optional[Sequence[int]] = None,
    shape: Optional[Sequence[int]] = None,
    applied: Mapping[str, Any] | object | None = None,
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
        if applied is None:
            return default
        if isinstance(applied, Mapping):
            return applied.get(attr, default)
        if hasattr(applied, attr):
            return getattr(applied, attr)
        return default

    level_raw = level
    if level_raw is None:
        level_raw = _applied_value("level")
    if level_raw is None:
        entry = ledger.get("multiscale", "main", "level")
        if entry is not None and isinstance(entry.value, int):
            level_raw = entry.value
    if level_raw is None:
        raise ValueError("level update requires a level index")
    level = int(level_raw)

    step_tuple: Optional[tuple[int, ...]] = None
    if step is not None:
        step_tuple = tuple(int(v) for v in step)
    elif (step_raw := _applied_value("step")) is not None:
        step_tuple = tuple(int(v) for v in step_raw)

    shape_source = level_shape if level_shape is not None else shape
    if shape_source is None:
        shape_source = _applied_value("level_shape")
    if shape_source is None:
        shape_source = _applied_value("shape")
    shape_tuple: Optional[tuple[int, ...]] = None
    if shape_source is not None:
        shape_tuple = tuple(int(v) for v in shape_source)

    spec = _get_dims_spec(ledger)
    current_step = spec.current_step
    if step_tuple is None:
        step_tuple = current_step

    level_shapes_payload: list[tuple[int, ...]] = []
    if spec.level_shapes:
        level_shapes_payload = [tuple(int(v) for v in shape) for shape in spec.level_shapes]
    if shape_tuple is not None:
        while len(level_shapes_payload) <= level:
            level_shapes_payload.append(shape_tuple)
        level_shapes_payload[level] = shape_tuple
    updated_level_shapes = tuple(level_shapes_payload) if level_shapes_payload else tuple()

    volume_payload = _plain_volume_state(volume_state)
    mode_value: Optional[str] = None
    if mode is not None:
        mode_value = mode.name if isinstance(mode, RenderMode) else str(mode)

    next_op_seq = _next_scene_op_seq(ledger)

    level_shapes_final: tuple[tuple[int, ...], ...]
    if updated_level_shapes:
        level_shapes_final = tuple(tuple(int(dim) for dim in shape) for shape in updated_level_shapes)
    else:
        level_shapes_final = tuple(tuple(int(dim) for dim in shape) for shape in spec.level_shapes)

    axis_count = len(spec.axes)
    base_step = step_tuple if step_tuple is not None else ()
    if level_shapes_final:
        previous_level = int(spec.current_level)
        if previous_level != int(level):
            step_tuple = dims_spec_remap_step_for_level(
                spec,
                step=base_step,
                prev_level=previous_level,
                next_level=int(level),
            )
        else:
            step_tuple = dims_spec_clamp_step(spec, int(level), base_step)
    else:
        step_tuple = dims_spec_clamp_step(spec, int(level), base_step)

    axes_list: list[DimsSpecAxis] = []
    for axis in spec.axes:
        axis_index = int(axis.index)
        current_step_value = step_tuple[axis_index] if axis_index < len(step_tuple) else axis.current_step
        per_level_steps = list(axis.per_level_steps)
        if level_shapes_final:
            per_level_steps = [int(shape[axis_index]) for shape in level_shapes_final]
        axes_list.append(
            replace(
                axis,
                current_step=int(current_step_value),
                per_level_steps=tuple(per_level_steps),
                per_level_world=_axis_extents_for_steps(per_level_steps),
            )
        )

    new_spec = replace(
        spec,
        current_level=int(level),
        current_step=step_tuple,
        level_shapes=tuple(level_shapes_final),
        axes=tuple(axes_list),
    )

    stored_entries = apply_level_switch_transaction(
        ledger=ledger,
        level=level,
        step=step_tuple,
        dims_spec_payload=_serialize_dims_spec(new_spec),
        level_shapes=updated_level_shapes if updated_level_shapes else None,
        downgraded=bool(downgraded) if downgraded is not None else None,
        origin=origin,
        timestamp=ts,
        op_seq=next_op_seq,
        op_kind="level-update",
    )

    plane_payload = _plain_plane_state(plane_state)
    if plane_payload is not None:
        plane_model = PlaneState(**plane_payload)
        _store_plane_state(
            ledger,
            plane_model,
            origin=origin,
            timestamp=ts,
            metadata=_metadata_from_intent(intent_id),
        )

    if volume_payload is not None:
        volume_model = VolumeState(**volume_payload)
        _store_volume_state(
            ledger,
            volume_model,
            origin=origin,
            timestamp=ts,
            metadata=_metadata_from_intent(intent_id),
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
        dims_spec=new_spec,
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
        spec = _get_dims_spec(ledger)
        plane_state.target_ndisplay = 2
        plane_state.target_level = int(spec.current_level)
        plane_state.applied_level = int(spec.current_level)
        plane_state.target_step = spec.current_step
        plane_state.applied_step = spec.current_step
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
    "resolve_axis_index",
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
