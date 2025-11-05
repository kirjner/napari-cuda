"""DimsSpec: canonical applied dims document.

This module defines the applied dims snapshot document and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from napari_cuda.server.state_ledger import LedgerEntry


_DIMS_SPEC_VERSION = 1


def _default_axis_labels(ndim: int) -> tuple[str, ...]:
    if ndim <= 0:
        return ()
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 2:
        return ("y", "x")
    if ndim == 1:
        return ("d0",)
    return tuple(f"d{i}" for i in range(ndim))


@dataclass(frozen=True)
class AxisExtent:
    start: float
    stop: float
    step: float


@dataclass(frozen=True)
class DimsSpecAxis:
    index: int
    label: str
    role: str
    displayed: bool
    order_position: int
    current_step: int
    margin_left_steps: float
    margin_right_steps: float
    margin_left_world: float
    margin_right_world: float
    per_level_steps: tuple[int, ...]
    per_level_world: tuple[AxisExtent, ...]


@dataclass(frozen=True)
class DimsSpec:
    version: int
    ndim: int
    ndisplay: int
    order: tuple[int, ...]
    displayed: tuple[int, ...]
    current_level: int
    current_step: tuple[int, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    plane_mode: bool
    axes: tuple[DimsSpecAxis, ...]

    def axis_by_label(self, label: str) -> DimsSpecAxis:
        target = str(label)
        for axis in self.axes:
            if axis.label == target:
                return axis
        raise KeyError(f"dims spec missing label={target}")

    def axis_by_index(self, index: int) -> DimsSpecAxis:
        target = int(index)
        for axis in self.axes:
            if axis.index == target:
                return axis
        raise KeyError(f"dims spec missing index={target}")


LedgerSnapshot = Mapping[tuple[str, str, str], Any]


def build_dims_spec_from_ledger(snapshot: LedgerSnapshot) -> DimsSpec:
    current_step_raw = _require_value(snapshot, "dims", "main", "current_step")
    level_shapes_raw = _require_value(snapshot, "multiscale", "main", "level_shapes")
    current_level_raw = _require_value(snapshot, "multiscale", "main", "level")
    ndisplay_raw = _require_value(snapshot, "view", "main", "ndisplay")
    order_raw = _require_value(snapshot, "dims", "main", "order")
    displayed_raw = _require_value(snapshot, "view", "main", "displayed")

    axis_labels_value = _optional_value(snapshot, "dims", "main", "axis_labels")
    margin_left_value = _optional_value(snapshot, "dims", "main", "margin_left")
    margin_right_value = _optional_value(snapshot, "dims", "main", "margin_right")

    current_step = _as_int_tuple(current_step_raw)
    level_shapes = _as_shape_tuple(level_shapes_raw)
    current_level = int(current_level_raw)
    ndisplay = int(ndisplay_raw)
    order = _as_int_tuple(order_raw)
    displayed = _as_int_tuple(displayed_raw)

    ndim_candidates = [
        len(current_step),
        len(order),
    ]
    if level_shapes:
        reference_level = min(max(current_level, 0), len(level_shapes) - 1)
        ndim_candidates.append(len(level_shapes[reference_level]))
    if axis_labels_value is not None:
        ndim_candidates.append(len(tuple(str(v) for v in axis_labels_value)))
    ndim = max([value for value in ndim_candidates if value > 0], default=1)

    axis_labels = _resolve_axis_labels(axis_labels_value, ndim)

    margin_left_steps = _coerce_margin_array(margin_left_value, ndim)
    margin_right_steps = _coerce_margin_array(margin_right_value, ndim)

    axes: list[DimsSpecAxis] = []
    for axis_index in range(ndim):
        axis_label = axis_labels[axis_index]
        role = _infer_axis_role(axis_label, axis_index)

        order_position = order.index(axis_index)
        displayed_flag = axis_index in displayed
        current_step_value = current_step[axis_index] if axis_index < len(current_step) else 0

        per_level_steps = _extract_axis_steps(level_shapes, axis_index)
        per_level_world = tuple(_extent_for_step(count) for count in per_level_steps)

        margin_left_step = margin_left_steps[axis_index]
        margin_right_step = margin_right_steps[axis_index]

        axes.append(
            DimsSpecAxis(
                index=axis_index,
                label=axis_label,
                role=role,
                displayed=displayed_flag,
                order_position=order_position,
                current_step=current_step_value,
                margin_left_steps=margin_left_step,
                margin_right_steps=margin_right_step,
                margin_left_world=margin_left_step,
                margin_right_world=margin_right_step,
                per_level_steps=per_level_steps,
                per_level_world=per_level_world,
            )
        )

    return DimsSpec(
        version=_DIMS_SPEC_VERSION,
        ndim=ndim,
        ndisplay=ndisplay,
        order=order,
        displayed=displayed,
        current_level=current_level,
        current_step=current_step,
        level_shapes=level_shapes,
        plane_mode=ndisplay < 3,
        axes=tuple(axes),
    )


def dims_spec_to_payload(spec: DimsSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    dims_payload = []
    for axis in spec.axes:
        dims_payload.append(
            {
                "index": axis.index,
                "label": axis.label,
                "role": axis.role,
                "displayed": axis.displayed,
                "order_pos": axis.order_position,
                "current_step": axis.current_step,
                "margin_left_steps": axis.margin_left_steps,
                "margin_right_steps": axis.margin_right_steps,
                "margin_left_world": axis.margin_left_world,
                "margin_right_world": axis.margin_right_world,
                "per_level_steps": list(axis.per_level_steps),
                "per_level_world": [
                    {"start": extent.start, "stop": extent.stop, "step": extent.step}
                    for extent in axis.per_level_world
                ],
            }
        )
    return {
        "version": spec.version,
        "ndim": spec.ndim,
        "ndisplay": spec.ndisplay,
        "order": list(spec.order),
        "displayed": list(spec.displayed),
        "current_level": spec.current_level,
        "current_step": list(spec.current_step),
        "level_shapes": [list(shape) for shape in spec.level_shapes],
        "plane_mode": spec.plane_mode,
        "axes": dims_payload,
    }


def dims_spec_from_payload(data: Mapping[str, Any] | None) -> DimsSpec | None:
    if data is None:
        return None
    dims_payload = []
    axes_seq = data.get("axes")
    assert isinstance(axes_seq, Sequence), "dims spec payload requires axes sequence"
    for entry in axes_seq:
        assert isinstance(entry, Mapping), "dims spec axis payload must be mapping"
        per_level_steps_entry = entry.get("per_level_steps")
        per_level_world_entry = entry.get("per_level_world")
        assert isinstance(per_level_steps_entry, Sequence), "per_level_steps must be sequence"
        assert isinstance(per_level_world_entry, Sequence), "per_level_world must be sequence"
        dims_payload.append(
            DimsSpecAxis(
                index=int(entry["index"]),
                label=str(entry["label"]),
                role=str(entry["role"]),
                displayed=bool(entry["displayed"]),
                order_position=int(entry["order_pos"]),
                current_step=int(entry["current_step"]),
                margin_left_steps=float(entry["margin_left_steps"]),
                margin_right_steps=float(entry["margin_right_steps"]),
                margin_left_world=float(entry["margin_left_world"]),
                margin_right_world=float(entry["margin_right_world"]),
                per_level_steps=tuple(int(v) for v in per_level_steps_entry),
                per_level_world=tuple(
                    AxisExtent(
                        start=float(extent_entry["start"]),
                        stop=float(extent_entry["stop"]),
                        step=float(extent_entry["step"]),
                    )
                    for extent_entry in per_level_world_entry
                ),
            )
        )
    return DimsSpec(
        version=int(data["version"]),
        ndim=int(data["ndim"]),
        ndisplay=int(data["ndisplay"]),
        order=tuple(int(v) for v in data["order"]),
        displayed=tuple(int(v) for v in data["displayed"]),
        current_level=int(data["current_level"]),
        current_step=tuple(int(v) for v in data["current_step"]),
        level_shapes=tuple(tuple(int(dim) for dim in shape) for shape in data["level_shapes"]),
        plane_mode=bool(data["plane_mode"]),
        axes=tuple(dims_payload),
    )


def dims_spec_from_notify_payload(payload: Any) -> DimsSpec | None:
    data = getattr(payload, "dims_spec", None)
    if isinstance(data, Mapping):
        return dims_spec_from_payload(data)
    if data is None:
        return None
    raise TypeError("notify.dims dims_spec payload must be mapping or None")


def validate_ledger_against_dims_spec(spec: DimsSpec, snapshot: LedgerSnapshot) -> None:
    current_step_raw = _require_value(snapshot, "dims", "main", "current_step")
    level_shapes_raw = _require_value(snapshot, "multiscale", "main", "level_shapes")
    current_level_raw = _require_value(snapshot, "multiscale", "main", "level")
    ndisplay_raw = _require_value(snapshot, "view", "main", "ndisplay")
    order_raw = _require_value(snapshot, "dims", "main", "order")
    displayed_raw = _require_value(snapshot, "view", "main", "displayed")
    axis_labels_raw = _optional_value(snapshot, "dims", "main", "axis_labels")

    assert spec.current_step == _as_int_tuple(current_step_raw), "ledger current_step mismatch"
    assert spec.level_shapes == _as_shape_tuple(level_shapes_raw), "ledger level_shapes mismatch"
    assert spec.current_level == int(current_level_raw), "ledger current_level mismatch"
    assert spec.ndisplay == int(ndisplay_raw), "ledger ndisplay mismatch"
    assert spec.order == _as_int_tuple(order_raw), "ledger order mismatch"
    assert spec.displayed == _as_int_tuple(displayed_raw), "ledger displayed mismatch"

    if axis_labels_raw is not None:
        axis_labels = tuple(str(v) for v in axis_labels_raw)
        for axis in spec.axes:
            assert axis.label == axis_labels[axis.index], "ledger axis_labels mismatch"


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    assert isinstance(value, Iterable), "expected iterable for integer tuple conversion"
    return tuple(int(v) for v in value)


def _as_shape_tuple(value: Any) -> tuple[tuple[int, ...], ...]:
    assert isinstance(value, Iterable), "expected iterable for level_shapes"
    shapes: list[tuple[int, ...]] = []
    for entry in value:
        assert isinstance(entry, Iterable), "level_shapes entry must be iterable"
        shapes.append(tuple(int(dim) for dim in entry))
    return tuple(shapes)


def _extent_for_step(count: int) -> AxisExtent:
    size = max(int(count), 0)
    if size <= 0:
        return AxisExtent(start=0.0, stop=0.0, step=1.0)
    stop = float(size - 1)
    return AxisExtent(start=0.0, stop=stop, step=1.0)


def _require_value(snapshot: LedgerSnapshot, scope: str, target: str, key: str) -> Any:
    value = _optional_value(snapshot, scope, target, key)
    assert value is not None, f"ledger missing {scope}/{key}"
    return value


def _optional_value(snapshot: LedgerSnapshot, scope: str, target: str, key: str) -> Any:
    entry = snapshot.get((scope, target, key))
    if entry is None:
        return None
    return entry.value if hasattr(entry, "value") else entry


def _resolve_axis_labels(raw: Any, ndim: int) -> tuple[str, ...]:
    if raw is None:
        return _default_axis_labels(ndim)
    labels = tuple(str(v) for v in raw)
    assert len(labels) == ndim, "axis_labels length mismatch"
    return labels


def _coerce_margin_array(raw: Any, ndim: int) -> tuple[float, ...]:
    if raw is None:
        return tuple(0.0 for _ in range(ndim))
    assert isinstance(raw, Iterable), "margin values must be iterable"
    values = tuple(float(v) for v in raw)
    assert len(values) == ndim, "margin array length mismatch"
    return values


def _extract_axis_steps(level_shapes: tuple[tuple[int, ...], ...], axis_index: int) -> tuple[int, ...]:
    entries: list[int] = []
    for shape in level_shapes:
        assert axis_index < len(shape), "axis index exceeds level shape"
        entries.append(int(shape[axis_index]))
    return tuple(entries)


def _infer_axis_role(label: str, index: int) -> str:
    text = label.strip().lower()
    if text in {"z", "depth"}:
        return "z"
    if text in {"y", "row"}:
        return "y"
    if text in {"x", "col", "column"}:
        return "x"
    if text in {"c", "channel"}:
        return "channel"
    if text in {"t", "time"}:
        return "time"
    return f"axis-{index}"


__all__ = [
    "DimsSpec",
    "DimsSpecAxis",
    "build_dims_spec_from_ledger",
    "dims_spec_from_notify_payload",
    "dims_spec_from_payload",
    "dims_spec_to_payload",
    "validate_ledger_against_dims_spec",
    # Shared
    "AxisExtent",
]
