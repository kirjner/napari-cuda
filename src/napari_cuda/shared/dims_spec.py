"""DimsSpec: canonical applied dims document.

This module defines the applied dims snapshot document and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

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
    levels: tuple[Mapping[str, Any], ...]
    downgraded: bool | None
    labels: tuple[str, ...] | None

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
    entry = snapshot.get(("dims", "main", "dims_spec"))
    assert entry is not None, "ledger missing dims_spec entry"
    assert isinstance(entry, LedgerEntry), "ledger dims_spec entry malformed"
    spec = dims_spec_from_payload(entry.value)
    assert spec is not None, "dims spec payload missing"
    return spec


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
        "levels": [dict(level) for level in spec.levels],
        "downgraded": spec.downgraded,
        "labels": list(spec.labels) if spec.labels is not None else None,
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
    levels_entry = data.get("levels")
    assert isinstance(levels_entry, Sequence), "dims spec payload requires levels sequence"
    levels: list[Mapping[str, Any]] = []
    for entry in levels_entry:
        assert isinstance(entry, Mapping), "dims spec levels entry must be mapping"
        levels.append(dict(entry))

    labels_entry = data.get("labels")
    labels: tuple[str, ...] | None
    if labels_entry is None:
        labels = None
    else:
        assert isinstance(labels_entry, Sequence), "dims spec labels entry must be sequence"
        labels = tuple(str(v) for v in labels_entry)

    downgraded_entry = data.get("downgraded")
    downgraded_value: bool | None
    if downgraded_entry is None:
        downgraded_value = None
    else:
        downgraded_value = bool(downgraded_entry)

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
        levels=tuple(levels),
        downgraded=downgraded_value,
        labels=labels,
    )


def dims_spec_from_notify_payload(payload: Any) -> DimsSpec | None:
    data = getattr(payload, "dims_spec", None)
    if isinstance(data, Mapping):
        return dims_spec_from_payload(data)
    if data is None:
        return None
    raise TypeError("notify.dims dims_spec payload must be mapping or None")


def validate_ledger_against_dims_spec(spec: DimsSpec, snapshot: LedgerSnapshot) -> None:
    entry = snapshot.get(("dims", "main", "dims_spec"))
    assert entry is not None, "ledger missing dims_spec entry"
    assert isinstance(entry, LedgerEntry), "ledger dims_spec entry malformed"
    snapshot_spec = dims_spec_from_payload(entry.value)
    assert snapshot_spec is not None, "dims spec payload missing"
    assert snapshot_spec == spec, "dims spec payload mismatch"


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
