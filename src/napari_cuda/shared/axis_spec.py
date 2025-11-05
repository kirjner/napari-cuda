"""Axis semantics shared between server and client stacks.

The ``AxisSpec`` dataclass consolidates all axis interpretation state
into an immutable structure so both control- and render-path code can
reason about dims ordering, displayed axes, world extents, and slab
thickness without ad-hoc heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Sequence

AxisRole = Literal["x", "y", "z", "channel", "time", "depth", "unknown"]

_DEFAULT_ROLE: AxisRole = "unknown"


@dataclass(frozen=True)
class WorldSpan:
    """World-coordinate span for an axis section."""

    start: float
    stop: float
    step: float
    scale: float | None = None

    def steps_to_world(self, steps: float) -> float:
        """Convert a delta (in step units) to world units."""
        return float(steps) * (self.scale if self.scale is not None else self.step)

    def clamp_step(self, value: float, *, step_count: int) -> int:
        """Clamp an index to available steps for the span."""
        if step_count <= 0:
            return 0
        return max(0, min(int(step_count - 1), int(value)))


@dataclass(frozen=True)
class AxisExtent:
    """Axis metadata for a single logical dimension."""

    index: int
    label: str
    role: AxisRole
    displayed: bool
    order_pos: int
    current_step: int
    margin_left_world: float
    margin_right_world: float
    margin_left_steps: float
    margin_right_steps: float
    per_level_steps: tuple[int, ...]
    per_level_world: tuple[WorldSpan | None, ...]

    def world_span(self, level: int) -> WorldSpan | None:
        if not self.per_level_world:
            return None
        if level < 0 or level >= len(self.per_level_world):
            return None
        return self.per_level_world[level]

    def step_count(self, level: int) -> int:
        if not self.per_level_steps:
            return 0
        if level < 0 or level >= len(self.per_level_steps):
            return 0
        return int(self.per_level_steps[level])


@dataclass(frozen=True)
class AxisSpec:
    """Immutable snapshot of axis semantics for a render scene."""

    axes: tuple[AxisExtent, ...]
    ndim: int
    ndisplay: int
    displayed: tuple[int, ...]
    order: tuple[int, ...]
    current_level: int
    level_shapes: tuple[tuple[int, ...], ...]
    plane_mode: bool
    version: int = 1

    def axis_by_index(self, idx: int) -> AxisExtent:
        for axis in self.axes:
            if axis.index == idx:
                return axis
        raise KeyError(f"axis index {idx} not present in spec")

    def axis_by_label(self, target: str) -> AxisExtent:
        lowered = target.strip().lower()
        for axis in self.axes:
            if axis.label == target or axis.label.lower() == lowered:
                return axis
        raise KeyError(f"axis label {target!r} not present in spec")


def _axis_from_index(spec: AxisSpec, axis: int | str) -> AxisExtent:
    if isinstance(axis, int):
        return spec.axis_by_index(axis)
    try:
        return spec.axis_by_label(str(axis))
    except KeyError as exc:
        raise ValueError(f"unknown axis {axis!r}") from exc


def axis_by_index(spec: AxisSpec, idx: int) -> AxisExtent:
    return spec.axis_by_index(idx)


def axis_by_label(spec: AxisSpec, target: str) -> AxisExtent:
    return spec.axis_by_label(target)


def axis_role(spec: AxisSpec, axis: int | str) -> AxisRole:
    return _axis_from_index(spec, axis).role


def clamp_index(spec: AxisSpec, axis: int | str, value: int, level: int | None = None) -> int:
    extent = _axis_from_index(spec, axis)
    lvl = spec.current_level if level is None else int(level)
    steps = extent.step_count(lvl)
    if steps <= 0:
        return 0
    return max(0, min(steps - 1, int(value)))


def margin_span(
    spec: AxisSpec,
    axis: int | str,
    *,
    level: int | None = None,
    prefer_world: bool = True,
) -> tuple[float, float]:
    extent = _axis_from_index(spec, axis)
    if prefer_world:
        return (float(extent.margin_left_world), float(extent.margin_right_world))
    lvl = spec.current_level if level is None else int(level)
    span = extent.world_span(lvl)
    if span is None:
        return (float(extent.margin_left_steps), float(extent.margin_right_steps))
    return (
        span.steps_to_world(extent.margin_left_steps),
        span.steps_to_world(extent.margin_right_steps),
    )


def world_to_steps(
    spec: AxisSpec,
    axis: int | str,
    delta: float,
    *,
    level: int | None = None,
) -> float:
    extent = _axis_from_index(spec, axis)
    lvl = spec.current_level if level is None else int(level)
    span = extent.world_span(lvl)
    if span is None:
        return float(delta)
    scale = span.scale if span.scale is not None else span.step
    if scale == 0.0:
        return 0.0
    return float(delta) / float(scale)


def steps_to_world(
    spec: AxisSpec,
    axis: int | str,
    steps: float,
    *,
    level: int | None = None,
) -> float:
    extent = _axis_from_index(spec, axis)
    lvl = spec.current_level if level is None else int(level)
    span = extent.world_span(lvl)
    if span is None:
        return float(steps)
    return span.steps_to_world(steps)


def with_updated_margins(
    spec: AxisSpec,
    axis: int | str,
    *,
    margin_left_world: float | None = None,
    margin_right_world: float | None = None,
    margin_left_steps: float | None = None,
    margin_right_steps: float | None = None,
) -> AxisSpec:
    extent = _axis_from_index(spec, axis)
    updated = replace(
        extent,
        margin_left_world=float(margin_left_world)
        if margin_left_world is not None
        else extent.margin_left_world,
        margin_right_world=float(margin_right_world)
        if margin_right_world is not None
        else extent.margin_right_world,
        margin_left_steps=float(margin_left_steps)
        if margin_left_steps is not None
        else extent.margin_left_steps,
        margin_right_steps=float(margin_right_steps)
        if margin_right_steps is not None
        else extent.margin_right_steps,
    )
    axes = list(spec.axes)
    axes[extent.index] = updated
    return replace(spec, axes=tuple(axes))


def axis_world_span(spec: AxisSpec, axis: int | str, level: int | None = None) -> WorldSpan | None:
    extent = _axis_from_index(spec, axis)
    lvl = spec.current_level if level is None else int(level)
    return extent.world_span(lvl)


def derive_axis_labels(spec: AxisSpec) -> tuple[str, ...]:
    labels = [""] * spec.ndim
    for axis in spec.axes:
        idx = axis.index
        if 0 <= idx < len(labels):
            labels[idx] = axis.label
    for idx, label in enumerate(labels):
        if not label:
            labels[idx] = f"axis-{idx}"
    return tuple(labels)


def derive_order(spec: AxisSpec) -> tuple[int, ...]:
    ordering = sorted(spec.axes, key=lambda ax: ax.order_pos)
    return tuple(axis.index for axis in ordering)


def derive_displayed(spec: AxisSpec) -> tuple[int, ...]:
    ordering = derive_order(spec)
    count = min(len(ordering), max(1, spec.ndisplay))
    return tuple(ordering[-count:])


def derive_margins(spec: AxisSpec, *, prefer_world: bool = False) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left: list[float] = []
    right: list[float] = []
    for axis in spec.axes:
        if prefer_world:
            left.append(float(axis.margin_left_world))
            right.append(float(axis.margin_right_world))
        else:
            left.append(float(axis.margin_left_steps))
            right.append(float(axis.margin_right_steps))
    return (tuple(left), tuple(right))


def derive_current_step(spec: AxisSpec) -> tuple[int, ...]:
    """Return the per-axis slider positions ordered by axis index."""
    if not spec.axes:
        return tuple()
    ndim = max(spec.ndim, max(axis.index for axis in spec.axes) + 1)
    steps = [0] * ndim
    for axis in spec.axes:
        idx = max(0, min(ndim - 1, int(axis.index)))
        steps[idx] = int(axis.current_step)
    return tuple(steps)


def _world_span_to_payload(span: WorldSpan | None) -> Mapping[str, float] | None:
    if span is None:
        return None
    payload: MutableMapping[str, float] = {
        "start": float(span.start),
        "stop": float(span.stop),
        "step": float(span.step),
    }
    if span.scale is not None:
        payload["scale"] = float(span.scale)
    return payload


def _world_span_from_payload(payload: Mapping[str, Any] | None) -> WorldSpan | None:
    if payload is None:
        return None
    start = float(payload.get("start", 0.0))
    stop = float(payload.get("stop", 0.0))
    step = float(payload.get("step", 1.0))
    scale = payload.get("scale")
    scale_val = None if scale is None else float(scale)
    return WorldSpan(start=start, stop=stop, step=step, scale=scale_val)


def axis_spec_to_payload(spec: AxisSpec) -> dict[str, Any]:
    axes_payload: list[dict[str, Any]] = []
    for axis in spec.axes:
        axes_payload.append(
            {
                "index": int(axis.index),
                "label": str(axis.label),
                "role": str(axis.role or _DEFAULT_ROLE),
                "displayed": bool(axis.displayed),
                "order_pos": int(axis.order_pos),
                "current_step": int(axis.current_step),
                "margin_left_world": float(axis.margin_left_world),
                "margin_right_world": float(axis.margin_right_world),
                "margin_left_steps": float(axis.margin_left_steps),
                "margin_right_steps": float(axis.margin_right_steps),
                "per_level_steps": [int(v) for v in axis.per_level_steps],
                "per_level_world": [
                    _world_span_to_payload(span) for span in axis.per_level_world
                ],
            }
        )
    return {
        "version": int(spec.version),
        "ndim": int(spec.ndim),
        "ndisplay": int(spec.ndisplay),
        "displayed": [int(v) for v in spec.displayed],
        "order": [int(v) for v in spec.order],
        "current_level": int(spec.current_level),
        "level_shapes": [[int(dim) for dim in shape] for shape in spec.level_shapes],
        "plane_mode": bool(spec.plane_mode),
        "axes": axes_payload,
    }


def axis_spec_from_payload(payload: Mapping[str, Any]) -> AxisSpec:
    version = int(payload.get("version", 1))
    ndim = int(payload.get("ndim", 0))
    ndisplay = int(payload.get("ndisplay", 2))
    displayed_seq = payload.get("displayed", [])
    order_seq = payload.get("order", [])
    current_level = int(payload.get("current_level", 0))
    level_shapes_field = payload.get("level_shapes", [])
    plane_mode = bool(payload.get("plane_mode", False))
    axes_field = payload.get("axes", [])

    level_shapes: list[tuple[int, ...]] = []
    for entry in level_shapes_field:
        if isinstance(entry, Sequence):
            level_shapes.append(tuple(int(dim) for dim in entry))

    axes: list[AxisExtent] = []
    for axis_entry in axes_field or ():
        if not isinstance(axis_entry, Mapping):
            continue
        per_level_world_field = axis_entry.get("per_level_world", [])
        per_level_world: list[WorldSpan | None] = []
        for span_entry in per_level_world_field or ():
            if isinstance(span_entry, Mapping):
                per_level_world.append(_world_span_from_payload(span_entry))
            else:
                per_level_world.append(None)

        per_level_steps_field = axis_entry.get("per_level_steps", [])
        per_level_steps = tuple(int(value) for value in per_level_steps_field or ())

        role_field = str(axis_entry.get("role", _DEFAULT_ROLE) or _DEFAULT_ROLE).strip().lower()
        role: AxisRole = role_field if role_field in {"x", "y", "z", "channel", "time", "depth"} else _DEFAULT_ROLE

        axes.append(
            AxisExtent(
                index=int(axis_entry.get("index", len(axes))),
                label=str(axis_entry.get("label", f"axis-{len(axes)}")),
                role=role,
                displayed=bool(axis_entry.get("displayed", False)),
                order_pos=int(axis_entry.get("order_pos", len(axes))),
                current_step=int(axis_entry.get("current_step", 0)),
                margin_left_world=float(axis_entry.get("margin_left_world", 0.0)),
                margin_right_world=float(axis_entry.get("margin_right_world", 0.0)),
                margin_left_steps=float(axis_entry.get("margin_left_steps", 0.0)),
                margin_right_steps=float(axis_entry.get("margin_right_steps", 0.0)),
                per_level_steps=per_level_steps,
                per_level_world=tuple(per_level_world),
            )
        )

    axes_sorted = sorted(axes, key=lambda axis: axis.index)

    spec = AxisSpec(
        axes=tuple(axes_sorted),
        ndim=int(ndim) if ndim > 0 else len(axes_sorted),
        ndisplay=ndisplay if ndisplay > 0 else 2,
        displayed=tuple(int(v) for v in displayed_seq or ()),
        order=tuple(int(v) for v in order_seq or ()),
        current_level=current_level,
        level_shapes=tuple(level_shapes),
        plane_mode=plane_mode,
        version=version,
    )

    # Populate derived fallbacks if missing.
    if not spec.order:
        spec = replace(spec, order=derive_order(spec))
    if not spec.displayed:
        spec = replace(spec, displayed=derive_displayed(spec))

    return spec


def fabricate_axis_spec(
    *,
    ndim: int,
    ndisplay: int,
    current_level: int,
    level_shapes: Sequence[Sequence[int]],
    order: Sequence[int] | None = None,
    displayed: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    current_step: Sequence[int] | None = None,
    margin_left_world: Sequence[float] | None = None,
    margin_right_world: Sequence[float] | None = None,
) -> AxisSpec:
    """Fabricate a minimal spec when upstream metadata is unavailable."""

    nd = max(1, int(ndim))
    lvl_shapes = [
        tuple(int(dim) for dim in shape) if isinstance(shape, Sequence) else tuple(0 for _ in range(nd))
        for shape in level_shapes
    ]
    while len(lvl_shapes) <= current_level:
        lvl_shapes.append(tuple(0 for _ in range(nd)))

    order_tuple = tuple(int(idx) for idx in order) if order is not None else tuple(range(nd))
    displayed_tuple = (
        tuple(int(idx) for idx in displayed)
        if displayed is not None
        else tuple(order_tuple[-min(len(order_tuple), max(1, int(ndisplay))):])
    )

    axes: list[AxisExtent] = []
    step_values = list(current_step[:nd]) if current_step is not None else [0] * nd
    if len(step_values) < nd:
        step_values.extend([0] * (nd - len(step_values)))

    left_values = list(margin_left_world[:nd]) if margin_left_world is not None else [0.0] * nd
    if len(left_values) < nd:
        left_values.extend([0.0] * (nd - len(left_values)))

    right_values = list(margin_right_world[:nd]) if margin_right_world is not None else [0.0] * nd
    if len(right_values) < nd:
        right_values.extend([0.0] * (nd - len(right_values)))

    for axis_idx in range(nd):
        label = (
            str(labels[axis_idx])
            if labels is not None and axis_idx < len(labels) and str(labels[axis_idx]).strip()
            else f"axis-{axis_idx}"
        )
        role_val = _DEFAULT_ROLE
        if roles is not None and axis_idx < len(roles):
            candidate = str(roles[axis_idx]).strip().lower()
            if candidate in {"x", "y", "z", "channel", "time", "depth"}:
                role_val = candidate  # type: ignore[assignment]

        per_level_steps = []
        per_level_world = []
        for shape in lvl_shapes:
            step_count = int(shape[axis_idx]) if axis_idx < len(shape) else 0
            per_level_steps.append(step_count)
            if step_count > 0:
                per_level_world.append(WorldSpan(start=0.0, stop=float(max(0, step_count - 1)), step=1.0, scale=1.0))
            else:
                per_level_world.append(None)

        axes.append(
            AxisExtent(
                index=axis_idx,
                label=label,
                role=role_val,  # type: ignore[arg-type]
                displayed=axis_idx in displayed_tuple,
                order_pos=order_tuple.index(axis_idx) if axis_idx in order_tuple else axis_idx,
                current_step=int(step_values[axis_idx]),
                margin_left_world=float(left_values[axis_idx]),
                margin_right_world=float(right_values[axis_idx]),
                margin_left_steps=float(left_values[axis_idx]),
                margin_right_steps=float(right_values[axis_idx]),
                per_level_steps=tuple(per_level_steps),
                per_level_world=tuple(per_level_world),
            )
        )

    return AxisSpec(
        axes=tuple(axes),
        ndim=nd,
        ndisplay=max(1, int(ndisplay)),
        displayed=displayed_tuple,
        order=order_tuple,
        current_level=int(current_level),
        level_shapes=tuple(lvl_shapes),
        plane_mode=bool(ndisplay < 3),
    )


__all__ = [
    "AxisExtent",
    "AxisRole",
    "AxisSpec",
    "WorldSpan",
    "axis_by_index",
    "axis_by_label",
    "axis_role",
    "axis_spec_from_payload",
    "axis_spec_to_payload",
    "axis_world_span",
    "clamp_index",
    "derive_axis_labels",
    "derive_displayed",
    "derive_margins",
    "derive_order",
    "derive_current_step",
    "fabricate_axis_spec",
    "margin_span",
    "steps_to_world",
    "world_to_steps",
    "with_updated_margins",
]
