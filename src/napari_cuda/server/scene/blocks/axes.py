from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypedDict


@dataclass(frozen=True)
class AxisExtentBlock:
    start: float
    stop: float
    step: float


@dataclass(frozen=True)
class AxisBlock:
    axis_id: int
    label: str
    role: str
    displayed: bool
    world_extent: AxisExtentBlock
    margin_left_world: float
    margin_right_world: float


@dataclass(frozen=True)
class AxesBlock:
    axes: tuple[AxisBlock, ...]

    def axis_labels(self) -> tuple[str, ...]:
        return tuple(axis.label for axis in self.axes)


class AxisExtentPayload(TypedDict):
    start: float
    stop: float
    step: float


class AxisPayload(TypedDict):
    axis_id: int
    label: str
    role: str
    displayed: bool
    world_extent: AxisExtentPayload
    margin_left_world: float
    margin_right_world: float


def axes_to_payload(block: AxesBlock) -> tuple[AxisPayload, ...]:
    payload: list[AxisPayload] = []
    for axis in block.axes:
        payload.append(
            AxisPayload(
                axis_id=axis.axis_id,
                label=axis.label,
                role=axis.role,
                displayed=axis.displayed,
                world_extent=AxisExtentPayload(
                    start=axis.world_extent.start,
                    stop=axis.world_extent.stop,
                    step=axis.world_extent.step,
                ),
                margin_left_world=axis.margin_left_world,
                margin_right_world=axis.margin_right_world,
            )
        )
    return tuple(payload)


def axes_from_payload(data: Iterable[AxisPayload]) -> AxesBlock:
    axes: list[AxisBlock] = []
    for entry in data:
        world_extent = entry["world_extent"]
        extent = AxisExtentBlock(
            start=world_extent["start"],
            stop=world_extent["stop"],
            step=world_extent["step"],
        )
        axes.append(
            AxisBlock(
                axis_id=entry["axis_id"],
                label=entry["label"],
                role=entry["role"],
                displayed=entry["displayed"],
                world_extent=extent,
                margin_left_world=entry["margin_left_world"],
                margin_right_world=entry["margin_right_world"],
            )
        )
    return AxesBlock(axes=tuple(axes))
