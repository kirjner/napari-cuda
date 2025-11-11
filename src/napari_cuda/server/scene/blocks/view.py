from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass(frozen=True)
class ViewBlock:
    """Rendered view metadata: mode + displayed axes."""

    mode: Literal["plane", "volume"]
    displayed_axes: tuple[int, ...]
    ndim: int


class ViewBlockPayload(TypedDict):
    mode: Literal["plane", "volume"]
    displayed_axes: tuple[int, ...]
    ndim: int


def view_block_to_payload(block: ViewBlock) -> ViewBlockPayload:
    return {
        "mode": block.mode,
        "displayed_axes": block.displayed_axes,
        "ndim": block.ndim,
    }


def view_block_from_payload(data: ViewBlockPayload) -> ViewBlock:
    return ViewBlock(mode=data["mode"], displayed_axes=data["displayed_axes"], ndim=data["ndim"])
