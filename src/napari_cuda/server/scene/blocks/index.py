from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypedDict


@dataclass(frozen=True)
class IndexBlock:
    value: tuple[int, ...]


class IndexBlockPayload(TypedDict):
    value: tuple[int, ...]


def index_block_to_payload(block: IndexBlock) -> IndexBlockPayload:
    return {"value": block.value}


def index_block_from_payload(data: IndexBlockPayload) -> IndexBlock:
    return IndexBlock(value=data["value"])
