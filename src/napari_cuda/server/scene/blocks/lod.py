from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class LodBlock:
    level: int
    roi: tuple[int, ...] | None
    policy: str | None


class LodBlockPayload(TypedDict):
    level: int
    roi: tuple[int, ...] | None
    policy: str | None


def lod_block_to_payload(block: LodBlock) -> LodBlockPayload:
    return {
        "level": block.level,
        "roi": block.roi,
        "policy": block.policy,
    }


def lod_block_from_payload(data: LodBlockPayload) -> LodBlock:
    return LodBlock(level=data["level"], roi=data["roi"], policy=data["policy"])
