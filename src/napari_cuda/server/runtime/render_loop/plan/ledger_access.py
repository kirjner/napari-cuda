"""Read helpers for ``ServerStateLedger`` values used by the runtime."""

from __future__ import annotations

from typing import Optional

from napari_cuda.server.control.state_ledger import ServerStateLedger


def step(ledger: Optional[ServerStateLedger]) -> Optional[tuple[int, ...]]:
    if ledger is None:
        return None
    entry = ledger.get("dims", "main", "current_step")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


def level(ledger: Optional[ServerStateLedger]) -> Optional[int]:
    if ledger is None:
        return None
    entry = ledger.get("multiscale", "main", "level")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, int):
        return int(value)
    return None


def axis_labels(ledger: Optional[ServerStateLedger]) -> Optional[tuple[str, ...]]:
    if ledger is None:
        return None
    entry = ledger.get("dims", "main", "axis_labels")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return None


def order(ledger: Optional[ServerStateLedger]) -> Optional[tuple[int, ...]]:
    if ledger is None:
        return None
    entry = ledger.get("dims", "main", "order")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


def ndisplay(ledger: Optional[ServerStateLedger]) -> Optional[int]:
    if ledger is None:
        return None
    entry = ledger.get("view", "main", "ndisplay")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, int):
        return int(value)
    return None


def displayed(ledger: Optional[ServerStateLedger]) -> Optional[tuple[int, ...]]:
    if ledger is None:
        return None
    entry = ledger.get("view", "main", "displayed")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


def level_shapes(ledger: Optional[ServerStateLedger]) -> Optional[tuple[tuple[int, ...], ...]]:
    if ledger is None:
        return None
    entry = ledger.get("multiscale", "main", "level_shapes")
    if entry is None:
        return None
    value = entry.value
    if isinstance(value, (list, tuple)):
        shapes: list[tuple[int, ...]] = []
        for shape in value:
            if isinstance(shape, (list, tuple)):
                shapes.append(tuple(int(v) for v in shape))
        if shapes:
            return tuple(shapes)
    return None


__all__ = [
    "axis_labels",
    "displayed",
    "level",
    "level_shapes",
    "ndisplay",
    "order",
    "step",
]
