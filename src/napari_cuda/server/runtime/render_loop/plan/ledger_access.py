"""Read helpers for ``ServerStateLedger`` values used by the runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import DimsSpec
from napari_cuda.shared.dims_spec import build_dims_spec_from_ledger


def dims_spec(ledger: ServerStateLedger | None) -> DimsSpec | None:
    if ledger is None:
        return None
    return _memoized_spec(id(ledger), ledger)


@lru_cache(maxsize=16)
def _memoized_spec(_ledger_id: int, ledger: ServerStateLedger | None) -> DimsSpec | None:
    if ledger is None:
        return None
    snapshot = ledger.snapshot()
    return build_dims_spec_from_ledger(snapshot)


def _require_spec(ledger: ServerStateLedger | None) -> DimsSpec:
    spec = dims_spec(ledger)
    assert spec is not None, "axes spec missing from ledger snapshot"
    return spec


def step(ledger: ServerStateLedger | None) -> tuple[int, ...]:
    return _require_spec(ledger).current_step


def level(ledger: ServerStateLedger | None) -> int:
    return int(_require_spec(ledger).current_level)


def axis_labels(ledger: ServerStateLedger | None) -> tuple[str, ...]:
    spec = _require_spec(ledger)
    return tuple(axis.label for axis in spec.axes)


def order(ledger: ServerStateLedger | None) -> tuple[int, ...]:
    return tuple(int(v) for v in _require_spec(ledger).order)


def ndisplay(ledger: ServerStateLedger | None) -> int:
    return int(_require_spec(ledger).ndisplay)


def displayed(ledger: ServerStateLedger | None) -> tuple[int, ...]:
    return tuple(int(v) for v in _require_spec(ledger).displayed)


def level_shapes(ledger: ServerStateLedger | None) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(dim) for dim in shape) for shape in _require_spec(ledger).level_shapes)


__all__ = [
    "dims_spec",
    "axis_labels",
    "displayed",
    "level",
    "level_shapes",
    "ndisplay",
    "order",
    "step",
]
