"""Read helpers for ``ServerStateLedger`` values used by the runtime."""

from __future__ import annotations

from napari_cuda.server.ledger import LedgerEntry, ServerStateLedger
from napari_cuda.shared.dims_spec import (
    DimsSpec,
    dims_spec_from_payload,
    validate_ledger_against_dims_spec,
)


def dims_spec(ledger: ServerStateLedger | None) -> DimsSpec | None:
    if ledger is None:
        return None
    snapshot = ledger.snapshot()
    spec_entry = snapshot.get(("dims", "main", "dims_spec"))
    if spec_entry is not None:
        assert isinstance(spec_entry, LedgerEntry), "ledger dims_spec entry malformed"
        spec = dims_spec_from_payload(spec_entry.value)
        assert spec is not None, "dims spec ledger entry missing payload"
        validate_ledger_against_dims_spec(spec, snapshot)
        return spec
    raise AssertionError("ledger missing dims spec for snapshot")


def _require_spec(ledger: ServerStateLedger | None) -> DimsSpec:
    spec = dims_spec(ledger)
    assert spec is not None, "dims spec missing from ledger snapshot"
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
