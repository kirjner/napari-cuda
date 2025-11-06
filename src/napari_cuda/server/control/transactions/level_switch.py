"""Ledger transaction helper for multiscale level switches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_level_switch_transaction(
    *,
    ledger: ServerStateLedger,
    level: int,
    step: Sequence[int],
    dims_spec_payload: Mapping[str, Any],
    level_shapes: Optional[Sequence[Sequence[int]]] = None,
    downgraded: Optional[bool] = None,
    origin: str = "worker.state.level",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> dict[PropertyKey, LedgerEntry]:
    """Record the ledger updates for a level switch and return stored entries."""

    step_tuple = tuple(int(v) for v in step)

    batch_entries: list[tuple] = []
    if op_seq is not None:
        batch_entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            batch_entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            batch_entries.append(("scene", "main", "op_kind", str(op_kind)))
    batch_entries.append(("multiscale", "main", "level", int(level)))

    if level_shapes is not None:
        normalized_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in level_shapes
        )
        batch_entries.append(
            ("multiscale", "main", "level_shapes", normalized_shapes),
        )
    if downgraded is not None:
        batch_entries.append(
            ("multiscale", "main", "downgraded", bool(downgraded)),
        )

    # Persist the requested step with the level switch so the worker can apply
    # the snapshot verbatim without additional remapping.
    batch_entries.append(("dims", "main", "current_step", step_tuple))

    batch_entries.append(("dims", "main", "dims_spec", dict(dims_spec_payload)))

    return ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = [
    "apply_level_switch_transaction",
]
