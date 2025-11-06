"""Ledger transaction helper for 2D ↔ 3D view toggles."""

from __future__ import annotations

from collections.abc import Mapping

from napari_cuda.server.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_view_toggle_transaction(
    *,
    ledger: ServerStateLedger,
    op_seq: int,
    target_ndisplay: int,
    order_value: tuple[int, ...],
    displayed_value: tuple[int, ...],
    dims_spec_payload: Mapping[str, Any],
    origin: str,
    timestamp: float,
) -> dict[PropertyKey, LedgerEntry]:
    """Record the ledger updates for a view toggle intent."""

    return ledger.batch_record_confirmed(
        (
            ("scene", "main", "op_seq", int(op_seq)),
            ("scene", "main", "op_state", "open"),
            ("scene", "main", "op_kind", "view-toggle"),
            ("dims", "main", "dims_spec", dict(dims_spec_payload)),
        ),
        origin=origin,
        timestamp=timestamp,
        dedupe=False,
    )


__all__ = ["apply_view_toggle_transaction"]
