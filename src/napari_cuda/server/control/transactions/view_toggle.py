"""Ledger transaction helper for 2D ↔ 3D view toggles."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock
from typing import Dict, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)
from napari_cuda.server.scene import (
    ServerSceneData,
    get_control_meta,
    increment_server_sequence,
)


def _normalize_order(
    *,
    baseline_order: Optional[Sequence[int]],
    requested_order: Optional[Sequence[int]],
    ndim: int,
) -> Tuple[int, ...]:
    existing_order: Tuple[int, ...] = ()
    if requested_order is not None:
        existing_order = tuple(int(idx) for idx in requested_order)
    elif baseline_order is not None:
        existing_order = tuple(int(idx) for idx in baseline_order)

    normalized: list[int] = []
    seen: set[int] = set()
    for axis in existing_order:
        axis_idx = int(axis)
        if 0 <= axis_idx < ndim and axis_idx not in seen:
            normalized.append(axis_idx)
            seen.add(axis_idx)
    for axis_idx in range(ndim):
        if axis_idx not in seen:
            normalized.append(axis_idx)
            seen.add(axis_idx)

    if not normalized:
        normalized = list(range(ndim))

    return tuple(normalized)


def _normalize_displayed(
    *,
    requested_displayed: Optional[Sequence[int]],
    order_value: Tuple[int, ...],
    target_ndisplay: int,
) -> Tuple[int, ...]:
    if requested_displayed is not None:
        return tuple(int(idx) for idx in requested_displayed)
    count = min(len(order_value), max(1, int(target_ndisplay)))
    if count <= 0:
        return tuple()
    return tuple(order_value[-count:])


def apply_view_toggle_transaction(
    *,
    store: ServerSceneData,
    ledger: ServerStateLedger,
    lock: Lock,
    target_ndisplay: int,
    baseline_step: Optional[Sequence[int]],
    baseline_order: Optional[Sequence[int]],
    requested_order: Optional[Sequence[int]],
    requested_displayed: Optional[Sequence[int]],
    origin: str,
    timestamp: float,
) -> Tuple[Dict[PropertyKey, LedgerEntry], Tuple[int, ...], Tuple[int, ...], int]:
    """Record the ledger updates for a view toggle intent."""

    entries: Dict[PropertyKey, LedgerEntry] = {}
    with lock:
        op_seq = increment_server_sequence(store)
    entries.update(
        ledger.batch_record_confirmed(
        (
            ("scene", "main", "op_seq", int(op_seq)),
            ("scene", "main", "op_state", "open"),
            ("scene", "main", "op_kind", "view-toggle"),
        ),
        origin=origin,
        timestamp=timestamp,
        dedupe=False,
    )
    )

    with lock:
        store.last_scene_seq = increment_server_sequence(store)
        server_seq = store.last_scene_seq
        meta = get_control_meta(store, "view", "main", "ndisplay")
        meta.last_server_seq = server_seq
        meta.last_timestamp = timestamp
        store.use_volume = bool(int(target_ndisplay) == 3)

    ndisplay_entry = ledger.record_confirmed(
        "view",
        "main",
        "ndisplay",
        int(target_ndisplay),
        origin=origin,
        timestamp=timestamp,
    )
    entries[("view", "main", "ndisplay")] = ndisplay_entry

    ndim = 0
    if baseline_step is not None:
        ndim = len(tuple(int(v) for v in baseline_step))
    if ndim <= 0 and baseline_order is not None:
        ndim = len(tuple(int(v) for v in baseline_order))
    if ndim <= 0:
        ndim = max(int(target_ndisplay), 1)

    order_value = _normalize_order(
        baseline_order=baseline_order,
        requested_order=requested_order,
        ndim=ndim,
    )
    displayed_value = _normalize_displayed(
        requested_displayed=requested_displayed,
        order_value=order_value,
        target_ndisplay=target_ndisplay,
    )

    dims_entry = ledger.record_confirmed(
        "dims",
        "main",
        "order",
        order_value,
        origin=origin,
        timestamp=timestamp,
    )
    entries[("dims", "main", "order")] = dims_entry
    displayed_entry = ledger.record_confirmed(
        "view",
        "main",
        "displayed",
        displayed_value,
        origin=origin,
        timestamp=timestamp,
    )
    entries[("view", "main", "displayed")] = displayed_entry

    with lock:
        snapshot = store.latest_state
        store.latest_state = replace(
            snapshot,
            ndisplay=int(target_ndisplay),
            order=order_value,
            displayed=displayed_value,
        )

    return entries, order_value, displayed_value, int(server_seq)


__all__ = ["apply_view_toggle_transaction"]
