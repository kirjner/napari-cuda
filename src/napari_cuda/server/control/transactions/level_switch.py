"""Ledger transaction helper for multiscale level switches."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def apply_level_switch_transaction(
    *,
    ledger: ServerStateLedger,
    level: int,
    step: Sequence[int],
    level_shapes: Optional[Sequence[Sequence[int]]] = None,
    downgraded: Optional[bool] = None,
    step_metadata: Optional[Mapping[str, object]] = None,
    level_metadata: Optional[Mapping[str, object]] = None,
    level_shapes_metadata: Optional[Mapping[str, object]] = None,
    downgraded_metadata: Optional[Mapping[str, object]] = None,
    viewport_mode: Optional[str] = None,
    viewport_plane_state: Optional[Mapping[str, object]] = None,
    viewport_volume_state: Optional[Mapping[str, object]] = None,
    viewport_metadata: Optional[Mapping[str, object]] = None,
    origin: str = "worker.state.level",
    timestamp: Optional[float] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Record the ledger updates for a level switch and return stored entries."""

    step_tuple = tuple(int(v) for v in step)

    batch_entries: list[tuple] = []
    if level_metadata is None:
        batch_entries.append(("multiscale", "main", "level", int(level)))
    else:
        batch_entries.append(
            ("multiscale", "main", "level", int(level), dict(level_metadata))
        )

    if level_shapes is not None:
        normalized_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in level_shapes
        )
        if level_shapes_metadata is None:
            batch_entries.append(
                ("multiscale", "main", "level_shapes", normalized_shapes),
            )
        else:
            batch_entries.append(
                (
                    "multiscale",
                    "main",
                    "level_shapes",
                    normalized_shapes,
                    dict(level_shapes_metadata),
                ),
            )
    if downgraded is not None:
        if downgraded_metadata is None:
            batch_entries.append(
                ("multiscale", "main", "downgraded", bool(downgraded)),
            )
        else:
            batch_entries.append(
                (
                    "multiscale",
                    "main",
                    "downgraded",
                    bool(downgraded),
                    dict(downgraded_metadata),
                ),
            )

    if step_metadata is not None:
        batch_entries.append(
            ("dims", "main", "current_step", step_tuple, dict(step_metadata)),
        )
    else:
        batch_entries.append(("dims", "main", "current_step", step_tuple))

    if viewport_mode is not None:
        if viewport_metadata is None:
            batch_entries.append(
                ("viewport", "state", "mode", str(viewport_mode)),
            )
        else:
            batch_entries.append(
                (
                    "viewport",
                    "state",
                    "mode",
                    str(viewport_mode),
                    dict(viewport_metadata),
                ),
            )
    if viewport_plane_state is not None:
        payload = dict(viewport_plane_state)
        if viewport_metadata is None:
            batch_entries.append(
                ("viewport", "plane", "state", payload),
            )
        else:
            batch_entries.append(
                (
                    "viewport",
                    "plane",
                    "state",
                    payload,
                    dict(viewport_metadata),
                ),
            )
        applied_level = payload.get("applied_level")
        if applied_level is not None:
            if viewport_metadata is None:
                batch_entries.append(
                    ("view_cache", "plane", "level", int(applied_level)),
                )
            else:
                batch_entries.append(
                    (
                        "view_cache",
                        "plane",
                        "level",
                        int(applied_level),
                        dict(viewport_metadata),
                    ),
                )
        applied_step = payload.get("applied_step")
        if applied_step is not None:
            step_tuple = tuple(int(v) for v in applied_step)
            if viewport_metadata is None:
                batch_entries.append(
                    ("view_cache", "plane", "step", step_tuple),
                )
            else:
                batch_entries.append(
                    (
                        "view_cache",
                        "plane",
                        "step",
                        step_tuple,
                        dict(viewport_metadata),
                    ),
                )
    if viewport_volume_state is not None:
        payload = dict(viewport_volume_state)
        if viewport_metadata is None:
            batch_entries.append(
                ("viewport", "volume", "state", payload),
            )
        else:
            batch_entries.append(
                (
                    "viewport",
                    "volume",
                    "state",
                    payload,
                    dict(viewport_metadata),
                ),
            )

    return ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )


__all__ = [
    "apply_level_switch_transaction",
]
