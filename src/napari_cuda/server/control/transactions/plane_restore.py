"""Ledger transaction helper for deterministic 3D → 2D plane restore."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from napari_cuda.server.control.state_ledger import (
    LedgerEntry,
    PropertyKey,
    ServerStateLedger,
)


def _as_level(value: int | float | str) -> int:
    return int(value)


def _as_step(step: Sequence[int | float | str]) -> Tuple[int, ...]:
    return tuple(int(v) for v in step)


def _as_center(center: Sequence[float | int | str]) -> Tuple[float, float, float]:
    if len(center) < 3:
        raise ValueError("plane restore center requires three components")
    return (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )


def _as_rect(rect: Sequence[float | int | str]) -> Tuple[float, float, float, float]:
    if len(rect) < 4:
        raise ValueError("plane restore rect requires four components")
    return (
        float(rect[0]),
        float(rect[1]),
        float(rect[2]),
        float(rect[3]),
    )


def apply_plane_restore_transaction(
    *,
    ledger: ServerStateLedger,
    level: int | float | str,
    step: Sequence[int | float | str],
    center: Sequence[float | int | str],
    zoom: float | int | str,
    rect: Sequence[float | int | str],
    viewport_plane_state: Optional[Mapping[str, object]] = None,
    viewport_metadata: Optional[Mapping[str, object]] = None,
    origin: str = "control.view.plane_restore",
    timestamp: Optional[float] = None,
    op_seq: Optional[int] = None,
    op_state: Optional[str] = None,
    op_kind: Optional[str] = None,
) -> Dict[PropertyKey, LedgerEntry]:
    """Batch ledger writes for a plane camera restore.

    Parameters
    ----------
    ledger
        Server ledger to update.
    level, step
        Target multiscale level and step indices for the restored plane.
    center, zoom, rect
        Plane camera pose expressed in world coordinates.
    origin
        Ledger origin identifier.
    timestamp
        Optional timestamp to stamp on the ledger entries.
    """

    level_idx = _as_level(level)
    step_tuple = _as_step(step)
    center_tuple = _as_center(center)
    rect_tuple = _as_rect(rect)
    zoom_value = float(zoom)

    batch_entries: list[tuple] = []
    if op_seq is not None:
        batch_entries.append(("scene", "main", "op_seq", int(op_seq)))
        if op_state is not None:
            batch_entries.append(("scene", "main", "op_state", str(op_state)))
        if op_kind is not None:
            batch_entries.append(("scene", "main", "op_kind", str(op_kind)))

    batch_entries.extend(
        [
            ("multiscale", "main", "level", level_idx),
            ("dims", "main", "current_step", step_tuple),
            ("camera", "main", "center", center_tuple),
            ("camera", "main", "zoom", zoom_value),
            ("camera", "main", "rect", rect_tuple),
            ("camera_plane", "main", "center", center_tuple),
            ("camera_plane", "main", "zoom", zoom_value),
            ("camera_plane", "main", "rect", rect_tuple),
            ("view_cache", "plane", "level", level_idx),
            ("view_cache", "plane", "step", step_tuple),
        ]
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

    stored = ledger.batch_record_confirmed(
        batch_entries,
        origin=origin,
        timestamp=timestamp,
    )

    return stored
