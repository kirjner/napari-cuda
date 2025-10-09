"""Data transfer objects for server control state flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ClientStateUpdateRequest:
    """Normalised state update submitted by a control-channel client."""

    scope: str
    target: str
    key: str
    value: Any
    intent_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class WorkerStateUpdateConfirmation:
    """State feedback originating from the render/worker pipeline."""

    scope: str
    target: str
    key: str
    step: Tuple[int, ...]
    ndisplay: int
    mode: str
    displayed: Optional[Tuple[int, ...]]
    order: Optional[Tuple[int, ...]]
    axis_labels: Optional[Tuple[str, ...]]
    labels: Optional[Tuple[str, ...]]
    current_level: int
    levels: Tuple[Mapping[str, Any], ...]
    level_shapes: Tuple[Tuple[int, ...], ...]
    downgraded: Optional[bool] = None
    timestamp: Optional[float] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ServerLedgerUpdate:
    """Authoritative state committed to the server ledger."""

    scope: str
    target: str
    key: str
    value: Any
    server_seq: int
    intent_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Mapping[str, Any]] = None
    origin: Optional[str] = None
    version: Any | None = None
    axis_index: Optional[int] = None
    current_step: Optional[tuple[int, ...]] = None
