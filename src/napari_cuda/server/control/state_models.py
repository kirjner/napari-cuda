"""Data transfer objects for server control state flows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


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
class ServerLedgerUpdate:
    """Authoritative state committed to the server ledger."""

    scope: str
    target: str
    key: str
    value: Any
    intent_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Mapping[str, Any]] = None
    origin: Optional[str] = None
    version: Optional[int] = None
    axis_index: Optional[int] = None
    current_step: Optional[tuple[int, ...]] = None
