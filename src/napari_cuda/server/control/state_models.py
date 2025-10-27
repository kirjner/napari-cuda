"""Data transfer objects for server control state flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class BootstrapSceneMetadata:
    """Snapshot of worker bootstrap state for seeding reducer intents."""

    step: tuple[int, ...]
    axis_labels: tuple[str, ...]
    order: tuple[int, ...]
    level_shapes: tuple[tuple[int, ...], ...]
    levels: tuple[dict[str, Any], ...]
    current_level: int
    ndisplay: int
    plane_rect: Optional[tuple[float, float, float, float]] = None
    plane_center: Optional[tuple[float, float]] = None
    plane_zoom: Optional[float] = None


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
