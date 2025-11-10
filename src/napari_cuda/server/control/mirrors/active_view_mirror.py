"""Mirror that projects ActiveView ledger state to notify.level."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.protocol.messages import NotifyLevelPayload
from napari_cuda.server.state_ledger import LedgerEntry, LedgerEvent, ServerStateLedger

ScheduleFn = Callable[[Awaitable[None], str], None]
BroadcastFn = Callable[[NotifyLevelPayload], Awaitable[None]]


@dataclass(frozen=True)
class _Key:
    scope: str
    target: str
    key: str


class ActiveViewMirror:
    """Project ActiveView ledger updates into notify.level broadcasts."""

    _WATCH_KEYS: tuple[_Key, ...] = (
        _Key("viewport", "active", "state"),
    )

    def __init__(
        self,
        *,
        ledger: ServerStateLedger,
        broadcaster: BroadcastFn,
        schedule: ScheduleFn,
    ) -> None:
        self._ledger = ledger
        self._broadcaster = broadcaster
        self._schedule = schedule
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ActiveViewMirror already started")
            self._started = True
        for key in self._WATCH_KEYS:
            self._ledger.subscribe(key.scope, key.target, key.key, self._on_ledger_event)

        # Emit initial state if present
        snapshot = self._ledger.snapshot()
        entry = snapshot.get(("viewport", "active", "state"))
        if isinstance(entry, LedgerEntry) and isinstance(entry.value, dict):
            payload = self._build_payload(entry.value)
            self._schedule(self._broadcaster(payload), "mirror-active-view-init")

    # ------------------------------------------------------------------
    def _on_ledger_event(self, event: LedgerEvent) -> None:
        # Build payload from the event value (authoritative)
        if not isinstance(event.value, dict):
            return
        payload = self._build_payload(event.value)
        self._schedule(self._broadcaster(payload), "mirror-active-view")

    @staticmethod
    def _build_payload(value: dict[str, Any]) -> NotifyLevelPayload:
        level_value = int(value.get("level", 0))
        mode_value = str(value.get("mode", "plane"))
        return NotifyLevelPayload(current_level=level_value, mode=mode_value)


__all__ = ["ActiveViewMirror"]
