"""Mirrors that project ledger state to downstream consumers."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.shared.dims_spec import DimsSpec, dims_spec_from_payload
from napari_cuda.server.state_ledger import LedgerEntry, LedgerEvent, ServerStateLedger
from napari_cuda.server.utils.signatures import SignatureToken, dims_content_signature

ScheduleFn = Callable[[Awaitable[None], str], None]
BroadcastFn = Callable[[NotifyDimsPayload], Awaitable[None]]
OnPayloadFn = Callable[[NotifyDimsPayload], None]


@dataclass(frozen=True)
class _Key:
    scope: str
    target: str
    key: str


class ServerDimsMirror:
    """Project server ledger updates into notify.dims broadcasts."""

    _WATCH_KEYS: tuple[_Key, ...] = (
        _Key("scene", "main", "op_state"),
        _Key("dims", "main", "dims_spec"),
    )

    def __init__(
        self,
        *,
        ledger: ServerStateLedger,
        broadcaster: BroadcastFn,
        schedule: ScheduleFn,
        on_payload: Optional[OnPayloadFn] = None,
    ) -> None:
        self._ledger = ledger
        self._broadcaster = broadcaster
        self._schedule = schedule
        self._on_payload = on_payload
        self._lock = threading.Lock()
        self._started = False
        self._last_signature: Optional[SignatureToken] = None
        self._latest_payload: Optional[NotifyDimsPayload] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ServerDimsMirror already started")
            self._started = True
        for key in self._WATCH_KEYS:
            self._ledger.subscribe(key.scope, key.target, key.key, self._on_ledger_event)
        snapshot = self._ledger.snapshot()
        spec = self._extract_spec(snapshot)
        signature = self._compute_state_signature(spec)
        payload = self._build_payload_from_spec(spec)
        with self._lock:
            self._latest_payload = payload
            self._last_signature = signature

    # ------------------------------------------------------------------
    def _on_ledger_event(self, event: LedgerEvent) -> None:
        # Gate notifications on op_state if present
        snapshot = self._ledger.snapshot()
        op_entry = snapshot.get(("scene", "main", "op_state"))
        if op_entry is not None:
            text = str(op_entry.value)
            if text != "applied":
                return

        spec = self._extract_spec(snapshot)
        token = self._compute_state_signature(spec)
        with self._lock:
            if not token.changed(self._last_signature):
                return
            self._last_signature = token
            payload = self._build_payload_from_spec(spec)
            self._latest_payload = payload

        if self._on_payload is not None:
            self._on_payload(payload)
        self._schedule(self._broadcaster(payload), "mirror-dims")

    # ------------------------------------------------------------------
    def latest_payload(self) -> Optional[NotifyDimsPayload]:
        with self._lock:
            if self._latest_payload is not None:
                return self._latest_payload
        snapshot = self._ledger.snapshot()
        spec = self._extract_spec(snapshot)
        token = self._compute_state_signature(spec)
        payload = self._build_payload_from_spec(spec)
        with self._lock:
            self._latest_payload = payload
            self._last_signature = token
            return payload

    def reset(self) -> None:
        with self._lock:
            self._last_signature = None
            self._latest_payload = None

    # ------------------------------------------------------------------
    def _extract_spec(
        self, snapshot: dict[tuple[str, str, str], LedgerEntry]
    ) -> DimsSpec:
        entry = snapshot.get(("dims", "main", "dims_spec"))
        if entry is None:
            raise ValueError("ledger missing dims_spec entry")
        spec_entry = entry
        assert isinstance(spec_entry, LedgerEntry), "ledger dims_spec entry malformed"
        spec_payload = spec_entry.value
        dims_spec = dims_spec_from_payload(spec_payload)
        assert isinstance(dims_spec, DimsSpec), "ledger dims_spec entry malformed"
        return dims_spec

    def _build_payload_from_spec(self, spec: DimsSpec) -> NotifyDimsPayload:
        return NotifyDimsPayload(
            current_step=spec.current_step,
            level_shapes=spec.level_shapes,
            levels=spec.levels,
            current_level=spec.current_level,
            mode="volume" if int(spec.ndisplay) >= 3 else "plane",
            ndisplay=spec.ndisplay,
            labels=spec.labels,
            dims_spec=spec,
        )

    # ------------------------------------------------------------------
    def _compute_state_signature(self, spec: DimsSpec) -> SignatureToken:
        return dims_content_signature(
            current_step=spec.current_step,
            current_level=spec.current_level,
            ndisplay=spec.ndisplay,
            mode="volume" if int(spec.ndisplay) >= 3 else "plane",
            displayed=spec.displayed,
            axis_labels=tuple(axis.label for axis in spec.axes),
            order=spec.order,
            labels=spec.labels,
            levels=spec.levels,
            level_shapes=spec.level_shapes,
        )


__all__ = ["ServerDimsMirror"]
