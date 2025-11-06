"""Mirrors that project ledger state to downstream consumers."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.shared.dims_spec import (
    DimsSpec,
    dims_spec_axis_labels,
    dims_spec_displayed,
    dims_spec_from_payload,
    dims_spec_order,
)
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
        state = self._extract_state(snapshot)
        signature = self._compute_state_signature(state)
        payload = self._build_payload_from_state(state)
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

        state = self._extract_state(snapshot)
        token = self._compute_state_signature(state)
        with self._lock:
            if not token.changed(self._last_signature):
                return
            self._last_signature = token
            payload = self._build_payload_from_state(state)
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
        state = self._extract_state(snapshot)
        token = self._compute_state_signature(state)
        payload = self._build_payload_from_state(state)
        with self._lock:
            self._latest_payload = payload
            self._last_signature = token
            return payload

    def reset(self) -> None:
        with self._lock:
            self._last_signature = None
            self._latest_payload = None

    # ------------------------------------------------------------------
    def _extract_state(
        self, snapshot: dict[tuple[str, str, str], LedgerEntry]
    ) -> dict[str, Any]:
        entry = snapshot.get(("dims", "main", "dims_spec"))
        if entry is None:
            raise ValueError("ledger missing dims_spec entry")
        spec_entry = entry
        assert isinstance(spec_entry, LedgerEntry), "ledger dims_spec entry malformed"
        spec_payload = spec_entry.value
        dims_spec = dims_spec_from_payload(spec_payload)
        assert isinstance(dims_spec, DimsSpec), "ledger dims_spec entry malformed"

        current_step = tuple(int(v) for v in dims_spec.current_step)
        current_level = int(dims_spec.current_level)
        level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in dims_spec.level_shapes)
        ndisplay = int(dims_spec.ndisplay)
        mode = "volume" if ndisplay >= 3 else "plane"

        displayed = dims_spec_displayed(dims_spec)
        order = dims_spec_order(dims_spec)
        axis_labels = dims_spec_axis_labels(dims_spec)
        labels = tuple(str(lbl) for lbl in dims_spec.labels) if dims_spec.labels is not None else None
        downgraded = dims_spec.downgraded
        levels = tuple(dict(level) for level in dims_spec.levels)

        return {
            "current_step": current_step,
            "current_level": current_level,
            "levels": levels,
            "level_shapes": level_shapes,
            "ndisplay": ndisplay,
            "mode": mode,
            "displayed": displayed,
            "order": order,
            "axis_labels": axis_labels,
            "labels": labels,
            "downgraded": downgraded,
            "dims_spec": dims_spec,
        }

    def _build_payload_from_state(self, state: dict[str, Any]) -> NotifyDimsPayload:
        return NotifyDimsPayload(
            current_step=state["current_step"],
            level_shapes=state["level_shapes"],
            levels=state["levels"],
            current_level=state["current_level"],
            downgraded=state["downgraded"],
            mode=state["mode"],
            ndisplay=state["ndisplay"],
            labels=state["labels"],
            dims_spec=state["dims_spec"],
        )

    # ------------------------------------------------------------------
    def _compute_state_signature(self, state: dict[str, Any]) -> SignatureToken:
        return dims_content_signature(
            current_step=state["current_step"],
            current_level=state["current_level"],
            ndisplay=state["ndisplay"],
            mode=state["mode"],
            displayed=state["displayed"],
            axis_labels=state["axis_labels"],
            order=state["order"],
            labels=state["labels"],
            levels=state["levels"],
            level_shapes=state["level_shapes"],
            downgraded=state["downgraded"],
        )


__all__ = ["ServerDimsMirror"]
