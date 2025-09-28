"""Reducer that manages optimistic + confirmed state for the streaming client."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple

from napari_cuda.protocol.messages import StateUpdateMessage


PropertyKey = Tuple[str, str, str]


@dataclass
class ConfirmedState:
    value: Any
    server_seq: Optional[int]
    timestamp: float


@dataclass
class PendingEntry:
    client_seq: int
    value: Any
    phase: str
    timestamp: float
    interaction_id: Optional[str] = None


@dataclass
class PropertyState:
    confirmed: Optional[ConfirmedState] = None
    pending: "OrderedDict[int, PendingEntry]" = field(default_factory=OrderedDict)


@dataclass
class ReconcileResult:
    projection_value: Any
    confirmed_value: Any
    pending_value: Optional[Any]
    is_self: bool
    overridden: bool
    pending_len: int


class StateStore:
    """Track optimistic + confirmed values keyed by ``(scope, target, key)``."""

    def __init__(
        self,
        *,
        client_id: str,
        next_client_seq: Callable[[], int],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client_id = client_id
        self._next_client_seq = next_client_seq
        self._clock = clock
        self._state: MutableMapping[PropertyKey, PropertyState] = {}

    # ------------------------------------------------------------------
    def apply_local(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        phase: Optional[str],
        interaction_id: Optional[str] = None,
    ) -> tuple[StateUpdateMessage, Any]:
        """Register a local mutation and return payload + projection.

        Parameters correspond to the outgoing ``state.update`` payload.  The
        reducer allocates a fresh ``client_seq`` and updates the optimistic
        queue before returning the message to transmit.
        """

        phase_normalized = (phase or "update").lower()
        property_key = (scope, target, key)
        state = self._state.setdefault(property_key, PropertyState())

        if phase_normalized in {"start", "reset"}:
            state.pending.clear()
        elif phase_normalized == "update" and state.pending:
            last_key = next(reversed(state.pending))
            state.pending.pop(last_key, None)

        client_seq = int(self._next_client_seq())
        timestamp = float(self._clock())

        entry = PendingEntry(
            client_seq=client_seq,
            value=value,
            phase=phase_normalized,
            timestamp=timestamp,
            interaction_id=interaction_id,
        )
        state.pending[client_seq] = entry

        payload = StateUpdateMessage(
            scope=scope,
            target=target,
            key=key,
            value=value,
            client_id=self._client_id,
            client_seq=client_seq,
            interaction_id=interaction_id,
            phase=phase_normalized,
            timestamp=timestamp,
        )

        projection_value = entry.value
        if state.pending:
            projection_value = next(reversed(state.pending.values())).value
        elif state.confirmed is not None:
            projection_value = state.confirmed.value

        return payload, projection_value

    # ------------------------------------------------------------------
    def apply_remote(self, message: StateUpdateMessage) -> ReconcileResult:
        """Reconcile an inbound ``state.update`` and compute projection."""

        property_key = (message.scope, message.target, message.key)
        state = self._state.setdefault(property_key, PropertyState())
        timestamp = float(message.timestamp) if message.timestamp is not None else float(self._clock())

        confirmed_value = message.value
        is_self = bool(message.client_id) and message.client_id == self._client_id
        pending_value: Optional[Any] = None
        overridden = False

        if is_self and message.client_seq is not None:
            ack_seq = int(message.client_seq)
            pending_entry = state.pending.pop(ack_seq, None)
            if pending_entry is not None:
                pending_value = pending_entry.value
                # Drop any even older optimistic entries—they have been
                # superseded by this acknowledgement.
                for seq in list(state.pending.keys()):
                    if seq < ack_seq:
                        state.pending.pop(seq, None)
                overridden = pending_entry.value != message.value
            else:
                # Stale acknowledgement; treat as overridden only if we still
                # have optimistic entries outstanding.
                overridden = bool(state.pending)
        else:
            if state.pending:
                overridden = True
                state.pending.clear()

        state.confirmed = ConfirmedState(
            value=confirmed_value,
            server_seq=message.server_seq,
            timestamp=timestamp,
        )

        if state.pending:
            projection_value = next(reversed(state.pending.values())).value
        else:
            projection_value = confirmed_value

        return ReconcileResult(
            projection_value=projection_value,
            confirmed_value=confirmed_value,
            pending_value=pending_value,
            is_self=is_self,
            overridden=overridden,
            pending_len=len(state.pending),
        )

    # ------------------------------------------------------------------
    def seed_confirmed(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        *,
        server_seq: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Prime a property with authoritative baseline state."""

        property_key = (scope, target, key)
        state = self._state.setdefault(property_key, PropertyState())
        state.pending.clear()
        state.confirmed = ConfirmedState(
            value=value,
            server_seq=server_seq,
            timestamp=float(timestamp) if timestamp is not None else float(self._clock()),
        )

    # ------------------------------------------------------------------
    def clear_pending_on_reconnect(self) -> None:
        """Drop optimistic state after a transport reconnect."""

        for property_state in self._state.values():
            property_state.pending.clear()

    # ------------------------------------------------------------------
    def dump_debug(self) -> Dict[str, Any]:  # pragma: no cover - diagnostic helper
        summary: Dict[str, Any] = {}
        for key, state in self._state.items():
            scope, target, prop = key
            k = f"{scope}:{target}:{prop}"
            confirmed = None
            if state.confirmed is not None:
                confirmed = {
                    "value": state.confirmed.value,
                    "server_seq": state.confirmed.server_seq,
                    "timestamp": state.confirmed.timestamp,
                }
            summary[k] = {
                "confirmed": confirmed,
                "pending": [
                    {
                        "client_seq": entry.client_seq,
                        "value": entry.value,
                        "phase": entry.phase,
                        "timestamp": entry.timestamp,
                        "interaction_id": entry.interaction_id,
                    }
                    for entry in state.pending.values()
                ],
            }
        return summary


__all__ = [
    "ConfirmedState",
    "PendingEntry",
    "PropertyState",
    "ReconcileResult",
    "StateStore",
]
