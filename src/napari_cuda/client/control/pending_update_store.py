"""Reducer that manages optimistic + confirmed state for the streaming client."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from napari_cuda.protocol import AckState


PropertyKey = Tuple[str, str, str]


@dataclass
class ConfirmedState:
    value: Any
    timestamp: float


@dataclass
class PendingEntry:
    intent_id: str
    frame_id: str
    value: Any
    update_phase: str
    timestamp: float
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PendingUpdate:
    scope: str
    target: str
    key: str
    value: Any
    update_phase: str
    intent_id: str
    frame_id: str
    timestamp: float
    projection_value: Any
    metadata: Dict[str, Any] | None

    def payload_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "target": self.target,
            "key": self.key,
            "value": self.value,
        }


@dataclass(frozen=True)
class AckOutcome:
    status: str
    intent_id: str
    ack_frame_id: Optional[str]
    in_reply_to: str
    scope: Optional[str]
    target: Optional[str]
    key: Optional[str]
    pending_value: Optional[Any]
    projection_value: Optional[Any]
    confirmed_value: Optional[Any]
    applied_value: Optional[Any]
    error: Optional[Dict[str, Any]]
    update_phase: Optional[str]
    metadata: Optional[Dict[str, Any]]
    pending_len: int
    was_pending: bool


@dataclass
class PropertyState:
    confirmed: Optional[ConfirmedState] = None
    pending: "OrderedDict[str, PendingEntry]" = field(default_factory=OrderedDict)


class StateStore:
    """Track optimistic + confirmed values keyed by ``(scope, target, key)``."""

    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock
        self._state: MutableMapping[PropertyKey, PropertyState] = {}
        self._pending_index: Dict[str, PropertyKey] = {}

    # ------------------------------------------------------------------
    def apply_local(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        update_phase: Optional[str],
        *,
        intent_id: str,
        frame_id: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PendingUpdate:
        """Register a local mutation and return a pending update descriptor."""

        phase_normalized = (update_phase or "update").lower()
        property_key = (scope, target, key)
        state = self._state.setdefault(property_key, PropertyState())

        if phase_normalized in {"start", "reset"}:
            for stale_frame in list(state.pending.keys()):
                self._pending_index.pop(stale_frame, None)
            state.pending.clear()
        elif phase_normalized == "update" and state.pending:
            last_frame_id = next(reversed(state.pending))
            state.pending.pop(last_frame_id, None)
            self._pending_index.pop(last_frame_id, None)

        timestamp_value = float(timestamp if timestamp is not None else self._clock())

        entry = PendingEntry(
            intent_id=intent_id,
            frame_id=frame_id,
            value=value,
            update_phase=phase_normalized,
            timestamp=timestamp_value,
            metadata=dict(metadata) if metadata is not None else None,
        )
        state.pending[frame_id] = entry
        self._pending_index[frame_id] = property_key

        if state.pending:
            projection_value = next(reversed(state.pending.values())).value
        elif state.confirmed is not None:
            projection_value = state.confirmed.value
        else:
            projection_value = value

        return PendingUpdate(
            scope=scope,
            target=target,
            key=key,
            value=value,
            update_phase=phase_normalized,
            intent_id=intent_id,
            frame_id=frame_id,
            timestamp=timestamp_value,
            projection_value=projection_value,
            metadata=dict(metadata) if metadata is not None else None,
        )

    # ------------------------------------------------------------------
    def apply_ack(self, ack: AckState) -> AckOutcome:
        """Reconcile an ``ack.state`` frame against pending optimistic entries."""

        payload = ack.payload
        frame_id = str(payload.in_reply_to)
        property_key = self._pending_index.pop(frame_id, None)
        property_state: Optional[PropertyState] = None
        pending_entry: Optional[PendingEntry] = None

        if property_key is not None:
            property_state = self._state.get(property_key)
            if property_state is not None:
                pending_entry = property_state.pending.pop(frame_id, None)

        scope: Optional[str] = None
        target: Optional[str] = None
        key: Optional[str] = None
        pending_value: Optional[Any] = None
        projection_value: Optional[Any] = None
        confirmed_value: Optional[Any] = None
        metadata: Optional[Dict[str, Any]] = None
        update_phase: Optional[str] = None

        if property_key is not None:
            scope, target, key = property_key

        if pending_entry is not None:
            pending_value = pending_entry.value
            metadata = dict(pending_entry.metadata) if pending_entry.metadata is not None else None
            update_phase = pending_entry.update_phase

        if property_state is not None:
            if property_state.pending:
                projection_value = next(reversed(property_state.pending.values())).value
            elif property_state.confirmed is not None:
                projection_value = property_state.confirmed.value

        applied_value = payload.applied_value
        error_dict: Optional[Dict[str, Any]] = None
        status = str(payload.status)

        if status == "accepted":
            if property_state is not None:
                value_to_store = applied_value if applied_value is not None else pending_value
                if value_to_store is not None:
                    property_state.confirmed = ConfirmedState(value=value_to_store, timestamp=float(self._clock()))
                    confirmed_value = property_state.confirmed.value
                    projection_value = property_state.confirmed.value if not property_state.pending else projection_value
                elif property_state.confirmed is not None:
                    confirmed_value = property_state.confirmed.value
            if confirmed_value is None and applied_value is not None:
                confirmed_value = applied_value
        else:
            if payload.error is not None:
                error_payload = payload.error.to_dict()
                error_dict = dict(error_payload)
            else:
                error_dict = None
            if property_state is not None and property_state.confirmed is not None:
                confirmed_value = property_state.confirmed.value
                projection_value = property_state.confirmed.value if not property_state.pending else projection_value

        envelope = ack.envelope
        return AckOutcome(
            status=status,
            intent_id=str(payload.intent_id),
            ack_frame_id=envelope.frame_id,
            in_reply_to=frame_id,
            scope=scope,
            target=target,
            key=key,
            pending_value=pending_value,
            projection_value=projection_value,
            confirmed_value=confirmed_value,
            applied_value=applied_value,
            error=error_dict,
            update_phase=update_phase,
            metadata=metadata,
            pending_len=len(property_state.pending) if property_state is not None else 0,
            was_pending=pending_entry is not None,
        )

    # ------------------------------------------------------------------
    def seed_confirmed(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        timestamp: Optional[float] = None,
    ) -> None:
        """Prime a property with authoritative baseline state."""

        property_key = (scope, target, key)
        state = self._state.setdefault(property_key, PropertyState())
        for stale_frame in list(state.pending.keys()):
            self._pending_index.pop(stale_frame, None)
        state.pending.clear()
        state.confirmed = ConfirmedState(
            value=value,
            timestamp=float(timestamp) if timestamp is not None else float(self._clock()),
        )

    # ------------------------------------------------------------------
    def clear_pending_on_reconnect(self) -> None:
        """Drop optimistic state after a transport reconnect."""

        for property_state in self._state.values():
            property_state.pending.clear()
        self._pending_index.clear()

    def discard_pending(self, frame_id: str) -> None:
        """Remove a pending entry when dispatch fails before acknowledgement."""

        property_key = self._pending_index.pop(frame_id, None)
        if property_key is None:
            return
        property_state = self._state.get(property_key)
        if property_state is None:
            return
        property_state.pending.pop(frame_id, None)

    # ------------------------------------------------------------------
    def has_pending(self, scope: str, target: str, key: str) -> bool:
        """Return whether there are outstanding optimistic values for a property."""

        state = self._state.get((scope, target, key))
        return bool(state and state.pending)

    def latest_pending_value(self, scope: str, target: str, key: str) -> Any | None:
        """Return the most recent optimistic value, if any."""

        state = self._state.get((scope, target, key))
        if not state or not state.pending:
            return None
        return next(reversed(state.pending.values())).value

    def confirmed_value(self, scope: str, target: str, key: str) -> Any | None:
        state = self._state.get((scope, target, key))
        if not state or state.confirmed is None:
            return None
        return state.confirmed.value

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
                    "timestamp": state.confirmed.timestamp,
                }
            summary[k] = {
                "confirmed": confirmed,
                "pending": [
                    {
                        "intent_id": entry.intent_id,
                        "frame_id": entry.frame_id,
                        "value": entry.value,
                        "update_phase": entry.update_phase,
                        "timestamp": entry.timestamp,
                    }
                    for entry in state.pending.values()
                ],
            }
        return summary


__all__ = [
    "ConfirmedState",
    "PendingEntry",
    "PendingUpdate",
    "AckOutcome",
    "PropertyState",
    "StateStore",
]
