"""Reducer that manages optimistic + confirmed state for the streaming client."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

from napari_cuda.protocol import AckState


PropertyKey = Tuple[str, str, str]


@dataclass
class ConfirmedState:
    value: Any
    timestamp: float
    origin: str
    version: Any | None
    metadata: Dict[str, Any] | None


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


@dataclass(frozen=True)
class StateStoreUpdate:
    scope: str
    target: str
    key: str
    value: Any
    origin: str
    version: Any | None
    timestamp: float
    metadata: Dict[str, Any] | None


logger = logging.getLogger(__name__)


class StateStore:
    """Track optimistic + confirmed values keyed by ``(scope, target, key)``."""

    def __init__(self, *, clock=time.time) -> None:
        self._clock = clock
        self._state: MutableMapping[PropertyKey, PropertyState] = {}
        self._pending_index: Dict[str, PropertyKey] = {}
        self._subscribers: Dict[PropertyKey, List[Callable[[StateStoreUpdate], None]]] = {}
        self._global_subscribers: List[Callable[[StateStoreUpdate], None]] = []

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
        dedupe: bool = True,
    ) -> PendingUpdate | None:
        """Register a local mutation and return a pending update descriptor."""

        phase_normalized = (update_phase or "update").lower()
        property_key = (scope, target, key)
        state = self._state.setdefault(property_key, PropertyState())

        metadata_dict = dict(metadata) if metadata is not None else None
        update_kind = None
        if metadata_dict is not None and "update_kind" in metadata_dict:
            raw_kind = metadata_dict["update_kind"]
            if raw_kind is not None:
                update_kind = str(raw_kind).lower()
        is_delta_update = update_kind in {"step", "delta"}

        if phase_normalized in {"start", "reset"}:
            for stale_frame in list(state.pending.keys()):
                self._pending_index.pop(stale_frame, None)
            state.pending.clear()
        elif phase_normalized == "update" and state.pending:
            last_frame_id = next(reversed(state.pending))
            state.pending.pop(last_frame_id, None)
            self._pending_index.pop(last_frame_id, None)

        if (
            dedupe
            and not state.pending
            and state.confirmed is not None
            and not is_delta_update
            and state.confirmed.value == value
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "store dedupe: scope=%s target=%s key=%s value=%r confirmed_ts=%s",
                    scope,
                    target,
                    key,
                    value,
                    state.confirmed.timestamp,
                )
            return None

        timestamp_value = float(timestamp if timestamp is not None else self._clock())

        entry_metadata = dict(metadata_dict) if metadata_dict is not None else None
        entry = PendingEntry(
            intent_id=intent_id,
            frame_id=frame_id,
            value=value,
            update_phase=phase_normalized,
            timestamp=timestamp_value,
            metadata=entry_metadata,
        )
        state.pending[frame_id] = entry
        self._pending_index[frame_id] = property_key

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "store pending add: scope=%s target=%s key=%s phase=%s value=%r pending_len=%d",
                scope,
                target,
                key,
                phase_normalized,
                value,
                len(state.pending),
            )

        if state.pending:
            projection_value = next(reversed(state.pending.values())).value
        elif state.confirmed is not None:
            projection_value = state.confirmed.value
        else:
            projection_value = value

        pending_metadata = dict(metadata_dict) if metadata_dict is not None else None
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
            metadata=pending_metadata,
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
                    property_state.confirmed = ConfirmedState(
                        value=value_to_store,
                        timestamp=float(self._clock()),
                        origin="remote_ack",
                        version=None,
                        metadata=dict(metadata) if metadata is not None else None,
                    )
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
                if metadata is None and property_state.confirmed.metadata is not None:
                    metadata = dict(property_state.confirmed.metadata)

        envelope = ack.envelope
        outcome = AckOutcome(
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

        if property_key is not None and property_state is not None and property_state.confirmed is not None:
            self._notify(property_key, property_state.confirmed)

        return outcome

    # ------------------------------------------------------------------
    def seed_confirmed(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        timestamp: Optional[float] = None,
        *,
        origin: str = "remote",
        version: Any | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
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
            origin=str(origin),
            version=version,
            metadata=dict(metadata) if metadata is not None else None,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "store seed_confirmed: scope=%s target=%s key=%s value=%r origin=%s",
                scope,
                target,
                key,
                value,
                origin,
            )
        self._notify(property_key, state.confirmed)

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
        """Return whether there are pending optimistic values for the property."""

        state = self._state.get((scope, target, key))
        return bool(state and state.pending)

    def latest_pending_value(self, scope: str, target: str, key: str) -> Any | None:
        """Return the newest optimistic value, if present."""

        state = self._state.get((scope, target, key))
        if not state or not state.pending:
            return None
        return next(reversed(state.pending.values())).value

    def confirmed_value(self, scope: str, target: str, key: str) -> Any | None:
        """Return the last confirmed value for the property, if present."""

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
                    "origin": state.confirmed.origin,
                    "version": state.confirmed.version,
                    "metadata": dict(state.confirmed.metadata) if state.confirmed.metadata is not None else None,
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

    # ------------------------------------------------------------------
    def subscribe(
        self,
        scope: str,
        target: str,
        key: str,
        callback: Callable[[StateStoreUpdate], None],
    ) -> None:
        assert callable(callback), "StateStore subscriber must be callable"
        property_key = (scope, target, key)
        listeners = self._subscribers.setdefault(property_key, [])
        assert callback not in listeners, "StateStore subscriber already registered"
        listeners.append(callback)

    def subscribe_all(self, callback: Callable[[StateStoreUpdate], None]) -> None:
        assert callable(callback), "StateStore subscriber must be callable"
        assert callback not in self._global_subscribers, "StateStore global subscriber already registered"
        self._global_subscribers.append(callback)

    # ------------------------------------------------------------------
    def _notify(self, property_key: PropertyKey, confirmed: ConfirmedState) -> None:
        scope, target, key = property_key
        update = StateStoreUpdate(
            scope=scope,
            target=target,
            key=key,
            value=confirmed.value,
            origin=confirmed.origin,
            version=confirmed.version,
            timestamp=confirmed.timestamp,
            metadata=dict(confirmed.metadata) if confirmed.metadata is not None else None,
        )
        for callback in tuple(self._global_subscribers):
            callback(update)
        listeners = self._subscribers.get(property_key)
        if not listeners:
            return
        for callback in tuple(listeners):
            callback(update)

    # ------------------------------------------------------------------
    def pending_state_snapshot(
        self,
        scope: str,
        target: str,
        key: str,
    ) -> tuple[Any, int, Optional[str], Optional[Any]] | None:
        """Return current projection value, pending len, and confirmed origin/value."""

        property_key = (scope, target, key)
        state = self._state.get(property_key)
        if state is None:
            return None
        if state.pending:
            projection_value = next(reversed(state.pending.values())).value
            pending_len = len(state.pending)
        elif state.confirmed is not None:
            projection_value = state.confirmed.value
            pending_len = 0
        else:
            return None
        origin = state.confirmed.origin if state.confirmed is not None else None
        confirmed_value = state.confirmed.value if state.confirmed is not None else None
        return projection_value, pending_len, origin, confirmed_value


__all__ = [
    "ConfirmedState",
    "PendingEntry",
    "PendingUpdate",
    "AckOutcome",
    "PropertyState",
    "StateStore",
    "StateStoreUpdate",
]
