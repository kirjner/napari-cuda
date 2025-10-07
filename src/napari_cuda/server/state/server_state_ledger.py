"""Authoritative property ledger for server-side state."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple


PropertyKey = Tuple[str, str, str]
Subscriber = Callable[["LedgerEvent"], None]


@dataclass(frozen=True)
class LedgerEntry:
    value: Any
    timestamp: float
    origin: str
    metadata: Optional[Dict[str, Any]]
    version: Any | None


@dataclass(frozen=True)
class LedgerEvent:
    scope: str
    target: str
    key: str
    value: Any
    timestamp: float
    origin: str
    metadata: Optional[Dict[str, Any]]
    version: Any | None


logger = logging.getLogger(__name__)


class ServerStateLedger:
    """Thread-safe ledger mirroring the client-side confirmed state store."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._state: Dict[PropertyKey, LedgerEntry] = {}
        self._subscribers: Dict[PropertyKey, List[Subscriber]] = {}
        self._global_subscribers: List[Subscriber] = []

    # ------------------------------------------------------------------
    def record_confirmed(
        self,
        scope: str,
        target: str,
        key: str,
        value: Any,
        *,
        origin: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        version: Any | None = None,
        dedupe: bool = True,
    ) -> LedgerEvent | None:
        """Record a single confirmed property update."""

        property_key = (str(scope), str(target), str(key))

        with self._lock:
            previous = self._state.get(property_key)
            metadata_dict = self._normalize_metadata(metadata)
            ts = float(timestamp) if timestamp is not None else float(self._clock())

            if (
                dedupe
                and previous is not None
                and previous.value == value
                and previous.metadata == metadata_dict
                and previous.version == version
            ):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ledger dedupe: scope=%s target=%s key=%s origin=%s",
                        scope,
                        target,
                        key,
                        origin,
                    )
                return None

            entry = LedgerEntry(
                value=value,
                timestamp=ts,
                origin=str(origin),
                metadata=metadata_dict,
                version=version,
            )
            self._state[property_key] = entry

            event = LedgerEvent(
                scope=str(scope),
                target=str(target),
                key=str(key),
                value=value,
                timestamp=ts,
                origin=str(origin),
                metadata=metadata_dict,
                version=version,
            )

            per_key = list(self._subscribers.get(property_key, ()))
            globals_copy = list(self._global_subscribers)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ledger record_confirmed: scope=%s target=%s key=%s origin=%s",
                scope,
                target,
                key,
                origin,
            )

        for callback in per_key:
            callback(event)
        for callback in globals_copy:
            callback(event)
        return event

    # ------------------------------------------------------------------
    def batch_record_confirmed(
        self,
        entries: Iterable[Tuple[Any, ...]],
        *,
        origin: str,
        timestamp: Optional[float] = None,
        dedupe: bool = True,
    ) -> List[LedgerEvent]:
        """Promote multiple confirmed properties in one batch."""

        materialized = list(entries)
        if not materialized:
            return []

        notifications: List[Tuple[LedgerEvent, List[Subscriber], List[Subscriber]]] = []

        with self._lock:
            for raw in materialized:
                length = len(raw)
                if length < 4 or length > 6:
                    raise ValueError("batch_record_confirmed entries must contain 4-6 items")

                scope = str(raw[0])
                target = str(raw[1])
                key = str(raw[2])
                value = raw[3]
                metadata_raw = raw[4] if length >= 5 else None
                version = raw[5] if length == 6 else None

                metadata_dict = self._normalize_metadata(metadata_raw)
                ts = float(timestamp) if timestamp is not None else float(self._clock())

                property_key = (scope, target, key)
                previous = self._state.get(property_key)

                if (
                    dedupe
                    and previous is not None
                    and previous.value == value
                    and previous.metadata == metadata_dict
                    and previous.version == version
                ):
                    continue

                entry = LedgerEntry(
                    value=value,
                    timestamp=ts,
                    origin=str(origin),
                    metadata=metadata_dict,
                    version=version,
                )
                self._state[property_key] = entry

                event = LedgerEvent(
                    scope=scope,
                    target=target,
                    key=key,
                    value=value,
                    timestamp=ts,
                    origin=str(origin),
                    metadata=metadata_dict,
                    version=version,
                )

                per_key = list(self._subscribers.get(property_key, ()))
                globals_copy = list(self._global_subscribers)
                notifications.append((event, per_key, globals_copy))

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ledger batch record_confirmed: scope=%s target=%s key=%s origin=%s",
                        scope,
                        target,
                        key,
                        origin,
                    )

        events: List[LedgerEvent] = []
        for event, per_key, globals_copy in notifications:
            for callback in per_key:
                callback(event)
            for callback in globals_copy:
                callback(event)
            events.append(event)
        return events

    # ------------------------------------------------------------------
    def subscribe(self, scope: str, target: str, key: str, callback: Subscriber) -> None:
        assert callable(callback), "ServerStateLedger subscriber must be callable"
        property_key = (str(scope), str(target), str(key))
        with self._lock:
            listeners = self._subscribers.setdefault(property_key, [])
            assert callback not in listeners, "ServerStateLedger subscriber already registered"
            listeners.append(callback)

    def unsubscribe(self, scope: str, target: str, key: str, callback: Subscriber) -> None:
        property_key = (str(scope), str(target), str(key))
        with self._lock:
            listeners = self._subscribers.get(property_key)
            if not listeners:
                return
            try:
                listeners.remove(callback)
            except ValueError:
                return
            if not listeners:
                self._subscribers.pop(property_key, None)

    def subscribe_all(self, callback: Subscriber) -> None:
        assert callable(callback), "ServerStateLedger global subscriber must be callable"
        with self._lock:
            assert callback not in self._global_subscribers, "ServerStateLedger global subscriber already registered"
            self._global_subscribers.append(callback)

    def unsubscribe_all(self, callback: Subscriber) -> None:
        with self._lock:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[PropertyKey, LedgerEntry]:
        """Return a shallow copy of the ledger state."""

        with self._lock:
            return dict(self._state)

    def get(self, scope: str, target: str, key: str) -> Optional[LedgerEntry]:
        property_key = (str(scope), str(target), str(key))
        with self._lock:
            return self._state.get(property_key)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_metadata(metadata: Optional[Any]) -> Optional[Dict[str, Any]]:
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            raise TypeError("ledger metadata must be a mapping or None")
        return dict(metadata)


__all__ = [
    "LedgerEntry",
    "LedgerEvent",
    "PropertyKey",
    "ServerStateLedger",
]
