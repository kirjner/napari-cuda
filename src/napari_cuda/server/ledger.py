"""Authoritative property ledger for server-side state."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

PropertyKey = tuple[str, str, str]
Subscriber = Callable[["LedgerEvent"], None]


@dataclass(frozen=True)
class LedgerEntry:
    value: Any
    timestamp: float
    origin: str
    metadata: Optional[dict[str, Any]]
    version: Any | None


@dataclass(frozen=True)
class LedgerEvent:
    scope: str
    target: str
    key: str
    value: Any
    timestamp: float
    origin: str
    metadata: Optional[dict[str, Any]]
    version: Any | None


logger = logging.getLogger(__name__)


def _summarize_value(value: Any) -> str:
    """Create a compact, human-friendly summary for large values.

    Specifically avoids logging full pixel arrays for thumbnails while still
    exposing useful metadata like shape/dtype/timestamp to aid debugging.
    """
    if isinstance(value, dict):
        # Protocol thumbnails are dicts with keys: array (list), shape, dtype, generated_at
        if "array" in value:
            shape = value.get("shape")
            dtype = value.get("dtype")
            generated_at = value.get("generated_at")
            return f"<thumbnail shape={shape} dtype={dtype} generated_at={generated_at}>"
        # Some payloads may wrap under "thumbnail": [...] — also summarize
        thumb = value.get("thumbnail")
        if isinstance(thumb, list):
            return "<thumbnail array>"
        if value.get("layer_type") and value.get("controls"):
            return "<layer block>"
    # Fallback to repr for small/normal values
    return repr(value)


class ServerStateLedger:
    """Thread-safe ledger mirroring the client-side confirmed state store."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._state: dict[PropertyKey, LedgerEntry] = {}
        self._subscribers: dict[PropertyKey, list[Subscriber]] = {}
        self._global_subscribers: list[Subscriber] = []
        self._versions: dict[PropertyKey, int] = {}

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
    ) -> LedgerEntry:
        """Record a single confirmed property update and return the stored entry."""

        property_key = (str(scope), str(target), str(key))

        with self._lock:
            previous = self._state.get(property_key)
            metadata_dict = self._normalize_metadata(metadata)
            ts = float(timestamp) if timestamp is not None else float(self._clock())

            same_value = (
                previous is not None
                and previous.value == value
                and previous.metadata == metadata_dict
            )
            same_version = version is not None and previous is not None and previous.version == version
            if dedupe and same_value and (version is None or same_version):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ledger dedupe: scope=%s target=%s key=%s origin=%s",
                        scope,
                        target,
                        key,
                        origin,
                    )
                assert previous is not None
                return previous

            if version is None:
                next_version = int(self._versions.get(property_key, 0)) + 1
                self._versions[property_key] = next_version
                resolved_version: Any | None = next_version
            else:
                resolved_version = version
                self._versions[property_key] = int(version)

            entry = LedgerEntry(
                value=value,
                timestamp=ts,
                origin=str(origin),
                metadata=metadata_dict,
                version=resolved_version,
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
                version=resolved_version,
            )

            per_key = list(self._subscribers.get(property_key, ()))
            globals_copy = list(self._global_subscribers)

        # Always avoid dumping full arrays for thumbnails/metadata. Summarize.
        value_for_log: str | Any
        if scope == "layer" and key in {"thumbnail", "metadata"}:
            value_for_log = _summarize_value(value)
        elif scope == "scene_layers" and key == "block":
            value_for_log = _summarize_value(value)
        else:
            value_for_log = value

        logger.info(
            "ledger write: scope=%s target=%s key=%s origin=%s value=%r",
            scope,
            target,
            key,
            origin,
            value_for_log,
        )
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
        return entry

    # ------------------------------------------------------------------
    def batch_record_confirmed(
        self,
        entries: Iterable[tuple[Any, ...]],
        *,
        origin: str,
        timestamp: Optional[float] = None,
        dedupe: bool = True,
    ) -> dict[PropertyKey, LedgerEntry]:
        """Promote multiple confirmed properties in one batch and return stored entries."""

        materialized = list(entries)
        if not materialized:
            return {}

        notifications: list[tuple[LedgerEvent, list[Subscriber], list[Subscriber]]] = []
        stored: dict[PropertyKey, LedgerEntry] = {}

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

                same_value = (
                    previous is not None
                    and previous.value == value
                    and previous.metadata == metadata_dict
                )
                same_version = version is not None and previous is not None and previous.version == version
                if dedupe and same_value and (version is None or same_version):
                    if previous is not None:
                        stored[property_key] = previous
                    continue

                if version is None:
                    next_version = int(self._versions.get(property_key, 0)) + 1
                    self._versions[property_key] = next_version
                    resolved_version: Any | None = next_version
                else:
                    resolved_version = version
                    self._versions[property_key] = int(version)

                entry = LedgerEntry(
                    value=value,
                    timestamp=ts,
                    origin=str(origin),
                    metadata=metadata_dict,
                    version=resolved_version,
                )
                self._state[property_key] = entry
                stored[property_key] = entry

                event = LedgerEvent(
                    scope=scope,
                    target=target,
                    key=key,
                    value=value,
                    timestamp=ts,
                    origin=str(origin),
                    metadata=metadata_dict,
                    version=resolved_version,
                )

                per_key = list(self._subscribers.get(property_key, ()))
                globals_copy = list(self._global_subscribers)
                notifications.append((event, per_key, globals_copy))

                # Summarize thumbnails in batch mode as well
                batched_value_for_log: str | Any
                if scope == "layer" and key in {"thumbnail", "metadata"}:
                    batched_value_for_log = _summarize_value(value)
                else:
                    batched_value_for_log = value

                logger.info(
                    "ledger write: scope=%s target=%s key=%s origin=%s value=%r",
                    scope,
                    target,
                    key,
                    origin,
                    batched_value_for_log,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ledger batch record_confirmed: scope=%s target=%s key=%s origin=%s",
                        scope,
                        target,
                        key,
                        origin,
                    )

        for event, per_key, globals_copy in notifications:
            for callback in per_key:
                callback(event)
            for callback in globals_copy:
                callback(event)
        return stored

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
    def snapshot(self) -> dict[PropertyKey, LedgerEntry]:
        """Return a shallow copy of the ledger state."""

        with self._lock:
            return dict(self._state)

    def clear_scope(
        self,
        scope: str,
        *,
        target: Optional[str] = None,
        target_prefix: Optional[str] = None,
    ) -> int:
        """Remove confirmed entries matching the requested scope/target filters.

        Returns the number of properties removed. Versions and cached entries are
        purged so follow-up writes restart from version 1. Subscriptions remain
        intact so future updates on the same property key continue delivering
        notifications.
        """

        if target is not None and target_prefix is not None:
            raise ValueError("clear_scope accepts either target or target_prefix, not both")

        removed = 0
        scope_key = str(scope)

        with self._lock:
            to_delete: list[PropertyKey] = []
            for property_key in self._state:
                prop_scope, prop_target, _ = property_key
                if prop_scope != scope_key:
                    continue
                if target is not None and prop_target != str(target):
                    continue
                if target_prefix is not None and not str(prop_target).startswith(str(target_prefix)):
                    continue
                to_delete.append(property_key)

            for property_key in to_delete:
                self._state.pop(property_key, None)
                self._versions.pop(property_key, None)
                removed += 1

        if removed and logger.isEnabledFor(logging.INFO):
            logger.info(
                "ledger clear_scope: scope=%s target=%s prefix=%s removed=%d",
                scope_key,
                target,
                target_prefix,
                removed,
            )
        return removed

    def get(self, scope: str, target: str, key: str) -> Optional[LedgerEntry]:
        property_key = (str(scope), str(target), str(key))
        with self._lock:
            return self._state.get(property_key)

    def current_version(self, scope: str, target: str, key: str) -> Optional[int]:
        property_key = (str(scope), str(target), str(key))
        with self._lock:
            if property_key not in self._versions:
                return None
            return int(self._versions[property_key])

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_metadata(metadata: Optional[Any]) -> Optional[dict[str, Any]]:
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
