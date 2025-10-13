"""Mirrors that project ledger state to downstream consumers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Tuple

from napari_cuda.protocol.messages import NotifyDimsPayload

from napari_cuda.server.control.state_ledger import LedgerEvent, ServerStateLedger


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

    _WATCH_KEYS: Tuple[_Key, ...] = (
        _Key("view", "main", "ndisplay"),
        _Key("view", "main", "displayed"),
        _Key("dims", "main", "current_step"),
        _Key("dims", "main", "mode"),
        _Key("dims", "main", "order"),
        _Key("dims", "main", "axis_labels"),
        _Key("dims", "main", "labels"),
        _Key("multiscale", "main", "level"),
        _Key("multiscale", "main", "levels"),
        _Key("multiscale", "main", "level_shapes"),
        _Key("multiscale", "main", "downgraded"),
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
        self._last_signature: Optional[Tuple[Any, ...]] = None
        self._latest_payload: Optional[NotifyDimsPayload] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ServerDimsMirror already started")
            self._started = True
        for key in self._WATCH_KEYS:
            self._ledger.subscribe(key.scope, key.target, key.key, self._on_ledger_event)
        payload = self._build_payload()
        signature = self._compute_signature(payload)
        with self._lock:
            self._latest_payload = payload
            self._last_signature = signature

    # ------------------------------------------------------------------
    def _on_ledger_event(self, event: LedgerEvent) -> None:
        payload = self._build_payload()

        signature = self._compute_signature(payload)
        with self._lock:
            if self._last_signature == signature:
                return
            self._last_signature = signature
            self._latest_payload = payload

        if self._on_payload is not None:
            self._on_payload(payload)
        self._schedule(self._broadcaster(payload), "mirror-dims")

    # ------------------------------------------------------------------
    def latest_payload(self) -> Optional[NotifyDimsPayload]:
        with self._lock:
            if self._latest_payload is not None:
                return self._latest_payload
        payload = self._build_payload()
        signature = self._compute_signature(payload)
        with self._lock:
            self._latest_payload = payload
            self._last_signature = signature
            return payload

    # ------------------------------------------------------------------
    def _build_payload(self) -> NotifyDimsPayload:
        snapshot = self._ledger.snapshot()
        required = {
            ("dims", "main", "current_step"),
            ("multiscale", "main", "level"),
            ("multiscale", "main", "levels"),
            ("multiscale", "main", "level_shapes"),
            ("dims", "main", "mode"),
            ("view", "main", "ndisplay"),
        }
        for key in required:
            if key not in snapshot:
                raise ValueError("ledger missing required dims entry")

        current_step = self._as_int_tuple(snapshot[("dims", "main", "current_step")].value)
        current_level = int(snapshot[("multiscale", "main", "level")].value)
        levels = self._as_level_sequence(snapshot[("multiscale", "main", "levels")].value)
        level_shapes = self._as_shape_sequence(snapshot[("multiscale", "main", "level_shapes")].value)
        mode = str(snapshot[("dims", "main", "mode")].value)
        ndisplay = int(snapshot[("view", "main", "ndisplay")].value)

        displayed = self._optional_int_tuple(snapshot.get(("view", "main", "displayed")))
        order = self._optional_int_tuple(snapshot.get(("dims", "main", "order")))
        axis_labels = self._optional_str_tuple(snapshot.get(("dims", "main", "axis_labels")))
        labels = self._optional_str_tuple(snapshot.get(("dims", "main", "labels")))
        downgraded = self._optional_bool(snapshot.get(("multiscale", "main", "downgraded")))

        return NotifyDimsPayload(
            current_step=current_step,
            level_shapes=level_shapes,
            levels=levels,
            current_level=current_level,
            downgraded=downgraded,
            mode=mode,
            ndisplay=ndisplay,
            axis_labels=axis_labels,
            order=order,
            displayed=displayed,
            labels=labels,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _as_int_tuple(value: Any) -> Tuple[int, ...]:
        if value is None:
            raise ValueError("expected integer sequence, received None")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(int(v) for v in value)
        raise TypeError(f"expected integer sequence, received {type(value)!r}")

    @staticmethod
    def _as_shape_sequence(value: Any) -> Tuple[Tuple[int, ...], ...]:
        if value is None:
            raise ValueError("expected shape sequence, received None")
        shapes: list[Tuple[int, ...]] = []
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for entry in value:
                shapes.append(ServerDimsMirror._as_int_tuple(entry))
            return tuple(shapes)
        raise TypeError(f"expected shape sequence, received {type(value)!r}")

    @staticmethod
    def _as_level_sequence(value: Any) -> Tuple[Dict[str, Any], ...]:
        if value is None:
            raise ValueError("expected level sequence, received None")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            levels: list[Dict[str, Any]] = []
            for entry in value:
                if isinstance(entry, dict):
                    levels.append(dict(entry))
                else:
                    raise TypeError("level descriptors must be mappings")
            return tuple(levels)
        raise TypeError(f"expected level sequence, received {type(value)!r}")

    @staticmethod
    def _optional_int_tuple(entry: Any) -> Optional[Tuple[int, ...]]:
        if entry is None:
            return None
        value = getattr(entry, "value", entry)
        if value is None:
            return None
        return ServerDimsMirror._as_int_tuple(value)

    @staticmethod
    def _optional_str_tuple(entry: Any) -> Optional[Tuple[str, ...]]:
        if entry is None:
            return None
        value = getattr(entry, "value", entry)
        if value is None:
            return None
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(str(v) for v in value)
        raise TypeError(f"expected string sequence, received {type(value)!r}")

    @staticmethod
    def _optional_bool(entry: Any) -> Optional[bool]:
        if entry is None:
            return None
        value = getattr(entry, "value", entry)
        if value is None:
            return None
        return bool(value)

    @staticmethod
    def _compute_signature(payload: NotifyDimsPayload) -> Tuple[Any, ...]:
        levels_sig = tuple(tuple(sorted(level.items())) for level in payload.levels)
        return (
            payload.current_step,
            payload.current_level,
            payload.ndisplay,
            payload.mode,
            payload.displayed,
            payload.axis_labels,
            payload.order,
            payload.labels,
            levels_sig,
            payload.level_shapes,
            payload.downgraded,
        )


__all__ = ["ServerDimsMirror"]
