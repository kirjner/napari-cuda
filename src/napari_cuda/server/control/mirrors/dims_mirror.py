"""Mirrors that project ledger state to downstream consumers."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Iterable
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
        _Key("view", "main", "ndisplay"),
        _Key("view", "main", "displayed"),
        _Key("dims", "main", "current_step"),
        _Key("dims", "main", "order"),
        _Key("dims", "main", "axis_labels"),
        _Key("dims", "main", "labels"),
        _Key("multiscale", "main", "level"),
        _Key("multiscale", "main", "levels"),
        _Key("multiscale", "main", "level_shapes"),
        _Key("multiscale", "main", "downgraded"),
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
        required = {
            ("dims", "main", "current_step"),
            ("multiscale", "main", "level"),
            ("multiscale", "main", "levels"),
            ("multiscale", "main", "level_shapes"),
            ("view", "main", "ndisplay"),
            ("dims", "main", "dims_spec"),
        }
        for key in required:
            if key not in snapshot:
                raise ValueError("ledger missing required dims entry")

        spec_entry = snapshot[("dims", "main", "dims_spec")]
        spec_payload = getattr(spec_entry, "value", spec_entry)
        dims_spec = dims_spec_from_payload(spec_payload)
        assert isinstance(dims_spec, DimsSpec), "ledger dims_spec entry malformed"

        current_step = tuple(int(v) for v in dims_spec.current_step)
        current_level = int(dims_spec.current_level)
        level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in dims_spec.level_shapes)
        ndisplay = int(dims_spec.ndisplay)
        mode = "volume" if ndisplay >= 3 else "plane"

        displayed = tuple(int(idx) for idx in dims_spec.displayed)
        order = tuple(int(idx) for idx in dims_spec.order)
        axis_labels = tuple(axis.label for axis in dims_spec.axes)
        labels = self._optional_str_tuple(snapshot.get(("dims", "main", "labels")))
        downgraded = self._optional_bool(snapshot.get(("multiscale", "main", "downgraded")))
        levels = self._as_level_sequence(snapshot[("multiscale", "main", "levels")].value)

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
            axis_labels=state["axis_labels"],
            order=state["order"],
            displayed=state["displayed"],
            labels=state["labels"],
            dims_spec=state["dims_spec"],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _as_int_tuple(value: Any) -> tuple[int, ...]:
        if value is None:
            raise ValueError("expected integer sequence, received None")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(int(v) for v in value)
        raise TypeError(f"expected integer sequence, received {type(value)!r}")

    @staticmethod
    def _as_shape_sequence(value: Any) -> tuple[tuple[int, ...], ...]:
        if value is None:
            raise ValueError("expected shape sequence, received None")
        shapes: list[tuple[int, ...]] = []
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for entry in value:
                shapes.append(ServerDimsMirror._as_int_tuple(entry))
            return tuple(shapes)
        raise TypeError(f"expected shape sequence, received {type(value)!r}")

    @staticmethod
    def _as_level_sequence(value: Any) -> tuple[dict[str, Any], ...]:
        if value is None:
            raise ValueError("expected level sequence, received None")
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            levels: list[dict[str, Any]] = []
            for entry in value:
                if isinstance(entry, dict):
                    levels.append(dict(entry))
                else:
                    raise TypeError("level descriptors must be mappings")
            return tuple(levels)
        raise TypeError(f"expected level sequence, received {type(value)!r}")

    @staticmethod
    def _optional_int_tuple(entry: Any) -> Optional[tuple[int, ...]]:
        if entry is None:
            return None
        value = getattr(entry, "value", entry)
        if value is None:
            return None
        return ServerDimsMirror._as_int_tuple(value)

    @staticmethod
    def _optional_str_tuple(entry: Any) -> Optional[tuple[str, ...]]:
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
