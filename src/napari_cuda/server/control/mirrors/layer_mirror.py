"""Mirror that projects ledger-backed layer state into notify.layers broadcasts."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Mapping, Optional, Tuple

from napari_cuda.server.state_ledger import LedgerEvent, ServerStateLedger


ScheduleFn = Callable[[Awaitable[None], str], None]
BroadcastFn = Callable[[str, Mapping[str, object], Optional[str], float], Awaitable[None]]
DefaultLayerResolver = Callable[[], Optional[str]]


_ALLOWED_LAYER_KEYS = {
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "gamma",
    "colormap",
    "contrast_limits",
    "attenuation",
    "iso_threshold",
    "metadata",
}


class ServerLayerMirror:
    """Project ledger updates into notify.layers payloads once transactions apply."""

    def __init__(
        self,
        *,
        ledger: ServerStateLedger,
        broadcaster: BroadcastFn,
        schedule: ScheduleFn,
        default_layer: DefaultLayerResolver,
    ) -> None:
        self._ledger = ledger
        self._broadcaster = broadcaster
        self._schedule = schedule
        self._default_layer = default_layer
        self._lock = threading.Lock()
        self._started = False
        self._op_open = False
        self._pending: Dict[str, Dict[str, object]] = {}
        self._pending_versions: Dict[Tuple[str, str], int] = {}
        self._pending_intents: Dict[str, Optional[str]] = {}
        self._last_versions: Dict[Tuple[str, str], int] = {}
        self._latest_controls: Dict[str, Dict[str, object]] = defaultdict(dict)
        self._ndisplay: Optional[int] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ServerLayerMirror already started")
            self._started = True
        snapshot = self._ledger.snapshot()
        nd_entry = snapshot.get(("view", "main", "ndisplay"))
        with self._lock:
            if nd_entry is not None and nd_entry.value is not None:
                self._ndisplay = int(nd_entry.value)
        for (scope, target, key), entry in snapshot.items():
            self._record_snapshot(scope, target, key, entry.value, entry.version)
        self._ledger.subscribe("scene", "main", "op_state", self._on_op_state)
        self._ledger.subscribe("view", "main", "ndisplay", self._on_ndisplay)
        self._ledger.subscribe_all(self._on_event)

    # ------------------------------------------------------------------
    def latest_controls(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            return {layer: dict(props) for layer, props in self._latest_controls.items() if props}

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._pending_versions.clear()
            self._pending_intents.clear()
            self._last_versions.clear()
            self._latest_controls.clear()
            self._op_open = False
            self._ndisplay = None

    # ------------------------------------------------------------------
    def _on_op_state(self, event: LedgerEvent) -> None:
        value = str(event.value)
        to_flush: Dict[str, Dict[str, object]] = {}
        intents: Dict[str, Optional[str]] = {}
        with self._lock:
            if value == "open":
                self._op_open = True
                return
            if value != "applied":
                return
            self._op_open = False
            if not self._pending:
                return
            to_flush = self._drain_pending_locked()
            intents = self._pending_intents
            self._pending_intents = {}
        self._flush(to_flush, intents, event.timestamp)

    # ------------------------------------------------------------------
    def _on_ndisplay(self, event: LedgerEvent) -> None:
        value = int(event.value)
        with self._lock:
            previous_enabled = self._volume_enabled_locked()
            self._ndisplay = value
            if previous_enabled and not self._volume_enabled_locked():
                for layer_id, props in list(self._latest_controls.items()):
                    for key in [k for k in props if k.startswith("volume.")]:
                        props.pop(key, None)
                    if not props:
                        self._latest_controls.pop(layer_id, None)

    # ------------------------------------------------------------------
    def _on_event(self, event: LedgerEvent) -> None:
        scope = str(event.scope)
        if scope == "view" and event.key == "ndisplay":
            self._on_ndisplay(event)
            return
        if scope not in {"layer", "volume", "multiscale"}:
            return
        to_flush: Optional[Dict[str, Dict[str, object]]] = None
        intents: Dict[str, Optional[str]] = {}
        with self._lock:
            layer_id, prop, value = self._map_event_locked(event)
            if layer_id is None or prop is None:
                return
            version = event.version
            if version is None:
                raise AssertionError("ledger layer event missing version")
            version_int = int(version)
            key = (layer_id, prop)
            last = self._last_versions.get(key)
            if last is not None and last == version_int:
                return
            self._latest_controls[layer_id][prop] = value
            pending_props = self._pending.setdefault(layer_id, {})
            pending_props[prop] = value
            self._pending_versions[key] = version_int
            if event.metadata and "intent_id" in event.metadata:
                self._pending_intents[layer_id] = event.metadata.get("intent_id")
            if not self._op_open:
                to_flush = self._drain_pending_locked()
                intents = self._pending_intents
                self._pending_intents = {}
        if not self._op_open and to_flush:
            self._flush(to_flush, intents, event.timestamp)

    # ------------------------------------------------------------------
    def _drain_pending_locked(self) -> Dict[str, Dict[str, object]]:
        pending = self._pending
        self._pending = {}
        return {layer: dict(props) for layer, props in pending.items() if props}

    # ------------------------------------------------------------------
    def _flush(
        self,
        pending: Dict[str, Dict[str, object]],
        intents: Dict[str, Optional[str]],
        timestamp: float,
    ) -> None:
        for layer_id, changes in pending.items():
            key = tuple(changes.keys())
            for prop in key:
                version = self._pending_versions.pop((layer_id, prop), None)
                if version is not None:
                    self._last_versions[(layer_id, prop)] = version
            intent_id = intents.get(layer_id)
            self._schedule(
                self._broadcaster(layer_id, changes, intent_id, timestamp),
                f"mirror-layer-{layer_id}",
            )

    # ------------------------------------------------------------------
    def _map_event(self, event: LedgerEvent) -> tuple[Optional[str], Optional[str], object]:
        return self._map_event_locked(event)

    def _map_event_locked(self, event: LedgerEvent) -> tuple[Optional[str], Optional[str], object]:
        scope = str(event.scope)
        target = str(event.target) if event.target is not None else ""
        key = str(event.key)
        value = event.value

        if scope == "layer":
            if key not in _ALLOWED_LAYER_KEYS:
                return None, None, value
            return target, key, value

        if not self._volume_enabled_locked():
            return None, None, value
        # volume/multiscale updates map to the default remote layer
        layer_id = target if target.startswith("layer") else self._default_layer()
        if layer_id is None:
            return None, None, value
        namespaced = f"{scope}.{key}"
        return layer_id, namespaced, value

    # ------------------------------------------------------------------
    def _record_snapshot(
        self,
        scope: str,
        target: str,
        key: str,
        value: object,
        version: object | None,
    ) -> None:
        with self._lock:
            if scope == "layer":
                if key not in _ALLOWED_LAYER_KEYS:
                    return
                layer_id = str(target)
                version_int = int(version) if version is not None else None
                self._latest_controls[layer_id][key] = value
                if version_int is not None:
                    self._last_versions[(layer_id, key)] = version_int
                return
            if scope in {"volume", "multiscale"} and self._volume_enabled_locked():
                layer_id = str(target) if str(target).startswith("layer") else self._default_layer()
                if layer_id is None:
                    return
                namespaced = f"{scope}.{key}"
                version_int = int(version) if version is not None else None
                self._latest_controls[layer_id][namespaced] = value
                if version_int is not None:
                    self._last_versions[(layer_id, namespaced)] = version_int

    def _volume_enabled_locked(self) -> bool:
        value = self._ndisplay
        return value is not None and int(value) >= 3


__all__ = ["ServerLayerMirror"]
