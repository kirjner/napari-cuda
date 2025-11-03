"""Mirror that projects ledger-backed layer state into notify.layers broadcasts."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Optional

from napari_cuda.server.scene import LayerVisualState, build_ledger_snapshot
from napari_cuda.server.state_ledger import LedgerEvent, ServerStateLedger
from napari_cuda.server.utils.signatures import layer_payload_signature

ScheduleFn = Callable[[Awaitable[None], str], None]
BroadcastFn = Callable[[str, LayerVisualState, Optional[str], float], Awaitable[None]]
DefaultLayerResolver = Callable[[], Optional[str]]


_ALLOWED_LAYER_KEYS = {
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "gamma",
    "colormap",
    "contrast_limits",
    "depiction",
    "rendering",
    "attenuation",
    "iso_threshold",
    "metadata",
    "thumbnail",
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
        self._pending: dict[str, set[str]] = {}
        self._pending_versions: dict[tuple[str, str], int] = {}
        self._pending_intents: dict[str, Optional[str]] = {}
        self._last_versions: dict[tuple[str, str], int] = {}
        self._latest_states: dict[str, LayerVisualState] = {}
        self._ndisplay: Optional[int] = None
        self._last_payload_sig: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ServerLayerMirror already started")
            self._started = True
        snapshot = self._ledger.snapshot()
        ledger_snapshot = build_ledger_snapshot(self._ledger, snapshot)
        nd_entry = snapshot.get(("view", "main", "ndisplay"))
        with self._lock:
            if nd_entry is not None and nd_entry.value is not None:
                self._ndisplay = int(nd_entry.value)
            self._pending.clear()
            self._pending_versions.clear()
            self._pending_intents.clear()
            self._last_versions.clear()
            self._latest_states.clear()
            self._last_payload_sig.clear()
            layer_values = ledger_snapshot.layer_values or {}
            for layer_id, layer_state in layer_values.items():
                norm_id = str(layer_id)
                self._latest_states[norm_id] = layer_state
                if layer_state.versions:
                    for prop, version in layer_state.versions.items():
                        self._last_versions[(norm_id, str(prop))] = int(version)
        for (scope, target, key), entry in snapshot.items():
            if scope not in {"volume", "multiscale"}:
                continue
            self._record_snapshot(
                scope,
                target,
                key,
                entry.value,
                entry.version,
                finalize=True,
            )
        self._ledger.subscribe("scene", "main", "op_state", self._on_op_state)
        self._ledger.subscribe("view", "main", "ndisplay", self._on_ndisplay)
        self._ledger.subscribe_all(self._on_event)

    # ------------------------------------------------------------------
    def latest_visual_states(self) -> dict[str, LayerVisualState]:
        with self._lock:
            return dict(self._latest_states)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._pending_versions.clear()
            self._pending_intents.clear()
            self._last_versions.clear()
            self._latest_states.clear()
            self._op_open = False
            self._ndisplay = None
            self._last_payload_sig.clear()

    # ------------------------------------------------------------------
    def _on_op_state(self, event: LedgerEvent) -> None:
        value = str(event.value)
        to_flush: dict[str, set[str]] = {}
        intents: dict[str, Optional[str]] = {}
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
                for layer_id, state in list(self._latest_states.items()):
                    volume_keys = [
                        key for key in state.extra.keys() if str(key).startswith("volume.")
                    ]
                    if not volume_keys:
                        continue
                    updates = {key: None for key in volume_keys}
                    version_updates = {key: None for key in volume_keys}
                    pruned = state.with_updates(
                        updates=updates,
                        versions=version_updates,
                    )
                    self._latest_states[layer_id] = pruned
                    for key in volume_keys:
                        self._last_versions.pop((layer_id, key), None)
                        self._pending_versions.pop((layer_id, key), None)

    # ------------------------------------------------------------------
    def _on_event(self, event: LedgerEvent) -> None:
        scope = str(event.scope)
        if scope == "view" and event.key == "ndisplay":
            self._on_ndisplay(event)
            return
        if scope not in {"layer", "volume", "multiscale"}:
            return
        to_flush: Optional[dict[str, set[str]]] = None
        intents: dict[str, Optional[str]] = {}
        with self._lock:
            layer_id, prop, value = self._map_event_locked(event)
            if layer_id is None or prop is None:
                return
            version_obj = event.version
            if version_obj is None:
                raise AssertionError("ledger layer event missing version")
            version_int = int(version_obj)
            key = (layer_id, prop)
            last = self._last_versions.get(key)
            if last is not None and last == version_int:
                return
            updates = {prop: value}
            version_map = {prop: version_int}
            self._update_state_locked(
                layer_id,
                updates=updates,
                versions=version_map,
                finalize=False,
            )
            pending_props = self._pending.setdefault(layer_id, set())
            pending_props.add(prop)
            self._pending_versions[key] = version_int
            if event.metadata and "intent_id" in event.metadata:
                intent_value = event.metadata.get("intent_id")
                self._pending_intents[layer_id] = (
                    None if intent_value is None else str(intent_value)
                )
            if not self._op_open:
                to_flush = self._drain_pending_locked()
                intents = self._pending_intents
                self._pending_intents = {}
        if not self._op_open and to_flush:
            self._flush(to_flush, intents, event.timestamp)

    # ------------------------------------------------------------------
    def _drain_pending_locked(self) -> dict[str, set[str]]:
        pending = self._pending
        self._pending = {}
        return {layer: set(props) for layer, props in pending.items() if props}

    # ------------------------------------------------------------------
    def _flush(
        self,
        pending: dict[str, set[str]],
        intents: dict[str, Optional[str]],
        timestamp: float,
    ) -> None:
        for layer_id, props in pending.items():
            state = self._latest_states.get(layer_id)
            if state is None or not props:
                continue
            subset = state.subset(props)
            if not subset.keys():
                continue
            # Content signature gating for emission (dedupe by values, not versions)
            sig = layer_payload_signature(subset)
            last = self._last_payload_sig.get(layer_id)
            if last is not None and last == sig:
                # No change in outward payload content; skip broadcast
                for prop in props:
                    _ = self._pending_versions.pop((layer_id, prop), None)
                continue
            self._last_payload_sig[layer_id] = sig
            for prop in props:
                version = self._pending_versions.pop((layer_id, prop), None)
                if version is not None:
                    self._last_versions[(layer_id, prop)] = version
            intent_id = intents.get(layer_id)
            self._schedule(
                self._broadcaster(layer_id, subset, intent_id, timestamp),
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
        *,
        finalize: bool = False,
    ) -> None:
        with self._lock:
            layer_id: Optional[str] = None
            prop_name: Optional[str] = None
            if scope == "layer":
                if key not in _ALLOWED_LAYER_KEYS:
                    return
                layer_id = str(target)
                prop_name = str(key)
            elif scope in {"volume", "multiscale"}:
                if not self._volume_enabled_locked():
                    return
                layer_id = str(target) if str(target).startswith("layer") else self._default_layer()
                if layer_id is None:
                    return
                prop_name = f"{scope}.{key}"
            else:
                return

            assert layer_id is not None and prop_name is not None

            version_map: dict[str, int | None] | None = None
            if version is not None:
                version_map = {prop_name: int(version)}

            updates = {prop_name: value}
            self._update_state_locked(
                layer_id,
                updates=updates,
                versions=version_map,
                finalize=finalize,
            )

    def _update_state_locked(
        self,
        layer_id: str,
        *,
        updates: Mapping[str, object] | None = None,
        versions: Mapping[str, int | None] | None = None,
        finalize: bool,
    ) -> LayerVisualState:
        state = self._latest_states.get(layer_id)
        if state is None:
            state = LayerVisualState(layer_id=layer_id)
        updates_payload = {str(key): value for key, value in (updates or {}).items()}
        versions_payload = (
            {str(key): (None if value is None else int(value)) for key, value in versions.items()}
            if versions
            else None
        )
        state = state.with_updates(updates=updates_payload, versions=versions_payload)
        self._latest_states[layer_id] = state
        if finalize and versions_payload:
            for key, value in versions_payload.items():
                if value is not None:
                    self._last_versions[(layer_id, key)] = int(value)
        return state

    def _volume_enabled_locked(self) -> bool:
        value = self._ndisplay
        return value is not None and int(value) >= 3


__all__ = ["ServerLayerMirror"]
