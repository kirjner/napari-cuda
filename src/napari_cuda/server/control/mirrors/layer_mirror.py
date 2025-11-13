"""Mirror that projects ledger-backed layer state into notify.layers broadcasts."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from collections.abc import Awaitable, Callable, Mapping
from typing import Optional

from napari_cuda.server.ledger import LedgerEvent, ServerStateLedger
from napari_cuda.server.scene import LayerVisualState
from napari_cuda.server.scene.blocks import LayerBlock, layer_block_from_payload
from napari_cuda.server.scene.builders import scene_blocks_from_snapshot
from napari_cuda.server.scene.layer_block_adapter import layer_block_to_visual_state
from napari_cuda.server.scene.layer_block_diff import (
    AppliedVersions,
    LayerBlockDelta,
    compute_layer_block_deltas,
    layer_block_delta_updates,
)
from napari_cuda.server.utils.signatures import SignatureToken, layer_content_signature
from napari_cuda.shared.dims_spec import dims_spec_from_payload

ScheduleFn = Callable[[Awaitable[None], str], None]
BroadcastFn = Callable[[str, LayerBlockDelta, Optional[str], float], Awaitable[None]]
DefaultLayerResolver = Callable[[], Optional[str]]


logger = logging.getLogger(__name__)


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
        self._pending: dict[str, LayerBlockDelta] = {}
        self._pending_intents: dict[str, Optional[str]] = {}
        self._latest_states: dict[str, LayerVisualState] = {}
        self._latest_blocks: dict[str, LayerBlock] = {}
        self._block_versions: AppliedVersions = {}
        self._ndisplay: Optional[int] = None
        self._last_payload_sig: dict[str, SignatureToken] = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("ServerLayerMirror already started")
            self._started = True
        snapshot = self._ledger.snapshot()
        blocks_snapshot = scene_blocks_from_snapshot(snapshot)
        assert blocks_snapshot is not None, "layer blocks missing from ledger snapshot"
        spec_entry = snapshot.get(("dims", "main", "dims_spec"))
        with self._lock:
            if spec_entry is not None and spec_entry.value is not None:
                spec = dims_spec_from_payload(spec_entry.value)
                if spec is not None:
                    self._ndisplay = int(spec.ndisplay)
            self._pending.clear()
            self._pending_intents.clear()
            self._latest_states.clear()
            self._latest_blocks.clear()
            self._block_versions.clear()
            self._last_payload_sig.clear()
            for block in blocks_snapshot.layers:
                layer_id = str(block.layer_id)
                self._latest_blocks[layer_id] = block
                self._latest_states[layer_id] = layer_block_to_visual_state(block)
                for prop, version in (block.versions or {}).items():
                    self._block_versions[("layer", layer_id, str(prop))] = int(version)
        self._ledger.subscribe("scene", "main", "op_state", self._on_op_state)
        self._ledger.subscribe("dims", "main", "dims_spec", self._on_dims_spec)
        self._ledger.subscribe_all(self._on_event)

    # ------------------------------------------------------------------
    def latest_visual_states(self) -> dict[str, LayerVisualState]:
        with self._lock:
            return dict(self._latest_states)

    def latest_layer_blocks(self) -> dict[str, LayerBlock]:
        with self._lock:
            return dict(self._latest_blocks)

    def latest_layer_blocks(self) -> dict[str, LayerBlock]:
        with self._lock:
            return dict(self._latest_blocks)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._pending_intents.clear()
            self._latest_states.clear()
            self._latest_blocks.clear()
            self._block_versions.clear()
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
    def _handle_ndisplay(self, value: int) -> None:
        with self._lock:
            previous_enabled = self._volume_enabled_locked()
            self._ndisplay = value
            if previous_enabled and not self._volume_enabled_locked():
                for layer_id, state in list(self._latest_states.items()):
                    volume_keys = tuple(
                        key for key in state.extra.keys() if str(key).startswith("volume.")
                    )
                    if not volume_keys:
                        continue
                    updates = {key: None for key in volume_keys}
                    pruned = state.with_updates(updates=updates, versions=None)
                    self._latest_states[layer_id] = pruned
                    block = self._latest_blocks.get(layer_id)
                    if block is not None and block.extras:
                        filtered = {
                            str(key): value
                            for key, value in block.extras.items()
                            if not str(key).startswith("volume.")
                        }
                        self._latest_blocks[layer_id] = replace(block, extras=filtered)
                    for key in volume_keys:
                        self._block_versions.pop(("layer", layer_id, str(key)), None)

    def _on_dims_spec(self, event: LedgerEvent) -> None:
        spec = dims_spec_from_payload(event.value if isinstance(event.value, Mapping) else None)
        if spec is None:
            return
        self._handle_ndisplay(int(spec.ndisplay))

    # ------------------------------------------------------------------
    def _on_event(self, event: LedgerEvent) -> None:
        scope = str(event.scope)
        if scope == "dims" and event.key == "dims_spec":
            spec = dims_spec_from_payload(event.value if isinstance(event.value, Mapping) else None)
            if spec is not None:
                self._handle_ndisplay(int(spec.ndisplay))
            return
        if scope != "scene_layers" or event.key != "block":
            return

        payload = event.value
        if not isinstance(payload, Mapping):
            raise AssertionError("scene_layers block payload must be a mapping")
        block = layer_block_from_payload(payload)
        layer_id = str(block.layer_id)

        to_flush: Optional[dict[str, LayerBlockDelta]] = None
        intents: dict[str, Optional[str]] = {}
        with self._lock:
            deltas = compute_layer_block_deltas(
                self._block_versions,
                (block,),
                previous_blocks=self._latest_blocks,
            )
            delta = deltas.get(layer_id)
            if delta is None:
                return
            self._latest_blocks[layer_id] = block
            self._latest_states[layer_id] = layer_block_to_visual_state(block)
            self._pending[layer_id] = delta
            if event.metadata and "intent_id" in event.metadata:
                raw = event.metadata.get("intent_id")
                self._pending_intents[layer_id] = None if raw is None else str(raw)
            if not self._op_open:
                to_flush = self._drain_pending_locked()
                intents = self._pending_intents
                self._pending_intents = {}
        if not self._op_open and to_flush:
            self._flush(to_flush, intents, event.timestamp)

    # ------------------------------------------------------------------
    def _drain_pending_locked(self) -> dict[str, LayerBlockDelta]:
        pending = self._pending
        self._pending = {}
        return dict(pending)

    # ------------------------------------------------------------------
    def _flush(
        self,
        pending: dict[str, LayerBlockDelta],
        intents: dict[str, Optional[str]],
        timestamp: float,
    ) -> None:
        for layer_id, delta in pending.items():
            state = self._latest_states.get(layer_id)
            if state is None:
                continue
            updates, versions = layer_block_delta_updates(delta)
            subset = LayerVisualState(layer_id=layer_id).with_updates(
                updates=updates,
                versions=versions,
            )
            if not subset.keys() and not subset.extra and not subset.metadata and subset.thumbnail is None:
                continue
            # Content signature gating for emission (dedupe by values, not versions)
            sig = layer_content_signature(subset)
            last = self._last_payload_sig.get(layer_id)
            if not sig.changed(last):
                continue
            self._last_payload_sig[layer_id] = sig
            intent_id = intents.get(layer_id)
            self._schedule(
                self._broadcaster(layer_id, delta, intent_id, timestamp),
                f"mirror-layer-{layer_id}",
            )

    def _volume_enabled_locked(self) -> bool:
        value = self._ndisplay
        return value is not None and int(value) >= 3


__all__ = ["ServerLayerMirror"]
