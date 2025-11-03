"""Registry that mirrors server-provided layer snapshots locally."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
)

import numpy as np

from napari_cuda.protocol.snapshots import LayerDelta, SceneSnapshot

from .remote_image_layer import RemoteImageLayer

logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> bool:
    flag = (os.getenv("NAPARI_CUDA_LAYER_DEBUG") or "").lower()
    if flag not in {"1", "true", "yes", "on", "dbg", "debug"}:
        return False
    has_local = any(getattr(handler, "_napari_cuda_local", False) for handler in logger.handlers)
    if not has_local:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
        handler.setLevel(logging.DEBUG)
        handler._napari_cuda_local = True
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return True


_LAYER_DEBUG = _maybe_enable_debug_logger()

_CONTROL_ONLY_KEYS = {
    "opacity",
    "visible",
    "blending",
    "interpolation",
    "colormap",
    "rendering",
    "gamma",
    "contrast_limits",
    "iso_threshold",
    "attenuation",
}

LayerListener = Callable[["RegistrySnapshot"], None]


@dataclass(frozen=True)
class LayerRecord:
    layer_id: str
    block: dict[str, Any]
    layer: RemoteImageLayer


@dataclass(frozen=True)
class RegistrySnapshot:
    layers: tuple[LayerRecord, ...]

    def iter(self) -> Iterable[LayerRecord]:
        return self.layers

    def ids(self) -> tuple[str, ...]:
        return tuple(record.layer_id for record in self.layers)


class RemoteLayerRegistry:
    """Thread-safe registry of remote layers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._layers: dict[str, RemoteImageLayer] = {}
        self._blocks: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._listeners: list[LayerListener] = []

    # ------------------------------------------------------------------
    def add_listener(self, callback: LayerListener) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: LayerListener) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    # ------------------------------------------------------------------
    def apply_snapshot(self, snapshot: SceneSnapshot) -> None:
        emitted: Optional[RegistrySnapshot] = None
        with self._lock:
            desired_order: list[str] = []
            changed = False
            for entry in snapshot.layers:
                layer_id = entry.layer_id
                desired_order.append(layer_id)
                block = dict(entry.block)
                self._blocks[layer_id] = block
                layer = self._layers.get(layer_id)
                if layer is None:
                    layer = self._create_layer(layer_id, block)
                    if layer is None:
                        self._blocks.pop(layer_id, None)
                        continue
                    self._layers[layer_id] = layer
                    changed = True
                    if _LAYER_DEBUG:
                        logger.debug("created remote layer: id=%s name=%s", layer_id, block.get("name"))
                else:
                    self._update_layer(layer, block)
                    changed = True
                    if _LAYER_DEBUG:
                        logger.debug("updated remote layer: id=%s", layer_id)

            current_ids = set(self._layers.keys())
            desired_ids = {layer_id for layer_id in desired_order if layer_id in self._layers}
            for removed_id in current_ids - desired_ids:
                self._layers.pop(removed_id, None)
                self._blocks.pop(removed_id, None)
                changed = True
                if _LAYER_DEBUG:
                    logger.debug("removed layer missing from snapshot: id=%s", removed_id)

            ordered_ids = [layer_id for layer_id in desired_order if layer_id in self._layers]
            if ordered_ids != self._order:
                self._order = ordered_ids
                changed = True

            if changed:
                emitted = self._snapshot_locked()
                if _LAYER_DEBUG and emitted is not None:
                    logger.debug("registry snapshot refreshed: ids=%s", emitted.ids())

        if emitted is not None:
            self._emit(emitted)

    def apply_delta(self, delta: LayerDelta) -> None:
        emitted: Optional[RegistrySnapshot] = None
        with self._lock:
            layer_id = delta.layer_id
            # Handle removal
            if getattr(delta, "removed", False):
                removed_layer = self._layers.pop(layer_id, None)
                self._blocks.pop(layer_id, None)
                if layer_id in self._order:
                    self._order.remove(layer_id)
                if removed_layer is None:
                    if _LAYER_DEBUG:
                        logger.debug("layer removal ignored (no cached layer): id=%s", layer_id)
                    return
                emitted = self._snapshot_locked()
                if _LAYER_DEBUG:
                    logger.debug("layer removed via delta: id=%s", layer_id)
            else:
                block = self._blocks.get(layer_id)
                if block is None:
                    if _LAYER_DEBUG:
                        logger.debug("layer delta ignored (no baseline): id=%s", layer_id)
                    return
                # TODO(protocol): consider emitting deltas in structured sections
                # (client consumes structured sections in v2).
                controls = block.get("controls")
                if not isinstance(controls, dict):
                    controls = {}
                    block["controls"] = controls
                # Structured sections
                controls_delta: dict[str, Any] = dict(delta.controls) if delta.controls else {}
                metadata_delta: dict[str, Any] | None = dict(delta.metadata) if delta.metadata else None
                data_delta: dict[str, Any] | None = dict(delta.data) if delta.data else None
                thumbnail_delta: dict[str, Any] | None = dict(delta.thumbnail) if delta.thumbnail else None

                needs_full_update = False
                # Apply data changes first (structural)
                if data_delta:
                    for key, value in data_delta.items():
                        block[key] = value
                    needs_full_update = True

                # Apply control deltas to block controls
                if controls_delta:
                    for key, value in controls_delta.items():
                        if key in _CONTROL_ONLY_KEYS:
                            controls[key] = value
                            if key == "contrast_limits":
                                if isinstance(value, Sequence):
                                    block["contrast_limits"] = [float(v) for v in value]
                                else:
                                    block["contrast_limits"] = value

                # Apply metadata (non-visual context)
                preview_payload = None
                if metadata_delta is not None:
                    block["metadata"] = dict(metadata_delta)
                    preview_raw = block["metadata"].get("thumbnail")
                    if preview_raw is not None:
                        arr = np.asarray(preview_raw, dtype=np.float32)
                        if arr.size > 0:
                            max_val = float(np.nanmax(arr))
                            if max_val > 0:
                                arr = arr / max_val
                        np.clip(arr, 0.0, 1.0, out=arr)
                        preview_payload = arr

                layer = self._layers.get(layer_id)
                if layer is None:
                    layer = self._create_layer(layer_id, block)
                    if layer is None:
                        return
                    self._layers[layer_id] = layer
                    if layer_id not in self._order:
                        self._order.append(layer_id)
                else:
                    # Route changes based on sections
                    if needs_full_update:
                        self._update_layer(layer, block)
                    elif controls_delta:
                        layer.apply_control_changes(block, controls_delta)
                    else:
                        pass

                # Apply/refresh thumbnail after layer existence ensured
                if thumbnail_delta is not None:
                    thumb = thumbnail_delta.get("array")
                    if thumb is None:
                        # Allow raw array under alternative key or pass-through mapping
                        # e.g., server may send already-normalized array with any key
                        # Prefer values under 'array', else attempt to coerce mapping itself
                        if "data" in thumbnail_delta:
                            thumb = thumbnail_delta.get("data")
                    if thumb is not None:
                        arr = np.asarray(thumb)
                    layer.update_thumbnail(arr)
                elif preview_payload is not None:
                    layer.update_thumbnail(preview_payload)

                emitted = self._snapshot_locked()
                if _LAYER_DEBUG:
                    logger.debug(
                        "layer delta applied: id=%s sections=(controls=%s, metadata=%s, data=%s, thumbnail=%s)",
                        layer_id,
                        bool(controls_delta),
                        metadata_delta is not None,
                        data_delta is not None,
                        thumbnail_delta is not None,
                    )

        if emitted is not None:
            self._emit(emitted)

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return self._snapshot_locked()

    # ------------------------------------------------------------------
    def _snapshot_locked(self) -> RegistrySnapshot:
        entries: list[LayerRecord] = []
        for layer_id in self._order:
            layer = self._layers.get(layer_id)
            block = self._blocks.get(layer_id)
            if layer is None or block is None:
                continue
            entries.append(LayerRecord(layer_id=layer_id, block=dict(block), layer=layer))
        return RegistrySnapshot(layers=tuple(entries))

    def _emit(self, snapshot: RegistrySnapshot) -> None:
        for callback in list(self._listeners):
            try:
                callback(snapshot)
            except Exception:
                logger.debug("layer registry listener failed", exc_info=True)

    def _create_layer(self, layer_id: str, block: dict[str, Any]) -> RemoteImageLayer | None:
        layer_type = str(block.get("layer_type", "")).lower()
        assert layer_type in {"image", "volume"}, f"unsupported layer type: {layer_type!r}"
        return RemoteImageLayer(layer_id=layer_id, block=block)

    def _update_layer(self, layer: RemoteImageLayer, block: dict[str, Any]) -> None:
        layer.update_from_block(block)
