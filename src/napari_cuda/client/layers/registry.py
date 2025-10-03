"""Registry that mirrors server-provided layer specifications locally."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from napari_cuda.protocol.messages import LayerSpec
from napari_cuda.protocol.snapshots import LayerDelta, SceneSnapshot

from .remote_image_layer import RemoteImageLayer


logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> bool:
    """Enable DEBUG logging for this module when env requests it."""

    flag = (os.getenv('NAPARI_CUDA_LAYER_DEBUG') or '').lower()
    if flag not in ('1', 'true', 'yes', 'on', 'dbg', 'debug'):
        return False
    has_local = any(getattr(h, '_napari_cuda_local', False) for h in logger.handlers)
    if not has_local:
        handler = logging.StreamHandler()
        fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(logging.DEBUG)
        setattr(handler, '_napari_cuda_local', True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return True


_LAYER_DEBUG = _maybe_enable_debug_logger()

LayerListener = Callable[["RegistrySnapshot"], None]


@dataclass(frozen=True)
class LayerRecord:
    layer_id: str
    spec: LayerSpec
    layer: RemoteImageLayer


@dataclass(frozen=True)
class RegistrySnapshot:
    layers: Tuple[LayerRecord, ...]

    def iter(self) -> Iterable[LayerRecord]:
        return self.layers

    def ids(self) -> Tuple[str, ...]:
        return tuple(record.layer_id for record in self.layers)


class RemoteLayerRegistry:
    """Thread-safe registry of remote layers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._layers: Dict[str, RemoteImageLayer] = {}
        self._specs: Dict[str, LayerSpec] = {}
        self._order: List[str] = []
        self._listeners: List[LayerListener] = []
        self._blocks: Dict[str, Dict[str, Any]] = {}

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
            desired_order: List[str] = []
            changed = False
            for entry in snapshot.layers:
                layer_id = entry.layer_id
                desired_order.append(layer_id)
                block = dict(entry.block)
                self._blocks[layer_id] = block
                spec = LayerSpec.from_dict(block)
                layer = self._layers.get(layer_id)
                if layer is None:
                    layer = self._create_layer(spec)
                    if layer is None:
                        self._blocks.pop(layer_id, None)
                        continue
                    self._layers[layer_id] = layer
                    changed = True
                    if _LAYER_DEBUG:
                        logger.debug("created remote layer from snapshot: id=%s name=%s", layer_id, spec.name)
                else:
                    layer.update_from_spec(spec)
                    changed = True
                    if _LAYER_DEBUG:
                        logger.debug("updated remote layer from snapshot: id=%s", layer_id)
                self._specs[layer_id] = spec

            current_ids = set(self._layers.keys())
            desired_ids = {layer_id for layer_id in desired_order if layer_id in self._layers}
            for removed_id in current_ids - desired_ids:
                self._layers.pop(removed_id, None)
                self._specs.pop(removed_id, None)
                self._blocks.pop(removed_id, None)
                changed = True
                if _LAYER_DEBUG:
                    logger.debug("removed layer missing from snapshot: id=%s", removed_id)

            if desired_order:
                ordered_ids = [layer_id for layer_id in desired_order if layer_id in self._layers]
            else:
                ordered_ids = []
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
            changes = dict(delta.changes)
            if not changes:
                return

            if changes.get("removed"):
                had_spec = layer_id in self._specs
                had_block = layer_id in self._blocks
                had_order = layer_id in self._order
                removed = self._layers.pop(layer_id, None)
                self._specs.pop(layer_id, None)
                self._blocks.pop(layer_id, None)
                if had_order:
                    self._order.remove(layer_id)
                if removed is None and not (had_spec or had_block or had_order):
                    if _LAYER_DEBUG:
                        logger.debug("layer removal ignored (no cached layer): id=%s", layer_id)
                    return
                emitted = self._snapshot_locked()
                if _LAYER_DEBUG and removed is not None:
                    logger.debug("layer removed via delta: id=%s", layer_id)
            else:
                block = self._blocks.get(layer_id)
                if block is None:
                    if _LAYER_DEBUG:
                        logger.debug("layer delta ignored (no baseline): id=%s", layer_id)
                    return
                controls = block.get("controls")
                if not isinstance(controls, dict):
                    controls = {}
                    block["controls"] = controls
                updated = False
                for key, value in changes.items():
                    if key == "removed":
                        continue
                    controls[key] = value
                    if key == "contrast_limits":
                        if isinstance(value, Sequence):
                            block["contrast_limits"] = [float(v) for v in value]
                        else:
                            block["contrast_limits"] = value
                    updated = True
                if not updated:
                    return

                spec = LayerSpec.from_dict(dict(block))
                self._specs[layer_id] = spec
                layer = self._layers.get(layer_id)
                if layer is None:
                    layer = self._create_layer(spec)
                    if layer is None:
                        return
                    self._layers[layer_id] = layer
                    if layer_id not in self._order:
                        self._order.append(layer_id)
                else:
                    layer.update_from_spec(spec)
                emitted = self._snapshot_locked()
                if _LAYER_DEBUG:
                    logger.debug("layer delta applied: id=%s keys=%s", layer_id, tuple(changes.keys()))

        if emitted is not None:
            self._emit(emitted)

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return self._snapshot_locked()

    # ------------------------------------------------------------------
    def _snapshot_locked(self) -> RegistrySnapshot:
        entries: List[LayerRecord] = []
        for layer_id in self._order:
            layer = self._layers.get(layer_id)
            spec = self._specs.get(layer_id)
            if layer is None or spec is None:
                continue
            entries.append(LayerRecord(layer_id=layer_id, spec=spec, layer=layer))
        return RegistrySnapshot(layers=tuple(entries))

    def _emit(self, snapshot: RegistrySnapshot) -> None:
        listeners: List[LayerListener]
        with self._lock:
            listeners = list(self._listeners)
        if _LAYER_DEBUG:
            logger.debug("emitting registry snapshot: ids=%s", snapshot.ids())
        for callback in listeners:
            try:
                callback(snapshot)
            except Exception:  # pragma: no cover - listeners should handle their own errors
                continue

    def _create_layer(self, spec: LayerSpec) -> RemoteImageLayer | None:
        layer_type = spec.layer_type.lower()
        if layer_type not in ("image", "volume"):
            if _LAYER_DEBUG:
                logger.debug("unsupported layer type: id=%s type=%s", spec.layer_id, spec.layer_type)
            return None
        if _LAYER_DEBUG:
            logger.debug("instantiating RemoteImageLayer: id=%s type=%s", spec.layer_id, spec.layer_type)
        return RemoteImageLayer(spec)
