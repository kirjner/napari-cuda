"""Registry that mirrors server-provided layer specifications locally."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from napari_cuda.protocol.messages import (
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    LayerRenderHints,
    SceneSpecMessage,
)

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


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _merge_hints(base: LayerRenderHints | None, update: LayerRenderHints | None) -> LayerRenderHints | None:
    if update is None:
        return base
    if base is None:
        return update
    overrides = {}
    for field in fields(LayerRenderHints):
        value = getattr(update, field.name)
        if value is not None:
            overrides[field.name] = value
    return replace(base, **overrides)


def _merge_specs(base: LayerSpec | None, update: LayerSpec, partial: bool) -> LayerSpec:
    if not partial or base is None:
        return update
    overrides = {}
    for field in fields(LayerSpec):
        name = field.name
        value = getattr(update, name)
        if name == "render":
            merged = _merge_hints(getattr(base, name), value)
            if merged is not None:
                overrides[name] = merged
            continue
        if name == "metadata" and isinstance(value, dict):
            merged_meta = dict(getattr(base, name) or {})
            merged_meta.update(value)
            overrides[name] = merged_meta
            continue
        if name == "extras" and isinstance(value, dict):
            merged_extras = dict(getattr(base, name) or {})
            merged_extras.update(value)
            overrides[name] = merged_extras
            continue
        if _has_value(value):
            overrides[name] = value
    return replace(base, **overrides)


class RemoteLayerRegistry:
    """Thread-safe registry of remote layers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._layers: Dict[str, RemoteImageLayer] = {}
        self._specs: Dict[str, LayerSpec] = {}
        self._order: List[str] = []
        self._listeners: List[LayerListener] = []

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
    def apply_scene(self, message: SceneSpecMessage) -> None:
        snapshot: Optional[RegistrySnapshot] = None
        with self._lock:
            desired_order: List[str] = []
            changed = False
            for spec in message.scene.layers:
                layer_id = spec.layer_id
                desired_order.append(layer_id)
                self._specs[layer_id] = spec
                existing = self._layers.get(layer_id)
                if existing is None:
                    layer = self._create_layer(spec)
                    if layer is None:
                        if _LAYER_DEBUG:
                            logger.debug("skipping unsupported layer type: id=%s type=%s", spec.layer_id, spec.layer_type)
                        continue
                    self._layers[layer_id] = layer
                    changed = True
                    if _LAYER_DEBUG:
                        logger.debug("created remote layer from scene: id=%s name=%s", spec.layer_id, spec.name)
                else:
                    existing.update_from_spec(spec)
                    if _LAYER_DEBUG:
                        logger.debug("updated existing layer from scene: id=%s", spec.layer_id)
            # Remove layers no longer present
            current_ids = set(self._layers.keys())
            desired_ids = set(desired_order)
            for removed_id in current_ids - desired_ids:
                self._layers.pop(removed_id, None)
                self._specs.pop(removed_id, None)
                changed = True
                if _LAYER_DEBUG:
                    logger.debug("removed layer missing from scene: id=%s", removed_id)
            if desired_order != self._order:
                self._order = [layer_id for layer_id in desired_order if layer_id in self._layers]
                changed = True
            if changed:
                snapshot = self._snapshot_locked()
                if _LAYER_DEBUG and snapshot is not None:
                    logger.debug("scene snapshot applied: ids=%s", snapshot.ids())
        if snapshot is not None:
            self._emit(snapshot)

    def apply_update(self, message: LayerUpdateMessage) -> None:
        spec = message.layer
        if spec is None:
            return
        snapshot: Optional[RegistrySnapshot] = None
        with self._lock:
            base = self._specs.get(spec.layer_id)
            merged = _merge_specs(base, spec, message.partial)
            self._specs[merged.layer_id] = merged
            layer = self._layers.get(merged.layer_id)
            changed = False
            if layer is None:
                layer = self._create_layer(merged)
                if layer is None:
                    if _LAYER_DEBUG:
                        logger.debug("layer.update ignored (unsupported type): id=%s type=%s", merged.layer_id, merged.layer_type)
                    return
                self._layers[merged.layer_id] = layer
                if merged.layer_id not in self._order:
                    self._order.append(merged.layer_id)
                changed = True
                if _LAYER_DEBUG:
                    logger.debug("layer.update created new layer: id=%s", merged.layer_id)
            else:
                layer.update_from_spec(merged)
                changed = True
                if _LAYER_DEBUG:
                    logger.debug("layer.update applied to existing layer: id=%s partial=%s", merged.layer_id, message.partial)
            if changed:
                snapshot = self._snapshot_locked()
        if snapshot is not None:
            self._emit(snapshot)

    def remove_layer(self, message: LayerRemoveMessage) -> None:
        snapshot: Optional[RegistrySnapshot] = None
        with self._lock:
            removed = self._layers.pop(message.layer_id, None)
            if removed is None:
                return
            self._specs.pop(message.layer_id, None)
            if message.layer_id in self._order:
                self._order.remove(message.layer_id)
            if _LAYER_DEBUG:
                logger.debug("layer.remove applied: id=%s reason=%s", message.layer_id, message.reason)
            snapshot = self._snapshot_locked()
        if snapshot is not None:
            self._emit(snapshot)

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
            return None
        if _LAYER_DEBUG:
            logger.debug("instantiating RemoteImageLayer: id=%s type=%s", spec.layer_id, spec.layer_type)
        return RemoteImageLayer(spec)
