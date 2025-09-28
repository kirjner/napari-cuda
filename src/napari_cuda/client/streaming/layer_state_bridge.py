"""Bridge Qt layer controls to the unified ``state.update`` protocol."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import math
import os
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING

from napari.utils.events import EventEmitter

from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
from napari_cuda.client.layers.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.streaming.client_loop.intents import ClientStateContext
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.state_store import StateStore
from napari_cuda.protocol.messages import LayerSpec, StateUpdateMessage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop


logger = logging.getLogger(__name__)


def _env_bridge_enabled() -> bool:
    flag = (os.getenv("NAPARI_CUDA_LAYER_BRIDGE") or "").lower()
    return flag in {"1", "true", "yes", "on"}


def _isclose(a: float, b: float, *, tol: float = 1e-5) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _tuples_close(a: Iterable[float], b: Iterable[float]) -> bool:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        return False
    return all(_isclose(x, y) for x, y in zip(a_list, b_list))


def _colormap_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _control_value(spec: LayerSpec, key: str) -> Optional[Any]:
    controls = getattr(spec, "controls", None)
    if isinstance(controls, dict):
        return controls.get(key)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _coerce_clim(spec: LayerSpec) -> Optional[tuple[float, float]]:
    value = _control_value(spec, "contrast_limits")
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


@dataclass(frozen=True)
class PropertyConfig:
    """Describe how to monitor and serialize a layer property."""

    key: str
    event_name: str
    encoder: Callable[[Any], Any]
    getter: Callable[[RemoteImageLayer], Any]
    setter: Callable[[RemoteImageLayer, Any], None]
    equals: Callable[[Any, Any], bool]
    spec_getter: Callable[[LayerSpec], Optional[Any]]


@dataclass
class PropertyRuntime:
    interaction_id: Optional[str] = None
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0


@dataclass
class LayerBinding:
    remote_id: str
    layer: RemoteImageLayer
    callbacks: list[tuple[EventEmitter, Callable]] = field(default_factory=list)
    handlers: dict[str, tuple[EventEmitter, Callable]] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)
    properties: dict[str, PropertyRuntime] = field(default_factory=dict)


PROPERTY_CONFIGS: tuple[PropertyConfig, ...] = (
    PropertyConfig(
        key="opacity",
        event_name="opacity",
        encoder=lambda value: float(value),
        getter=lambda layer: float(getattr(layer, "opacity", 0.0)),
        setter=lambda layer, value: setattr(layer, "opacity", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "opacity")),
    ),
    PropertyConfig(
        key="visible",
        event_name="visible",
        encoder=lambda value: bool(value),
        getter=lambda layer: bool(getattr(layer, "visible", False)),
        setter=lambda layer, value: setattr(layer, "visible", bool(value)),
        equals=lambda a, b: bool(a) is bool(b),
        spec_getter=lambda spec: _coerce_bool(_control_value(spec, "visible")),
    ),
    PropertyConfig(
        key="rendering",
        event_name="rendering",
        encoder=lambda value: str(value),
        getter=lambda layer: str(getattr(layer, "rendering", "")),
        setter=lambda layer, value: setattr(layer, "rendering", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: _control_value(spec, "rendering"),
    ),
    PropertyConfig(
        key="colormap",
        event_name="colormap",
        encoder=lambda value: str(value),
        getter=lambda layer: str(_colormap_name(getattr(layer, "colormap", None)) or ""),
        setter=lambda layer, value: setattr(layer, "colormap", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: _colormap_name(_control_value(spec, "colormap")),
    ),
    PropertyConfig(
        key="gamma",
        event_name="gamma",
        encoder=lambda value: float(value),
        getter=lambda layer: float(getattr(layer, "gamma", 1.0)),
        setter=lambda layer, value: setattr(layer, "gamma", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b), tol=1e-4),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "gamma")),
    ),
    PropertyConfig(
        key="contrast_limits",
        event_name="contrast_limits",
        encoder=lambda value: tuple(float(v) for v in value) if value is not None else None,
        getter=lambda layer: tuple(getattr(layer, "contrast_limits", (0.0, 1.0))),
        setter=lambda layer, value: setattr(layer, "contrast_limits", tuple(float(v) for v in value)),
        equals=lambda a, b: _tuples_close(a, b),
        spec_getter=_coerce_clim,
    ),
)

PROPERTY_BY_KEY: dict[str, PropertyConfig] = {cfg.key: cfg for cfg in PROPERTY_CONFIGS}


class LayerStateBridge:
    """Bridge layer property changes to the reducer-backed state pipeline."""

    def __init__(
        self,
        loop: ClientStreamLoop,
        presenter,
        registry,
        *,
        intent_state: ClientStateContext,
        loop_state: ClientLoopState,
        enabled: Optional[bool] = None,
    ) -> None:
        self._loop = loop
        self._registry = registry
        self._intent_state = intent_state
        self._loop_state = loop_state
        self._enabled = enabled if enabled is not None else _env_bridge_enabled()
        self._bindings: Dict[str, LayerBinding] = {}
        self._state_store = StateStore(
            client_id=intent_state.client_id,
            next_client_seq=intent_state.next_client_seq,
            clock=time.time,
        )

        if not self._enabled:
            logger.debug("LayerStateBridge disabled via environment")
            return

        registry.add_listener(self._on_registry_snapshot)
        logger.info("LayerStateBridge activated")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if not self._enabled:
            return
        for remote_id in list(self._bindings.keys()):
            self._unbind_layer(remote_id)
        try:
            self._registry.remove_listener(self._on_registry_snapshot)
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("LayerStateBridge remove_listener failed", exc_info=True)
        logger.info("LayerStateBridge shut down")

    # ------------------------------------------------------------------
    def clear_pending_on_reconnect(self) -> None:
        if not self._enabled:
            return
        self._state_store.clear_pending_on_reconnect()
        for binding in self._bindings.values():
            for runtime in binding.properties.values():
                runtime.active = False
                runtime.interaction_id = None

    # ------------------------------------------------------------------
    def handle_state_update(self, message: StateUpdateMessage) -> None:
        if not self._enabled:
            return
        if message.scope != "layer":
            return
        binding = self._bindings.get(message.target)
        if binding is None:
            return
        config = PROPERTY_BY_KEY.get(message.key)
        if config is None:
            return

        runtime = binding.properties.setdefault(config.key, PropertyRuntime())
        result = self._state_store.apply_remote(message)
        projection = result.projection_value
        self._apply_projection(binding, config, projection)

        runtime.last_phase = message.phase or runtime.last_phase
        if result.is_self:
            if result.pending_len == 0:
                runtime.active = False
                runtime.interaction_id = None
        else:
            runtime.active = False
            runtime.interaction_id = None
            runtime.last_send_ts = 0.0

        logger.debug(
            "state.update applied: id=%s key=%s value=%s is_self=%s pending=%d overridden=%s",
            binding.remote_id,
            config.key,
            projection,
            result.is_self,
            result.pending_len,
            result.overridden,
        )

    # ------------------------------------------------------------------
    def _on_registry_snapshot(self, snapshot: RegistrySnapshot) -> None:
        if not self._enabled:
            return

        desired_ids = {record.layer_id for record in snapshot.iter()}
        for record in snapshot.iter():
            if record.layer_id not in self._bindings:
                if isinstance(record.layer, RemoteImageLayer):
                    self._bind_layer(record)
        for remote_id in list(self._bindings.keys()):
            if remote_id not in desired_ids:
                self._unbind_layer(remote_id)

    # ------------------------------------------------------------------
    def _bind_layer(self, record: LayerRecord) -> None:
        layer = record.layer
        remote_id = record.layer_id
        binding = LayerBinding(remote_id=remote_id, layer=layer)
        for cfg in PROPERTY_CONFIGS:
            binding.properties[cfg.key] = PropertyRuntime()

        self._seed_binding(binding, record.spec)

        for config in PROPERTY_CONFIGS:
            emitter = getattr(layer.events, config.event_name, None)
            if emitter is None or not isinstance(emitter, EventEmitter):
                continue
            callback = self._make_property_handler(binding, config)
            emitter.connect(callback)
            binding.callbacks.append((emitter, callback))
            binding.handlers[config.key] = (emitter, callback)
        self._bindings[remote_id] = binding
        logger.debug("LayerStateBridge bound layer %s", remote_id)

    # ------------------------------------------------------------------
    def _seed_binding(self, binding: LayerBinding, spec: LayerSpec) -> None:
        for config in PROPERTY_CONFIGS:
            runtime = binding.properties.setdefault(config.key, PropertyRuntime())
            raw_value: Any = None
            try:
                raw_value = config.spec_getter(spec)
            except Exception:
                raw_value = None
            if raw_value is None:
                try:
                    raw_value = config.getter(binding.layer)
                except Exception:
                    raw_value = None
            encoded = self._encode_value(config, raw_value)
            if encoded is None:
                continue
            self._state_store.seed_confirmed("layer", binding.remote_id, config.key, encoded)
            runtime.last_phase = "seed"
            self._apply_projection(binding, config, encoded, suppress_blocker=True)

    # ------------------------------------------------------------------
    def _unbind_layer(self, remote_id: str) -> None:
        binding = self._bindings.pop(remote_id, None)
        if binding is None:
            return
        for emitter, callback in binding.callbacks:
            with suppress_exception():
                emitter.disconnect(callback)
        binding.handlers.clear()
        binding.properties.clear()
        logger.debug("LayerStateBridge unbound layer %s", remote_id)

    # ------------------------------------------------------------------
    def _make_property_handler(
        self, binding: LayerBinding, config: PropertyConfig
    ) -> Callable[[Any], None]:
        def _handler(event: Any = None) -> None:
            self._on_property_change(binding, config)

        return _handler

    # ------------------------------------------------------------------
    def _on_property_change(self, binding: LayerBinding, config: PropertyConfig) -> None:
        if config.key in binding.suspended:
            return
        try:
            current_value = config.getter(binding.layer)
        except Exception:
            logger.debug(
                "LayerStateBridge getter failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
            return

        encoded = self._encode_value(config, current_value)
        if encoded is None:
            return
        runtime = binding.properties.setdefault(config.key, PropertyRuntime())
        phase = "start" if not runtime.active else "update"
        interaction_id = runtime.interaction_id or uuid.uuid4().hex

        payload, projection = self._state_store.apply_local(
            "layer",
            binding.remote_id,
            config.key,
            encoded,
            phase,
            interaction_id=interaction_id,
        )
        self._apply_projection(binding, config, projection)

        runtime.active = True
        runtime.interaction_id = interaction_id
        runtime.last_phase = phase
        runtime.last_send_ts = time.perf_counter()

        ok = self._loop.post(payload.to_dict())
        if not ok:
            logger.warning(
                "LayerStateBridge failed to enqueue state.update: id=%s key=%s",
                binding.remote_id,
                config.key,
            )

    # ------------------------------------------------------------------
    def _apply_projection(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        value: Any,
        *,
        suppress_blocker: bool = False,
    ) -> None:
        if value is None:
            return
        try:
            current_value = config.getter(binding.layer)
        except Exception:
            current_value = None

        if current_value is not None and config.equals(current_value, value):
            return

        binding.suspended.add(config.key)
        try:
            blocker_ctx = nullcontext()
            if not suppress_blocker:
                handler = binding.handlers.get(config.key)
                if handler is not None:
                    emitter, callback = handler
                    if hasattr(emitter, "blocker"):
                        blocker_ctx = emitter.blocker(callback)
            with blocker_ctx:
                config.setter(binding.layer, value)
        except Exception:
            logger.debug(
                "LayerStateBridge projection failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
        finally:
            binding.suspended.discard(config.key)

    # ------------------------------------------------------------------
    def _encode_value(self, config: PropertyConfig, value: Any) -> Optional[Any]:
        if value is None:
            return None
        try:
            return config.encoder(value)
        except Exception:
            logger.debug(
                "LayerStateBridge value encode failed: key=%s value=%r",
                config.key,
                value,
                exc_info=True,
            )
            return None


class suppress_exception:
    """Context manager that silences all exceptions (logging debug only)."""

    def __enter__(self) -> "suppress_exception":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            logger.debug("LayerStateBridge suppressed exception", exc_info=True)
        return True


__all__ = [
    "LayerStateBridge",
]
