"""Layer intent bridge wiring Qt layer controls to server intents.

The bridge observes :class:`RemoteImageLayer` events, converts local UI
mutations into ``layer.intent.*`` payloads, and defers visible changes until
the server acknowledges them via ``layer.update`` messages. This keeps the
client UI in lock-step with the authoritative scene state while reusing the
existing intent rate limiting and acknowledgement tracking built into the
stream loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING

from napari.utils.events import EventEmitter

from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
from napari_cuda.client.layers.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.streaming.client_loop.intents import IntentState
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.presenter_facade import PresenterFacade
from napari_cuda.protocol.messages import LayerSpec, LayerUpdateMessage

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


def _spec_value(spec: LayerSpec, *paths: tuple[str, ...]) -> Optional[Any]:
    for path in paths:
        node: Any = spec
        missing = False
        for part in path:
            if isinstance(node, dict):
                if part not in node:
                    missing = True
                    break
                node = node[part]
            else:
                if not hasattr(node, part):
                    missing = True
                    break
                node = getattr(node, part)
            if node is None:
                missing = True
                break
        if not missing and node is not None:
            return node
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
    value = _spec_value(spec, ("extras", "contrast_limits"), ("contrast_limits",))
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
    intent_type: str
    payload_builder: Callable[[Any], Dict[str, Any]]
    getter: Callable[[RemoteImageLayer], Any]
    setter: Callable[[RemoteImageLayer, Any], None]
    equals: Callable[[Any, Any], bool]
    spec_getter: Callable[[LayerSpec], Optional[Any]]


@dataclass
class PendingIntent:
    seq: int
    value: Any
    timestamp: float


@dataclass
class LayerBinding:
    remote_id: str
    layer: RemoteImageLayer
    callbacks: list[tuple[EventEmitter, Callable]] = field(default_factory=list)
    last_values: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, list[PendingIntent]] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)


PROPERTY_CONFIGS: tuple[PropertyConfig, ...] = (
    PropertyConfig(
        key="opacity",
        event_name="opacity",
        intent_type="layer.intent.set_opacity",
        payload_builder=lambda value: {"opacity": float(value)},
        getter=lambda layer: float(getattr(layer, "opacity", 0.0)),
        setter=lambda layer, value: setattr(layer, "opacity", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(
            _spec_value(
                spec,
                ("extras", "opacity"),
                ("render", "opacity"),
            )
        ),
    ),
    PropertyConfig(
        key="visibility",
        event_name="visible",
        intent_type="layer.intent.set_visibility",
        payload_builder=lambda value: {"visible": bool(value)},
        getter=lambda layer: bool(getattr(layer, "visible", False)),
        setter=lambda layer, value: setattr(layer, "visible", bool(value)),
        equals=lambda a, b: bool(a) is bool(b),
        spec_getter=lambda spec: _coerce_bool(
            _spec_value(
                spec,
                ("extras", "visibility"),
                ("render", "visibility"),
            )
        ),
    ),
    PropertyConfig(
        key="rendering",
        event_name="rendering",
        intent_type="layer.intent.set_render_mode",
        payload_builder=lambda value: {"mode": str(value)},
        getter=lambda layer: str(getattr(layer, "rendering", "")),
        setter=lambda layer, value: setattr(layer, "rendering", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: spec.render.mode if spec.render and spec.render.mode else None,
    ),
    PropertyConfig(
        key="colormap",
        event_name="colormap",
        intent_type="layer.intent.set_colormap",
        payload_builder=lambda value: {"name": str(value)},
        getter=lambda layer: str(_colormap_name(getattr(layer, "colormap", None)) or ""),
        setter=lambda layer, value: setattr(layer, "colormap", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: _colormap_name(
            _spec_value(
                spec,
                ("extras", "colormap"),
                ("render", "colormap"),
            )
        ),
    ),
    PropertyConfig(
        key="gamma",
        event_name="gamma",
        intent_type="layer.intent.set_gamma",
        payload_builder=lambda value: {"gamma": float(value)},
        getter=lambda layer: float(getattr(layer, "gamma", 1.0)),
        setter=lambda layer, value: setattr(layer, "gamma", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b), tol=1e-4),
        spec_getter=lambda spec: _coerce_float(
            _spec_value(
                spec,
                ("extras", "gamma"),
                ("render", "gamma"),
            )
        ),
    ),
    PropertyConfig(
        key="iso_threshold",
        event_name="iso_threshold",
        intent_type="layer.intent.set_iso_threshold",
        payload_builder=lambda value: {"iso_threshold": float(value)},
        getter=lambda layer: float(getattr(layer, "iso_threshold", 0.0)),
        setter=lambda layer, value: setattr(layer, "iso_threshold", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(
            _spec_value(
                spec,
                ("extras", "iso_threshold"),
                ("render", "iso_threshold"),
            )
        ),
    ),
    PropertyConfig(
        key="attenuation",
        event_name="attenuation",
        intent_type="layer.intent.set_attenuation",
        payload_builder=lambda value: {"attenuation": float(value)},
        getter=lambda layer: float(getattr(layer, "attenuation", 0.0)),
        setter=lambda layer, value: setattr(layer, "attenuation", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(
            _spec_value(
                spec,
                ("extras", "attenuation"),
                ("render", "attenuation"),
            )
        ),
    ),
    PropertyConfig(
        key="contrast_limits",
        event_name="contrast_limits",
        intent_type="layer.intent.set_contrast_limits",
        payload_builder=lambda value: {
            "lo": float(value[0]) if value is not None else 0.0,
            "hi": float(value[1]) if value is not None else 0.0,
        },
        getter=lambda layer: tuple(float(v) for v in getattr(layer, "contrast_limits", (0.0, 1.0))),
        setter=lambda layer, value: setattr(
            layer, "contrast_limits", (float(value[0]), float(value[1]))
        ),
        equals=lambda a, b: _tuples_close(a, b),
        spec_getter=lambda spec: _coerce_clim(spec),
    ),
)


class LayerIntentBridge:
    """Bridge layer property changes to remote intents."""

    def __init__(
        self,
        loop: ClientStreamLoop,
        presenter: PresenterFacade,
        registry,
        *,
        intent_state: IntentState,
        loop_state: ClientLoopState,
        enabled: Optional[bool] = None,
    ) -> None:
        self._loop = loop
        self._presenter = presenter
        self._registry = registry
        self._intent_state = intent_state
        self._loop_state = loop_state
        self._enabled = enabled if enabled is not None else _env_bridge_enabled()
        self._bindings: Dict[str, LayerBinding] = {}
        self._prev_dispatcher: Optional[Callable[[str, Any], None]] = None

        if not self._enabled:
            logger.debug("LayerIntentBridge disabled via environment")
            return

        self._prev_dispatcher = presenter.set_intent_dispatcher(self._dispatch)
        registry.add_listener(self._on_registry_snapshot)
        logger.info("LayerIntentBridge activated")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if not self._enabled:
            return
        for remote_id in list(self._bindings.keys()):
            self._unbind_layer(remote_id)
        if self._prev_dispatcher is not None:
            self._presenter.set_intent_dispatcher(self._prev_dispatcher)
            self._prev_dispatcher = None
        logger.info("LayerIntentBridge shut down")

    # ------------------------------------------------------------------
    def _dispatch(self, kind: str, payload: Any) -> None:
        if self._prev_dispatcher is not None and self._prev_dispatcher is not self._dispatch:
            try:
                self._prev_dispatcher(kind, payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug("LayerIntentBridge prev dispatcher failed", exc_info=True)

        if not self._enabled:
            return
        if kind == "layer-update":
            msg = payload if isinstance(payload, LayerUpdateMessage) else None
            if msg is not None:
                self.handle_layer_update(msg)

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
        for config in PROPERTY_CONFIGS:
            emitter = getattr(layer.events, config.event_name, None)
            if emitter is None or not isinstance(emitter, EventEmitter):
                continue

            callback = self._make_property_handler(binding, config)
            emitter.connect(callback)
            binding.callbacks.append((emitter, callback))
            try:
                binding.last_values[config.key] = config.getter(layer)
            except Exception:
                binding.last_values[config.key] = None
        self._bindings[remote_id] = binding
        logger.debug("LayerIntentBridge bound layer %s", remote_id)

    # ------------------------------------------------------------------
    def _unbind_layer(self, remote_id: str) -> None:
        binding = self._bindings.pop(remote_id, None)
        if binding is None:
            return
        for emitter, callback in binding.callbacks:
            with suppress_exception():
                emitter.disconnect(callback)
        pending_map = self._loop_state.pending_intents
        for queue in binding.pending.values():
            for info in queue:
                pending_map.pop(info.seq, None)
        logger.debug("LayerIntentBridge unbound layer %s", remote_id)

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
                "LayerIntentBridge getter failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
            return

        last_value = binding.last_values.get(config.key)
        pending_queue = binding.pending.get(config.key)

        if last_value is not None and config.equals(current_value, last_value):
            return

        if not self._rate_gate_settings():
            logger.debug(
                "LayerIntentBridge rate gated: id=%s key=%s",
                binding.remote_id,
                config.key,
            )
            return

        seq = self._send_intent(binding, config, current_value)
        if seq is None:
            self._restore_property(binding, config, last_value)
            return

        queue = pending_queue if pending_queue is not None else []
        queue.append(PendingIntent(seq=seq, value=current_value, timestamp=time.perf_counter()))
        binding.pending[config.key] = queue
        binding.last_values[config.key] = current_value

    # ------------------------------------------------------------------
    def _restore_property(
        self, binding: LayerBinding, config: PropertyConfig, value: Any
    ) -> None:
        if value is None:
            return
        binding.suspended.add(config.key)
        try:
            emitter = getattr(binding.layer.events, config.event_name, None)
            blocker = emitter.blocker() if hasattr(emitter, "blocker") else suppress_exception()
            with blocker:
                config.setter(binding.layer, value)
        except Exception:
            logger.debug(
                "LayerIntentBridge restore failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
        finally:
            binding.suspended.discard(config.key)

    # ------------------------------------------------------------------
    def _send_intent(
        self, binding: LayerBinding, config: PropertyConfig, value: Any
    ) -> Optional[int]:
        payload = {
            "type": config.intent_type,
            "layer_id": binding.remote_id,
            "client_id": self._intent_state.client_id,
        }
        try:
            payload.update(config.payload_builder(value))
        except Exception:
            logger.debug(
                "LayerIntentBridge payload build failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
            return None

        seq = self._intent_state.next_client_seq()
        payload["client_seq"] = seq
        ok = self._loop.post(payload)
        if ok:
            self._loop_state.pending_intents[seq] = {
                "kind": f"layer.{config.key}",
                "layer_id": binding.remote_id,
                "value": payload.get(config.key) or payload,
            }
            logger.info(
                "layer.intent dispatched: id=%s type=%s seq=%s payload=%s",
                binding.remote_id,
                config.intent_type,
                seq,
                payload,
            )
            return seq

        logger.warning(
            "LayerIntentBridge failed to send intent: id=%s payload=%s",
            binding.remote_id,
            payload,
        )
        return None

    # ------------------------------------------------------------------
    def _rate_gate_settings(self) -> bool:
        now = time.perf_counter()
        self._intent_state.last_settings_send = now
        return True

    # ------------------------------------------------------------------
    def handle_layer_update(self, message: LayerUpdateMessage) -> None:
        if not self._enabled:
            return
        spec = message.layer
        if spec is None:
            return
        binding = self._bindings.get(spec.layer_id)
        if binding is None:
            return

        intent_seq = _intent_seq_from_message(message)
        if intent_seq is None:
            intent_seq = _extract_intent_seq(spec)
        changed_keys: set[str] = set()
        for config in PROPERTY_CONFIGS:
            new_value = config.spec_getter(spec)
            if new_value is None:
                continue
            changed_keys.add(config.key)
            self._apply_remote_value(binding, config, new_value, intent_seq)

        if intent_seq is None and changed_keys:
            # Drop any pending entries for keys we just processed
            for key in changed_keys:
                info = binding.pending.pop(key, None)
                if info is not None:
                    self._loop_state.pending_intents.pop(info.seq, None)

    # ------------------------------------------------------------------
    def _apply_remote_value(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        new_value: Any,
        intent_seq: Optional[int],
    ) -> None:
        queue = binding.pending.get(config.key)
        removed_seq: Optional[int] = None
        if queue:
            if intent_seq is not None:
                new_queue = [item for item in queue if item.seq != intent_seq]
                if len(new_queue) != len(queue):
                    removed_seq = intent_seq
                    queue = new_queue
                else:
                    item = queue.pop(0)
                    removed_seq = item.seq
            else:
                latest_intent = queue[-1]
                pending_value = latest_intent.value
                binding.last_values[config.key] = pending_value
                if not config.equals(new_value, pending_value):
                    self._restore_property(binding, config, pending_value)
                return
            if queue:
                binding.pending[config.key] = queue
            else:
                binding.pending.pop(config.key, None)
        if removed_seq is not None:
            self._loop_state.pending_intents.pop(removed_seq, None)

        binding.last_values[config.key] = new_value

        current_value = config.getter(binding.layer)
        if current_value is not None and config.equals(current_value, new_value):
            return

        binding.suspended.add(config.key)
        try:
            emitter = getattr(binding.layer.events, config.event_name, None)
            blocker = emitter.blocker() if hasattr(emitter, "blocker") else suppress_exception()
            with blocker:
                config.setter(binding.layer, new_value)
        except Exception:
            logger.debug(
                "LayerIntentBridge remote apply failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )
        finally:
            binding.suspended.discard(config.key)

def _extract_intent_seq(spec: LayerSpec) -> Optional[int]:
    target_keys = (
        "intent_seq",
        "napari_cuda.intent_seq",
    )
    containers = [getattr(spec, "metadata", None), getattr(spec, "extras", None)]
    for container in containers:
        if isinstance(container, dict):
            for key in target_keys:
                if key in container:
                    try:
                        return int(container[key])
                    except Exception:
                        continue
    return None


def _intent_seq_from_message(message: LayerUpdateMessage) -> Optional[int]:
    # Preferred contract: LayerUpdateMessage exposes intent_seq as part of the
    # StreamProtocol payload. Honour that and rely on the protocol to raise if
    # the field is malformed.
    if not hasattr(message, "intent_seq") or message.intent_seq is None:
        return None
    return int(message.intent_seq)


class suppress_exception:
    """Context manager that silences all exceptions (logging debug only)."""

    def __enter__(self) -> "suppress_exception":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            logger.debug("LayerIntentBridge suppressed exception", exc_info=True)
        return True
