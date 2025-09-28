"""Layer intent bridge wiring Qt layer controls to server intents.

The bridge observes :class:`RemoteImageLayer` events, converts local UI
mutations into ``layer.intent.*`` payloads, and defers visible changes until
the server acknowledges them via ``layer.update`` messages. This keeps the
client UI in lock-step with the authoritative scene state while reusing the
existing intent rate limiting and acknowledgement tracking built into the
stream loop.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

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
class InteractionSession:
    key: str
    pending: List[PendingIntent] = field(default_factory=list)
    target_value: Any = None
    confirmed_value: Any = None
    last_sent_seq: Optional[int] = None
    last_confirmed_seq: Optional[int] = None
    last_send_ts: float = 0.0
    dirty: bool = False

    def mark_target(self, value: Any) -> None:
        self.target_value = value
        self.dirty = True

    def push_pending(self, seq: int, value: Any) -> None:
        now = time.perf_counter()
        self.pending.append(PendingIntent(seq=seq, value=value, timestamp=now))
        self.last_sent_seq = seq
        self.dirty = False
        self.last_send_ts = now

    def pop_pending(self, seq: Optional[int]) -> Optional[PendingIntent]:
        if not self.pending:
            return None
        if seq is None:
            return self.pending.pop(0)
        for idx, item in enumerate(self.pending):
            if item.seq == seq:
                return self.pending.pop(idx)
        return None

    def clear_pending(self) -> None:
        self.pending.clear()
        self.last_sent_seq = None

    def mark_confirmed(self, value: Any, seq: Optional[int]) -> None:
        self.confirmed_value = value
        if seq is not None:
            self.last_confirmed_seq = seq
        self.dirty = False


@dataclass
class LayerBinding:
    remote_id: str
    layer: RemoteImageLayer
    callbacks: list[tuple[EventEmitter, Callable]] = field(default_factory=list)
    handlers: dict[str, tuple[EventEmitter, Callable]] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)
    sessions: dict[str, InteractionSession] = field(default_factory=dict)


PROPERTY_CONFIGS: tuple[PropertyConfig, ...] = (
    PropertyConfig(
        key="opacity",
        event_name="opacity",
        intent_type="layer.intent.set_opacity",
        payload_builder=lambda value: {"opacity": float(value)},
        getter=lambda layer: float(getattr(layer, "opacity", 0.0)),
        setter=lambda layer, value: setattr(layer, "opacity", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "opacity")),
    ),
    PropertyConfig(
        key="visible",
        event_name="visible",
        intent_type="layer.intent.set_visibility",
        payload_builder=lambda value: {"visible": bool(value)},
        getter=lambda layer: bool(getattr(layer, "visible", False)),
        setter=lambda layer, value: setattr(layer, "visible", bool(value)),
        equals=lambda a, b: bool(a) is bool(b),
        spec_getter=lambda spec: _coerce_bool(_control_value(spec, "visible")),
    ),
    PropertyConfig(
        key="rendering",
        event_name="rendering",
        intent_type="layer.intent.set_render_mode",
        payload_builder=lambda value: {"mode": str(value)},
        getter=lambda layer: str(getattr(layer, "rendering", "")),
        setter=lambda layer, value: setattr(layer, "rendering", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: _control_value(spec, "rendering"),
    ),
    PropertyConfig(
        key="colormap",
        event_name="colormap",
        intent_type="layer.intent.set_colormap",
        payload_builder=lambda value: {"name": str(value)},
        getter=lambda layer: str(_colormap_name(getattr(layer, "colormap", None)) or ""),
        setter=lambda layer, value: setattr(layer, "colormap", str(value)),
        equals=lambda a, b: str(a) == str(b),
        spec_getter=lambda spec: _colormap_name(_control_value(spec, "colormap")),
    ),
    PropertyConfig(
        key="gamma",
        event_name="gamma",
        intent_type="layer.intent.set_gamma",
        payload_builder=lambda value: {"gamma": float(value)},
        getter=lambda layer: float(getattr(layer, "gamma", 1.0)),
        setter=lambda layer, value: setattr(layer, "gamma", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b), tol=1e-4),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "gamma")),
    ),
    PropertyConfig(
        key="iso_threshold",
        event_name="iso_threshold",
        intent_type="layer.intent.set_iso_threshold",
        payload_builder=lambda value: {"iso_threshold": float(value)},
        getter=lambda layer: float(getattr(layer, "iso_threshold", 0.0)),
        setter=lambda layer, value: setattr(layer, "iso_threshold", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "iso_threshold")),
    ),
    PropertyConfig(
        key="attenuation",
        event_name="attenuation",
        intent_type="layer.intent.set_attenuation",
        payload_builder=lambda value: {"attenuation": float(value)},
        getter=lambda layer: float(getattr(layer, "attenuation", 0.0)),
        setter=lambda layer, value: setattr(layer, "attenuation", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b)),
        spec_getter=lambda spec: _coerce_float(_control_value(spec, "attenuation")),
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


SESSION_PROPERTY_KEYS: set[str] = {config.key for config in PROPERTY_CONFIGS}


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
    def _uses_session(self, config: PropertyConfig) -> bool:
        return config.key in SESSION_PROPERTY_KEYS

    # ------------------------------------------------------------------
    def _ensure_session(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        *,
        initial_value: Any = None,
    ) -> InteractionSession:
        session = binding.sessions.get(config.key)
        if session is None:
            session = InteractionSession(key=config.key)
            binding.sessions[config.key] = session
        if initial_value is not None and session.target_value is None:
            session.target_value = initial_value
            session.confirmed_value = initial_value
        return session

    # ------------------------------------------------------------------
    def _settings_min_dt(self) -> float:
        min_dt = float(getattr(self._intent_state, "settings_min_dt", 0.0) or 0.0)
        if min_dt <= 0.0:
            min_dt = 1.0 / 60.0
        return min_dt

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
        processed_payload = payload
        suppressed_keys: set[str] = set()

        if self._enabled and kind == "layer-update":
            msg = payload if isinstance(payload, LayerUpdateMessage) else None
            if msg is not None:
                suppressed_keys = self.handle_layer_update(msg)
                if suppressed_keys:
                    self._strip_suppressed_controls(msg, suppressed_keys)
                processed_payload = msg

        if self._prev_dispatcher is not None and self._prev_dispatcher is not self._dispatch:
            try:
                self._prev_dispatcher(kind, processed_payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug("LayerIntentBridge prev dispatcher failed", exc_info=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_suppressed_controls(
        message: LayerUpdateMessage, keys: set[str]
    ) -> None:
        if not keys:
            return

        controls = message.controls if isinstance(message.controls, dict) else None
        spec = message.layer
        spec_controls = (
            spec.controls if (spec is not None and isinstance(spec.controls, dict)) else None
        )

        for key in keys:
            if controls is not None:
                controls.pop(key, None)
            if spec_controls is not None:
                spec_controls.pop(key, None)

        if controls is not None and not controls:
            message.controls = None
        if spec is not None and spec_controls is not None and not spec_controls:
            spec.controls = None

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
            binding.handlers[config.key] = (emitter, callback)
            try:
                value = config.getter(layer)
            except Exception:
                value = None
            if self._uses_session(config):
                self._ensure_session(binding, config, initial_value=value)
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
        binding.handlers.clear()
        pending_map = self._loop_state.pending_intents
        for session in binding.sessions.values():
            while session.pending:
                info = session.pop_pending(None)
                if info is None:
                    break
                pending_map.pop(info.seq, None)
        binding.sessions.clear()
        logger.debug("LayerIntentBridge unbound layer %s", remote_id)

    # ------------------------------------------------------------------
    def _make_property_handler(
        self, binding: LayerBinding, config: PropertyConfig
    ) -> Callable[[Any], None]:
        def _handler(event: Any = None) -> None:
            self._on_property_change(binding, config)

        return _handler

    # ------------------------------------------------------------------
    def _maybe_send_session(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        session: InteractionSession,
        *,
        force: bool = False,
    ) -> None:
        target = session.target_value
        if target is None:
            return

        if session.pending and config.equals(session.pending[-1].value, target):
            session.dirty = False
            return

        if not force and not self._rate_gate_settings():
            session.dirty = True
            return

        seq = self._send_intent(binding, config, target)
        if seq is None:
            session.dirty = True
            fallback = session.confirmed_value
            if fallback is not None and not config.equals(target, fallback):
                self._restore_property(binding, config, fallback)
            return

        session.push_pending(seq, target)
        # Keep rate limiter in sync with the last successful send.
        self._intent_state.last_settings_send = session.last_send_ts

    # ------------------------------------------------------------------
    def _handle_session_property_change(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        current_value: Any,
    ) -> None:
        session = self._ensure_session(binding, config)

        if (
            not session.pending
            and session.confirmed_value is not None
            and config.equals(session.confirmed_value, current_value)
        ):
            session.dirty = False
            session.target_value = current_value
            return

        session.mark_target(current_value)

        self._maybe_send_session(binding, config, session)

    # ------------------------------------------------------------------
    def _apply_session_remote(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        new_value: Any,
        intent_seq: Optional[int],
    ) -> bool:
        session = self._ensure_session(binding, config)

        if intent_seq is not None:
            removed = session.pop_pending(intent_seq)
            if removed is None and session.last_confirmed_seq is not None:
                if intent_seq <= session.last_confirmed_seq:
                    logger.debug(
                        "LayerIntentBridge stale ack ignored: id=%s key=%s seq=%s <= confirmed=%s",
                        binding.remote_id,
                        config.key,
                        intent_seq,
                        session.last_confirmed_seq,
                    )
                    return True
            elif removed is not None:
                self._loop_state.pending_intents.pop(removed.seq, None)
        else:
            # Reset pending when the server pushes an unsolicited update.
            while session.pending:
                info = session.pop_pending(None)
                if info is None:
                    break
                self._loop_state.pending_intents.pop(info.seq, None)

        if session.pending:
            if intent_seq is not None and session.pending[-1].seq < intent_seq:
                # Pending intents are older than the confirmed seq; drop them as stale.
                while session.pending:
                    stale = session.pop_pending(None)
                    if stale is None:
                        break
                    self._loop_state.pending_intents.pop(stale.seq, None)
            else:
                latest = session.pending[-1]
                session.mark_confirmed(new_value, intent_seq)
                current_value = None
                try:
                    current_value = config.getter(binding.layer)
                except Exception:
                    logger.debug(
                        "LayerIntentBridge getter failed during pending restore: id=%s key=%s",
                        binding.remote_id,
                        config.key,
                        exc_info=True,
                    )
                if current_value is None or not config.equals(current_value, latest.value):
                    self._restore_property(binding, config, latest.value)
                return False

        session.mark_confirmed(new_value, intent_seq)

        try:
            current_value = config.getter(binding.layer)
        except Exception:
            current_value = None
            logger.debug(
                "LayerIntentBridge getter failed: id=%s key=%s",
                binding.remote_id,
                config.key,
                exc_info=True,
            )

        if current_value is None or not config.equals(current_value, new_value):
            binding.suspended.add(config.key)
            try:
                handler = binding.handlers.get(config.key)
                blocker = nullcontext()
                if handler is not None:
                    emitter, callback = handler
                    if hasattr(emitter, "blocker"):
                        blocker = emitter.blocker(callback)
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

        # If the confirmed value lags behind the latest target, flush it now.
        if session.target_value is not None and not config.equals(session.target_value, new_value):
            session.dirty = True
            self._maybe_send_session(binding, config, session, force=True)
            return False

        session.target_value = new_value
        session.dirty = False
        return True

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

        self._handle_session_property_change(binding, config, current_value)

    # ------------------------------------------------------------------
    def _restore_property(
        self, binding: LayerBinding, config: PropertyConfig, value: Any
    ) -> None:
        if value is None:
            return
        binding.suspended.add(config.key)
        try:
            handler = binding.handlers.get(config.key)
            blocker = nullcontext()
            if handler is not None:
                emitter, callback = handler
                if hasattr(emitter, "blocker"):
                    blocker = emitter.blocker(callback)
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
        min_dt = float(getattr(self._intent_state, "settings_min_dt", 0.0) or 0.0)
        if min_dt <= 0.0:
            # Fall back to ~60Hz when no rate limit is configured
            min_dt = 1.0 / 60.0
        last = float(getattr(self._intent_state, "last_settings_send", 0.0) or 0.0)
        if (now - last) < min_dt:
            return False
        self._intent_state.last_settings_send = now
        return True

    # ------------------------------------------------------------------
    def handle_layer_update(self, message: LayerUpdateMessage) -> set[str]:
        suppressed: set[str] = set()
        if not self._enabled:
            return suppressed
        spec = message.layer
        if spec is None:
            return suppressed
        binding = self._bindings.get(spec.layer_id)
        if binding is None:
            return suppressed

        intent_seq = _intent_seq_from_message(message)
        if intent_seq is None:
            intent_seq = _extract_intent_seq(spec)
        changed_keys: set[str] = set()
        for config in PROPERTY_CONFIGS:
            new_value = config.spec_getter(spec)
            if new_value is None:
                continue
            changed_keys.add(config.key)
            applied = self._apply_remote_value(binding, config, new_value, intent_seq)
            if not applied:
                suppressed.add(config.key)

        return suppressed

    # ------------------------------------------------------------------
    def _apply_remote_value(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        new_value: Any,
        intent_seq: Optional[int],
    ) -> bool:
        return self._apply_session_remote(binding, config, new_value, intent_seq)

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
