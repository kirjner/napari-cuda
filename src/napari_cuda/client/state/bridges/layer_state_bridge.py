"""Bridge Qt layer controls to the unified ``state.update`` protocol."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from napari.utils.events import EventEmitter

from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.data.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.pending_update_store import StateStore, PendingUpdate, AckOutcome

if TYPE_CHECKING:  # pragma: no cover - typing only
    from napari_cuda.client.runtime.stream_runtime import ClientStreamLoop


logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> bool:
    """Enable DEBUG logging when the layer debug flag is set."""

    flag = (os.getenv("NAPARI_CUDA_LAYER_DEBUG") or "").strip().lower()
    if flag not in {"1", "true", "yes", "on", "dbg", "debug"}:
        return False
    has_local = any(getattr(h, "_napari_cuda_local", False) for h in logger.handlers)
    if not has_local:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(logging.DEBUG)
        setattr(handler, "_napari_cuda_local", True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return True


_BRIDGE_DEBUG = _maybe_enable_debug_logger()


def _env_bridge_enabled() -> bool:
    """Return whether the layer bridge should be active."""

    flag = (os.getenv("NAPARI_CUDA_LAYER_BRIDGE") or "").strip().lower()
    if not flag:
        return True
    if flag in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return True


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


def _control_value(block: Mapping[str, Any], key: str) -> Optional[Any]:
    controls = block.get("controls") if isinstance(block.get("controls"), dict) else None
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


def _coerce_clim(block: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    value = _control_value(block, "contrast_limits")
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
    block_getter: Callable[[Mapping[str, Any]], Optional[Any]]


@dataclass
class PropertyRuntime:
    active: bool = False
    last_phase: Optional[str] = None
    last_send_ts: float = 0.0
    active_intent_id: Optional[str] = None
    active_frame_id: Optional[str] = None


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
        block_getter=lambda block: _coerce_float(_control_value(block, "opacity")),
    ),
    PropertyConfig(
        key="visible",
        event_name="visible",
        encoder=lambda value: bool(value),
        getter=lambda layer: bool(getattr(layer, "visible", False)),
        setter=lambda layer, value: setattr(layer, "visible", bool(value)),
        equals=lambda a, b: bool(a) is bool(b),
        block_getter=lambda block: _coerce_bool(_control_value(block, "visible")),
    ),
    PropertyConfig(
        key="rendering",
        event_name="rendering",
        encoder=lambda value: str(value),
        getter=lambda layer: str(getattr(layer, "rendering", "")),
        setter=lambda layer, value: setattr(layer, "rendering", str(value)),
        equals=lambda a, b: str(a) == str(b),
        block_getter=lambda block: _control_value(block, "rendering"),
    ),
    PropertyConfig(
        key="colormap",
        event_name="colormap",
        encoder=lambda value: str(value),
        getter=lambda layer: str(_colormap_name(getattr(layer, "colormap", None)) or ""),
        setter=lambda layer, value: setattr(layer, "colormap", str(value)),
        equals=lambda a, b: str(a) == str(b),
        block_getter=lambda block: _colormap_name(_control_value(block, "colormap")),
    ),
    PropertyConfig(
        key="gamma",
        event_name="gamma",
        encoder=lambda value: float(value),
        getter=lambda layer: float(getattr(layer, "gamma", 1.0)),
        setter=lambda layer, value: setattr(layer, "gamma", float(value)),
        equals=lambda a, b: _isclose(float(a), float(b), tol=1e-4),
        block_getter=lambda block: _coerce_float(_control_value(block, "gamma")),
    ),
    PropertyConfig(
        key="contrast_limits",
        event_name="contrast_limits",
        encoder=lambda value: tuple(float(v) for v in value) if value is not None else None,
        getter=lambda layer: tuple(getattr(layer, "contrast_limits", (0.0, 1.0))),
        setter=lambda layer, value: setattr(layer, "contrast_limits", tuple(float(v) for v in value)),
        equals=lambda a, b: _tuples_close(a, b),
        block_getter=_coerce_clim,
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
        control_state: ControlStateContext,
        loop_state: ClientLoopState,
        *,
        enabled: Optional[bool] = None,
        state_store: Optional[StateStore] = None,
    ) -> None:
        self._loop = loop
        self._registry = registry
        self._control_state = control_state
        self._loop_state = loop_state
        self._enabled = enabled if enabled is not None else _env_bridge_enabled()
        self._bindings: Dict[str, LayerBinding] = {}
        self._state_store = state_store or StateStore(clock=time.time)
        self._mute_depth = 0

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
        logger.debug("LayerStateBridge shut down")

    # ------------------------------------------------------------------
    def clear_pending_on_reconnect(self) -> None:
        if not self._enabled:
            return
        self._state_store.clear_pending_on_reconnect()
        for binding in self._bindings.values():
            for runtime in binding.properties.values():
                runtime.active = False
                runtime.active_intent_id = None
                runtime.active_frame_id = None

    # ------------------------------------------------------------------
    def handle_ack(self, outcome: AckOutcome) -> None:
        if not self._enabled:
            return
        if outcome.scope != "layer" or outcome.target is None or outcome.key is None:
            return

        binding = self._bindings.get(outcome.target)
        if binding is None:
            return

        config = PROPERTY_BY_KEY.get(outcome.key)
        if config is None:
            return

        runtime = binding.properties.setdefault(config.key, PropertyRuntime())
        runtime.last_phase = outcome.update_phase or runtime.last_phase
        runtime.last_send_ts = time.perf_counter()

        if outcome.in_reply_to == runtime.active_frame_id:
            runtime.active_frame_id = None
            runtime.active_intent_id = None

        runtime.active = outcome.pending_len > 0
        if not runtime.active:
            runtime.last_phase = None

        if outcome.status == "accepted":
            logger.debug(
                "layer ack accepted: id=%s key=%s pending=%d",
                binding.remote_id,
                config.key,
                outcome.pending_len,
            )
            return

        error = outcome.error or {}
        logger.warning(
            "layer intent rejected: id=%s key=%s code=%s message=%s details=%s",
            binding.remote_id,
            config.key,
            error.get("code"),
            error.get("message"),
            error.get("details"),
        )

        revert_value = outcome.confirmed_value
        if revert_value is None:
            revert_value = outcome.pending_value

        if revert_value is not None:
            self._apply_projection(binding, config, revert_value, suppress_blocker=True)

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

        self._seed_binding(binding, record.block)

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
    def _seed_binding(self, binding: LayerBinding, block: Mapping[str, Any]) -> None:
        for config in PROPERTY_CONFIGS:
            runtime = binding.properties.setdefault(config.key, PropertyRuntime())
            raw_value: Any = None
            try:
                raw_value = config.block_getter(block)
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

    # ------------------------------------------------------------------
    @staticmethod
    def sync_viewer_layers(viewer: Any, snapshot: RegistrySnapshot) -> None:
        """Mirror the registry snapshot into the viewer's layer list."""

        assert hasattr(viewer, "layers"), "proxy viewer must expose a layers collection"
        layers_obj = viewer.layers
        events = layers_obj.events
        assert hasattr(events, "blocker"), "LayerList events missing blocker()"
        ctx_manager = events.blocker()
        had_flag = hasattr(viewer, "_suppress_forward")
        previous_flag = getattr(viewer, "_suppress_forward", False) if had_flag else False
        if had_flag:
            setattr(viewer, "_suppress_forward", True)
        try:
            with ctx_manager:
                desired_records = tuple(snapshot.iter())
                desired_ids = [record.layer_id for record in desired_records]

                for layer in list(layers_obj):
                    assert hasattr(layer, "remote_id"), "encountered non-remote layer in proxy viewer"
                    remote_id = str(layer.remote_id)
                    if remote_id not in desired_ids:
                        layers_obj.remove(layer)

                existing_map: dict[str, Any] = {}
                for layer in list(layers_obj):
                    remote_id = str(layer.remote_id)
                    existing_map[remote_id] = layer

                for idx, record in enumerate(desired_records):
                    layer_id = record.layer_id
                    current = existing_map.get(layer_id)

                    if current is None:
                        insert_at = min(idx, len(layers_obj))
                        layers_obj.insert(insert_at, record.layer)
                        current = record.layer
                        existing_map[layer_id] = current
                    else:
                        if current is not record.layer:
                            assert hasattr(current, "update_from_block"), "remote layer missing update_from_block"
                            current.update_from_block(record.block)

                    current_index = layers_obj.index(current)
                    if current_index != idx:
                        layers_obj.move(current_index, idx)

                    controls = record.block.get("controls") if isinstance(record.block.get("controls"), dict) else None
                    if isinstance(controls, dict) and "visible" in controls:
                        emitter = current.events.visible
                        block_ctx = emitter.blocker()
                        with block_ctx:
                            current.visible = bool(controls["visible"])
        finally:
            if had_flag:
                setattr(viewer, "_suppress_forward", previous_flag)

    def _make_property_handler(
        self, binding: LayerBinding, config: PropertyConfig
    ) -> Callable[[Any], None]:
        def _handler(event: Any = None) -> None:
            self._on_property_change(binding, config)

        return _handler

    # ------------------------------------------------------------------
    def _on_property_change(self, binding: LayerBinding, config: PropertyConfig) -> None:
        if not self._enabled:
            return
        if self._mute_depth > 0:
            return
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

        intent_id, frame_id = self._control_state.next_intent_ids()

        pending = self._state_store.apply_local(
            "layer",
            binding.remote_id,
            config.key,
            encoded,
            phase,
            intent_id=intent_id,
            frame_id=frame_id,
            metadata={
                "layer_id": binding.remote_id,
                "property": config.key,
            },
        )
        self._apply_projection(binding, config, pending.projection_value)

        runtime.active = True
        runtime.last_phase = phase
        runtime.last_send_ts = time.perf_counter()
        runtime.active_intent_id = pending.intent_id
        runtime.active_frame_id = pending.frame_id

        ok = self._loop._dispatch_state_update(pending, origin=f"layer:{config.key}")
        if not ok:
            logger.warning(
                "LayerStateBridge failed to enqueue state.update: id=%s key=%s",
                binding.remote_id,
                config.key,
            )
            runtime.active = False
            runtime.active_intent_id = None
            runtime.active_frame_id = None

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

    @contextmanager
    def remote_sync(self):
        if not self._enabled:
            yield
            return
        self._mute_depth += 1
        try:
            yield
        finally:
            self._mute_depth -= 1

    def seed_remote_values(self, layer_id: str, changes: Mapping[str, Any]) -> None:
        if not self._enabled or not changes:
            return
        binding = self._bindings.get(layer_id)
        for key, raw_value in changes.items():
            if key == "removed":
                continue
            config = PROPERTY_BY_KEY.get(key)
            if config is None:
                continue
            encoded = self._encode_value(config, raw_value)
            if encoded is None:
                continue
            self._state_store.seed_confirmed("layer", layer_id, config.key, encoded)
            if binding is None:
                continue
            runtime = binding.properties.setdefault(config.key, PropertyRuntime())
            runtime.active = False
            runtime.active_intent_id = None
            runtime.active_frame_id = None
            runtime.last_phase = None


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
