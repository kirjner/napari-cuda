"""Emit layer intents derived from local ``RemoteImageLayer`` interactions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from qtpy import QtCore

from napari.utils.events import EventEmitter
from napari_cuda.client.control.client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
)
from napari_cuda.client.control.state_update_actions import (
    ControlStateContext,
    _emit_state_update,
    _update_runtime_from_ack_outcome,
)
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Property helpers (mirrors ``LayerStateBridge`` definitions but without the
# bridge scaffolding). We keep the helpers explicit to avoid rampant
# ``getattr`` usage; each property exposes a direct getter/setter pair.


def _event_visible(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.visible


def _event_opacity(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.opacity


def _event_rendering(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.rendering


def _event_colormap(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.colormap


def _event_gamma(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.gamma


def _event_contrast_limits(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.contrast_limits


def _get_visible(layer: RemoteImageLayer) -> bool:
    return bool(layer.visible)


def _set_visible(layer: RemoteImageLayer, value: bool) -> None:
    layer.visible = bool(value)


def _get_opacity(layer: RemoteImageLayer) -> float:
    return float(layer.opacity)


def _set_opacity(layer: RemoteImageLayer, value: float) -> None:
    layer.opacity = float(value)


def _get_rendering(layer: RemoteImageLayer) -> str:
    return str(layer.rendering)


def _set_rendering(layer: RemoteImageLayer, value: str) -> None:
    layer.rendering = str(value)


def _get_colormap(layer: RemoteImageLayer) -> str:
    cmap = layer.colormap
    assert hasattr(cmap, "name"), "RemoteImageLayer colormap missing 'name'"
    name = cmap.name  # type: ignore[attr-defined]
    assert isinstance(name, str), "RemoteImageLayer colormap name must be str"
    return name


def _set_colormap(layer: RemoteImageLayer, value: str) -> None:
    layer.colormap = str(value)


def _get_gamma(layer: RemoteImageLayer) -> float:
    return float(layer.gamma)


def _set_gamma(layer: RemoteImageLayer, value: float) -> None:
    layer.gamma = float(value)


def _get_contrast_limits(layer: RemoteImageLayer) -> tuple[float, float]:
    low, high = layer.contrast_limits
    return (float(low), float(high))


def _set_contrast_limits(layer: RemoteImageLayer, value: Mapping[str, Any] | tuple[Any, Any] | list[Any]) -> None:
    lo, hi = value  # type: ignore[assignment]
    layer.contrast_limits = (float(lo), float(hi))


def _encode_bool(value: Any) -> bool:
    return bool(value)


def _encode_float(value: Any) -> float:
    return float(value)


def _encode_str(value: Any) -> str:
    return str(value)


def _encode_colormap(value: Any) -> str:
    if isinstance(value, str):
        return value
    assert hasattr(value, "name"), "Colormap payload missing 'name'"
    name = value.name  # type: ignore[attr-defined]
    assert isinstance(name, str), "Colormap name must be str"
    return name


def _encode_limits(value: Any) -> tuple[float, float]:
    lo, hi = value
    return (float(lo), float(hi))


def _block_visible(block: Mapping[str, Any]) -> bool | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "visible" in controls:
        return _encode_bool(controls["visible"])
    return None


def _block_opacity(block: Mapping[str, Any]) -> float | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "opacity" in controls:
        return _encode_float(controls["opacity"])
    return None


def _block_rendering(block: Mapping[str, Any]) -> str | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "rendering" in controls:
        return _encode_str(controls["rendering"])
    return None


def _block_colormap(block: Mapping[str, Any]) -> str | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "colormap" in controls:
        return _encode_colormap(controls["colormap"])
    return None


def _block_gamma(block: Mapping[str, Any]) -> float | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "gamma" in controls:
        return _encode_float(controls["gamma"])
    return None


def _block_contrast_limits(block: Mapping[str, Any]) -> tuple[float, float] | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "contrast_limits" in controls:
        return _encode_limits(controls["contrast_limits"])
    if "contrast_limits" in block:
        return _encode_limits(block["contrast_limits"])
    return None


def _equals_bool(lhs: Any, rhs: Any) -> bool:
    return bool(lhs) is bool(rhs)


def _equals_float(lhs: Any, rhs: Any, *, tol: float = 1e-5) -> bool:
    return abs(float(lhs) - float(rhs)) <= tol


def _equals_str(lhs: Any, rhs: Any) -> bool:
    return str(lhs) == str(rhs)


def _equals_limits(lhs: Any, rhs: Any) -> bool:
    left = tuple(float(v) for v in lhs)
    right = tuple(float(v) for v in rhs)
    return len(left) == len(right) and all(_equals_float(a, b) for a, b in zip(left, right, strict=False))


@dataclass(frozen=True)
class PropertyConfig:
    """Describe how to monitor and encode a layer property."""

    key: str
    event_getter: Callable[[RemoteImageLayer], EventEmitter]
    getter: Callable[[RemoteImageLayer], Any]
    setter: Callable[[RemoteImageLayer, Any], None]
    encoder: Callable[[Any], Any]
    equals: Callable[[Any, Any], bool]
    block_getter: Callable[[Mapping[str, Any]], Any | None]


PROPERTY_CONFIGS: tuple[PropertyConfig, ...] = (
    PropertyConfig(
        key="visible",
        event_getter=_event_visible,
        getter=_get_visible,
        setter=_set_visible,
        encoder=_encode_bool,
        equals=_equals_bool,
        block_getter=_block_visible,
    ),
    PropertyConfig(
        key="opacity",
        event_getter=_event_opacity,
        getter=_get_opacity,
        setter=_set_opacity,
        encoder=_encode_float,
        equals=_equals_float,
        block_getter=_block_opacity,
    ),
    PropertyConfig(
        key="rendering",
        event_getter=_event_rendering,
        getter=_get_rendering,
        setter=_set_rendering,
        encoder=_encode_str,
        equals=_equals_str,
        block_getter=_block_rendering,
    ),
    PropertyConfig(
        key="colormap",
        event_getter=_event_colormap,
        getter=_get_colormap,
        setter=_set_colormap,
        encoder=_encode_colormap,
        equals=_equals_str,
        block_getter=_block_colormap,
    ),
    PropertyConfig(
        key="gamma",
        event_getter=_event_gamma,
        getter=_get_gamma,
        setter=_set_gamma,
        encoder=_encode_float,
        equals=_equals_float,
        block_getter=_block_gamma,
    ),
    PropertyConfig(
        key="contrast_limits",
        event_getter=_event_contrast_limits,
        getter=_get_contrast_limits,
        setter=_set_contrast_limits,
        encoder=_encode_limits,
        equals=_equals_limits,
        block_getter=_block_contrast_limits,
    ),
)


PROPERTY_BY_KEY: dict[str, PropertyConfig] = {cfg.key: cfg for cfg in PROPERTY_CONFIGS}


@dataclass
class LayerBinding:
    layer_id: str
    layer: RemoteImageLayer
    handlers: dict[str, tuple[EventEmitter, Callable[[Any], None]]] = field(default_factory=dict)
    suspended: set[str] = field(default_factory=set)
    pending: dict[str, Any] = field(default_factory=dict)
    timer: QtCore.QTimer | None = None


class NapariLayerIntentEmitter:
    """Translate layer UI changes into coordinator intents (dims-style)."""

    def __init__(
        self,
        *,
        ledger: ClientStateLedger,
        state: ControlStateContext,
        loop_state: ClientLoopState,
        dispatch_state_update: Callable[[Any, str], bool],
        ui_call: Optional[Any],
        log_layers_info: bool,
        tx_interval_ms: int,
    ) -> None:
        assert ui_call is not None, "NapariLayerIntentEmitter requires a GUI call proxy"
        app = QtCore.QCoreApplication.instance()
        assert app is not None, "Qt application instance must exist"
        self._ui_thread = app.thread()
        self._ledger = ledger
        self._state = state
        self._loop_state = loop_state
        self._dispatch_state_update = dispatch_state_update
        self._log_layers_info = bool(log_layers_info)
        self._tx_interval_ms = max(0, int(tx_interval_ms))
        self._bindings: dict[str, LayerBinding] = {}
        self._suppress_depth = 0

    # ------------------------------------------------------------------ configuration
    def set_logging(self, enabled: bool) -> None:
        self._log_layers_info = bool(enabled)

    def set_tx_interval_ms(self, value: int) -> None:
        self._tx_interval_ms = max(0, int(value))

    # ------------------------------------------------------------------ lifecycle
    def attach_layer(self, layer: RemoteImageLayer) -> None:
        self._assert_gui_thread()
        layer_id = layer.remote_id
        assert layer_id not in self._bindings, f"layer {layer_id!r} already attached"
        binding = LayerBinding(layer_id=layer_id, layer=layer)
        for config in PROPERTY_CONFIGS:
            emitter = config.event_getter(layer)
            handler = self._make_property_handler(binding, config)
            emitter.connect(handler)
            binding.handlers[config.key] = (emitter, handler)
        if self._tx_interval_ms > 0:
            binding.timer = QtCore.QTimer()
            binding.timer.setSingleShot(True)
            binding.timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
            binding.timer.timeout.connect(lambda bid=layer_id: self._flush_pending(bid))
        self._bindings[layer_id] = binding

    def detach_layer(self, layer_id: str) -> None:
        self._assert_gui_thread()
        binding = self._bindings.pop(layer_id, None)
        if binding is None:
            return
        for emitter, callback in binding.handlers.values():
            emitter.disconnect(callback)
        if binding.timer is not None:
            if binding.timer.isActive():
                binding.timer.stop()
            binding.timer.deleteLater()

    def shutdown(self) -> None:
        self._assert_gui_thread()
        for layer_id in list(self._bindings.keys()):
            self.detach_layer(layer_id)

    # ------------------------------------------------------------------ suppression API
    def suppress_forward(self) -> None:
        self._suppress_depth += 1

    def resume_forward(self) -> None:
        assert self._suppress_depth > 0, "resume_forward without matching suppress"
        self._suppress_depth -= 1

    def suppressing(self):
        """Context manager that suppresses outgoing intents temporarily."""

        class _Suppressor:
            def __init__(self, emitter: NapariLayerIntentEmitter) -> None:
                self._emitter = emitter

            def __enter__(self) -> None:
                self._emitter.suppress_forward()

            def __exit__(self, exc_type, exc, tb) -> None:
                self._emitter.resume_forward()

        return _Suppressor(self)

    # ------------------------------------------------------------------ intent entry points
    def handle_ack(self, outcome: AckReconciliation) -> None:
        self._assert_gui_thread()
        if outcome.scope != "layer" or outcome.target is None or outcome.key is None:
            return
        binding = self._bindings.get(outcome.target)
        if binding is None:
            return
        config = PROPERTY_BY_KEY.get(outcome.key)
        if config is None:
            return

        _update_runtime_from_ack_outcome(self._state, outcome)

        if outcome.status == "accepted":
            if self._log_layers_info:
                logger.info(
                    "layer intent accepted: id=%s key=%s pending=%d",
                    binding.layer_id,
                    config.key,
                    outcome.pending_len,
                )
            return

        revert_value = outcome.confirmed_value if outcome.confirmed_value is not None else outcome.pending_value
        if revert_value is None:
            return
        self._apply_projection(binding, config, revert_value)

    def prime_from_block(self, layer_id: str, block: Mapping[str, Any]) -> None:
        self._assert_gui_thread()
        binding = self._bindings.get(layer_id)
        if binding is None:
            return
        for config in PROPERTY_CONFIGS:
            value = config.block_getter(block)
            if value is None:
                continue
            encoded = config.encoder(value)
            self._ledger.record_confirmed(
                "layer",
                layer_id,
                config.key,
                encoded,
            )
            self._apply_projection(binding, config, encoded)

    def apply_remote_values(self, layer_id: str, changes: Mapping[str, Any]) -> None:
        self._assert_gui_thread()
        binding = self._bindings.get(layer_id)
        if binding is None:
            return
        for key, value in changes.items():
            if key == "removed":
                continue
            config = PROPERTY_BY_KEY.get(key)
            if config is None:
                continue
            encoded = config.encoder(value)
            self._ledger.record_confirmed("layer", layer_id, key, encoded)
            self._apply_projection(binding, config, encoded)

    # ------------------------------------------------------------------ internals
    def _make_property_handler(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
    ) -> Callable[[Any], None]:
        def _handler(_event: Any = None) -> None:
            self._on_property_change(binding, config)

        return _handler

    def _on_property_change(self, binding: LayerBinding, config: PropertyConfig) -> None:
        self._assert_gui_thread()
        if self._suppress_depth > 0:
            return
        if config.key in binding.suspended:
            return

        current_value = config.getter(binding.layer)
        encoded = config.encoder(current_value)
        if self._tx_interval_ms > 0:
            binding.pending[config.key] = encoded
            assert binding.timer is not None, "coalescing timer missing"
            binding.timer.start(max(1, self._tx_interval_ms))
            return

        self._emit_layer_update(binding, config, encoded)

    def _flush_pending(self, layer_id: str) -> None:
        self._assert_gui_thread()
        binding = self._bindings.get(layer_id)
        if binding is None:
            return
        pending_items = list(binding.pending.items())
        binding.pending.clear()
        for key, encoded in pending_items:
            config = PROPERTY_BY_KEY.get(key)
            if config is None:
                continue
            self._emit_layer_update(binding, config, encoded)

    def _emit_layer_update(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        value: Any,
    ) -> None:
        self._assert_gui_thread()
        ok, projection = _emit_state_update(
            self._state,
            self._loop_state,
            self._ledger,
            self._dispatch_state_update,
            scope="layer",
            target=binding.layer_id,
            key=config.key,
            value=value,
            origin=f"layer:{config.key}",
            metadata={
                "layer_id": binding.layer_id,
                "property": config.key,
            },
        )
        if not ok:
            return
        projection_value = projection if projection is not None else value
        if self._log_layers_info:
            logger.info(
                "layer intent -> state.update id=%s key=%s value=%r",
                binding.layer_id,
                config.key,
                projection_value,
            )
        self._apply_projection(binding, config, projection_value)

    def _apply_projection(
        self,
        binding: LayerBinding,
        config: PropertyConfig,
        value: Any,
    ) -> None:
        self._assert_gui_thread()
        current_value = config.getter(binding.layer)
        if config.equals(current_value, value):
            return
        binding.suspended.add(config.key)
        emitter, callback = binding.handlers[config.key]
        with emitter.blocker(callback):
            config.setter(binding.layer, value)
        binding.suspended.discard(config.key)

    def _assert_gui_thread(self) -> None:
        current = QtCore.QThread.currentThread()
        assert (
            current is self._ui_thread
        ), "NapariLayerIntentEmitter methods must run on the Qt GUI thread"


__all__ = ["NapariLayerIntentEmitter"]
