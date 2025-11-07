"""Emit layer intents derived from local ``RemoteImageLayer`` interactions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from qtpy import QtCore
import contextlib

from napari.utils.events import EventEmitter
from napari_cuda.client.control.client_state_ledger import (
    AckReconciliation,
    ClientStateLedger,
)
from napari_cuda.client.control.control_state import ControlStateContext, _emit_state_update, _update_runtime_from_ack_outcome
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


def _event_blending(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.blending


def _event_rendering(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.rendering


class _CompositeEmitter:
    """Minimal emitter proxy that multiplexes two EventEmitters.

    Provides ``connect``, ``disconnect`` and ``blocker`` used by the
    NapariLayerIntentEmitter. The composite is used to subscribe to both
    ``interpolation2d`` and ``interpolation3d`` events while retaining the
    PropertyConfig wiring pattern.
    """

    def __init__(self, a: EventEmitter, b: EventEmitter) -> None:
        self._a = a
        self._b = b

    def connect(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._a.connect(callback)
        self._b.connect(callback)

    def disconnect(self, callback) -> None:  # type: ignore[no-untyped-def]
        with contextlib.suppress(Exception):
            self._a.disconnect(callback)
        with contextlib.suppress(Exception):
            self._b.disconnect(callback)

    def blocker(self, callback=None):  # type: ignore[no-untyped-def]
        a_blk = self._a.blocker(callback)
        b_blk = self._b.blocker(callback)

        class _Ctx:
            def __enter__(self):  # type: ignore[no-untyped-def]
                self._a_tok = a_blk.__enter__()
                self._b_tok = b_blk.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
                ok_b = b_blk.__exit__(exc_type, exc, tb)
                ok_a = a_blk.__exit__(exc_type, exc, tb)
                return bool(ok_a and ok_b)

        return _Ctx()


def _event_interpolation(layer: RemoteImageLayer) -> EventEmitter:
    # Subscribe to both non-deprecated events and avoid the consolidated
    # `interpolation` WarningEmitter to prevent deprecation noise.
    return _CompositeEmitter(layer.events.interpolation2d, layer.events.interpolation3d)  # type: ignore[return-value]


def _event_depiction(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.depiction


def _event_iso_threshold(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.iso_threshold


def _event_attenuation(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.attenuation


def _event_colormap(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.colormap


def _event_gamma(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.gamma


def _event_contrast_limits(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.contrast_limits

def _event_projection_mode(layer: RemoteImageLayer) -> EventEmitter:
    return layer.events.projection_mode

def _event_plane_thickness(layer: RemoteImageLayer) -> EventEmitter:
    # Plane thickness changes are emitted on layer.plane.events.thickness
    return layer.plane.events.thickness  # type: ignore[return-value]


def _get_visible(layer: RemoteImageLayer) -> bool:
    return bool(layer.visible)


def _set_visible(layer: RemoteImageLayer, value: bool) -> None:
    layer.visible = bool(value)


def _get_opacity(layer: RemoteImageLayer) -> float:
    return float(layer.opacity)


def _set_opacity(layer: RemoteImageLayer, value: float) -> None:
    layer.opacity = float(value)


def _get_blending(layer: RemoteImageLayer) -> str:
    return str(layer.blending)


def _set_blending(layer: RemoteImageLayer, value: str) -> None:
    layer.blending = str(value)


def _get_rendering(layer: RemoteImageLayer) -> str:
    return str(layer.rendering)


def _set_rendering(layer: RemoteImageLayer, value: str) -> None:
    layer.rendering = str(value)


def _get_interpolation(layer: RemoteImageLayer) -> str:
    # RemoteImageLayer mirrors napari Image and maintains 2D/3D values.
    # In our stack we keep them equal; read the 2D value.
    return str(getattr(layer, "interpolation2d"))


def _set_interpolation(layer: RemoteImageLayer, value: str) -> None:
    token = str(value)
    # Set both 2D and 3D to keep them in sync for intents/state.
    # We subscribe to the consolidated `interpolation` event, so
    # these per-dimension event emissions won't loop back.
    layer.interpolation2d = token
    layer.interpolation3d = token


def _get_depiction(layer: RemoteImageLayer) -> str:
    return str(layer.depiction)


def _set_depiction(layer: RemoteImageLayer, value: str) -> None:
    layer.depiction = str(value)


def _get_iso_threshold(layer: RemoteImageLayer) -> float:
    return float(layer.iso_threshold)


def _set_iso_threshold(layer: RemoteImageLayer, value: float) -> None:
    layer.iso_threshold = float(value)


def _get_attenuation(layer: RemoteImageLayer) -> float:
    return float(layer.attenuation)


def _set_attenuation(layer: RemoteImageLayer, value: float) -> None:
    layer.attenuation = float(value)


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

def _get_projection_mode(layer: RemoteImageLayer) -> str:
    return str(layer.projection_mode)

def _set_projection_mode(layer: RemoteImageLayer, value: str) -> None:
    layer.projection_mode = str(value)

def _get_plane_thickness(layer: RemoteImageLayer) -> float:
    return float(layer.plane.thickness)

def _set_plane_thickness(layer: RemoteImageLayer, value: float) -> None:
    layer.plane.thickness = float(value)


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


def _block_interpolation(block: Mapping[str, Any]) -> Any | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "interpolation" in controls:
        return _encode_str(controls["interpolation"])  # type: ignore[return-value]
    if "interpolation" in block:
        return _encode_str(block["interpolation"])  # type: ignore[return-value]
    return None


def _block_depiction(block: Mapping[str, Any]) -> Any | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "depiction" in controls:
        return _encode_str(controls["depiction"])  # type: ignore[return-value]
    if "depiction" in block:
        return _encode_str(block["depiction"])  # type: ignore[return-value]
    return None


def _block_iso_threshold(block: Mapping[str, Any]) -> Any | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "iso_threshold" in controls:
        return _encode_float(controls["iso_threshold"])  # type: ignore[return-value]
    if "iso_threshold" in block:
        return _encode_float(block["iso_threshold"])  # type: ignore[return-value]
    return None


def _block_attenuation(block: Mapping[str, Any]) -> Any | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "attenuation" in controls:
        return _encode_float(controls["attenuation"])  # type: ignore[return-value]
    if "attenuation" in block:
        return _encode_float(block["attenuation"])  # type: ignore[return-value]
    return None


def _block_opacity(block: Mapping[str, Any]) -> float | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "opacity" in controls:
        return _encode_float(controls["opacity"])
    return None


def _block_blending(block: Mapping[str, Any]) -> str | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "blending" in controls:
        return _encode_str(controls["blending"])
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

def _block_projection_mode(block: Mapping[str, Any]) -> str | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "projection_mode" in controls:
        return _encode_str(controls["projection_mode"])  # type: ignore[return-value]
    if "projection_mode" in block:
        return _encode_str(block["projection_mode"])  # type: ignore[return-value]
    return None

def _block_plane_thickness(block: Mapping[str, Any]) -> float | None:
    controls = block.get("controls")
    if isinstance(controls, Mapping) and "plane_thickness" in controls:
        return _encode_float(controls["plane_thickness"])  # type: ignore[return-value]
    if "plane_thickness" in block:
        return _encode_float(block["plane_thickness"])  # type: ignore[return-value]
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
        key="blending",
        event_getter=_event_blending,
        getter=_get_blending,
        setter=_set_blending,
        encoder=_encode_str,
        equals=_equals_str,
        block_getter=_block_blending,
    ),
    PropertyConfig(
        key="interpolation",
        event_getter=_event_interpolation,
        getter=_get_interpolation,
        setter=_set_interpolation,
        encoder=_encode_str,
        equals=_equals_str,
        block_getter=_block_interpolation,
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
        key="depiction",
        event_getter=_event_depiction,
        getter=_get_depiction,
        setter=_set_depiction,
        encoder=_encode_str,
        equals=_equals_str,
        block_getter=_block_depiction,
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
    PropertyConfig(
        key="iso_threshold",
        event_getter=_event_iso_threshold,
        getter=_get_iso_threshold,
        setter=_set_iso_threshold,
        encoder=_encode_float,
        equals=_equals_float,
        block_getter=_block_iso_threshold,
    ),
    PropertyConfig(
        key="attenuation",
        event_getter=_event_attenuation,
        getter=_get_attenuation,
        setter=_set_attenuation,
        encoder=_encode_float,
        equals=_equals_float,
        block_getter=_block_attenuation,
    ),
    PropertyConfig(
        key="projection_mode",
        event_getter=_event_projection_mode,
        getter=_get_projection_mode,
        setter=_set_projection_mode,
        encoder=_encode_str,
        equals=_equals_str,
        block_getter=_block_projection_mode,
    ),
    PropertyConfig(
        key="plane_thickness",
        event_getter=_event_plane_thickness,
        getter=_get_plane_thickness,
        setter=_set_plane_thickness,
        encoder=_encode_float,
        equals=_equals_float,
        block_getter=_block_plane_thickness,
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
            self._on_property_change(binding, config, event=_event)

        return _handler

    def _on_property_change(self, binding: LayerBinding, config: PropertyConfig, *, event: Any | None = None) -> None:
        self._assert_gui_thread()
        if self._suppress_depth > 0:
            return
        if config.key in binding.suspended:
            return

        # Special handling for interpolation: use the event payload to avoid
        # sampling only the 2D value when the 3D signal fires.
        if config.key == "interpolation" and event is not None and hasattr(event, "value"):
            raw = getattr(event, "value")
            # `raw` is an Interpolation enum in napari; prefer its `.value`.
            raw_str = str(getattr(raw, "value", raw))
            encoded = config.encoder(raw_str)
        else:
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
