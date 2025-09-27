from __future__ import annotations

import pytest

from napari_cuda.client.layers.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
from napari_cuda.client.streaming.client_loop.intents import IntentState
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.layer_intent_bridge import LayerIntentBridge
from napari_cuda.client.streaming.presenter_facade import PresenterFacade
from napari_cuda.protocol.messages import LayerRenderHints, LayerSpec, LayerUpdateMessage


class DummyLoop:
    def __init__(self) -> None:
        self.posted: list[dict] = []

    def post(self, payload: dict) -> bool:
        self.posted.append(payload)
        return True


class DummyRegistry:
    def __init__(self) -> None:
        self._listeners: list = []

    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def emit(self, snapshot: RegistrySnapshot) -> None:
        for callback in list(self._listeners):
            callback(snapshot)


@pytest.fixture
def intent_state() -> IntentState:
    state = IntentState()
    state.client_id = "client-test"
    state.settings_min_dt = 0.0
    state.last_settings_send = 0.0
    return state


def _make_layer(remote_id: str = "layer-1") -> RemoteImageLayer:
    spec = LayerSpec(
        layer_id=remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        contrast_limits=[0.0, 1.0],
        metadata={},
        render=LayerRenderHints(mode="mip", opacity=0.5, visibility=True, gamma=1.0),
        extras={"data_id": "demo"},
    )
    return RemoteImageLayer(spec)


def test_opacity_intent_roundtrip(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    # Local UI change emits intent and snaps back to previous value until ack
    layer.opacity = 0.25

    assert loop.posted, "intent payload not dispatched"
    payload = loop.posted[-1]
    assert payload["type"] == "layer.intent.set_opacity"
    assert payload["layer_id"] == layer.remote_id
    assert payload["opacity"] == pytest.approx(0.25)

    # Local layer reflects optimistic value until ack arrives
    assert layer.opacity == pytest.approx(0.25)

    seq = payload["client_seq"]
    assert seq in loop_state.pending_intents

    # Simulate server ack via layer.update
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        contrast_limits=None,
        metadata={"intent_seq": seq},
        render=LayerRenderHints(opacity=0.25, visibility=True),
    )
    message = LayerUpdateMessage(layer=ack_spec, partial=True, ack=True, intent_seq=seq)
    bridge.handle_layer_update(message)

    assert layer.opacity == pytest.approx(0.25)
    assert not loop_state.pending_intents


def test_contrast_intent(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-contrast")
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    layer.contrast_limits = (0.1, 0.9)
    payload = loop.posted[-1]
    assert payload["type"] == "layer.intent.set_contrast_limits"
    assert payload["lo"] == pytest.approx(0.1)
    assert payload["hi"] == pytest.approx(0.9)

    # Local layer keeps optimistic limits until ack arrives
    assert tuple(layer.contrast_limits) == pytest.approx((0.1, 0.9))

    seq = payload["client_seq"]
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        contrast_limits=[0.1, 0.9],
        metadata={"intent_seq": seq},
        render=LayerRenderHints(opacity=0.5, visibility=True),
    )
    message = LayerUpdateMessage(layer=ack_spec, partial=True, ack=True, intent_seq=seq)
    bridge.handle_layer_update(message)

    assert tuple(layer.contrast_limits) == pytest.approx((0.1, 0.9))
    assert not loop_state.pending_intents


def test_colormap_intent(intent_state: IntentState) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerIntentBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer("layer-color")
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))

    layer.colormap = "magma"
    payload = loop.posted[-1]
    assert payload["type"] == "layer.intent.set_colormap"
    assert payload["name"] == "magma"

    seq = payload["client_seq"]
    ack_spec = LayerSpec(
        layer_id=layer.remote_id,
        layer_type="image",
        name="demo",
        ndim=2,
        shape=[1, 1],
        dtype="float32",
        contrast_limits=[0.0, 1.0],
        metadata={"intent_seq": seq},
        render=LayerRenderHints(colormap="magma", opacity=0.5, visibility=True),
    )
    bridge.handle_layer_update(LayerUpdateMessage(layer=ack_spec, partial=True, ack=True, intent_seq=seq))

    # Value committed after ack
    assert getattr(layer.colormap, "name", str(layer.colormap)) == "magma"
    assert not loop_state.pending_intents
