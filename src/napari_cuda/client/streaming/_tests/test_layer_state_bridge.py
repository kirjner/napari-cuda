from __future__ import annotations

import time

import pytest

from napari_cuda.client.layers.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
from napari_cuda.client.control.state_update_actions import ClientStateContext
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.viewer_layer_adapter import LayerStateBridge
from napari_cuda.client.streaming.presenter_facade import PresenterFacade
from napari_cuda.protocol.messages import LayerRenderHints, LayerSpec, StateUpdateMessage


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

    def remove_listener(self, callback) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def emit(self, snapshot: RegistrySnapshot) -> None:
        for callback in list(self._listeners):
            callback(snapshot)


@pytest.fixture
def intent_state() -> ClientStateContext:
    state = ClientStateContext()
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
        contrast_limits=None,
        metadata={},
        render=LayerRenderHints(mode="mip"),
        controls={
            "visible": True,
            "opacity": 0.5,
            "rendering": "mip",
            "colormap": "gray",
            "gamma": 1.0,
            "contrast_limits": [0.0, 1.0],
        },
        extras={"data_id": "demo"},
    )
    return RemoteImageLayer(spec)


def _bind_layer(bridge: LayerStateBridge, registry: DummyRegistry, layer: RemoteImageLayer) -> None:
    record = LayerRecord(layer_id=layer.remote_id, spec=layer._remote_spec, layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))


def test_local_opacity_dispatches_state_update(intent_state: ClientStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    layer.opacity = 0.25

    assert loop.posted, "state.update not dispatched"
    payload = loop.posted[-1]
    assert payload["type"] == "state.update"
    assert payload["scope"] == "layer"
    assert payload["target"] == layer.remote_id
    assert payload["key"] == "opacity"
    assert payload["value"] == pytest.approx(0.25)
    assert payload["phase"] == "start"
    assert layer.opacity == pytest.approx(0.25)


def test_remote_ack_clears_pending(intent_state: ClientStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    layer.opacity = 0.25
    payload = loop.posted[-1]

    ack = StateUpdateMessage(
        scope="layer",
        target=layer.remote_id,
        key="opacity",
        value=payload["value"],
        client_id=payload["client_id"],
        client_seq=payload["client_seq"],
        interaction_id=payload.get("interaction_id"),
        phase="update",
        timestamp=payload.get("timestamp"),
        server_seq=42,
    )
    bridge.handle_state_update(ack)

    assert layer.opacity == pytest.approx(0.25)


def test_foreign_update_overrides_projection(intent_state: ClientStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(
        loop,
        presenter,
        registry,
        intent_state=intent_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    message = StateUpdateMessage(
        scope="layer",
        target=layer.remote_id,
        key="gamma",
        value=1.75,
        client_id="foreign",
        client_seq=5,
        interaction_id="abc",
        phase="update",
        timestamp=time.time(),
        server_seq=9,
    )
    bridge.handle_state_update(message)

    assert layer.gamma == pytest.approx(1.75)
