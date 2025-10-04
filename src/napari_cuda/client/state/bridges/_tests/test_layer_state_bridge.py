from __future__ import annotations

import pytest

from napari_cuda.client.data.registry import LayerRecord, RegistrySnapshot
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.state import LayerStateBridge
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.protocol import build_ack_state


class DummyLoop:
    def __init__(self) -> None:
        self.pending: list[tuple] = []

    def _dispatch_state_update(self, pending, origin: str) -> bool:
        self.pending.append((pending, origin))
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
def control_state() -> ControlStateContext:
    env = type('Env', (), {
        'dims_rate_hz': 60.0,
        'wheel_step': 1,
        'settings_rate_hz': 30.0,
        'dims_z': None,
        'dims_z_min': None,
        'dims_z_max': None,
    })
    state = ControlStateContext.from_env(env)
    state.session_id = 'session-test'
    return state


def _make_layer(remote_id: str = 'layer-1') -> RemoteImageLayer:
    block = {
        "layer_id": remote_id,
        "layer_type": "image",
        "name": "demo",
        "ndim": 2,
        "shape": [1, 1],
        "dtype": "float32",
        "axis_labels": ["y", "x"],
        "scale": None,
        "translate": None,
        "contrast_limits": [0.0, 1.0],
        "metadata": {},
        "render": {"mode": "mip"},
        "controls": {
            "visible": True,
            "opacity": 0.5,
            "rendering": "mip",
            "colormap": "gray",
            "gamma": 1.0,
            "contrast_limits": [0.0, 1.0],
        },
        "source": {"data_id": "demo"},
    }
    return RemoteImageLayer(layer_id=remote_id, block=block)


def _bind_layer(bridge: LayerStateBridge, registry: DummyRegistry, layer: RemoteImageLayer) -> None:
    record = LayerRecord(layer_id=layer.remote_id, block=dict(layer._remote_block), layer=layer)
    registry.emit(RegistrySnapshot(layers=(record,)))


def test_local_opacity_dispatches_pending_update(control_state: ControlStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(
        loop,
        presenter,
        registry,
        control_state=control_state,
        loop_state=loop_state,
        enabled=True,
    )

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    layer.opacity = 0.25

    pending, origin = loop.pending[-1]
    assert origin == 'layer:opacity'
    assert pending.scope == 'layer'
    assert pending.target == layer.remote_id
    assert pending.key == 'opacity'
    assert pending.value == pytest.approx(0.25)
    assert pending.metadata == {'layer_id': layer.remote_id, 'property': 'opacity'}
    assert layer.opacity == pytest.approx(0.25)


def test_handle_ack_accept_clears_runtime(control_state: ControlStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(loop, presenter, registry, control_state=control_state, loop_state=loop_state, enabled=True)

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    layer.opacity = 0.3
    pending, _ = loop.pending[-1]

    ack = build_ack_state(
        session_id='session-test',
        frame_id='ack-1',
        payload={
            'intent_id': pending.intent_id,
            'in_reply_to': pending.frame_id,
            'status': 'accepted',
            'applied_value': 0.3,
        },
        timestamp=10.0,
    )
    outcome = bridge._state_store.apply_ack(ack)
    bridge.handle_ack(outcome)

    binding = bridge._bindings[layer.remote_id]
    runtime = binding.properties['opacity']
    assert runtime.active is False
    assert runtime.active_frame_id is None
    assert runtime.active_intent_id is None


def test_handle_ack_rejected_reverts_property(control_state: ControlStateContext) -> None:
    presenter = PresenterFacade()
    registry = DummyRegistry()
    loop = DummyLoop()
    loop_state = ClientLoopState()

    bridge = LayerStateBridge(loop, presenter, registry, control_state=control_state, loop_state=loop_state, enabled=True)

    layer = _make_layer()
    _bind_layer(bridge, registry, layer)

    original_opacity = layer.opacity
    layer.opacity = 0.1
    pending, _ = loop.pending[-1]

    ack = build_ack_state(
        session_id='session-test',
        frame_id='ack-2',
        payload={
            'intent_id': pending.intent_id,
            'in_reply_to': pending.frame_id,
            'status': 'rejected',
            'error': {'code': 'state.rejected', 'message': 'not allowed'},
        },
        timestamp=11.0,
    )
    outcome = bridge._state_store.apply_ack(ack)
    bridge.handle_ack(outcome)

    assert layer.opacity == pytest.approx(original_opacity)
