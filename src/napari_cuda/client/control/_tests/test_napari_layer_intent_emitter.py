from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.emitters import NapariLayerIntentEmitter
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.runtime.client_loop.scheduler import CallProxy
from napari_cuda.protocol import build_ack_state


class FakeDispatch:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def __call__(self, pending: Any, origin: str) -> bool:
        self.calls.append((pending, origin))
        return True


@dataclass
class EmitterSetup:
    emitter: NapariLayerIntentEmitter
    layer: RemoteImageLayer
    dispatch: FakeDispatch
    ledger: ClientStateLedger
    state: ControlStateContext


def _make_layer(remote_id: str = "layer-1") -> RemoteImageLayer:
    block = {
        "layer_id": remote_id,
        "layer_type": "image",
        "name": "demo",
        "ndim": 2,
        "shape": [1, 1],
        "dtype": "float32",
        "axis_labels": ["y", "x"],
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
    }
    return RemoteImageLayer(layer_id=remote_id, block=block)


@pytest.fixture
def emitter_setup(qtbot) -> EmitterSetup:
    env = type(
        "Env",
        (),
        {
            "dims_rate_hz": 60.0,
            "wheel_step": 1,
            "settings_rate_hz": 30.0,
            "dims_z": None,
            "dims_z_min": None,
            "dims_z_max": None,
        },
    )
    state = ControlStateContext.from_env(env)
    state.session_id = "session-test"
    ledger = ClientStateLedger(clock=lambda: 0.0)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    dispatch = FakeDispatch()
    call_proxy = CallProxy()
    emitter = NapariLayerIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=call_proxy,
        log_layers_info=False,
        tx_interval_ms=0,
    )
    layer = _make_layer()
    emitter.attach_layer(layer)
    emitter.prime_from_block(layer.remote_id, layer._remote_block)  # type: ignore[attr-defined]
    yield EmitterSetup(emitter=emitter, layer=layer, dispatch=dispatch, ledger=ledger, state=state)
    call_proxy.deleteLater()


def test_layer_opacity_dispatches(emitter_setup: EmitterSetup) -> None:
    emitter = emitter_setup.emitter
    layer = emitter_setup.layer
    dispatch = emitter_setup.dispatch

    layer.opacity = 0.25

    pending, origin = dispatch.calls[-1]
    assert origin == "layer:opacity"
    assert pending.scope == "layer"
    assert pending.target == layer.remote_id
    assert pending.key == "opacity"
    assert pending.value == pytest.approx(0.25)
    assert pending.metadata == {"layer_id": layer.remote_id, "property": "opacity"}


def test_handle_ack_accept_clears_runtime(emitter_setup: EmitterSetup) -> None:
    emitter = emitter_setup.emitter
    layer = emitter_setup.layer
    dispatch = emitter_setup.dispatch
    ledger = emitter_setup.ledger
    state = emitter_setup.state

    layer.opacity = 0.3
    pending, _ = dispatch.calls[-1]

    ack = build_ack_state(
        session_id="session-test",
        frame_id="ack-1",
        payload={
            "intent_id": pending.intent_id,
            "in_reply_to": pending.frame_id,
            "status": "accepted",
            "applied_value": 0.3,
        },
        timestamp=10.0,
    )
    outcome = ledger.apply_ack(ack)
    emitter.handle_ack(outcome)

    runtime_key = "layer:layer-1:opacity"
    runtime = state.control_runtimes[runtime_key]
    assert runtime.active is False
    assert runtime.active_frame_id is None
    assert runtime.active_intent_id is None


def test_handle_ack_rejected_reverts_property(emitter_setup: EmitterSetup) -> None:
    emitter = emitter_setup.emitter
    layer = emitter_setup.layer
    dispatch = emitter_setup.dispatch
    ledger = emitter_setup.ledger

    original_opacity = layer.opacity
    layer.opacity = 0.1
    pending, _ = dispatch.calls[-1]

    ack = build_ack_state(
        session_id="session-test",
        frame_id="ack-2",
        payload={
            "intent_id": pending.intent_id,
            "in_reply_to": pending.frame_id,
            "status": "rejected",
            "error": {"code": "state.rejected", "message": "not allowed"},
        },
        timestamp=11.0,
    )
    outcome = ledger.apply_ack(ack)
    emitter.handle_ack(outcome)

    assert layer.opacity == pytest.approx(original_opacity)

