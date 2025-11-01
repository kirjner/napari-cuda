from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from napari_cuda.client.control.client_state_ledger import (
    ClientStateLedger,
    IntentRecord,
)
from napari_cuda.client.control.emitters import NapariDimsIntentEmitter
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState


class FakeDispatch:
    def __init__(self) -> None:
        self.calls: list[tuple[IntentRecord, str]] = []

    def __call__(self, pending: IntentRecord, origin: str) -> bool:
        self.calls.append((pending, origin))
        return True


class DummySignal:
    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self._callbacks.remove(callback)

    def emit(self, event=None) -> None:
        for cb in list(self._callbacks):
            cb(event)


class DummyDims:
    def __init__(self, ndim: int = 3) -> None:
        self.ndim = ndim
        self.current_step = tuple(0 for _ in range(ndim))
        self.ndisplay = 2
        self.events = SimpleNamespace(
            current_step=DummySignal(),
            ndisplay=DummySignal(),
        )


class DummyViewer:
    def __init__(self, ndim: int = 3) -> None:
        self.dims = DummyDims(ndim)
        self.window = SimpleNamespace(_qt_viewer=SimpleNamespace(dims=SimpleNamespace(is_playing=False)))


@pytest.fixture
def emitter_setup() -> tuple[NapariDimsIntentEmitter, ControlStateContext, ClientLoopState, ClientStateLedger, FakeDispatch]:
    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
        dims_z=None,
        dims_z_min=None,
        dims_z_max=None,
    )
    state = ControlStateContext.from_env(env)
    state.dims_ready = True
    state.dims_meta.update(
        {
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'ndim': 3,
        }
    )
    loop_state = ClientLoopState()
    loop_state.control_state = state
    ledger = ClientStateLedger(clock=lambda: 0.0)
    dispatch = FakeDispatch()
    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    return emitter, state, loop_state, ledger, dispatch


def test_dims_step_dispatches_delta(emitter_setup) -> None:
    emitter, _state, _loop_state, _ledger, dispatch = emitter_setup

    ok = emitter.dims_step('primary', 1, origin='ui')

    assert ok is True
    pending, origin = dispatch.calls[-1]
    assert origin == 'ui'
    assert pending.scope == 'dims'
    assert pending.key == 'step'
    assert pending.value == 1
    assert pending.metadata['axis_index'] == 0


def test_dims_set_index_dispatches_absolute(emitter_setup) -> None:
    emitter, _state, _loop_state, _ledger, dispatch = emitter_setup

    ok = emitter.dims_set_index(2, 5, origin='ui')

    assert ok is True
    pending, origin = dispatch.calls[-1]
    assert pending.scope == 'dims'
    assert pending.key == 'index'
    assert pending.value == 5
    assert pending.metadata['axis_index'] == 2


def test_handle_wheel_forwards_primary_axis(emitter_setup) -> None:
    emitter, state, _loop_state, _ledger, dispatch = emitter_setup
    state.wheel_step = 2

    sent = emitter.handle_wheel({'angle_y': 120})

    assert sent is True
    pending, origin = dispatch.calls[-1]
    assert pending.key == 'step'
    assert pending.value == 2
    assert origin == 'wheel'


def test_view_set_ndisplay_dispatches(emitter_setup) -> None:
    emitter, state, _loop_state, _ledger, dispatch = emitter_setup
    state.dims_meta['ndisplay'] = 2

    ok = emitter.view_set_ndisplay(3, origin='ui')

    assert ok is True
    pending, origin = dispatch.calls[-1]
    assert pending.scope == 'view'
    assert pending.key == 'ndisplay'
    assert pending.value == 3
    assert origin == 'ui'


def test_toggle_ndisplay_flips_target(emitter_setup) -> None:
    emitter, state, _loop_state, _ledger, dispatch = emitter_setup
    state.dims_meta['ndisplay'] = 2

    ok = emitter.toggle_ndisplay(origin='ui')

    assert ok is True
    pending, _ = dispatch.calls[-1]
    assert pending.value == 3


def test_attach_viewer_wires_events(emitter_setup) -> None:
    emitter, _state, _loop_state, _ledger, dispatch = emitter_setup
    viewer = DummyViewer(ndim=3)
    emitter.attach_viewer(viewer)

    viewer.dims.current_step = (0, 1, 0)
    viewer.dims.events.current_step.emit()

    pending, origin = dispatch.calls[-1]
    assert pending.scope == 'dims'
    assert pending.key == 'index'
    assert origin == 'ui'

    viewer.dims.events.ndisplay.emit(SimpleNamespace(value=3))
    pending, origin = dispatch.calls[-1]
    assert pending.scope == 'view'
    assert pending.key == 'ndisplay'
    assert pending.value == 3
    assert origin == 'ui'
