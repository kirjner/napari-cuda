from __future__ import annotations

from types import SimpleNamespace
from typing import List, Tuple

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import AckReconciliation, ClientStateLedger, IntentRecord
from napari_cuda.client.control.emitters import NapariCameraIntentEmitter
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState


def _make_emitter() -> tuple[NapariCameraIntentEmitter, ControlStateContext, ClientStateLedger, List[Tuple[IntentRecord, str]]]:
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    ledger = ClientStateLedger()
    dispatched: List[Tuple[IntentRecord, str]] = []

    def dispatch(pending_update: IntentRecord, origin: str) -> bool:
        dispatched.append((pending_update, origin))
        return True

    emitter = NapariCameraIntentEmitter(
        ledger=ledger,
        state=control_state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_camera_info=False,
    )
    return emitter, control_state, ledger, dispatched


@pytest.mark.usefixtures("qtbot")
def test_zoom_emits_intent() -> None:
    emitter, control_state, _ledger, dispatched = _make_emitter()

    ok = emitter.zoom(factor=1.5, anchor_px=(10.0, 20.0), origin='test')

    assert ok is True
    assert dispatched, "expected state.update dispatch"
    pending, origin = dispatched[-1]
    assert origin == 'test'
    assert pending.scope == 'camera'
    assert pending.key == 'zoom'
    assert pending.value == {'factor': 1.5, 'anchor_px': [10.0, 20.0]}
    assert control_state.camera_state['zoom'] == {'factor': 1.5, 'anchor_px': [10.0, 20.0]}


@pytest.mark.usefixtures("qtbot")
def test_handle_ack_updates_state() -> None:
    emitter, control_state, _ledger, _dispatched = _make_emitter()
    emitter.zoom(factor=1.2, anchor_px=(0.0, 0.0), origin='ui')

    outcome = AckReconciliation(
        status='accepted',
        intent_id='intent-1',
        ack_frame_id='ack-1',
        in_reply_to='state-1',
        scope='camera',
        target='main',
        key='zoom',
        pending_value={'factor': 1.2, 'anchor_px': [0.0, 0.0]},
        projection_value=None,
        confirmed_value={'factor': 2.0, 'anchor_px': [1.0, 2.0]},
        applied_value={'factor': 2.0, 'anchor_px': [1.0, 2.0]},
        error=None,
        update_phase='update',
        metadata=None,
        pending_len=0,
        was_pending=True,
    )

    emitter.handle_ack(outcome)

    assert control_state.camera_state['zoom'] == {'factor': 2.0, 'anchor_px': [1.0, 2.0]}


@pytest.mark.usefixtures("qtbot")
def test_handle_ack_reverts_on_rejection() -> None:
    emitter, control_state, _ledger, _dispatched = _make_emitter()
    emitter.pan(dx_px=5.0, dy_px=3.0, origin='drag')
    control_state.camera_state['pan'] = {'dx_px': 10.0, 'dy_px': -4.0}

    rejected = AckReconciliation(
        status='rejected',
        intent_id='intent-2',
        ack_frame_id='ack-2',
        in_reply_to='state-2',
        scope='camera',
        target='main',
        key='pan',
        pending_value={'dx_px': 5.0, 'dy_px': 3.0},
        projection_value=None,
        confirmed_value={'dx_px': 1.0, 'dy_px': -1.0},
        applied_value=None,
        error={'code': 'camera.out_of_bounds'},
        update_phase='update',
        metadata=None,
        pending_len=0,
        was_pending=True,
    )

    emitter.handle_ack(rejected)

    assert control_state.camera_state['pan'] == {'dx_px': 1.0, 'dy_px': -1.0}


@pytest.mark.usefixtures("qtbot")
def test_record_confirmed_updates_state() -> None:
    emitter, control_state, _ledger, _dispatched = _make_emitter()

    emitter.record_confirmed('orbit', {'d_az_deg': 3.5, 'd_el_deg': -1.25})

    assert control_state.camera_state['orbit'] == {'d_az_deg': 3.5, 'd_el_deg': -1.25}
