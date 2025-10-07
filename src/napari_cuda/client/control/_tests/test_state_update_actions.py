from __future__ import annotations

from collections import deque
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

from napari_cuda.client.control import state_update_actions as control_actions
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.client_state_ledger import ClientStateLedger, IntentRecord
from napari_cuda.client.control.mirrors.napari_dims_mirror import NapariDimsMirror
from napari_cuda.client.control.emitters import NapariDimsIntentEmitter
from napari_cuda.protocol import build_ack_state, build_notify_dims
from napari_cuda.protocol.messages import NotifyDimsFrame

@pytest.fixture(autouse=True)
def _ensure_qapp(qtbot):  # noqa: D401
    """Guarantee a Qt application instance for mirror tests."""
    yield

class FakeDispatch:
    def __init__(self) -> None:
        self.calls: list[tuple[IntentRecord, str]] = []

    def __call__(self, pending: IntentRecord, origin: str) -> bool:
        self.calls.append((pending, origin))
        return True


class PresenterStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.multiscale_calls: list[dict[str, Any]] = []

    def apply_dims_update(self, payload: dict[str, Any]) -> None:
        self.calls.append(dict(payload))

    def apply_multiscale_policy(self, payload: dict[str, Any]) -> None:
        self.multiscale_calls.append(dict(payload))


class ViewerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _apply_remote_dims_update(self, **kwargs: Any) -> None:
        self.calls.append(dict(kwargs))


def _make_notify_dims_frame(
    *,
    current_step: Sequence[int],
    ndisplay: int,
    level_shapes: Sequence[Sequence[int]],
    mode: str = 'plane',
    session_id: str = 'session-test',
    frame_id: str = 'dims-test',
    timestamp: float = 1.5,
    levels: Sequence[dict[str, Any]] | None = None,
    current_level: int = 0,
    downgraded: bool | None = None,
    axis_labels: Sequence[str] | None = None,
    order: Sequence[int] | None = None,
    displayed: Sequence[int] | None = None,
) -> NotifyDimsFrame:
    step_values = [int(value) for value in current_step]

    if levels is None:
        level_entries = [
            {
                'index': idx,
                'shape': list(shape),
            }
            for idx, shape in enumerate(level_shapes)
        ]
    else:
        level_entries = [dict(entry) for entry in levels]

    payload: dict[str, Any] = {
        'step': step_values,
        'levels': level_entries,
        'level_shapes': [list(shape) for shape in level_shapes],
        'current_level': int(current_level),
        'mode': mode,
        'ndisplay': int(ndisplay),
    }

    if downgraded is not None:
        payload['downgraded'] = bool(downgraded)
    if axis_labels is not None:
        payload['axis_labels'] = [str(label) for label in axis_labels]
    if order is not None:
        payload['order'] = [int(index) for index in order]
    if displayed is not None:
        payload['displayed'] = [int(index) for index in displayed]

    return build_notify_dims(
        session_id=session_id,
        payload=payload,
        timestamp=timestamp,
        frame_id=frame_id,
    )





def _make_state() -> tuple[
    control_actions.ControlStateContext,
    ClientLoopState,
    ClientStateLedger,
    FakeDispatch,
]:
    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
        dims_z=None,
        dims_z_min=None,
        dims_z_max=None,
    )
    state = control_actions.ControlStateContext.from_env(env)
    loop_state = ClientLoopState()
    loop_state.control_state = state
    clock_steps = deque(float(x) for x in range(100, 200))
    ledger = ClientStateLedger(clock=lambda: clock_steps.popleft())
    dispatch = FakeDispatch()
    return state, loop_state, ledger, dispatch


def _ack_from_pending(pending: IntentRecord, *, status: str, applied_value: Any | None = None, error: dict[str, Any] | None = None) -> Any:
    payload: dict[str, Any] = {
        'intent_id': pending.intent_id,
        'in_reply_to': pending.frame_id,
        'status': status,
    }
    if status == 'accepted' and applied_value is not None:
        payload['applied_value'] = applied_value
    if status == 'rejected':
        payload['error'] = error or {'code': 'state.rejected', 'message': 'rejected'}
    return build_ack_state(
        session_id='session-test',
        frame_id='ack-1',
        payload=payload,
        timestamp=5.0,
    )


def test_handle_dims_update_seeds_state_ledger() -> None:
    state, loop_state, ledger, dispatch = _make_state()

    presenter = PresenterStub()
    ready_calls: list[None] = []

    state.dims_meta.update(
        {
            'ndim': 3,
            'order': [0, 1, 2],
            'axis_labels': ['z', 'y', 'x'],
            'range': [(0, 10), (0, 5), (0, 3)],
                'displayed': [1, 2],
        }
    )

    frame = _make_notify_dims_frame(
        current_step=[1, 2, 3],
        ndisplay=2,
        level_shapes=[[11, 6, 4]],
        axis_labels=['z', 'y', 'x'],
        order=[0, 1, 2],
        displayed=[1, 2],
    )

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: None,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
        notify_first_ready=lambda: ready_calls.append(None),
    )
    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    mirror.ingest_dims_notify(frame)

    assert state.dims_ready is True
    assert loop_state.last_dims_payload == {
        'current_step': [1, 2, 3],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [[0, 10], [0, 5], [0, 3]],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'displayed': [1, 2],
        'mode': 'plane',
        'volume': False,
        'source': None,
    }
    assert presenter.calls
    debug = ledger.dump_debug()
    assert debug['view:main:ndisplay']['confirmed']['value'] == 2
    assert ready_calls, "first dims update should trigger readiness"


def test_handle_dims_update_modes_volume_flag() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    presenter = PresenterStub()

    frame = _make_notify_dims_frame(
        current_step=[0, 0, 0],
        ndisplay=3,
        level_shapes=[[32, 32, 32]],
        mode='volume',
    )

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: None,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    mirror.ingest_dims_notify(frame)

    assert state.dims_meta['mode'] == 'volume'
    assert state.dims_meta['volume'] is True
    assert loop_state.last_dims_payload['volume'] is True
    assert presenter.calls[-1]['mode'] == 'volume'


def test_handle_dims_update_volume_adjusts_viewer_axes() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    viewer = ViewerStub()
    presenter = PresenterStub()

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    state.dims_meta.update(
        {
            'ndim': 3,
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'range': [[0, 5], [0, 5], [0, 5]],
        }
    )

    frame = _make_notify_dims_frame(
        current_step=[1, 2, 3],
        ndisplay=3,
        level_shapes=[[32, 32, 32]],
        mode='volume',
    )

    mirror.ingest_dims_notify(frame)

    assert viewer.calls, "viewer should receive dims update"
    payload = viewer.calls[-1]
    assert payload['ndisplay'] == 3
    assert tuple(payload['current_step']) == (1, 2, 3)
    assert payload['axis_labels'] == ['z', 'y', 'x']

    assert presenter.calls[-1]['ndisplay'] == 3
    assert loop_state.last_dims_payload['ndisplay'] == 3
    assert loop_state.last_dims_payload['mode'] == 'volume'


def test_scene_level_then_dims_updates_slider_bounds() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    viewer = ViewerStub()
    presenter = PresenterStub()

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)


    state.dims_meta.update(
        {
            'ndim': 3,
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'range': [[0, 511], [0, 255], [0, 63]],
        }
    )

    frame = _make_notify_dims_frame(
        current_step=[0, 0, 0],
        ndisplay=2,
        level_shapes=[[512, 256, 64], [256, 128, 32], [128, 64, 16]],
        current_level=1,
        downgraded=True,
    )

    mirror.ingest_dims_notify(frame)

    dims_range = state.dims_meta['range']
    assert dims_range and dims_range[2][1] == 31

    assert viewer.calls, 'viewer should receive updated dims metadata'
    payload = viewer.calls[-1]
    assert payload['dims_range'][2][1] == 31



def test_hud_snapshot_carries_volume_state() -> None:
    state, _, ledger, _dispatch = _make_state()
    state.dims_meta['ndisplay'] = 3
    state.dims_meta['mode'] = 'volume'
    state.dims_meta['volume'] = True

    snap = control_actions.hud_snapshot(
        state,
        video_size=(None, None),
        zoom_state={},
    )

    assert snap['volume'] is True
    assert snap['vol_mode'] is True



def test_dims_notify_preserves_optional_metadata() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_meta.update(
        {
            'ndim': 3,
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'displayed': [1, 2],
            'range': [[0, 255], [0, 127], [0, 63]],
        }
    )

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: None,
        ui_call=None,
        presenter=None,
        log_dims_info=False,
    )
    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    frame = _make_notify_dims_frame(
        current_step=[10, 20, 30],
        ndisplay=2,
        level_shapes=[[256, 128, 64]],
    )

    mirror.ingest_dims_notify(frame)

    assert state.dims_meta['axis_labels'] == ['z', 'y', 'x']
    assert state.dims_meta['order'] == [0, 1, 2]
    assert state.dims_meta['displayed'] == [1, 2]


def test_dims_notify_populates_multiscale_state() -> None:
    state, loop_state, ledger, _dispatch = _make_state()
    viewer = ViewerStub()
    presenter = PresenterStub()

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )
    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=_dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    frame = _make_notify_dims_frame(
        current_step=[0, 0, 0],
        ndisplay=2,
        level_shapes=[[512, 256, 64], [256, 128, 32], [128, 64, 16]],
        current_level=1,
        levels=[
            {'index': 0, 'path': 'level_00', 'shape': [512, 256, 64]},
            {'index': 1, 'path': 'level_01', 'shape': [256, 128, 32]},
            {'index': 2, 'path': 'level_02', 'shape': [128, 64, 16]},
        ],
        downgraded=True,
    )

    mirror.ingest_dims_notify(frame)

    ms_state = state.multiscale_state
    assert ms_state['level'] == 1
    assert ms_state['current_level'] == 1
    assert ms_state['downgraded'] is True
    assert len(ms_state['levels']) == 3
    meta = state.dims_meta.get('multiscale')
    assert isinstance(meta, dict)
    assert meta['current_level'] == 1
    assert meta['downgraded'] is True
    assert len(meta['levels']) == 3
    assert presenter.multiscale_calls, 'presenter should receive multiscale snapshot'
    last_payload = presenter.multiscale_calls[-1]
    assert last_payload['current_level'] == 1
    assert len(last_payload['levels']) == 3


def test_dims_ack_accepted_updates_viewer_with_applied_value() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_ready = True
    state.dims_meta.update(
        {
            'current_step': [148, 0, 0],
            'axis_labels': ['z', 'y', 'x'],
            'ndim': 3,
        }
    )
    viewer = ViewerStub()
    presenter = PresenterStub()

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)


    pending = ledger.apply_local(
        'dims',
        '0',
        'index',
        148,
        'start',
        intent_id='intent-1',
        frame_id='state-1',
        metadata={'axis_index': 0, 'axis_target': '0', 'update_kind': 'index'},
    )
    assert pending is not None
    assert pending.metadata is not None

    ack = build_ack_state(
        session_id='sess-1',
        frame_id='ack-1',
        payload={
            'intent_id': pending.intent_id,
            'in_reply_to': pending.frame_id,
            'status': 'accepted',
            'applied_value': 271,
        },
        timestamp=6.0,
    )

    outcome = ledger.apply_ack(ack)
    control_actions.handle_dims_ack(
        state,
        loop_state,
        outcome,
        presenter=presenter,
        viewer_ref=lambda: viewer,
        ui_call=None,
        log_dims_info=True,
    )

    assert state.dims_meta['current_step'][0] == 271
    assert state.dims_state[('0', 'index')] == 271
    assert len(viewer.calls) >= 1, "viewer should receive applied dims update"
    assert len(presenter.calls) >= 1, "presenter should receive applied dims update"


def test_dims_step_attaches_axis_metadata() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_ready = True
    state.dims_meta['current_step'] = [5, 0, 0]

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )

    sent = emitter.dims_step(axis=0, delta=1, origin='ui')

    assert sent is True
    pending, _ = dispatch.calls[-1]
    assert pending.scope == 'dims'
    assert pending.metadata == {
        'axis_index': 0,
        'axis_target': '0',
        'update_kind': 'step',
    }


def test_dims_set_index_suppresses_duplicate_value() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_ready = True
    state.dims_meta['current_step'] = [7]
    ledger.record_confirmed('dims', '0', 'index', 7, metadata={'axis_index': 0})

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )

    sent = emitter.dims_set_index(axis=0, value=7, origin='ui')

    assert sent is False
    assert dispatch.calls == []


def test_handle_dims_ack_rejected_reverts_projection() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_ready = False

    # Seed baseline metadata/state
    state.dims_meta.update(
        {
            'ndim': 1,
            'order': [0],
            'axis_labels': ['z'],
            'range': [(0, 8)],
            'range': [[0, 8]],
            'displayed': [0],
        }
    )
    presenter = PresenterStub()
    viewer_updates: list[dict[str, Any]] = []

    class Viewer:
        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            viewer_updates.append(dict(kwargs))

    viewer = Viewer()

    mirror = NapariDimsMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
        presenter=presenter,
        log_dims_info=False,
    )

    emitter = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    mirror.attach_emitter(emitter)

    frame = _make_notify_dims_frame(current_step=[4], ndisplay=2, level_shapes=[[9]])

    mirror.ingest_dims_notify(frame)

    # Emit a new intent that should be rejected
    emitter.dims_set_index('primary', 7, origin='ui')

    pending, _ = dispatch.calls[-1]
    ack = _ack_from_pending(
        pending,
        status='rejected',
        error={'code': 'state.bad_value', 'message': 'invalid axis'},
    )
    outcome = ledger.apply_ack(ack)

    control_actions.handle_dims_ack(
        state,
        loop_state,
        outcome,
        presenter=presenter,
        viewer_ref=lambda: viewer,
        ui_call=None,
        log_dims_info=False,
    )

    assert state.dims_meta['current_step'][0] == 4
    assert viewer_updates


def test_handle_generic_ack_updates_view_metadata() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    state.dims_ready = True

    emitter2 = NapariDimsIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_dims_info=False,
        tx_interval_ms=0,
    )
    emitter2.view_set_ndisplay(3, origin='ui')
    pending, _ = dispatch.calls[-1]

    ack = _ack_from_pending(pending, status='accepted', applied_value=2)
    outcome = ledger.apply_ack(ack)

    presenter = PresenterStub()
    control_actions.handle_generic_ack(state, loop_state, outcome, presenter=presenter)

    assert state.dims_meta['ndisplay'] == 2


def test_camera_zoom_bypasses_dedupe() -> None:
    state, loop_state, ledger, dispatch = _make_state()
    # Seed a confirmed camera zoom to mimic the server acknowledging a prior zoom.
    zoom_value = {"factor": 1.2, "anchor_px": [128.0, 256.0]}
    ledger.record_confirmed('camera', 'main', 'zoom', zoom_value)

    sent = control_actions.camera_zoom(
        state,
        loop_state,
        ledger,
        dispatch,
        factor=1.2,
        anchor_px=(128.0, 256.0),
        origin='ui',
    )

    assert sent is True
    pending, origin = dispatch.calls[-1]
    assert origin == 'ui'
    assert pending.metadata is not None
    assert pending.metadata.get('update_kind') == 'delta'
