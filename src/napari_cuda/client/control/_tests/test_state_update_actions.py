from __future__ import annotations

from collections import deque
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

from napari_cuda.client.control import state_update_actions as control_actions
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.pending_update_store import StateStore, PendingUpdate
from napari_cuda.protocol import build_ack_state, build_notify_dims
from napari_cuda.protocol.messages import NotifyDimsFrame


class FakeDispatch:
    def __init__(self) -> None:
        self.calls: list[tuple[PendingUpdate, str]] = []

    def __call__(self, pending: PendingUpdate, origin: str) -> bool:
        self.calls.append((pending, origin))
        return True


class PresenterStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_dims_update(self, payload: dict[str, Any]) -> None:
        self.calls.append(dict(payload))


class ViewerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _apply_remote_dims_update(self, **kwargs: Any) -> None:
        self.calls.append(dict(kwargs))


def _make_notify_dims_frame(
    *,
    current_step: Sequence[int],
    ndisplay: int,
    mode: str = 'plane',
    source: str = 'test-suite',
    session_id: str = 'session-test',
    frame_id: str = 'dims-test',
    timestamp: float = 1.5,
) -> NotifyDimsFrame:
    return build_notify_dims(
        session_id=session_id,
        payload={
            'current_step': tuple(current_step),
            'ndisplay': int(ndisplay),
            'mode': mode,
            'source': source,
        },
        timestamp=timestamp,
        frame_id=frame_id,
    )


def _make_state() -> tuple[
    control_actions.ControlStateContext,
    ClientLoopState,
    StateStore,
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
    store = StateStore(clock=lambda: clock_steps.popleft())
    dispatch = FakeDispatch()
    return state, loop_state, store, dispatch


def _ack_from_pending(pending: PendingUpdate, *, status: str, applied_value: Any | None = None, error: dict[str, Any] | None = None) -> Any:
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


def test_toggle_ndisplay_requires_ready() -> None:
    state, loop_state, store, dispatch = _make_state()

    sent = control_actions.toggle_ndisplay(state, loop_state, store, dispatch, origin='ui')

    assert sent is False
    assert dispatch.calls == []


def test_toggle_ndisplay_dispatches_state_update() -> None:
    state, loop_state, store, dispatch = _make_state()
    state.dims_ready = True

    sent = control_actions.toggle_ndisplay(state, loop_state, store, dispatch, origin='ui')

    assert sent is True
    pending, origin = dispatch.calls[-1]
    assert origin == 'ui'
    assert pending.scope == 'view'
    assert pending.target == 'main'
    assert pending.key == 'ndisplay'
    assert pending.value == 3
    assert pending.update_phase == 'start'


def test_handle_dims_update_seeds_state_store() -> None:
    state, loop_state, store, dispatch = _make_state()

    presenter = PresenterStub()
    ready_calls: list[None] = []

    state.dims_meta.update(
        {
            'ndim': 3,
            'order': [0, 1, 2],
            'axis_labels': ['z', 'y', 'x'],
            'range': [(0, 10), (0, 5), (0, 3)],
            'sizes': [11, 6, 4],
            'displayed': [1, 2],
        }
    )

    frame = _make_notify_dims_frame(current_step=[1, 2, 3], ndisplay=2)

    control_actions.handle_dims_update(
        state,
        loop_state,
        frame,
        presenter=presenter,
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: ready_calls.append(None),
        log_dims_info=False,
        state_store=store,
    )

    assert state.dims_ready is True
    assert loop_state.last_dims_payload == {
        'current_step': [1, 2, 3],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 10), (0, 5), (0, 3)],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
        'mode': 'plane',
        'volume': False,
        'source': 'test-suite',
    }
    assert presenter.calls
    debug = store.dump_debug()
    assert debug['view:main:ndisplay']['confirmed']['value'] == 2
    assert ready_calls, "first dims update should trigger readiness"


def test_handle_dims_update_modes_volume_flag() -> None:
    state, loop_state, store, _dispatch = _make_state()
    presenter = PresenterStub()

    frame = _make_notify_dims_frame(current_step=[0, 0, 0], ndisplay=3, mode='volume')

    control_actions.handle_dims_update(
        state,
        loop_state,
        frame,
        presenter=presenter,
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: None,
        log_dims_info=False,
        state_store=store,
    )

    assert state.dims_meta['mode'] == 'volume'
    assert state.dims_meta['volume'] is True
    assert loop_state.last_dims_payload['volume'] is True
    assert presenter.calls[-1]['mode'] == 'volume'


def test_handle_dims_update_volume_adjusts_viewer_axes() -> None:
    state, loop_state, store, _dispatch = _make_state()
    viewer = ViewerStub()
    presenter = PresenterStub()

    state.dims_meta.update(
        {
            'ndim': 3,
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'range': [[0, 5], [0, 5], [0, 5]],
            'sizes': [6, 6, 6],
        }
    )

    frame = _make_notify_dims_frame(current_step=[1, 2, 3], ndisplay=3, mode='volume')

    control_actions.handle_dims_update(
        state,
        loop_state,
        frame,
        presenter=presenter,
        viewer_ref=lambda: viewer,
        ui_call=None,
        notify_first_dims_ready=lambda: None,
        log_dims_info=False,
        state_store=store,
    )

    assert viewer.calls, "viewer should receive dims update"
    payload = viewer.calls[-1]
    assert payload['ndisplay'] == 3
    assert payload['current_step'] == (1, 2, 3)
    assert payload['axis_labels'] == ['z', 'y', 'x']

    assert presenter.calls[-1]['ndisplay'] == 3
    assert loop_state.last_dims_payload['ndisplay'] == 3
    assert loop_state.last_dims_payload['mode'] == 'volume'


def test_scene_level_then_dims_updates_slider_bounds() -> None:
    state, loop_state, store, _dispatch = _make_state()
    viewer = ViewerStub()
    presenter = PresenterStub()

    state.dims_meta.update(
        {
            'ndim': 3,
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'sizes': [512, 256, 64],
            'range': [[0, 511], [0, 255], [0, 63]],
        }
    )

    policies = {
        'multiscale': {
            'current_level': 1,
            'active_level': 1,
            'levels': [
                {'index': 0, 'shape': [512, 256, 64]},
                {'index': 1, 'shape': [256, 128, 32]},
            ],
        }
    }

    control_actions.apply_scene_policies(state, policies)

    frame = _make_notify_dims_frame(current_step=[0, 0, 0], ndisplay=2)

    control_actions.handle_dims_update(
        state,
        loop_state,
        frame,
        presenter=presenter,
        viewer_ref=lambda: viewer,
        ui_call=None,
        notify_first_dims_ready=lambda: None,
        log_dims_info=False,
        state_store=store,
    )

    sizes = state.dims_meta['sizes']
    assert sizes == [256, 128, 32]
    dims_range = state.dims_meta['range']
    assert dims_range and dims_range[2][1] == 31

    assert viewer.calls, 'viewer should receive updated dims metadata'
    payload = viewer.calls[-1]
    assert payload['sizes'][2] == 32
    assert payload['dims_range'][2][1] == 31

    assert loop_state.last_dims_payload['sizes'][2] == 32


def test_hud_snapshot_carries_volume_state() -> None:
    state, _, store, _dispatch = _make_state()
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


def test_apply_scene_policies_populates_multiscale_state() -> None:
    state, _, _, _ = _make_state()
    policies = {
        'multiscale': {
            'policy': 'oversampling',
            'active_level': 3,
            'downgraded': True,
            'index_space': 'zyx',
            'levels': [
                {'index': 0, 'path': 'level_00'},
                {'index': 1, 'path': 'level_01'},
            ],
        }
    }

    control_actions.apply_scene_policies(state, policies)

    assert state.scene_policies['multiscale']['policy'] == 'oversampling'
    assert state.multiscale_state['policy'] == 'oversampling'
    assert state.multiscale_state['level'] == 3
    meta = state.dims_meta.get('multiscale')
    assert isinstance(meta, dict)
    assert meta['current_level'] == 3
    assert meta['downgraded'] is True
    assert meta['index_space'] == 'zyx'
    levels = meta['levels']
    assert isinstance(levels, list)
    assert levels[0]['path'] == 'level_00'


def test_dims_ack_accepted_updates_viewer_with_applied_value() -> None:
    state, loop_state, store, _ = _make_state()
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

    pending = store.apply_local(
        'dims',
        '0',
        'index',
        148,
        'start',
        intent_id='intent-1',
        frame_id='state-1',
        metadata={'axis_index': 0, 'axis_target': '0', 'update_kind': 'index'},
    )
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

    outcome = store.apply_ack(ack)
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
    assert viewer.calls, "viewer should receive applied dims update"
    assert presenter.calls, "presenter should receive applied dims update"


def test_dims_step_attaches_axis_metadata() -> None:
    state, loop_state, store, dispatch = _make_state()
    state.dims_ready = True
    state.dims_meta['current_step'] = [5, 0, 0]

    sent = control_actions.dims_step(
        state,
        loop_state,
        store,
        dispatch,
        axis=0,
        delta=1,
        origin='ui',
        viewer_ref=lambda: None,
        ui_call=None,
    )

    assert sent is True
    pending, _ = dispatch.calls[-1]
    assert pending.scope == 'dims'
    assert pending.metadata == {
        'axis_index': 0,
        'axis_target': '0',
        'update_kind': 'step',
    }


def test_handle_dims_ack_rejected_reverts_projection() -> None:
    state, loop_state, store, dispatch = _make_state()
    state.dims_ready = False

    # Seed baseline metadata/state
    state.dims_meta.update(
        {
            'ndim': 1,
            'order': [0],
            'axis_labels': ['z'],
            'range': [(0, 8)],
            'sizes': [9],
            'displayed': [0],
        }
    )
    presenter = PresenterStub()
    viewer_updates: list[dict[str, Any]] = []

    frame = _make_notify_dims_frame(current_step=[4], ndisplay=2)

    control_actions.handle_dims_update(
        state,
        loop_state,
        frame,
        presenter=presenter,
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: None,
        log_dims_info=False,
        state_store=store,
    )

    # Emit a new intent that should be rejected
    control_actions.dims_set_index(
        state,
        loop_state,
        store,
        dispatch,
        axis='primary',
        value=7,
        origin='ui',
        viewer_ref=lambda: None,
        ui_call=None,
    )

    pending, _ = dispatch.calls[-1]
    ack = _ack_from_pending(
        pending,
        status='rejected',
        error={'code': 'state.bad_value', 'message': 'invalid axis'},
    )
    outcome = store.apply_ack(ack)

    class Viewer:
        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            viewer_updates.append(dict(kwargs))

    viewer = Viewer()

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
    state, loop_state, store, dispatch = _make_state()
    state.dims_ready = True

    control_actions.view_set_ndisplay(state, loop_state, store, dispatch, 3, origin='ui')
    pending, _ = dispatch.calls[-1]

    ack = _ack_from_pending(pending, status='accepted', applied_value=2)
    outcome = store.apply_ack(ack)

    presenter = PresenterStub()
    control_actions.handle_generic_ack(state, loop_state, outcome, presenter=presenter)

    assert state.dims_meta['ndisplay'] == 2
