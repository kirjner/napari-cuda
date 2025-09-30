from __future__ import annotations

from collections import deque
from itertools import count
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from napari_cuda.client.streaming.client_loop import control as control_actions
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.pending_update_store import StateStore, PendingUpdate
from napari_cuda.protocol import build_ack_state


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

    payload = {
        'seq': 9,
        'current_step': [1, 2, 3],
        'ndim': 3,
        'ndisplay': 2,
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'range': [(0, 10), (0, 5), (0, 3)],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
    }

    control_actions.handle_dims_update(
        state,
        loop_state,
        payload,
        presenter=presenter,
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: ready_calls.append(None),
        log_dims_info=False,
        state_store=store,
    )

    assert state.dims_ready is True
    assert loop_state.last_dims_seq == 9
    assert presenter.calls
    debug = store.dump_debug()
    assert debug['view:main:ndisplay']['confirmed']['value'] == 2
    assert ready_calls, "first dims update should trigger readiness"


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
    payload = {
        'seq': 1,
        'current_step': [4],
        'ndim': 1,
        'ndisplay': 2,
        'order': [0],
        'axis_labels': ['z'],
        'range': [(0, 8)],
        'sizes': [9],
        'displayed': [0],
    }
    presenter = PresenterStub()
    viewer_updates: list[dict[str, Any]] = []

    control_actions.handle_dims_update(
        state,
        loop_state,
        payload,
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
