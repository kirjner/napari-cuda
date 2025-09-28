from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from napari_cuda.client.streaming.client_loop import intents
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.state_store import StateStore
from napari_cuda.protocol.messages import StateUpdateMessage


class FakeChannel:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> bool:
        self.payloads.append(dict(payload))
        return True


@dataclass
class FakeUICall:
    emit: Callable[[Callable[[], None]], None]

    @property
    def call(self) -> "FakeUICall":  # pragma: no cover - simple shim
        return self


def _make_state() -> tuple[
    intents.ClientStateContext,
    ClientLoopState,
    FakeChannel,
    StateStore,
]:
    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
        dims_z=None,
        dims_z_min=None,
        dims_z_max=None,
    )
    state = intents.ClientStateContext.from_env(env)
    loop_state = ClientLoopState()
    channel = FakeChannel()
    loop_state.state_channel = channel
    store = StateStore(
        client_id=state.client_id,
        next_client_seq=state.next_client_seq,
    )
    return state, loop_state, channel, store


def test_toggle_ndisplay_requires_ready() -> None:
    state, loop_state, channel, store = _make_state()

    sent = intents.toggle_ndisplay(state, loop_state, store, origin='ui')

    assert sent is False
    assert channel.payloads == []


def test_toggle_ndisplay_flips_between_2d_and_3d() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True

    # Initial call toggles to 3D when no cached value exists
    assert intents.toggle_ndisplay(state, loop_state, store, origin='ui') is True
    payload = channel.payloads[-1]
    assert payload['type'] == 'state.update'
    assert payload['scope'] == 'view'
    assert payload['target'] == 'main'
    assert payload['key'] == 'ndisplay'
    assert payload['value'] == 3

    # 2D -> 3D
    state.dims_meta['ndisplay'] = 2
    state.last_settings_send = 0.0
    assert intents.toggle_ndisplay(state, loop_state, store, origin='ui') is True
    payload = channel.payloads[-1]
    assert payload['value'] == 3

    # 3D -> 2D
    state.dims_meta['ndisplay'] = 3
    state.last_settings_send = 0.0
    assert intents.toggle_ndisplay(state, loop_state, store, origin='ui') is True
    payload = channel.payloads[-1]
    assert payload['value'] == 2


def test_handle_dims_update_seeds_metadata() -> None:
    state, loop_state, channel, _store = _make_state()

    ready_calls: list[None] = []
    presenter_calls: list[dict[str, Any]] = []

    class Presenter:
        def apply_dims_update(self, payload: dict[str, Any]) -> None:
            presenter_calls.append(payload)

    payload = {
        'seq': 7,
        'current_step': [1, 2],
        'ndim': 3,
        'ndisplay': 2,
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'range': [(0, 10), (0, 5), (0, 3)],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
    }

    intents.handle_dims_update(
        state,
        loop_state,
        payload,
        presenter=Presenter(),
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: ready_calls.append(None),
        log_dims_info=False,
    )

    assert state.dims_ready is True
    assert loop_state.last_dims_seq == 7
    assert loop_state.last_dims_payload == {
        'current_step': [1, 2],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 10), (0, 5), (0, 3)],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
    }
    assert presenter_calls[-1]['seq'] == 7
    assert ready_calls  # first dims update triggered readiness
    assert channel.payloads == []  # metadata path no longer sends commands


def test_dims_step_emits_state_update() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True

    sent = intents.dims_step(
        state,
        loop_state,
        store,
        axis=0,
        delta=1,
        origin='ui',
        viewer_ref=lambda: None,
        ui_call=None,
    )

    assert sent is True
    assert len(channel.payloads) == 1
    payload = channel.payloads[-1]
    assert payload['type'] == 'state.update'
    assert payload['scope'] == 'dims'
    assert payload['target'] == '0'
    assert payload['key'] == 'step'
    assert payload['value'] == 1
    runtime = state.control_runtimes['dims:0:step']
    assert runtime.active is True
    assert runtime.interaction_id


def test_handle_dims_state_update_clears_runtime() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True

    intents.dims_step(
        state,
        loop_state,
        store,
        axis=0,
        delta=2,
        origin='ui',
        viewer_ref=lambda: None,
        ui_call=None,
    )

    payload = channel.payloads[-1]
    msg = StateUpdateMessage(
        scope='dims',
        target=payload['target'],
        key=payload['key'],
        value=payload['value'],
        client_id=payload['client_id'],
        client_seq=payload['client_seq'],
        phase='update',
        server_seq=10,
    )

    intents.handle_dims_state_update(state, store, msg)

    runtime = state.control_runtimes['dims:0:step']
    assert runtime.active is False
    assert runtime.interaction_id is None
    debug_state = store.dump_debug()
    dims_key = 'dims:0:step'
    assert debug_state[dims_key]['pending'] == []


def test_view_set_ndisplay_emits_state_update() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True

    ok = intents.view_set_ndisplay(state, loop_state, store, 3, origin='ui')

    assert ok is True
    payload = channel.payloads[-1]
    assert payload['type'] == 'state.update'
    assert payload['scope'] == 'view'
    assert payload['target'] == 'main'
    assert payload['key'] == 'ndisplay'
    assert payload['value'] == 3


def test_volume_set_clim_emits_state_update() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True
    state.dims_meta['volume'] = True
    state.dims_meta['ndisplay'] = 3

    ok = intents.volume_set_clim(state, loop_state, store, 0.1, 0.9, origin='ui')

    assert ok is True
    payload = channel.payloads[-1]
    assert payload['type'] == 'state.update'
    assert payload['scope'] == 'volume'
    assert payload['target'] == 'main'
    assert payload['key'] == 'contrast_limits'
    assert tuple(payload['value']) == (0.1, 0.9)


def test_generic_state_update_clears_runtime() -> None:
    state, loop_state, channel, store = _make_state()
    state.dims_ready = True

    intents.view_set_ndisplay(state, loop_state, store, 3, origin='ui')
    payload = channel.payloads[-1]

    msg = StateUpdateMessage(
        scope='view',
        target='main',
        key='ndisplay',
        value=payload['value'],
        client_id=payload['client_id'],
        client_seq=payload['client_seq'],
        phase='update',
        server_seq=5,
    )

    intents.handle_generic_state_update(state, store, msg)

    runtime = state.control_runtimes['view:main:ndisplay']
    assert runtime.active is False
    assert runtime.interaction_id is None


def test_replay_last_dims_payload_invokes_viewer() -> None:
    state, loop_state, _, _ = _make_state()
    calls: list[tuple] = []

    class Viewer:
        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            calls.append(tuple(sorted(kwargs.items())))

    viewer = Viewer()
    loop_state.last_dims_payload = {
        'current_step': [3, 1],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 8), (0, 4), (0, 2)],
        'order': [0, 2, 1],
        'axis_labels': ['z', 'x', 'y'],
        'sizes': [9, 5, 3],
        'displayed': [2, 1],
    }

    intents.replay_last_dims_payload(
        state,
        loop_state,
        viewer_ref=lambda: viewer,
        ui_call=None,
    )

    assert len(calls) == 1
    keys = {k for k, _ in calls[0]}
    assert keys == {'axis_labels', 'current_step', 'displayed', 'dims_range', 'ndim', 'ndisplay', 'order', 'sizes'}
