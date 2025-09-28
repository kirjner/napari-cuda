from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from napari_cuda.client.streaming.client_loop import intents
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.control_sessions import ControlSession
from napari_cuda.protocol.messages import CONTROL_COMMAND_TYPE


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


def _make_state() -> tuple[intents.IntentState, ClientLoopState, FakeChannel]:
    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
        dims_z=None,
        dims_z_min=None,
        dims_z_max=None,
    )
    state = intents.IntentState.from_env(env)
    loop_state = ClientLoopState()
    channel = FakeChannel()
    loop_state.state_channel = channel
    return state, loop_state, channel


def test_toggle_ndisplay_requires_ready() -> None:
    state, loop_state, channel = _make_state()

    sent = intents.toggle_ndisplay(state, loop_state, origin='ui')

    assert sent is False
    assert channel.payloads == []


def test_toggle_ndisplay_flips_between_2d_and_3d() -> None:
    state, loop_state, channel = _make_state()
    state.dims_ready = True

    # Initial call toggles to 3D when no cached value exists
    assert intents.toggle_ndisplay(state, loop_state, origin='ui') is True
    assert channel.payloads[-1]['ndisplay'] == 3

    # 2D -> 3D
    state.dims_meta['ndisplay'] = 2
    state.last_settings_send = 0.0
    assert intents.toggle_ndisplay(state, loop_state, origin='ui') is True
    assert channel.payloads[-1]['ndisplay'] == 3

    # 3D -> 2D
    state.dims_meta['ndisplay'] = 3
    state.last_settings_send = 0.0
    assert intents.toggle_ndisplay(state, loop_state, origin='ui') is True
    assert channel.payloads[-1]['ndisplay'] == 2


def test_handle_dims_update_caches_payload_and_clears_pending_ack() -> None:
    state, loop_state, channel = _make_state()
    session = ControlSession(key="dims:step:0")
    session.interaction_id = "sess-1"
    session.push_pending(41, value=1, phase="update")
    session.awaiting_commit = True
    state.control_sessions[session.key] = session
    loop_state.pending_intents = {
        41: {
            'kind': 'dims.step',
            'axis': 0,
            'phase': 'update',
            'value': 1,
            'interaction_id': session.interaction_id,
        }
    }

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
        'ack': True,
        'server_seq': 3,
        'phase': 'update',
        'control_versions': {
            'step': {
                'server_seq': 3,
                'source_client_id': state.client_id,
                'source_client_seq': 41,
                'interaction_id': session.interaction_id,
                'phase': 'update',
            }
        },
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
    assert loop_state.pending_intents
    commit_entry = next(iter(loop_state.pending_intents.values()))
    assert commit_entry['phase'] == 'commit'
    # Commit command should have been scheduled
    assert channel.payloads
    commit_payload = channel.payloads[-1]
    assert commit_payload['type'] == CONTROL_COMMAND_TYPE
    assert commit_payload['phase'] == 'commit'
    commit_seq = commit_payload['client_seq']

    commit_ack = dict(payload)
    commit_ack['server_seq'] = 4
    commit_ack['phase'] = 'commit'
    commit_ack['control_versions'] = {
        'step': {
            'server_seq': 4,
            'source_client_id': state.client_id,
            'source_client_seq': commit_seq,
            'interaction_id': session.interaction_id,
            'phase': 'commit',
        }
    }

    intents.handle_dims_update(
        state,
        loop_state,
        commit_ack,
        presenter=Presenter(),
        viewer_ref=lambda: None,
        ui_call=None,
        notify_first_dims_ready=lambda: ready_calls.append(None),
        log_dims_info=False,
    )

    assert loop_state.pending_intents == {}
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


def test_replay_last_dims_payload_invokes_viewer() -> None:
    state, loop_state, _ = _make_state()
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
