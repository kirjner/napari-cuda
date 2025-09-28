from __future__ import annotations

from typing import Any

from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.client_loop.intents import ClientStateContext
from napari_cuda.client.streaming.control_sessions import ControlSession
from napari_cuda.protocol.messages import CONTROL_COMMAND_TYPE


def _make_loop() -> ClientStreamLoop:
    loop = ClientStreamLoop.__new__(ClientStreamLoop)
    loop._dims_meta = {}
    loop._dims_ready = False
    loop._log_dims_info = False
    loop._primary_axis_index = None
    loop._loop_state = ClientLoopState()
    loop._viewer_mirror = lambda: None
    loop._mirror_dims_to_viewer = lambda *args, **kwargs: None
    loop._first_dims_ready_cb = None
    loop._first_dims_notified = False
    loop._ui_call = None
    loop._loop_state.last_dims_payload = None
    loop._compute_primary_axis_index = lambda: 0
    loop._intent_state = ClientStateContext()
    loop._intent_state.dims_meta = loop._dims_meta
    loop._loop_state.intents = loop._intent_state

    def _mark_dims_ready() -> None:
        loop._dims_ready = True
        loop._intent_state.dims_ready = True

    loop._notify_first_dims_ready = _mark_dims_ready

    class _StubPresenter:
        def apply_dims_update(self, payload: dict[str, Any]) -> None:  # pragma: no cover - stub
            return

    loop._presenter_facade = _StubPresenter()
    class _StateChannelStub:
        def __init__(self) -> None:
            self.posted: list[dict[str, Any]] = []

        def post(self, payload: dict[str, Any]) -> bool:
            self.posted.append(dict(payload))
            return True

        def request_keyframe_once(self) -> None:  # pragma: no cover - stub
            return

    loop._state_channel_stub = _StateChannelStub()
    loop._loop_state.state_channel = loop._state_channel_stub
    return loop


def test_toggle_ndisplay_requires_ready() -> None:
    loop = _make_loop()
    assert loop.toggle_ndisplay(origin='ui') is False
    assert loop._state_channel_stub.posted == []


def test_toggle_ndisplay_flips_between_2d_and_3d() -> None:
    loop = _make_loop()
    loop._intent_state.dims_ready = True

    # Missing meta defaults to 3D target
    assert loop.toggle_ndisplay(origin='ui') is True
    assert loop._state_channel_stub.posted[-1]['ndisplay'] == 3

    # 2D -> 3D
    loop._dims_meta['ndisplay'] = 2
    assert loop.toggle_ndisplay(origin='ui') is True
    assert loop._state_channel_stub.posted[-1]['ndisplay'] == 3

    # 3D -> 2D
    loop._dims_meta['ndisplay'] = 3
    assert loop.toggle_ndisplay(origin='ui') is True
    assert loop._state_channel_stub.posted[-1]['ndisplay'] == 2


def test_handle_dims_update_clears_ack_and_caches_payload() -> None:
    loop = _make_loop()
    session = ControlSession(key="dims:step:0")
    session.interaction_id = "sess-1"
    session.push_pending(41, value=1, phase="update")
    session.awaiting_commit = True
    loop._intent_state.control_sessions[session.key] = session
    loop._loop_state.pending_intents = {
        41: {
            'kind': 'dims.step',
            'axis': 0,
            'phase': 'update',
            'value': 1,
            'interaction_id': session.interaction_id,
        }
    }

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
        'server_seq': 5,
        'phase': 'update',
        'control_versions': {
            'step': {
                'server_seq': 5,
                'source_client_id': loop._intent_state.client_id,
                'source_client_seq': 41,
                'interaction_id': session.interaction_id,
                'phase': 'update',
            }
        },
    }

    loop._handle_dims_update(payload)

    assert loop._loop_state.pending_intents
    commit_entry = next(iter(loop._loop_state.pending_intents.values()))
    assert commit_entry['phase'] == 'commit'
    assert loop._loop_state.last_dims_seq == 7
    assert loop._loop_state.last_dims_payload == {
        'current_step': [1, 2],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 10), (0, 5), (0, 3)],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
    }
    assert loop._dims_ready is True
    assert loop._intent_state.dims_ready is True
    # Commit command should be dispatched after ack
    assert loop._state_channel_stub.posted
    commit_payload = loop._state_channel_stub.posted[-1]
    assert commit_payload['type'] == CONTROL_COMMAND_TYPE
    assert commit_payload['phase'] == 'commit'
    commit_seq = commit_payload['client_seq']

    commit_ack = dict(payload)
    commit_ack['server_seq'] = 6
    commit_ack['phase'] = 'commit'
    commit_ack['control_versions'] = {
        'step': {
            'server_seq': 6,
            'source_client_id': loop._intent_state.client_id,
            'source_client_seq': commit_seq,
            'interaction_id': session.interaction_id,
            'phase': 'commit',
        }
    }

    loop._handle_dims_update(commit_ack)

    assert loop._loop_state.pending_intents == {}


def test_replay_last_dims_payload_forwards_to_viewer() -> None:
    class ViewerStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            self.calls.append(dict(kwargs))

    viewer = ViewerStub()
    loop = _make_loop()
    loop._viewer_mirror = lambda: viewer
    loop._loop_state.last_dims_payload = {
        'current_step': [3, 1],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 8), (0, 4), (0, 2)],
        'order': [0, 2, 1],
        'axis_labels': ['z', 'x', 'y'],
        'sizes': [9, 5, 3],
        'displayed': [2, 1],
    }

    loop._replay_last_dims_payload()

    assert viewer.calls == [
        {
            'current_step': [3, 1],
            'ndisplay': 2,
            'ndim': 3,
            'dims_range': [(0, 8), (0, 4), (0, 2)],
            'order': [0, 2, 1],
            'axis_labels': ['z', 'x', 'y'],
            'sizes': [9, 5, 3],
            'displayed': [2, 1],
        }
    ]
