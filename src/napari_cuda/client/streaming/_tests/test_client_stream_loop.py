from __future__ import annotations

from typing import Any

from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.state_update_actions import ClientStateContext
from napari_cuda.client.control.pending_update_store import StateStore


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
    loop._state_store = StateStore(
        client_id=loop._intent_state.client_id,
        next_client_seq=loop._intent_state.next_client_seq,
    )
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
    payload = loop._state_channel_stub.posted[-1]
    assert payload['type'] == 'state.update'
    assert payload['scope'] == 'view'
    assert payload['target'] == 'main'
    assert payload['key'] == 'ndisplay'
    assert payload['value'] == 3

    # 2D -> 3D
    loop._dims_meta['ndisplay'] = 2
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.posted[-1]
    assert payload['value'] == 3

    # 3D -> 2D
    loop._dims_meta['ndisplay'] = 3
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.posted[-1]
    assert payload['value'] == 2


def test_handle_dims_update_caches_payload() -> None:
    loop = _make_loop()

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

    loop._handle_dims_update(payload)

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
    # Legacy commands are no longer dispatched for metadata updates
    assert loop._state_channel_stub.posted == []


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
