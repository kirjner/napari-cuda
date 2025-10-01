from __future__ import annotations

import threading
from itertools import count
from types import SimpleNamespace
from typing import Any

from napari_cuda.client.streaming.client_stream_loop import ClientStreamLoop
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.control.pending_update_store import StateStore
from napari_cuda.client.control.control_channel_client import SessionMetadata, ResumeCursor
from napari_cuda.protocol import FeatureToggle, build_notify_dims, build_reply_command
from napari_cuda.protocol.messages import NotifyDimsFrame


class _StateChannelStub:
    def __init__(self) -> None:
        self.sent_frames: list[Any] = []
        self.posted: list[dict[str, Any]] = []

    def send_frame(self, frame: Any) -> bool:
        self.sent_frames.append(frame)
        return True

    def post(self, payload: dict[str, Any]) -> bool:
        self.posted.append(dict(payload))
        return True

    def feature_enabled(self, name: str) -> bool:
        return name == "call.command"


class _PresenterStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_dims_update(self, payload: dict[str, Any]) -> None:
        self.calls.append(dict(payload))

    def set_viewer_mirror(self, _viewer: Any) -> None:  # pragma: no cover - simple stub
        return None


def _make_loop() -> ClientStreamLoop:
    loop = ClientStreamLoop.__new__(ClientStreamLoop)
    loop._log_dims_info = False
    loop._loop_state = ClientLoopState()
    loop._loop_state.last_dims_payload = None
    loop._loop_state.state_channel = _StateChannelStub()
    loop._state_channel_stub = loop._loop_state.state_channel
    loop._viewer_mirror = lambda: None
    loop._ui_call = None
    loop._first_dims_ready_cb = None
    loop._first_dims_notified = False

    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
        dims_z=None,
        dims_z_min=None,
        dims_z_max=None,
    )
    loop._control_state = ControlStateContext.from_env(env)
    loop._control_state.session_id = "session-test"
    loop._loop_state.control_state = loop._control_state
    loop._command_lock = threading.Lock()
    loop._pending_commands = {}
    loop._command_catalog = ()

    clock_counter = count(100)
    loop._state_store = StateStore(clock=lambda: float(next(clock_counter)))

    loop._presenter_facade = _PresenterStub()
    loop._layer_bridge = SimpleNamespace(
        handle_ack=lambda outcome: None,
        clear_pending_on_reconnect=lambda: None,
    )
    loop._layer_registry = SimpleNamespace(
        apply_scene=lambda msg: None,
        apply_update=lambda msg: None,
        remove_layer=lambda msg: None,
    )

    return loop


def test_toggle_ndisplay_requires_ready() -> None:
    loop = _make_loop()

    assert loop.toggle_ndisplay(origin='ui') is False
    assert loop._state_channel_stub.sent_frames == []


def test_toggle_ndisplay_flips_between_2d_and_3d() -> None:
    loop = _make_loop()
    loop._control_state.dims_ready = True

    assert loop.toggle_ndisplay(origin='ui') is True
    frame = loop._state_channel_stub.sent_frames[-1]
    payload = frame.payload
    assert payload.scope == 'view'
    assert payload.target == 'main'
    assert payload.key == 'ndisplay'
    assert payload.value == 3

    loop._control_state.dims_meta['ndisplay'] = 2
    loop._control_state.last_settings_send = 0.0
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.sent_frames[-1].payload
    assert payload.value == 3

    loop._control_state.dims_meta['ndisplay'] = 3
    loop._control_state.last_settings_send = 0.0
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.sent_frames[-1].payload
    assert payload.value == 2


def test_handle_dims_update_caches_payload() -> None:
    loop = _make_loop()

    meta = loop._control_state.dims_meta
    meta.update(
        {
            'ndim': 3,
            'order': [0, 1, 2],
            'axis_labels': ['z', 'y', 'x'],
            'range': [(0, 10), (0, 5), (0, 3)],
            'sizes': [11, 6, 4],
            'displayed': [1, 2],
        }
    )

    frame = build_notify_dims(
        session_id='session-test',
        payload={'current_step': (1, 2), 'ndisplay': 2, 'mode': 'slice', 'source': 'test-suite'},
        timestamp=1.5,
        frame_id='dims-1',
    )

    loop._handle_dims_update(frame)

    assert loop._loop_state.last_dims_payload == {
        'current_step': [1, 2],
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': [(0, 10), (0, 5), (0, 3)],
        'order': [0, 1, 2],
        'axis_labels': ['z', 'y', 'x'],
        'sizes': [11, 6, 4],
        'displayed': [1, 2],
        'mode': 'slice',
        'source': 'test-suite',
    }
    assert loop._control_state.dims_ready is True
    assert loop._presenter_facade.calls
    assert loop._state_channel_stub.sent_frames == []


def test_replay_last_dims_payload_forwards_to_viewer() -> None:
    loop = _make_loop()

    class ViewerStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            self.calls.append(dict(kwargs))

    viewer = ViewerStub()
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


def test_session_metadata_propagated_to_loop_state() -> None:
    loop = _make_loop()
    metadata = SessionMetadata(
        session_id='sess-meta',
        heartbeat_s=3.0,
        ack_timeout_ms=250,
        resume_tokens={
            'notify.scene': ResumeCursor(seq=1, delta_token='scene-a'),
            'notify.layers': None,
            'notify.stream': None,
        },
        features={'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',))},
    )

    loop._handle_session_ready(metadata)
    assert loop._loop_state.state_session_metadata is metadata
    assert loop._command_catalog == ('napari.pixel.request_keyframe',)

    loop._on_state_disconnect(None)
    assert loop._loop_state.state_session_metadata is None


def test_request_keyframe_command_future_resolves() -> None:
    loop = _make_loop()
    metadata = SessionMetadata(
        session_id='sess-meta',
        heartbeat_s=3.0,
        ack_timeout_ms=250,
        resume_tokens={
            'notify.scene': None,
            'notify.layers': None,
            'notify.stream': None,
        },
        features={'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',))},
    )
    loop._handle_session_ready(metadata)

    future = loop.request_keyframe(origin='test')
    assert future is not None
    assert not future.done()

    call_frame = loop._state_channel_stub.sent_frames[-1]
    frame_id = call_frame.envelope.frame_id
    reply = build_reply_command(
        session_id='sess-meta',
        frame_id='reply-test',
        payload={
            'in_reply_to': frame_id,
            'status': 'ok',
            'result': None,
        },
    )

    loop._handle_reply_command(reply)
    assert future.done()
    payload = future.result()
    assert payload.status == 'ok'
    assert payload.in_reply_to == frame_id
