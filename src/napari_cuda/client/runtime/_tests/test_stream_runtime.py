from __future__ import annotations

import threading
from itertools import count
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from qtpy import QtCore

pytestmark = pytest.mark.usefixtures("qtbot")


class _UiCallStub:
    class _Emitter:
        def emit(self, func):
            func()

    def __init__(self) -> None:
        self.call = self._Emitter()

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.control_channel_client import (
    ResumeCursor,
    SessionMetadata,
)
from napari_cuda.client.control.control_state import ControlStateContext
from napari_cuda.client.data.registry import RemoteLayerRegistry
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.runtime.stream_runtime import (
    ClientStreamLoop,
    CommandError,
)
from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_to_payload,
)
from napari_cuda.protocol import (
    FeatureToggle,
    build_error_command,
    build_notify_dims,
    build_reply_command,
)


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
        self.layer_updates: list[Any] = []

    def apply_dims_update(self, *, spec: DimsSpec, viewer_update: Mapping[str, Any]) -> None:
        self.calls.append({'spec': spec, 'viewer': dict(viewer_update)})

    def set_viewer_mirror(self, _viewer: Any) -> None:  # pragma: no cover - simple stub
        return None

    def apply_layer_delta(self, message: Any) -> None:
        self.layer_updates.append(message)

    def apply_multiscale_policy(self, payload: dict[str, Any]) -> None:
        self.calls.append(dict(payload))


def _make_loop() -> ClientStreamLoop:
    loop = ClientStreamLoop.__new__(ClientStreamLoop)
    loop._log_dims_info = False
    loop._loop_state = ClientLoopState()
    loop._loop_state.state_channel = _StateChannelStub()
    loop._state_channel_stub = loop._loop_state.state_channel
    loop._viewer_mirror = lambda: None
    loop._ui_call = _UiCallStub()
    loop._ui_thread = QtCore.QThread.currentThread()
    loop._first_dims_ready_cb = None
    loop._first_dims_notified = False

    env = SimpleNamespace(
        dims_rate_hz=60.0,
        wheel_step=1,
        settings_rate_hz=30.0,
                            )
    loop._control_state = ControlStateContext.from_env(env)
    loop._control_state.session_id = "session-test"
    loop._loop_state.control_state = loop._control_state
    loop._command_lock = threading.Lock()
    loop._pending_commands = {}
    loop._command_catalog = ()

    clock_counter = count(100)
    loop._state_ledger = ClientStateLedger(clock=lambda: float(next(clock_counter)))

    _seed_default_spec(loop._control_state)

    loop._presenter_facade = _PresenterStub()
    loop._layer_registry = RemoteLayerRegistry()
    loop._widget_to_video = lambda x, y: (float(x), float(y))
    loop._video_delta_to_canvas = lambda dx, dy: (float(dx), float(dy))
    loop._dims_mirror = None
    loop._dims_emitter = None
    loop._layer_mirror = None
    loop._layer_emitter = None
    loop._camera_mirror = None
    loop._camera_emitter = None
    loop._multiscale_mirror = None
    loop._slider_tx_interval_ms = 10

    loop._initialize_mirrors_and_emitters()

    return loop


def _seed_default_spec(state: ControlStateContext) -> None:
    spec = _make_spec(
        current_step=[0, 0, 0],
        ndisplay=2,
        level_shapes=[[10, 10, 10]],
        axis_labels=['z', 'y', 'x'],
        order=[0, 1, 2],
        displayed=[1, 2],
    )
    state.dims_spec = spec


def _make_spec(
    *,
    current_step: Sequence[int],
    ndisplay: int,
    level_shapes: Sequence[Sequence[int]],
    axis_labels: Sequence[str],
    order: Sequence[int],
    displayed: Sequence[int],
) -> DimsSpec:
    ndim = len(axis_labels)
    axes: list[DimsSpecAxis] = []
    for idx in range(ndim):
        per_level_steps = [int(shape[idx]) for shape in level_shapes]
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=str(axis_labels[idx]),
                role=str(axis_labels[idx]),
                displayed=idx in displayed,
                order_position=order.index(idx),
                current_step=current_step[idx] if idx < len(current_step) else 0,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=tuple(per_level_steps),
                per_level_world=tuple(
                    AxisExtent(0.0, float(max(count - 1, 0)), 1.0) for count in per_level_steps
                ),
            )
        )
    level_entries = [{'index': idx, 'shape': list(shape)} for idx, shape in enumerate(level_shapes)]
    return DimsSpec(
        version=1,
        ndim=len(level_shapes[0]) if level_shapes else len(current_step),
        ndisplay=int(ndisplay),
        order=tuple(int(v) for v in order),
        displayed=tuple(int(v) for v in displayed),
        current_level=0,
        current_step=tuple(int(v) for v in current_step),
        level_shapes=tuple(tuple(int(dim) for dim in shape) for shape in level_shapes),
        plane_mode=ndisplay < 3,
        axes=tuple(axes),
        levels=tuple(level_entries),
        downgraded=None,
        labels=None,
    )


def test_toggle_ndisplay_requires_ready() -> None:
    loop = _make_loop()

    with pytest.raises(AssertionError):
        loop.toggle_ndisplay(origin='ui')
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

    loop._control_state.last_settings_send = 0.0
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.sent_frames[-1].payload
    assert payload.value == 2

    loop._control_state.last_settings_send = 0.0
    assert loop.toggle_ndisplay(origin='ui') is True
    payload = loop._state_channel_stub.sent_frames[-1].payload
    assert payload.value == 3


def test_handle_dims_update_caches_spec() -> None:
    loop = _make_loop()

    class ViewerStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            self.calls.append(dict(kwargs))

    viewer = ViewerStub()
    loop._viewer_mirror = lambda: viewer
    loop._dims_mirror._viewer_ref = lambda: viewer

    spec = _make_spec(
        current_step=[1, 2, 0],
        ndisplay=2,
        level_shapes=[[11, 6, 4]],
        axis_labels=['z', 'y', 'x'],
        order=[0, 1, 2],
        displayed=[1, 2],
    )
    spec_payload = dims_spec_to_payload(spec)
    assert spec_payload is not None
    payload = {
        'dims_spec': spec_payload,
        'mode': 'plane' if spec.plane_mode else 'volume',
        'step': list(spec.current_step),
        'current_level': int(spec.current_level),
        'level_shapes': [list(shape) for shape in spec.level_shapes],
        'levels': [dict(entry) for entry in spec.levels],
        'ndisplay': int(spec.ndisplay),
    }
    frame = build_notify_dims(
        session_id='session-test',
        payload=payload,
        timestamp=1.5,
        frame_id='dims-1',
    )

    loop._dims_mirror.ingest_dims_notify(frame)

    assert loop._loop_state.last_dims_spec == spec
    call = loop._presenter_facade.calls[-1]
    assert call['spec'] == spec
    assert call['viewer'] == {
        'current_step': (1, 2, 0),
        'ndisplay': 2,
        'ndim': 3,
        'dims_range': ((0.0, 10.0, 1.0), (0.0, 5.0, 1.0), (0.0, 3.0, 1.0)),
        'order': (0, 1, 2),
        'axis_labels': ('z', 'y', 'x'),
        'displayed': (1, 2),
    }
    assert viewer.calls
    assert loop._control_state.dims_ready is True
    assert loop._presenter_facade.calls
    assert loop._state_channel_stub.sent_frames == []


def test_replay_last_dims_spec_forwards_to_viewer() -> None:
    loop = _make_loop()

    class ViewerStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def _apply_remote_dims_update(self, **kwargs: Any) -> None:
            self.calls.append(dict(kwargs))

    viewer = ViewerStub()
    loop._viewer_mirror = lambda: viewer
    spec = _make_spec(
        current_step=[3, 1, 0],
        ndisplay=2,
        level_shapes=[[9, 5, 3]],
        axis_labels=['z', 'x', 'y'],
        order=[0, 2, 1],
        displayed=[2, 1],
    )
    loop._loop_state.last_dims_spec = spec

    loop._control_state.dims_spec = spec

    loop._replay_last_dims_spec()

    assert viewer.calls == [
        {
            'current_step': (3, 1, 0),
            'ndisplay': 2,
            'ndim': 3,
            'dims_range': ((0.0, 8.0, 1.0), (0.0, 4.0, 1.0), (0.0, 2.0, 1.0)),
            'order': (0, 2, 1),
            'axis_labels': ('z', 'x', 'y'),
            'displayed': (2, 1),
        }
    ]


def test_pointer_ignored_until_dims_ready() -> None:
    loop = _make_loop()

    payload = {
        'type': 'input.pointer',
        'phase': 'move',
        'x_px': 10.0,
        'y_px': 20.0,
        'mods': 0,
        'buttons': 0,
        'width_px': 100,
        'height_px': 80,
        'ts': 1.0,
    }

    loop._on_pointer(payload)

    assert loop._state_channel_stub.posted == []


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
        features={
            'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',)),
        },
    )

    loop._on_state_session_ready(metadata)
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
        features={
            'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',)),
        },
    )
    loop._on_state_session_ready(metadata)

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

    loop._ingest_reply_command(reply)
    assert future.done()
    payload = future.result()
    assert payload.status == 'ok'
    assert payload.in_reply_to == frame_id


def test_request_keyframe_command_future_errors() -> None:
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
        features={
            'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',)),
        },
    )
    loop._on_state_session_ready(metadata)

    future = loop.request_keyframe(origin='test')
    assert future is not None
    assert not future.done()

    call_frame = loop._state_channel_stub.sent_frames[-1]
    frame_id = call_frame.envelope.frame_id
    error = build_error_command(
        session_id='sess-meta',
        frame_id='error-test',
        payload={
            'in_reply_to': frame_id,
            'status': 'error',
            'code': 'command.busy',
            'message': 'encoder busy',
            'details': {'retry_after_ms': 250},
        },
    )

    loop._ingest_error_command(error)

    assert future.done()
    with pytest.raises(CommandError) as excinfo:
        future.result()
    assert excinfo.value.code == 'command.busy'
    assert excinfo.value.details == {'retry_after_ms': 250}
    assert str(excinfo.value) == 'encoder busy'


def test_state_disconnect_aborts_pending_commands() -> None:
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
        features={
            'call.command': FeatureToggle(enabled=True, version=1, commands=('napari.pixel.request_keyframe',)),
        },
    )
    loop._on_state_session_ready(metadata)

    future = loop.request_keyframe(origin='test')
    assert future is not None
    assert not future.done()

    loop._on_state_disconnect(RuntimeError('disconnect'))

    assert future.done()
    with pytest.raises(CommandError) as excinfo:
        future.result()
    assert excinfo.value.code == 'command.session_closed'
