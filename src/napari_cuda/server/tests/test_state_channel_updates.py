from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, Coroutine, List

import time
import pytest

from napari_cuda.protocol import (
    PROTO_VERSION,
    FeatureToggle,
    NotifyStreamPayload,
    build_call_command,
    build_state_update,
)
from napari_cuda.protocol.envelopes import build_session_hello
from napari_cuda.protocol.messages import HelloClientInfo, NotifyDimsPayload
from napari_cuda.server.control.resumable_history_store import (
    EnvelopeSnapshot,
    ResumableHistoryStore,
    ResumableRetention,
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.command_registry import COMMAND_REGISTRY

from napari_cuda.server.control import control_channel_server as state_channel_handler
from napari_cuda.server.state.layer_manager import ViewerSceneManager
from napari_cuda.server.rendering.viewer_builder import canonical_axes_from_source
from napari_cuda.server.runtime.runtime_mailbox import RenderDelta
from napari_cuda.server.state.server_scene import create_server_scene_data
from napari_cuda.server.control.state_update_engine import apply_layer_state_update
from napari_cuda.server.runtime.worker_notifications import WorkerSceneNotification


class _CaptureWorker:
    def __init__(self) -> None:
        self.deltas: list[RenderDelta] = []
        self.policy_calls: list[str] = []
        self.level_requests: list[tuple[int, Any]] = []
        self.force_idr_calls = 0
        self.use_volume = False
        self._is_ready = True
        self._data_wh = (640, 480)
        self._zarr_axes = "yx"
        self._zarr_shape = (480, 640)
        self._zarr_dtype = "float32"
        self.volume_dtype = "float32"
        self._active_ms_level = 0
        self._plane_restore_state = None
        self._last_step = (0, 0)
        self._scene_source = None
        self._canonical_axes = canonical_axes_from_source(
            axes=("y", "x"),
            shape=(self._zarr_shape[0], self._zarr_shape[1]),
            step=(0, 0),
            use_volume=False,
        )

    def enqueue_update(self, delta: RenderDelta) -> None:
        self.deltas.append(delta)

    def set_policy(self, policy: str) -> None:
        self.policy_calls.append(str(policy))

    def request_multiscale_level(self, level: int, path: Any) -> None:
        self.level_requests.append((int(level), path))

    def force_idr(self) -> None:
        self.force_idr_calls += 1

    def viewer_model(self) -> None:
        return None

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class _FakeWS(SimpleNamespace):
    __hash__ = object.__hash__


def _make_server() -> tuple[SimpleNamespace, List[Coroutine[Any, Any, None]], List[dict[str, Any]]]:
    scene = create_server_scene_data()
    manager = ViewerSceneManager((640, 480))
    worker = _CaptureWorker()
    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        scene_source=None,
        layer_controls=None,
    )

    server = SimpleNamespace()
    server._scene = scene
    server._scene_manager = manager
    server._state_lock = threading.RLock()
    server._worker = worker
    server._log_state_traces = False
    server._log_dims_info = False
    server._log_cam_info = False
    server._log_cam_debug = False
    server._allowed_render_modes = {"mip"}
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)
    server.width = 640
    server.height = 480
    server.cfg = SimpleNamespace(fps=60.0)
    server.use_volume = False
    captured: list[dict[str, Any]] = []

    async def _state_send(_ws: Any, text: str) -> None:
        captured.append(json.loads(text))

    server._state_send = _state_send

    features = {
        "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
        "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
        "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
        "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
        "notify.camera": FeatureToggle(enabled=True, version=1, resume=False),
        "call.command": FeatureToggle(
            enabled=True,
            version=1,
            resume=False,
            commands=COMMAND_REGISTRY.command_names(),
        ),
    }

    fake_ws = _FakeWS(
        _napari_cuda_session="test-session",
        _napari_cuda_features=features,
        _napari_cuda_sequencers={},
        _napari_cuda_resume_plan={},
    )

    server._state_clients = {fake_ws}
    server._scene.last_dims_payload = NotifyDimsPayload.from_dict(
        {
            "step": [0, 0, 0],
            "current_step": [0, 0, 0],
            "level_shapes": [[10, 10, 10]],
            "axis_labels": ["z", "y", "x"],
            "order": [0, 1, 2],
            "displayed": [0, 1, 2],
            "ndisplay": 2,
            "mode": "plane",
            "levels": [{"index": 0, "shape": [10, 10, 10]}],
            "current_level": 0,
        }
    )
    server._pixel = SimpleNamespace(bypass_until_key=False)

    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)

    def _update_scene_manager() -> None:
        return

    server._update_scene_manager = _update_scene_manager

    server._ndisplay_calls: list[int] = []

    async def _handle_set_ndisplay(ndisplay: int) -> None:
        value = 3 if int(ndisplay) >= 3 else 2
        server._ndisplay_calls.append(value)
        server.use_volume = bool(value == 3)
        server._scene.use_volume = bool(value == 3)

    server._handle_set_ndisplay = _handle_set_ndisplay

    server._resumable_store = ResumableHistoryStore(
        {
            "notify.scene": ResumableRetention(),
            "notify.layers": ResumableRetention(min_deltas=512, max_deltas=2048, max_age_s=300.0),
            "notify.stream": ResumableRetention(min_deltas=1, max_deltas=32),
        }
    )
    server._ensure_keyframe_calls = 0

    async def _ensure_keyframe() -> None:
        server._ensure_keyframe_calls += 1

    server._ensure_keyframe = _ensure_keyframe
    server._idr_on_reset = True

    def _enqueue_camera_command(cmd: Any) -> None:
        scene.camera_commands.append(cmd)

    server._enqueue_camera_command = _enqueue_camera_command

    return server, scheduled, captured


def _drain_scheduled(tasks: list[Coroutine[Any, Any, None]]) -> None:
    while tasks:
        coro = tasks.pop(0)
        asyncio.run(coro)


def _frames_of_type(frames: list[dict[str, Any]], frame_type: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == frame_type]

def _make_dims_snapshot(**overrides: Any) -> dict[str, Any]:
    base = {
        "step": [0, 0, 0],
        "current_step": [0, 0, 0],
        "axis_labels": ["z", "y", "x"],
        "order": [0, 1, 2],
        "displayed": [1, 2],
        "ndisplay": 2,
        "mode": "plane",
        "levels": [
            {"index": 0, "shape": [512, 256, 64]},
            {"index": 1, "shape": [256, 128, 32]},
        ],
        "current_level": 0,
        "level_shapes": [[512, 256, 64], [256, 128, 32]],
    }
    base.update(overrides)
    return base



def _build_state_update(payload: dict[str, Any], *, intent_id: str, frame_id: str) -> dict[str, Any]:
    frame = build_state_update(
        session_id="test-session",
        intent_id=intent_id,
        frame_id=frame_id,
        payload=payload,
    )
    return frame.to_dict()


def test_layer_update_emits_ack_and_notify() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "layer",
        "target": "layer-0",
        "key": "colormap",
        "value": "red",
    }

    frame = _build_state_update(payload, intent_id="layer-intent", frame_id="state-layer-1")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    assert server._worker.deltas, "expected worker delta"
    snapshot = server._worker.deltas[-1].scene_state
    assert snapshot is not None
    assert snapshot.layer_updates == {"layer-0": {"colormap": "red"}}

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "layer-intent",
        "in_reply_to": "state-layer-1",
        "status": "accepted",
        "applied_value": "red",
    }

    layer_frames = _frames_of_type(captured, "notify.layers")
    assert layer_frames, "expected notify.layers frame"
    notify_payload = layer_frames[-1]["payload"]
    assert notify_payload["layer_id"] == "layer-0"
    assert notify_payload["changes"]["colormap"] == "red"


def test_call_command_requests_keyframe() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-test",
        payload={"command": "napari.pixel.request_keyframe"},
    )

    asyncio.run(state_channel_handler._handle_call_command(server, frame.to_dict(), fake_ws))

    assert server._ensure_keyframe_calls == 1
    replies = _frames_of_type(captured, "reply.command")
    assert replies, "expected reply.command frame"
    payload = replies[-1]["payload"]
    assert payload["status"] == "ok"
    assert payload["in_reply_to"] == "cmd-test"


def test_call_command_unknown_command_errors() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-missing",
        payload={"command": "napari.pixel.unknown"},
    )

    asyncio.run(state_channel_handler._handle_call_command(server, frame.to_dict(), fake_ws))

    errors = _frames_of_type(captured, "error.command")
    assert errors, "expected error.command frame"
    payload = errors[-1]["payload"]
    assert payload["status"] == "error"
    assert payload["in_reply_to"] == "cmd-missing"
    assert payload["code"] == "command.not_found"


def test_call_command_rejection_propagates_error() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    async def _reject_keyframe() -> None:
        raise state_channel_handler.CommandRejected(
            code="command.busy",
            message="encoder busy",
            details={"retry_after_ms": 250},
        )

    server._ensure_keyframe = _reject_keyframe

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-reject",
        payload={"command": "napari.pixel.request_keyframe"},
    )

    asyncio.run(state_channel_handler._handle_call_command(server, frame.to_dict(), fake_ws))

    errors = _frames_of_type(captured, "error.command")
    assert errors, "expected error.command frame"
    payload = errors[-1]["payload"]
    assert payload["status"] == "error"
    assert payload["code"] == "command.busy"
    assert payload["message"] == "encoder busy"
    assert payload["details"] == {"retry_after_ms": 250}


def test_layer_update_rejects_unknown_key() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "layer",
        "target": "layer-0",
        "key": "unknown",
        "value": "noop",
    }

    frame = _build_state_update(payload, intent_id="bad-layer", frame_id="state-layer-err")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["status"] == "rejected"
    assert ack_payload["intent_id"] == "bad-layer"
    assert ack_payload["in_reply_to"] == "state-layer-err"
    assert ack_payload["error"]["code"] == "state.invalid"


def test_dims_update_emits_ack_and_notify() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "dims",
        "target": "z",
        "key": "step",
        "value": 5,
    }

    frame = _build_state_update(payload, intent_id="dims-intent", frame_id="state-dims-1")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    assert server._scene.latest_state.current_step[0] == 5
    assert server._scene.last_dims_payload is not None
    assert server._scene.last_dims_payload.current_step[0] == 0

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "dims-intent",
        "in_reply_to": "state-dims-1",
        "status": "accepted",
        "applied_value": 5,
    }

    assert not _frames_of_type(captured, "notify.dims")


def test_view_ndisplay_update_ack_is_immediate() -> None:
    server, scheduled, captured = _make_server()

    captured.clear()

    payload = {
        "scope": "view",
        "target": "main",
        "key": "ndisplay",
        "value": 3,
    }

    frame = _build_state_update(payload, intent_id="view-intent", frame_id="state-view-1")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    assert server._ndisplay_calls == [3]
    assert server.use_volume is True
    assert server._scene.last_dims_payload is not None
    assert server._scene.last_dims_payload.ndisplay == 2
    assert server._scene.last_dims_payload.mode == 'plane'

    acks = _frames_of_type(captured, "ack.state")
    assert acks, "expected immediate ack"
    ack_payload = acks[-1]["payload"]
    assert ack_payload == {
        "intent_id": "view-intent",
        "in_reply_to": "state-view-1",
        "status": "accepted",
        "applied_value": 3,
    }

    assert not _frames_of_type(captured, "notify.dims")

    notify_payload = NotifyDimsPayload.from_dict(
        _make_dims_snapshot(
            step=[0, 0, 0],
            current_step=[0, 0, 0],
            level_shapes=[[512, 256, 64], [8, 8, 8]],
            ndisplay=3,
            mode="volume",
            displayed=[0, 1, 2],
            current_level=1,
        )
    )
    state_channel_handler.process_worker_notifications(
        server,
        [
            WorkerSceneNotification(
                kind="dims_snapshot",
                seq=1,
                payload=notify_payload,
                timestamp=time.time(),
            )
        ],
    )
    _drain_scheduled(scheduled)

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames, "expected dims broadcast after worker refresh"
    dims_payload = dims_frames[-1]["payload"]
    assert dims_payload["ndisplay"] == 3
    assert dims_payload["mode"] == "volume"
    assert server._scene.last_dims_payload is not None
    assert server._scene.last_dims_payload.ndisplay == 3
    assert server._scene.last_dims_payload.mode == "volume"




def test_worker_dims_snapshot_updates_multiscale_state() -> None:
    server, scheduled, captured = _make_server()
    captured.clear()

    snapshot_payload = _make_dims_snapshot(
        level_shapes=[[512, 256, 64], [256, 128, 32]],
        displayed=[0, 1, 2],
        current_level=1,
        downgraded=True,
    )
    snapshot_payload["levels"] = [
        {"index": 0, "shape": [512, 256, 64]},
        {"index": 1, "shape": [256, 128, 32]},
    ]
    notify_payload = NotifyDimsPayload.from_dict(snapshot_payload)

    state_channel_handler.process_worker_notifications(
        server,
        [
            WorkerSceneNotification(
                kind="dims_snapshot",
                seq=5,
                payload=notify_payload,
                timestamp=time.time(),
            )
        ],
    )
    _drain_scheduled(scheduled)

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames, "expected dims broadcast"
    payload = dims_frames[-1]["payload"]
    assert payload["current_level"] == 1
    assert payload.get("downgraded") is True

    assert server._scene.last_dims_payload is not None
    assert server._scene.last_dims_payload.level_shapes[1] == (256, 128, 32)

    ms_state = server._scene.multiscale_state
    assert ms_state.get("current_level") == 1
    assert ms_state.get("levels")[1]["shape"] == [256, 128, 32]
    assert ms_state.get("downgraded") is True



def test_volume_render_mode_update() -> None:
    server, scheduled, captured = _make_server()
    server._allowed_render_modes = {"mip", "iso"}

    payload = {
        "scope": "volume",
        "target": "main",
        "key": "render_mode",
        "value": "iso",
    }

    frame = _build_state_update(payload, intent_id="volume-intent", frame_id="state-volume-1")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    assert server._scene.volume_state.get("mode") == "iso"

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "volume-intent",
        "in_reply_to": "state-volume-1",
        "status": "accepted",
        "applied_value": "iso",
    }

    layer_frames = _frames_of_type(captured, "notify.layers")
    assert layer_frames
    payload = layer_frames[-1]["payload"]
    assert payload["changes"].get("volume.render_mode") == "iso"


def test_camera_zoom_update_emits_notify() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "zoom",
        "value": {"factor": 1.2, "anchor_px": [12, 24]},
    }

    frame = _build_state_update(payload, intent_id="cam-zoom", frame_id="state-cam-1")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    commands = server._scene.camera_commands
    assert commands, "expected camera command queued"
    cmd = commands[-1]
    assert cmd.kind == 'zoom'
    assert cmd.factor == 1.2
    assert cmd.anchor_px == (12.0, 24.0)

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "cam-zoom",
        "in_reply_to": "state-cam-1",
        "status": "accepted",
        "applied_value": {"factor": 1.2, "anchor_px": [12.0, 24.0]},
    }

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames, "expected notify.camera frame"
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "zoom"
    assert camera_payload["delta"]["factor"] == 1.2
    assert camera_payload["delta"]["anchor_px"] == [12.0, 24.0]
    assert camera_payload["origin"] == "state.update"


def test_camera_reset_triggers_keyframe() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "reset",
        "value": {"reason": "ui"},
    }

    frame = _build_state_update(payload, intent_id="cam-reset", frame_id="state-cam-2")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    commands = server._scene.camera_commands
    assert commands and commands[-1].kind == 'reset'
    assert server._ensure_keyframe_calls == 1

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "cam-reset"
    assert ack_payload["applied_value"] == {"reason": "ui"}

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "reset"
    assert camera_payload["delta"] == {"reason": "ui"}


def test_camera_set_updates_latest_state() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "set",
        "value": {
            "center": [1, 2, 3],
            "zoom": 2.5,
            "angles": [0.1, 0.2, 0.3],
        },
    }

    frame = _build_state_update(payload, intent_id="cam-set", frame_id="state-cam-3")

    asyncio.run(state_channel_handler._handle_state_update(server, frame, None))
    _drain_scheduled(scheduled)

    latest = server._scene.latest_state
    assert latest.center == (1.0, 2.0, 3.0)
    assert latest.zoom == 2.5
    assert latest.angles == (0.1, 0.2, 0.3)

    acks = _frames_of_type(captured, "ack.state")
    assert acks
    applied = acks[-1]["payload"]["applied_value"]
    assert applied == {
        "center": [1.0, 2.0, 3.0],
        "zoom": 2.5,
        "angles": [0.1, 0.2, 0.3],
    }

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "set"
    assert camera_payload["delta"] == applied


def test_parse_failure_emits_rejection_ack() -> None:
    server, scheduled, captured = _make_server()

    bad_frame = {
        "type": "state.update",
        "version": PROTO_VERSION,
        "session": "test-session",
        "frame_id": "state-bad-1",
        "timestamp": 0.0,
        "intent_id": "bad-intent",
        "payload": {"scope": "layer", "key": "opacity"},
    }

    asyncio.run(state_channel_handler._handle_state_update(server, bad_frame, None))
    _drain_scheduled(scheduled)

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["status"] == "rejected"
    assert ack_payload["intent_id"] == "bad-intent"
    assert ack_payload["in_reply_to"] == "state-bad-1"
    assert ack_payload["error"]["code"] == "state.invalid"


def test_send_state_baseline_emits_notifications(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        server, scheduled, captured = _make_server()
        server._scene.use_volume = False
        apply_layer_state_update(
            server._scene,
            server._state_lock,
            layer_id="layer-0",
            prop="opacity",
            value=0.25,
        )
        apply_layer_state_update(
            server._scene,
            server._state_lock,
            layer_id="layer-0",
            prop="colormap",
            value="viridis",
        )

        server._scene_manager.update_from_sources(
            worker=server._worker,
            scene_state=server._scene.latest_state,
            multiscale_state=None,
            volume_state=None,
            current_step=None,
            ndisplay=2,
            zarr_path=None,
            scene_source=None,
            layer_controls=server._scene.layer_controls,
        )

        server._await_worker_bootstrap = lambda _timeout: asyncio.sleep(0)
        server._state_send = lambda _ws, text: captured.append(json.loads(text))
        server._update_scene_manager = lambda: None
        server._schedule_coro = lambda coro, _label: scheduled.append(coro)
        server._pixel_channel = SimpleNamespace(last_avcc=None)
        server._pixel_config = SimpleNamespace()

        monkeypatch.setattr(state_channel_handler.pixel_channel, 'mark_stream_config_dirty', lambda *a, **k: None)

        class _CaptureWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, payload: str) -> None:
                self.sent.append(payload)

        ws = _CaptureWS()
        ws._napari_cuda_session = "baseline-session"
        ws._napari_cuda_features = {
            "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
        }
        ws._napari_cuda_sequencers = {}
        ws._napari_cuda_resume_plan = {}

        loop.run_until_complete(state_channel_handler._send_state_baseline(server, ws))
        _drain_scheduled(scheduled)

        scene_frames = _frames_of_type(captured, "notify.scene")
        assert scene_frames, "expected notify.scene snapshot"

        layer_frames = _frames_of_type(captured, "notify.layers")
        assert layer_frames, "expected notify.layers baseline"
        changes = layer_frames[-1]["payload"]["changes"]
        assert changes["opacity"] == 0.25
        assert changes["colormap"] == "viridis"

        dims_frames = _frames_of_type(captured, "notify.dims")
        assert dims_frames, "expected notify.dims baseline"
        dims_payload = dims_frames[-1]["payload"]
        assert dims_payload["step"] == list(server._scene.last_dims_payload.current_step)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _prepare_resumable_payloads(server: Any) -> tuple[EnvelopeSnapshot, EnvelopeSnapshot, EnvelopeSnapshot]:
    store: ResumableHistoryStore = server._resumable_store
    scene_payload = state_channel_handler.build_notify_scene_payload(
        server._scene,
        server._scene_manager,
        viewer_settings={"fps_target": 60.0},
    )
    scene_snapshot = store.snapshot_envelope(
        state_channel_handler.NOTIFY_SCENE_TYPE,
        payload=scene_payload.to_dict(),
        timestamp=time.time(),
    )

    layer_payload = state_channel_handler.build_notify_layers_payload(
        layer_id="layer-0",
        changes={"opacity": 0.42},
    )
    layer_snapshot = store.delta_envelope(
        state_channel_handler.NOTIFY_LAYERS_TYPE,
        payload=layer_payload.to_dict(),
        timestamp=time.time(),
    )

    stream_payload = NotifyStreamPayload(
        codec="h264",
        format="annexb",
        fps=60.0,
        frame_size=(640, 480),
        nal_length_size=4,
        avcc="AAAA",
        latency_policy={"mode": "low"},
    )
    stream_snapshot = store.snapshot_envelope(
        state_channel_handler.NOTIFY_STREAM_TYPE,
        payload=stream_payload.to_dict(),
        timestamp=time.time(),
    )

    return scene_snapshot, layer_snapshot, stream_snapshot


def test_handshake_stashes_resume_plan(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        server, scheduled, captured = _make_server()
        scene_snap, layer_snap, stream_snap = _prepare_resumable_payloads(server)
        resume_tokens = {
            state_channel_handler.NOTIFY_SCENE_TYPE: scene_snap.delta_token,
            state_channel_handler.NOTIFY_LAYERS_TYPE: layer_snap.delta_token,
            state_channel_handler.NOTIFY_STREAM_TYPE: stream_snap.delta_token,
        }

        hello = build_session_hello(
            client=HelloClientInfo(name="tests", version="1.0", platform="test"),
            features={
                state_channel_handler.NOTIFY_SCENE_TYPE: True,
                state_channel_handler.NOTIFY_LAYERS_TYPE: True,
                state_channel_handler.NOTIFY_STREAM_TYPE: True,
            },
            resume_tokens=resume_tokens,
        )

        hello_json = json.dumps(hello.to_dict(), separators=(",", ":"))

        class _WS:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.transport = SimpleNamespace(get_extra_info=lambda *_: None)

            async def recv(self) -> str:
                return hello_json

            async def close(self) -> None:  # pragma: no cover - defensive
                pass

        ws = _WS()

        async def _state_send(_ws: Any, text: str) -> None:
            captured.append(json.loads(text))

        server._state_send = _state_send

        loop.run_until_complete(state_channel_handler._perform_state_handshake(server, ws))
        _drain_scheduled(scheduled)

        plan = getattr(ws, "_napari_cuda_resume_plan")
        assert plan[state_channel_handler.NOTIFY_SCENE_TYPE].decision == ResumeDecision.REPLAY
        assert plan[state_channel_handler.NOTIFY_LAYERS_TYPE].decision == ResumeDecision.REPLAY
        assert plan[state_channel_handler.NOTIFY_STREAM_TYPE].decision == ResumeDecision.REPLAY

        welcome = _frames_of_type(captured, "session.welcome")
        assert welcome, "expected handshake welcome frame"
        command_feature = welcome[0]["payload"]["features"].get("call.command")
        assert command_feature is not None
        assert command_feature.get("commands") == ["napari.pixel.request_keyframe"]
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_send_state_baseline_replays_store(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        server, scheduled, captured = _make_server()
        scene_snap, layer_snap, stream_snap = _prepare_resumable_payloads(server)

        server._await_worker_bootstrap = lambda _timeout: asyncio.sleep(0)
        server._state_send = lambda _ws, text: captured.append(json.loads(text))
        server._pixel_channel = SimpleNamespace(last_avcc=b"avcc")
        server._pixel_config = SimpleNamespace()

        ws = _FakeWS(
            _napari_cuda_session="resume-session",
            _napari_cuda_features={
                "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
            },
            _napari_cuda_sequencers={},
        )
        ws._napari_cuda_resume_plan = {
            state_channel_handler.NOTIFY_SCENE_TYPE: ResumePlan(
                topic=state_channel_handler.NOTIFY_SCENE_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[scene_snap],
            ),
            state_channel_handler.NOTIFY_LAYERS_TYPE: ResumePlan(
                topic=state_channel_handler.NOTIFY_LAYERS_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[layer_snap],
            ),
            state_channel_handler.NOTIFY_STREAM_TYPE: ResumePlan(
                topic=state_channel_handler.NOTIFY_STREAM_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[stream_snap],
            ),
        }

        loop.run_until_complete(state_channel_handler._send_state_baseline(server, ws))
        _drain_scheduled(scheduled)

        scene_frames = _frames_of_type(captured, "notify.scene")
        assert scene_frames and scene_frames[-1]["seq"] == scene_snap.seq

        layer_frames = _frames_of_type(captured, "notify.layers")
        assert layer_frames and layer_frames[-1]["seq"] == layer_snap.seq

        stream_frames = _frames_of_type(captured, "notify.stream")
        assert stream_frames and stream_frames[-1]["seq"] == stream_snap.seq
    finally:
        asyncio.set_event_loop(None)
        loop.close()
