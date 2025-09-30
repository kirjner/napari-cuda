from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, Coroutine, List

from napari_cuda.protocol import PROTO_VERSION, FeatureToggle, build_state_update

from napari_cuda.server.control import control_channel_server as state_channel_handler
from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.render_mailbox import RenderDelta
from napari_cuda.server.server_scene import create_server_scene_data
from napari_cuda.server.control.state_update_engine import apply_layer_state_update


class _CaptureWorker:
    def __init__(self) -> None:
        self.deltas: list[RenderDelta] = []
        self.policy_calls: list[str] = []
        self.level_requests: list[tuple[int, Any]] = []
        self.force_idr_calls = 0

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


class _FakeWS(SimpleNamespace):
    __hash__ = object.__hash__


def _make_server() -> tuple[SimpleNamespace, List[Coroutine[Any, Any, None]], List[dict[str, Any]]]:
    scene = create_server_scene_data()
    manager = ViewerSceneManager((640, 480))
    manager.update_from_sources(
        worker=None,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        layer_controls=None,
    )

    server = SimpleNamespace()
    server._scene = scene
    server._scene_manager = manager
    server._state_lock = threading.RLock()
    server._worker = _CaptureWorker()
    server._log_state_traces = False
    server._log_dims_info = False
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
    }

    fake_ws = _FakeWS(
        _napari_cuda_session="test-session",
        _napari_cuda_features=features,
        _napari_cuda_sequencers={},
    )

    server._state_clients = {fake_ws}
    server._dims_metadata = lambda: {"ndim": 3, "order": ["z", "y", "x"], "range": [[0, 9], [0, 9], [0, 9]]}
    server._pixel = SimpleNamespace(bypass_until_key=False)

    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)

    server._ndisplay_calls: list[int] = []

    async def _handle_set_ndisplay(ndisplay: int) -> None:
        value = 3 if int(ndisplay) >= 3 else 2
        server._ndisplay_calls.append(value)
        server.use_volume = bool(value == 3)
        server._scene.use_volume = bool(value == 3)

    server._handle_set_ndisplay = _handle_set_ndisplay

    return server, scheduled, captured


def _drain_scheduled(tasks: list[Coroutine[Any, Any, None]]) -> None:
    while tasks:
        coro = tasks.pop(0)
        asyncio.run(coro)


def _frames_of_type(frames: list[dict[str, Any]], frame_type: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == frame_type]


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

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "dims-intent",
        "in_reply_to": "state-dims-1",
        "status": "accepted",
        "applied_value": 5,
    }

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames, "expected notify.dims frame"
    notify_payload = dims_frames[-1]["payload"]
    assert notify_payload["current_step"][0] == 5
    assert notify_payload["source"] == "state.update"


def test_view_ndisplay_update() -> None:
    server, scheduled, captured = _make_server()

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

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload == {
        "intent_id": "view-intent",
        "in_reply_to": "state-view-1",
        "status": "accepted",
        "applied_value": 3,
    }

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames
    dims_payload = dims_frames[-1]["payload"]
    assert dims_payload["ndisplay"] == 3
    assert dims_payload["mode"] == "volume"


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
            worker=None,
            scene_state=server._scene.latest_state,
            multiscale_state=None,
            volume_state=None,
            current_step=None,
            ndisplay=2,
            zarr_path=None,
            layer_controls=server._scene.layer_controls,
        )

        server._await_adapter_level_ready = lambda _timeout: asyncio.sleep(0)
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
        assert dims_frames[-1]["payload"]["source"] == "server.bootstrap"
    finally:
        asyncio.set_event_loop(None)
        loop.close()
