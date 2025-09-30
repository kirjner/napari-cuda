from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, Coroutine, List
import json

from napari_cuda.protocol import FeatureToggle

from napari_cuda.server import state_channel_handler
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
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]

    server._ndisplay_calls: list[tuple[int, Any, Any]] = []

    async def _handle_set_ndisplay(ndisplay: int, client_id: object, client_seq: object) -> None:
        value = 3 if int(ndisplay) >= 3 else 2
        server._ndisplay_calls.append((value, client_id, client_seq))
        server.use_volume = bool(value == 3)
        server._scene.use_volume = bool(value == 3)

    server._handle_set_ndisplay = _handle_set_ndisplay

    return server, scheduled, captured


def _drain_scheduled(scheduled: list[Coroutine[Any, Any, None]]) -> None:
    while scheduled:
        coro = scheduled.pop(0)
        asyncio.run(coro)


def _frames_of_type(frames: list[dict[str, Any]], frame_type: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == frame_type]


def test_state_update_layer_applies_scene_state() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "type": "state.update",
        "scope": "layer",
        "target": "layer-0",
        "key": "colormap",
        "value": "red",
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    assert server._worker.deltas, "worker should receive at least one delta"
    snapshot = server._worker.deltas[-1].scene_state
    assert snapshot is not None
    assert snapshot.layer_updates == {"layer-0": {"colormap": "red"}}
    frames = _frames_of_type(captured, "notify.layers")
    assert frames, "expected notify.layers frame"
    frame = frames[-1]
    payload = frame["payload"]
    assert payload["layer_id"] == "layer-0"
    assert payload["changes"]["colormap"] == "red"


def test_state_update_layer_stale_seq_ignored() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "type": "state.update",
        "scope": "layer",
        "target": "layer-0",
        "key": "gamma",
        "value": 1.3,
        "client_id": "client-a",
        "client_seq": 10,
        "interaction_id": "drag-1",
        "phase": "update",
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    stale_payload = dict(payload)
    stale_payload["value"] = 0.8
    stale_payload["client_seq"] = 8

    asyncio.run(state_channel_handler._handle_state_update(server, stale_payload, None))
    _drain_scheduled(scheduled)

    assert len(server._worker.deltas) == 1
    frames = _frames_of_type(captured, "notify.layers")
    assert len(frames) == 1
    frame = frames[0]
    payload = frame["payload"]
    assert payload["changes"]["gamma"] == 1.3
    assert frame.get("intent_id") == "drag-1"


def test_state_update_dims_broadcasts() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "type": "state.update",
        "scope": "dims",
        "target": "z",
        "key": "step",
        "value": 5,
        "client_id": "client-z",
        "client_seq": 3,
        "interaction_id": "dims-1",
        "phase": "commit",
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    assert server._scene.latest_state.current_step[0] == 5
    frames = _frames_of_type(captured, "notify.dims")
    assert frames, "expected notify.dims frame"
    frame = frames[-1]
    payload = frame["payload"]
    assert payload["current_step"][0] == 5
    assert payload["source"] == "state.update"
    assert frame.get("intent_id") == "dims-1"


def test_state_update_view_ndisplay_broadcasts() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "type": "state.update",
        "scope": "view",
        "target": "main",
        "key": "ndisplay",
        "value": 3,
        "client_id": "client-view",
        "client_seq": 6,
        "interaction_id": "toggle-1",
        "phase": "commit",
        "intent_seq": 42,
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    assert server._ndisplay_calls == [(3, "client-view", 6)]
    assert server.use_volume is True
    frames = _frames_of_type(captured, "notify.dims")
    assert frames, "expected notify.dims frame"
    frame = frames[-1]
    payload = frame["payload"]
    assert payload["ndisplay"] == 3
    assert payload["mode"] == "volume"
    assert frame.get("intent_id") == "toggle-1"


def test_state_update_volume_render_mode_broadcasts() -> None:
    server, scheduled, captured = _make_server()
    server._allowed_render_modes = {"mip", "iso"}

    payload = {
        "type": "state.update",
        "scope": "volume",
        "target": "main",
        "key": "render_mode",
        "value": "iso",
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    assert server._scene.volume_state.get("mode") == "iso"
    assert server._worker.deltas, "expected worker delta for volume update"
    latest_delta = server._worker.deltas[-1].scene_state
    assert latest_delta is not None
    assert latest_delta.volume_mode == "iso"

    frames = _frames_of_type(captured, "notify.layers")
    assert frames, "expected notify.layers frame"
    frame = frames[-1]
    payload = frame["payload"]
    assert payload["changes"].get("volume.render_mode") == "iso"


def test_state_update_multiscale_level_updates_state() -> None:
    server, scheduled, captured = _make_server()

    payload = {
        "type": "state.update",
        "scope": "multiscale",
        "target": "main",
        "key": "level",
        "value": 1,
    }

    asyncio.run(state_channel_handler._handle_state_update(server, payload, None))
    _drain_scheduled(scheduled)

    assert server._scene.multiscale_state.get("current_level") == 1
    assert server._pixel.bypass_until_key is True
    assert server._worker.level_requests, "expected multiscale level request"
    last_level, _ = server._worker.level_requests[-1]
    assert last_level == 1

    frames = _frames_of_type(captured, "notify.layers")
    assert frames, "expected notify.layers frame"
    frame = frames[-1]
    payload = frame["payload"]
    assert payload["changes"].get("multiscale.level") == 1


def test_send_state_baseline_emits_state_updates(monkeypatch) -> None:
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
            client_id="client-a",
            client_seq=1,
        )
        apply_layer_state_update(
            server._scene,
            server._state_lock,
            layer_id="layer-0",
            prop="colormap",
            value="viridis",
            client_id="client-a",
            client_seq=2,
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
        server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]
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

        frames = _frames_of_type(captured, "notify.scene")
        assert frames, "expected notify.scene snapshot"

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
