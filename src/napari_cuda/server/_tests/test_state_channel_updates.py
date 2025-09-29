from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, Coroutine, List

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
    server._state_clients = set()
    server._dims_metadata = lambda: {"ndim": 3, "order": ["z", "y", "x"], "range": [[0, 9], [0, 9], [0, 9]]}
    server._pixel = SimpleNamespace(bypass_until_key=False)

    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]

    captured: list[dict[str, Any]] = []

    async def _broadcast_state_json(payload: dict[str, Any]) -> None:
        captured.append(payload)

    server._broadcast_state_json = _broadcast_state_json

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


def _unwrap_notify(payload: dict[str, Any]) -> dict[str, Any]:
    msg_type = payload.get("type") if isinstance(payload, dict) else None
    if msg_type in {"notify.state", "notify.scene", "notify.stream"}:
        inner = payload.get("payload") if isinstance(payload, dict) else None
        if isinstance(inner, dict):
            return inner
    return payload


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
    assert captured, "expected broadcast invocation"
    event = captured[0]
    assert event["type"] == "state.update"
    assert event["scope"] == "layer"
    assert event["target"] == "layer-0"
    versions = event.get("control_versions") or {}
    assert versions.get("colormap", {}).get("server_seq") == event["server_seq"]


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
    assert len(captured) == 1
    event = captured[0]
    assert event["interaction_id"] == "drag-1"
    assert event["phase"] == "update"
    # metadata should remain linked to first payload
    versions = event.get("control_versions") or {}
    assert versions.get("gamma", {}).get("source_client_seq") == 10


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
    assert captured, "expected dims broadcast"
    event = [p for p in captured if p["scope"] == "dims"][0]
    assert event["value"] == 5
    versions = event.get("control_versions") or {}
    assert versions.get("step", {}).get("source_client_seq") == 3
    assert event["interaction_id"] == "dims-1"
    assert event["phase"] == "commit"
    assert isinstance(event.get("meta"), dict)
    assert event.get("ack") is True
    assert event.get("axis_index") == 0
    assert event.get("current_step")[:1] == [5]


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
    assert captured, "expected view broadcast"
    event = [p for p in captured if p["scope"] == "view"][0]
    assert event["target"] == "main"
    assert event["key"] == "ndisplay"
    assert event["value"] == 3
    assert event.get("ack") is True
    assert event.get("intent_seq") == 42
    versions = event.get("control_versions") or {}
    assert versions.get("ndisplay", {}).get("server_seq") == event["server_seq"]


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

    notifications = [_unwrap_notify(evt) for evt in captured]
    volume_events = [evt for evt in notifications if evt.get("scope") == "volume"]
    assert volume_events, "expected volume state update broadcast"
    event = volume_events[-1]
    assert event["key"] == "render_mode"
    assert event["value"] == "iso"


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

    notifications = [_unwrap_notify(evt) for evt in captured]
    ms_events = [evt for evt in notifications if evt.get("scope") == "multiscale"]
    assert ms_events, "expected multiscale state update broadcast"
    event = ms_events[-1]
    assert event["key"] == "level"
    assert event["value"] == 1


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

        monkeypatch.setattr(state_channel_handler.pixel_channel, 'build_video_config_payload', lambda *a, **k: {'type': 'video.config'})
        monkeypatch.setattr(state_channel_handler.pixel_channel, 'mark_config_dirty', lambda *a, **k: None)

        class _CaptureWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, payload: str) -> None:
                self.sent.append(payload)

        ws = _CaptureWS()

        loop.run_until_complete(state_channel_handler._send_state_baseline(server, ws))
        _drain_scheduled(scheduled)

        payloads = captured + [json.loads(data) for data in ws.sent]
        decoded = [_unwrap_notify(p) for p in payloads]
        updates = [p for p in decoded if p.get("type") == "state.update" and p.get("scope") == "layer"]
        assert updates, "expected at least one layer state.update payload"
        merged = {}
        for entry in updates:
            merged.update(entry.get("controls") or {})
        assert merged.get("opacity") == 0.25
        assert merged.get("colormap") == "viridis"
    finally:
        asyncio.set_event_loop(None)
        loop.close()
