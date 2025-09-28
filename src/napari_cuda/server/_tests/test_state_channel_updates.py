from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Coroutine

from napari_cuda.protocol.messages import CONTROL_COMMAND_TYPE
from napari_cuda.server import server_scene_intents as intents
from napari_cuda.server import state_channel_handler
from napari_cuda.server.render_mailbox import RenderDelta
from napari_cuda.server.server_scene import create_server_scene_data


class _CaptureWorker:
    def __init__(self) -> None:
        self.deltas: list[RenderDelta] = []

    def enqueue_update(self, delta: RenderDelta) -> None:
        self.deltas.append(delta)


async def _noop_broadcast(*_args, **_kwargs) -> None:
    return None


def test_layer_intent_pushes_scene_state(monkeypatch) -> None:
    server = SimpleNamespace()
    server._scene = create_server_scene_data()
    server._state_lock = threading.RLock()
    server._worker = _CaptureWorker()
    server._log_state_traces = False
    server._allowed_render_modes = {"mip"}
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)

    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]

    captured: list[dict[str, Any]] = []

    async def _capture_broadcast(_server, **kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(
        state_channel_handler,
        "broadcast_layer_update",
        _capture_broadcast,
    )

    payload = {
        "type": CONTROL_COMMAND_TYPE,
        "scope": "layer",
        "target": "layer-0",
        "prop": "colormap",
        "value": "red",
    }

    asyncio.run(state_channel_handler._handle_control_command(server, payload, None))
    for coro in scheduled:
        asyncio.run(coro)
    scheduled.clear()

    assert server._worker.deltas, "worker should receive at least one delta"
    snapshot = server._worker.deltas[-1].scene_state
    assert snapshot is not None
    assert snapshot.layer_updates == {"layer-0": {"colormap": "red"}}
    assert captured, "expected broadcast invocation"
    meta = captured[0]
    assert meta["server_seq"] == 1
    assert meta["source_client_seq"] is None
    assert meta["layer_id"] == "layer-0"
    assert meta.get("interaction_id") is None


def test_layer_intent_stale_seq_ignored(monkeypatch) -> None:
    server = SimpleNamespace()
    server._scene = create_server_scene_data()
    server._state_lock = threading.RLock()
    server._worker = _CaptureWorker()
    server._log_state_traces = False
    server._allowed_render_modes = {"mip"}
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)
    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]

    captured: list[dict[str, Any]] = []

    async def _capture_broadcast(_server, **kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(state_channel_handler, "broadcast_layer_update", _capture_broadcast)

    payload = {
        "type": CONTROL_COMMAND_TYPE,
        "scope": "layer",
        "target": "layer-0",
        "prop": "gamma",
        "value": 1.3,
        "client_id": "client-a",
        "client_seq": 10,
        "interaction_id": "drag-1",
        "phase": "update",
    }

    asyncio.run(state_channel_handler._handle_control_command(server, payload, None))
    for coro in scheduled:
        asyncio.run(coro)
    scheduled.clear()

    stale_payload = dict(payload)
    stale_payload["value"] = 0.8
    stale_payload["client_seq"] = 8

    asyncio.run(state_channel_handler._handle_control_command(server, stale_payload, None))
    for coro in scheduled:
        asyncio.run(coro)
    scheduled.clear()

    assert len(server._worker.deltas) == 1
    assert len(captured) == 1
    assert captured[0]["server_seq"] == 1
    assert captured[0]["interaction_id"] == "drag-1"
    assert captured[0]["phase"] == "update"


def test_control_command_dims_updates(monkeypatch) -> None:
    server = SimpleNamespace()
    server._scene = create_server_scene_data()
    server._state_lock = threading.RLock()
    server._worker = _CaptureWorker()
    server._log_state_traces = False
    server._dims_metadata = lambda: {"ndim": 3, "order": ["z", "y", "x"], "range": [[0, 9], [0, 9], [0, 9]]}
    scheduled: list[Coroutine[Any, Any, None]] = []
    server._schedule_coro = lambda coro, _label: scheduled.append(coro)  # type: ignore[arg-type]
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)

    captured: list[dict[str, Any]] = []

    async def _capture_dims(_server, step_list, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["step_list"] = list(step_list)
        captured.append(kwargs)

    monkeypatch.setattr(state_channel_handler, "broadcast_dims_update", _capture_dims)

    payload = {
        "type": CONTROL_COMMAND_TYPE,
        "scope": "dims",
        "target": "z",
        "prop": "step",
        "value": 5,
        "client_id": "client-z",
        "client_seq": 3,
        "interaction_id": "dims-1",
        "phase": "commit",
    }

    asyncio.run(state_channel_handler._handle_control_command(server, payload, None))

    for coro in scheduled:
        asyncio.run(coro)
    scheduled.clear()

    assert server._scene.latest_state.current_step[0] == 5
    assert captured, "expected dims broadcast"
    entry = captured[0]
    assert entry["step_list"][0] == 5
    assert entry["server_seq"] == 0
    assert entry["source_client_seq"] == 3
    assert entry["interaction_id"] == "dims-1"
    assert entry["phase"] == "commit"
    assert entry["control_prop"] == "step"

async def _noop_level_ready(_timeout: float) -> None:
    return None


class _CaptureWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def test_send_state_baseline_pushes_layer_controls(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        from napari_cuda.server.layer_manager import ViewerSceneManager
        from types import SimpleNamespace
        import threading
        import json

        server = SimpleNamespace()
        server._scene = create_server_scene_data()
        server._scene.use_volume = False
        server._state_lock = threading.RLock()
        result_one = intents.apply_layer_intent(
            server._scene,
            server._state_lock,
            layer_id="layer-0",
            prop="opacity",
            value=0.25,
            client_id="client-a",
            client_seq=1,
        )
        result_two = intents.apply_layer_intent(
            server._scene,
            server._state_lock,
            layer_id="layer-0",
            prop="colormap",
            value="viridis",
            client_id="client-a",
            client_seq=2,
        )
        assert result_one is not None and result_two is not None
        server._scene_manager = ViewerSceneManager((640, 480))
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
        server._dims_metadata = lambda: {'ndim': 2}
        server._await_adapter_level_ready = _noop_level_ready
        server._schedule_coro = lambda coro, _label: None
        server._pixel_channel = SimpleNamespace(last_avcc=None)
        server._pixel_config = SimpleNamespace()
        server._state_clients = set()
        server._log_dims_info = False
        server._log_state_traces = False
        server._log_cam_info = False
        server.metrics = SimpleNamespace(inc=lambda *args, **kwargs: None)

        sent_safe: list[str] = []

        async def _safe_send(_ws, text: str) -> None:
            sent_safe.append(text)

        server._safe_state_send = _safe_send
        server._update_scene_manager = lambda: server._scene_manager.update_from_sources(
            worker=None,
            scene_state=server._scene.latest_state,
            multiscale_state=None,
            volume_state=None,
            current_step=None,
            ndisplay=2,
            zarr_path=None,
            layer_controls=server._scene.layer_controls,
        )

        monkeypatch.setattr(state_channel_handler.pixel_channel, 'build_video_config_payload', lambda *a, **k: {'type': 'video.config'})
        monkeypatch.setattr(state_channel_handler.pixel_channel, 'mark_config_dirty', lambda *a, **k: None)

        ws = _CaptureWS()

        loop.run_until_complete(state_channel_handler._send_state_baseline(server, ws))

        payloads = [json.loads(p) for p in ws.sent + sent_safe]
        layer_updates = [p for p in payloads if isinstance(p, dict) and p.get('type') == 'layer.update']
        assert layer_updates, 'expected at least one layer.update baseline payload'
        first = layer_updates[0]
        controls = first.get('controls') or {}
        assert controls.get('opacity') == 0.25
        assert controls.get('colormap') == 'viridis'
        versions = first.get('control_versions') or {}
        assert versions.get('opacity', {}).get('server_seq') == 1
        assert versions.get('opacity', {}).get('source_client_seq') == 1
        assert versions.get('colormap', {}).get('server_seq') == 2
        assert versions.get('colormap', {}).get('source_client_seq') == 2
    finally:
        asyncio.set_event_loop(None)
        loop.close()
