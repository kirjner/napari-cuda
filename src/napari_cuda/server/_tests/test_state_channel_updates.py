from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

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
    server._state_lock = threading.Lock()
    server._worker = _CaptureWorker()
    server._log_state_traces = False
    server._allowed_render_modes = {"mip"}
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)

    # Schedule helper closes the coroutine immediately so the handler can proceed.
    server._schedule_coro = lambda coro, _label: coro.close()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        state_channel_handler,
        "broadcast_layer_update",
        _noop_broadcast,
    )

    payload = {
        "type": "layer.intent.set_colormap",
        "layer_id": "layer-0",
        "name": "red",
    }

    asyncio.run(state_channel_handler._handle_layer_intent(server, payload, None))

    assert server._worker.deltas, "worker should receive at least one delta"
    snapshot = server._worker.deltas[-1].scene_state
    assert snapshot is not None
    assert snapshot.layer_updates == {"layer-0": {"colormap": "red"}}



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
        from napari_cuda.server.server_scene import LayerControlState
        from types import SimpleNamespace
        import threading
        import json

        server = SimpleNamespace()
        server._scene = create_server_scene_data()
        server._scene.use_volume = False
        server._scene.layer_controls['layer-0'] = LayerControlState(opacity=0.25, colormap='viridis')
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
        server._state_lock = threading.Lock()
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
    finally:
        asyncio.set_event_loop(None)
        loop.close()
