"""Unit tests for worker lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.server_scene_queue import WorkerSceneNotificationQueue
from napari_cuda.server.server_scene import create_server_scene_data
from napari_cuda.server.worker_lifecycle import WorkerLifecycleState, start_worker, stop_worker


class DummyMetrics:
    def __init__(self) -> None:
        self.samples: list[tuple[str, float]] = []

    def observe_ms(self, name: str, value: float) -> None:
        self.samples.append((name, value))


def make_fake_server(loop: asyncio.AbstractEventLoop, tmp_path: Path):
    metrics = DummyMetrics()
    notifications = WorkerSceneNotificationQueue()
    processed: list = []

    async def broadcast_state_json(payload):
        broadcasts.append(payload)

    broadcasts: list = []

    class FakeServer:
        pass

    server = FakeServer()
    server.metrics = metrics
    server._ctx = SimpleNamespace(debug_policy=SimpleNamespace(encoder=SimpleNamespace(log_nals=False)))
    server._param_cache = {}
    server._pixel_channel = object()
    server._pixel_config = object()
    server._broadcast_state_json = broadcast_state_json
    server._schedule_coro = lambda coro, label: loop.create_task(coro)
    server._dump_remaining = 0
    server._dump_dir = str(tmp_path)
    server._dump_path = None
    server.width = 640
    server.height = 480
    server.cfg = SimpleNamespace(fps=60)
    server.use_volume = False
    server._animate = False
    server._animate_dps = 0.0
    server._zarr_path = None
    server._zarr_level = None
    server._zarr_axes = None
    server._zarr_z = None
    server._ctx_env = {}
    server._log_dims_info = False
    server._log_state_traces = False
    server._log_cam_info = False
    server._log_cam_debug = False
    server._publish_policy_metrics = lambda: None
    server._state_lock = threading.Lock()
    scene = create_server_scene_data(policy_event_path=tmp_path / "policy_events.jsonl")
    scene.latest_state = ServerSceneState(current_step=(0,))
    scene.camera_commands = deque()
    scene.policy_metrics_snapshot = {}
    scene.multiscale_state = {}
    scene.policy_event_path.parent.mkdir(parents=True, exist_ok=True)
    server._scene = scene
    server._worker_notifications = notifications
    server._dims_metadata = lambda: {}
    server.processed_notifications = processed

    def process_worker_notifications() -> None:
        drained = server._worker_notifications.drain()
        if drained:
            processed.extend(drained)

    server._process_worker_notifications = process_worker_notifications
    server.broadcasts = broadcasts
    server.loop = loop

    return server


def _run_loop_once(loop: asyncio.AbstractEventLoop) -> None:
    loop.run_until_complete(asyncio.sleep(0))


def test_scene_refresh_pushes_dims_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    holder: dict[str, object] = {"stop_event": state.stop_event}

    class StubWorker:
        _orientation_ready = True
        _z_index = None

        def __init__(self, *, scene_refresh_cb, **kwargs):
            holder["refresh_cb"] = scene_refresh_cb
            holder["stop_event"].set()

        def apply_state(self, state):
            raise AssertionError("apply_state should not run in this test")

        def process_camera_commands(self, commands):
            raise AssertionError("process_camera_commands should not run in this test")

        def capture_and_encode_packet(self):
            raise AssertionError("capture_and_encode_packet should not run in this test")

        def cleanup(self):
            pass

    async def dummy_send(*args, **kwargs):
        return None

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", StubWorker)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_video_config",
        lambda *a, **k: dummy_send(*a, **k),
    )

    start_worker(server, loop, state)
    state.thread.join(timeout=1.0)

    refresh_cb = holder["refresh_cb"]
    refresh_cb([4, 5])
    _run_loop_once(loop)

    assert server.processed_notifications
    note = server.processed_notifications[0]
    assert note.kind == "dims_update"
    assert note.step == (4, 5)
    assert server._scene.latest_state.current_step == (4, 5)

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_pushes_meta_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    holder: dict[str, object] = {"stop_event": state.stop_event}

    class StubWorker:
        _orientation_ready = True
        _z_index = None

        def __init__(self, *, scene_refresh_cb, **kwargs):
            holder["refresh_cb"] = scene_refresh_cb
            holder["stop_event"].set()

        def apply_state(self, state):
            raise AssertionError

        def process_camera_commands(self, commands):
            raise AssertionError

        def capture_and_encode_packet(self):
            raise AssertionError

        def cleanup(self):
            pass

    async def dummy_send(*args, **kwargs):
        return None

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", StubWorker)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_video_config",
        lambda *a, **k: dummy_send(*a, **k),
    )

    start_worker(server, loop, state)
    state.thread.join(timeout=1.0)

    refresh_cb = holder["refresh_cb"]
    refresh_cb(None)
    _run_loop_once(loop)

    assert server.processed_notifications
    note = server.processed_notifications[0]
    assert note.kind == "meta_refresh"
    assert note.step is None

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_on_frame_schedules_video_config(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    holder: dict[str, object] = {"stop_event": state.stop_event}
    send_calls: list[tuple[tuple, dict]] = []

    class PacketWorker:
        _orientation_ready = True
        _z_index = None

        def __init__(self, *, scene_refresh_cb, **kwargs):
            holder["refresh_cb"] = scene_refresh_cb

        def apply_state(self, state):
            pass

        def process_camera_commands(self, commands):
            pass

        def capture_and_encode_packet(self):
            holder["stop_event"].set()
            timings = SimpleNamespace(
                render_ms=1.0,
                blit_gpu_ns=None,
                blit_cpu_ms=0.0,
                map_ms=0.0,
                copy_ms=0.0,
                convert_ms=0.0,
                encode_ms=0.0,
                pack_ms=0.0,
                total_ms=1.0,
                capture_wall_ts=time.perf_counter(),
            )
            return timings, b"raw", 0, 7

        def cleanup(self):
            pass

    async def fake_send(*args, **kwargs):
        send_calls.append((args, kwargs))

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", PacketWorker)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"avcc", True))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: b"cfg")
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_video_config",
        lambda *a, **k: fake_send(*a, **k),
    )

    start_worker(server, loop, state)
    state.thread.join(timeout=1.0)
    _run_loop_once(loop)

    assert send_calls
    assert server.metrics.samples

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)
