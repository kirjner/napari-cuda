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
from napari_cuda.server.worker_notifications import WorkerSceneNotificationQueue
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

    async def broadcast_stream_config(payload):
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
    server._broadcast_stream_config = broadcast_stream_config
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
    server._state_lock = threading.RLock()
    scene = create_server_scene_data(policy_event_path=tmp_path / "policy_events.jsonl")
    scene.latest_state = ServerSceneState(current_step=(0,))
    scene.camera_commands = deque()
    scene.policy_metrics_snapshot = {}
    scene.multiscale_state = {}
    scene.policy_event_path.parent.mkdir(parents=True, exist_ok=True)
    server._scene = scene
    server._worker_notifications = notifications
    server._scene.last_dims_payload = {
        'ndim': 1,
        'order': ['z'],
        'range': [[0, 0]],
        'current_step': [0],
        'ndisplay': 2,
        'mode': 'plane',
        'volume': False,
    }
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


def _wait_for_worker(state: WorkerLifecycleState, *, timeout_s: float = 1.0) -> object:
    deadline = time.time() + timeout_s
    while state.worker is None and time.time() < deadline:
        time.sleep(0.01)
    assert state.worker is not None, "render worker failed to start"
    return state.worker


class _FakeWorkerBase:
    """Minimal render worker stub that satisfies lifecycle expectations."""

    _orientation_ready = True
    _z_index = None
    default_meta = {
        'ndim': 2,
        'axes': ['y', 'x'],
        'axis_labels': ['y', 'x'],
        'order': [0, 1],
        'range': [[0, 9], [0, 9]],
        'ndisplay': 2,
        'mode': 'plane',
    }
    default_step = (0, 0)

    def __init__(self, *, scene_refresh_cb, **kwargs):
        self._scene_refresh_cb = scene_refresh_cb
        self._meta = dict(self.default_meta)
        self._stop_event = None
        self.applied_states: list = []
        self.camera_commands: list = []
        self.frames: list[bytes] = []
        self._bootstrapped = True
        self._bootstrapping = False
        self._last_step = tuple(int(value) for value in self.default_step)

    def snapshot_dims_metadata(self) -> dict[str, object]:
        return dict(self._meta)

    @property
    def is_bootstrapped(self) -> bool:
        return self._bootstrapped

    @property
    def is_bootstrapping(self) -> bool:
        return self._bootstrapping

    def set_scene_refresh_callback(self, callback) -> None:  # noqa: ANN001 - test stub
        self._scene_refresh_cb = callback

    def bootstrap(self) -> None:
        self._bootstrapping = True
        self._bootstrapped = True
        self._bootstrapping = False

    def apply_state(self, state) -> None:  # noqa: ANN001 - signature mirrors concrete worker
        self.applied_states.append(state)

    def process_camera_commands(self, commands) -> None:  # noqa: ANN001
        if commands:
            self.camera_commands.extend(commands)

    def capture_and_encode_packet(self):
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
        payload = b"frame"
        self.frames.append(payload)
        if self._stop_event is not None:
            self._stop_event.set()
        return timings, payload, 0, 0

    def cleanup(self) -> None:
        pass


def test_snapshot_dims_metadata_prefers_scene_shape():
    from napari_cuda.server.render_worker import EGLRendererWorker

    dims = SimpleNamespace(
        current_step=(5, 2, 1),
        ndim=3,
        ndisplay=2,
        order=(0, 1, 2),
        axis_labels=("z", "y", "x"),
        displayed=(0, 1),
        nsteps=(60, 256, 256),
    )

    level0 = SimpleNamespace(shape=(60, 256, 256))
    level1 = SimpleNamespace(shape=(8, 128, 128))
    scene_source = SimpleNamespace(axes=("z", "y", "x"), level_descriptors=[level0, level1])

    worker = object.__new__(EGLRendererWorker)
    worker._viewer = SimpleNamespace(dims=dims)
    worker._scene_source = scene_source
    worker._zarr_axes = "zyx"
    worker._zarr_shape = (8, 256, 256)
    worker._active_ms_level = 0
    worker._last_step = tuple(int(v) for v in dims.current_step)

    meta = EGLRendererWorker.snapshot_dims_metadata(worker)

    assert meta["axes"] == ["z", "y", "x"]
    assert meta["axis_labels"] == ["z", "y", "x"]
    assert meta["sizes"] == [8, 256, 256]
    assert meta["range"] == [[0, 7], [0, 255], [0, 255]]
    assert "current_step" not in meta
    assert meta["order"] == [0, 1, 2]


class _FakePacketWorker(_FakeWorkerBase):
    default_meta = {
        'ndim': 1,
        'axes': ['z'],
        'axis_labels': ['z'],
        'order': [0],
        'range': [[0, 0]],
        'current_step': [0],
        'ndisplay': 2,
        'mode': 'plane',
    }


def test_scene_refresh_pushes_dims_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", _FakeWorkerBase)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]
    worker._last_step = (4, 5)

    _run_loop_once(loop)
    server.processed_notifications.clear()

    worker._scene_refresh_cb([4, 5])  # type: ignore[attr-defined]
    _run_loop_once(loop)

    assert server.processed_notifications, "expected dims notification"
    note = server.processed_notifications[-1]
    assert note.kind == "dims_update"
    assert note.step == (4, 5)
    assert "current_step" not in note.meta

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_pushes_meta_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", _FakeWorkerBase)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]
    worker._last_step = (7,)

    _run_loop_once(loop)
    server.processed_notifications.clear()

    worker._scene_refresh_cb(None)  # type: ignore[attr-defined]
    _run_loop_once(loop)

    assert server.processed_notifications, "expected dims notification"
    note = server.processed_notifications[-1]
    assert note.kind == "dims_update"
    assert note.step == (7,)
    assert "current_step" not in note.meta

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_remaps_z_axis_only(monkeypatch, tmp_path):
    class _DummySource:
        def __init__(self) -> None:
            self.axes = ('z', 'y', 'x')

        def level_shape(self, index: int) -> tuple[int, int, int]:
            if int(index) == 1:
                return (8, 256, 256)
            return (60, 256, 256)

    class _FakeMultiscaleWorker(_FakeWorkerBase):
        default_meta = {
            'ndim': 3,
            'axes': ['z', 'y', 'x'],
            'axis_labels': ['z', 'y', 'x'],
            'order': [0, 1, 2],
            'sizes': [60, 256, 256],
            'range': [[0, 59], [0, 255], [0, 255]],
            'current_step': [10, 0, 0],
            'ndisplay': 2,
            'mode': 'plane',
        }

        def __init__(self, *, scene_refresh_cb, **kwargs):
            super().__init__(scene_refresh_cb=scene_refresh_cb, **kwargs)
            self._scene_source = _DummySource()
            self._active_ms_level = 0
            self._zarr_axes = 'zyx'
            self._z_index = 10

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", _FakeMultiscaleWorker)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]

    _run_loop_once(loop)
    server.processed_notifications.clear()

    payload = {
        'scene_level': {
            'current_level': 1,
        }
    }
    worker._meta['sizes'][0] = 8  # type: ignore[index]
    worker._meta['range'][0] = [0, 7]  # type: ignore[index]
    worker._meta['current_step'][0] = 60  # type: ignore[index]
    worker._last_step = tuple(int(x) for x in worker._meta['current_step'])
    worker._scene_refresh_cb(payload)  # type: ignore[attr-defined]
    _run_loop_once(loop)

    cached = server._scene.last_dims_payload
    assert cached['sizes'][0] == worker._meta['sizes'][0]
    assert cached['range'][0] == worker._meta['range'][0]
    assert cached['axes'] == worker._meta['axes']

    assert server.processed_notifications
    dims_note = server.processed_notifications[-1]
    assert dims_note.kind == "dims_update"
    assert dims_note.step[0] == worker._meta['current_step'][0]

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_on_frame_schedules_stream_config(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()
    send_calls: list[tuple[tuple, dict]] = []

    async def fake_send(*args, **kwargs):
        send_calls.append((args, kwargs))

    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.EGLRendererWorker", _FakePacketWorker)
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"avcc", True))
    monkeypatch.setattr("napari_cuda.server.worker_lifecycle.build_avcc_config", lambda cache: b"cfg")
    monkeypatch.setattr(
        "napari_cuda.server.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: fake_send(*a, **k),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]

    state.thread.join(timeout=1.0)  # type: ignore[union-attr]
    _run_loop_once(loop)

    assert send_calls, "expected stream config send"
    assert server.metrics.samples, "expected timing samples recorded"

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)
