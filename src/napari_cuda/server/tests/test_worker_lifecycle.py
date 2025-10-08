"""Unit tests for worker lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Awaitable

import pytest

from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.rendering.viewer_builder import CanonicalAxes

from napari_cuda.server.runtime.scene_ingest import RenderSceneSnapshot
from napari_cuda.server.scene import create_server_scene_data
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_reducers import _dims_entries_from_payload
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.runtime.worker_lifecycle import WorkerLifecycleState, start_worker, stop_worker
from napari_cuda.server.rendering.debug_tools import DebugConfig


class DummyMetrics:
    def __init__(self) -> None:
        self.samples: list[tuple[str, float]] = []

    def observe_ms(self, name: str, value: float) -> None:
        self.samples.append((name, value))


def make_fake_server(loop: asyncio.AbstractEventLoop, tmp_path: Path):
    metrics = DummyMetrics()
    async def broadcast_stream_config(payload):
        broadcasts.append(payload)

    broadcasts: list = []
    dims_broadcasts: list[NotifyDimsPayload] = []

    class FakeServer:
        pass

    server = FakeServer()
    server.metrics = metrics
    server._ctx = SimpleNamespace(
        debug_policy=SimpleNamespace(
            encoder=SimpleNamespace(log_nals=False),
            worker=SimpleNamespace(force_tight_pitch=False),
            dumps=SimpleNamespace(raw_budget=0),
        )
    )
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
    server._state_ledger = ServerStateLedger()
    scene = create_server_scene_data(policy_event_path=tmp_path / "policy_events.jsonl")
    scene.latest_state = RenderSceneSnapshot(current_step=(0,))
    scene.camera_commands = deque()
    scene.policy_metrics_snapshot = {}
    scene.multiscale_state = {}
    scene.policy_event_path.parent.mkdir(parents=True, exist_ok=True)
    server._scene = scene
    baseline_dims = NotifyDimsPayload.from_dict(
        {
            'step': [0],
            'current_step': [0],
            'level_shapes': [[1]],
            'axis_labels': ['z'],
            'order': [0],
            'displayed': [0],
            'ndisplay': 2,
            'mode': 'plane',
            'levels': [{'index': 0, 'shape': [1]}],
            'current_level': 0,
        }
    )
    server._state_ledger.batch_record_confirmed(
        _dims_entries_from_payload(baseline_dims, axis_index=0, axis_target='z'),
        origin="test.bootstrap",
    )
    scene.multiscale_state["current_level"] = 0
    scene.multiscale_state["levels"] = [{'index': 0, 'shape': [1]}]
    server.broadcasts = broadcasts
    server.broadcasted_dims = dims_broadcasts
    server.loop = loop

    def _mirror_schedule(coro: Awaitable[None], label: str) -> None:
        def _create_task() -> None:
            task = loop.create_task(coro)  # type: ignore[arg-type]
            def _on_done(t: asyncio.Task) -> None:
                try:
                    t.result()
                except Exception as exc:  # pragma: no cover - propagate loudly
                    raise RuntimeError(f"mirror task {label} failed") from exc
            task.add_done_callback(_on_done)
        loop.call_soon_threadsafe(_create_task)

    async def _mirror_broadcast(payload: NotifyDimsPayload) -> None:
        dims_broadcasts.append(payload)

    def _mirror_apply(payload: NotifyDimsPayload) -> None:
        with server._state_lock:
            multiscale_state = server._scene.multiscale_state
            multiscale_state["current_level"] = payload.current_level
            multiscale_state["levels"] = [dict(level) for level in payload.levels]
            if payload.downgraded is not None:
                multiscale_state["downgraded"] = bool(payload.downgraded)
            else:
                multiscale_state.pop("downgraded", None)

    server._dims_mirror = ServerDimsMirror(
        ledger=server._state_ledger,
        broadcaster=_mirror_broadcast,
        schedule=_mirror_schedule,
        on_payload=_mirror_apply,
    )
    server._dims_mirror.start()

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

    _z_index = None
    default_meta = {
        'step': [0, 0],
        'current_step': [0, 0],
        'level_shapes': [[10, 10]],
        'axis_labels': ['y', 'x'],
        'order': [0, 1],
        'displayed': [0, 1],
        'ndisplay': 2,
        'mode': 'plane',
        'levels': [{'index': 0, 'shape': [10, 10]}],
        'current_level': 0,
    }
    default_step = (0, 0)

    def __init__(self, *, scene_refresh_cb, **kwargs):
        self._scene_refresh_cb = scene_refresh_cb
        self._meta = dict(self.default_meta)
        self._stop_event = None
        self.applied_states: list = []
        self.camera_commands: list = []
        self.frames: list[bytes] = []
        self._ready = True
        self._is_ready = True
        self._last_step = tuple(int(value) for value in self.default_step)
        self._plane_restore_state = None
        self._active_ms_level = 0
        self._scene_source = None
        self._zarr_axes = kwargs.get('zarr_axes')
        self._zarr_path = kwargs.get('zarr_path')
        self.width = int(kwargs.get('width', 640))
        self.height = int(kwargs.get('height', 480))
        self.fps = int(kwargs.get('fps', 60))
        self._animate = bool(kwargs.get('animate', False))
        self._raw_dump_budget = 0
        self._debug_policy = SimpleNamespace(worker=SimpleNamespace(force_tight_pitch=False))
        self._debug_config = DebugConfig()
        self._capture = SimpleNamespace(
            pipeline=SimpleNamespace(
                set_debug=lambda debug: None,
                set_raw_dump_budget=lambda budget: None,
                enc_input_format="NV12",
            ),
            cuda=SimpleNamespace(set_force_tight_pitch=lambda enabled: None),
        )
        level_shapes = self._meta.get('level_shapes') or []
        if level_shapes:
            sizes = [max(1, int(dim)) for dim in level_shapes[0]]
        else:
            base_len = len(self._last_step) if self._last_step else len(self._meta.get('step', []))
            if base_len <= 0:
                base_len = 1
            sizes = [1] * base_len
        ndim = len(sizes)
        axis_labels_meta = self._meta.get('axis_labels') or []
        axis_labels = [str(lbl) for lbl in axis_labels_meta[:ndim]]
        if len(axis_labels) < ndim:
            axis_labels.extend(f"axis-{idx}" for idx in range(len(axis_labels), ndim))
        order_meta = self._meta.get('order') or list(range(ndim))
        order_values = [int(idx) for idx in order_meta[:ndim]]
        if len(order_values) < ndim:
            order_values.extend(range(len(order_values), ndim))
        step_meta = (
            self._meta.get('current_step')
            or self._meta.get('step')
            or list(self._last_step)
            or [0] * ndim
        )
        current_step = [int(value) for value in step_meta[:ndim]]
        if len(current_step) < ndim:
            current_step.extend(0 for _ in range(ndim - len(current_step)))
        ranges = [(0.0, float(max(0, size - 1)), 1.0) for size in sizes]
        ndisplay_value = int(self._meta.get('ndisplay', min(2, ndim)))
        self._canonical_axes = CanonicalAxes(
            ndim=ndim,
            axis_labels=tuple(axis_labels),
            order=tuple(order_values),
            ndisplay=ndisplay_value,
            current_step=tuple(current_step),
            ranges=tuple((float(lo), float(hi), float(step)) for lo, hi, step in ranges),
            sizes=tuple(int(size) for size in sizes),
        )

    # --- Lifecycle hooks -------------------------------------------------
    def _init_cuda(self) -> None:
        return

    def _init_vispy_scene(self) -> None:
        return

    def _init_egl(self) -> None:
        return

    def _init_capture(self) -> None:
        return

    def _init_cuda_interop(self) -> None:
        return

    def _init_encoder(self) -> None:
        return

    def _log_debug_policy_once(self) -> None:
        return

    def build_notify_dims_payload(self) -> NotifyDimsPayload:
        snapshot = dict(self._meta)
        step = tuple(int(v) for v in self._last_step) if getattr(self, '_last_step', None) is not None else tuple()
        snapshot['step'] = [int(value) for value in step] if step else list(snapshot.get('step', []))
        snapshot['current_step'] = list(snapshot['step'])
        return NotifyDimsPayload.from_dict(snapshot)

    def _notify_scene_refresh(self) -> None:
        callback = getattr(self, "_scene_refresh_cb", None)
        if callback is not None:
            callback(None)

    @property
    def is_ready(self) -> bool:
        if hasattr(self, "_is_ready"):
            return bool(self._is_ready)
        return self._ready

    def set_scene_refresh_callback(self, callback) -> None:  # noqa: ANN001 - test stub
        self._scene_refresh_cb = callback

    def viewer_model(self):  # noqa: ANN001
        return None

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
        self._ready = False
        self._is_ready = False


def test_build_notify_dims_payload_prefers_scene_shape():
    from napari_cuda.server.runtime.egl_worker import EGLRendererWorker

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
    worker._canonical_axes = CanonicalAxes(
        ndim=3,
        axis_labels=("z", "y", "x"),
        order=(0, 1, 2),
        ndisplay=2,
        current_step=tuple(int(v) for v in dims.current_step),
        ranges=((0.0, 59.0, 1.0), (0.0, 255.0, 1.0), (0.0, 255.0, 1.0)),
        sizes=(60, 256, 256),
    )

    payload = EGLRendererWorker.build_notify_dims_payload(worker)

    assert payload.axis_labels == ("z", "y", "x")
    assert payload.level_shapes == ((60, 256, 256), (8, 128, 128))
    assert payload.level_shapes[payload.current_level] == (60, 256, 256)
    assert payload.current_step == (5, 2, 1)
    assert payload.order == (0, 1, 2)


class _FakePacketWorker(_FakeWorkerBase):
    default_meta = {
        'step': [0],
        'current_step': [0],
        'level_shapes': [[1]],
        'axis_labels': ['z'],
        'order': [0],
        'displayed': [0],
        'ndisplay': 2,
        'mode': 'plane',
        'levels': [{'index': 0, 'shape': [1]}],
        'current_level': 0,
    }


def test_scene_refresh_pushes_dims_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.EGLRendererWorker", _FakeWorkerBase)
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]
    _run_loop_once(loop)
    server.broadcasted_dims.clear()

    worker._last_step = (4, 5)
    worker._scene_refresh_cb([4, 5])  # type: ignore[attr-defined]
    _run_loop_once(loop)

    assert server.broadcasted_dims, "expected dims notification"
    note = server.broadcasted_dims[-1]
    assert isinstance(note, NotifyDimsPayload)
    assert list(note.current_step) == [4, 5]

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_pushes_meta_notification(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.EGLRendererWorker", _FakeWorkerBase)
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]
    _run_loop_once(loop)
    server.broadcasted_dims.clear()

    worker._last_step = (7,)
    worker._scene_refresh_cb(None)  # type: ignore[attr-defined]
    _run_loop_once(loop)

    assert server.broadcasted_dims, "expected dims notification"
    note = server.broadcasted_dims[-1]
    assert isinstance(note, NotifyDimsPayload)
    assert note.current_step[0] == 7

    stop_worker(state)
    _run_loop_once(loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_skips_duplicate_payload(monkeypatch, tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    state = WorkerLifecycleState()

    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.EGLRendererWorker", _FakeWorkerBase)
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]

    _run_loop_once(loop)
    server.broadcasted_dims.clear()

    worker._last_step = (3, 4)
    worker._scene_refresh_cb(None)  # type: ignore[attr-defined]
    _run_loop_once(loop)
    assert len(server.broadcasted_dims) == 1

    server.broadcasted_dims.clear()
    worker._scene_refresh_cb(None)  # type: ignore[attr-defined]
    _run_loop_once(loop)
    assert not server.broadcasted_dims

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
            'step': [10, 0, 0],
        'current_step': [10, 0, 0],
        'level_shapes': [[60, 256, 256], [8, 128, 128]],
        'axis_labels': ['z', 'y', 'x'],
        'order': [0, 1, 2],
        'displayed': [1, 2],
        'ndisplay': 2,
        'mode': 'plane',
        'levels': [
            {'index': 0, 'shape': [60, 256, 256]},
            {'index': 1, 'shape': [8, 128, 128]},
        ],
        'current_level': 0,
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

    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.EGLRendererWorker", _FakeMultiscaleWorker)
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"", False))
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.build_avcc_config", lambda cache: None)
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker_lifecycle.pixel_channel.maybe_send_stream_config",
        lambda *a, **k: asyncio.sleep(0),
    )

    start_worker(server, loop, state)
    worker = _wait_for_worker(state)
    worker._stop_event = state.stop_event  # type: ignore[attr-defined]

    _run_loop_once(loop)
    server.broadcasted_dims.clear()

    worker._meta['level_shapes'][0][0] = 8  # type: ignore[index]
    worker._meta['step'][0] = 60  # type: ignore[index]
    worker._meta['current_step'][0] = 60  # type: ignore[index]
    worker._last_step = tuple(int(x) for x in worker._meta['current_step'])
    worker._scene_refresh_cb(None)  # type: ignore[attr-defined]
    _run_loop_once(loop)

    assert server.broadcasted_dims, 'expected dims notification'
    dims_note = server.broadcasted_dims[-1]
    assert isinstance(dims_note, NotifyDimsPayload)
    assert dims_note.level_shapes[0][0] == worker._meta['level_shapes'][0][0]
    assert dims_note.current_step[0] == worker._meta['current_step'][0]

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

    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.EGLRendererWorker", _FakePacketWorker)
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.pack_to_avcc", lambda *a, **k: (b"avcc", True))
    monkeypatch.setattr("napari_cuda.server.runtime.worker_lifecycle.build_avcc_config", lambda cache: b"cfg")
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker_lifecycle.pixel_channel.maybe_send_stream_config",
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
