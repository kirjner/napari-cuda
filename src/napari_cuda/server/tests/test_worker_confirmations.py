"""Unit tests for worker lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Awaitable, Sequence, Mapping

import pytest

from napari_cuda.protocol.messages import NotifyDimsPayload
from napari_cuda.server.app.egl_headless_server import EGLHeadlessServer
from napari_cuda.server.rendering.viewer_builder import CanonicalAxes

from napari_cuda.server.runtime.scene_ingest import RenderSceneSnapshot
from napari_cuda.server.scene import create_server_scene_data
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_models import WorkerStateUpdateConfirmation
from napari_cuda.server.control.state_reducers import _dims_entries_from_payload
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.runtime.worker_lifecycle import (
    WorkerLifecycleState,
    start_worker,
    stop_worker,
    _build_dims_confirmation,
)
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
    server._control_loop = loop
    server._worker_updates = asyncio.Queue(maxsize=8)
    server._submit_worker_confirmation = MethodType(EGLHeadlessServer._submit_worker_confirmation, server)
    server._apply_worker_confirmation = MethodType(EGLHeadlessServer._apply_worker_confirmation, server)
    server._dispatch_worker_updates = MethodType(EGLHeadlessServer._dispatch_worker_updates, server)
    server._worker_dispatch_task = loop.create_task(server._dispatch_worker_updates())
    server._seq = 0

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


def _await_condition(loop: asyncio.AbstractEventLoop, predicate, *, attempts: int = 5) -> None:
    for _ in range(max(1, attempts)):
        if predicate():
            return
        _run_loop_once(loop)
    assert predicate(), "event loop condition not satisfied"


def _wait_for_worker(state: WorkerLifecycleState, *, timeout_s: float = 1.0) -> object:
    deadline = time.time() + timeout_s
    while state.worker is None and time.time() < deadline:
        time.sleep(0.01)
    assert state.worker is not None, "render worker failed to start"
    return state.worker


def _shutdown_fake_server(server: object, loop: asyncio.AbstractEventLoop) -> None:
    task = getattr(server, "_worker_dispatch_task", None)
    if task is not None:
        task.cancel()
    try:
        pending = [t for t in asyncio.all_tasks(loop) if t is not task]
    except RuntimeError:
        pending = []
    for pending_task in pending:
        pending_task.cancel()
    try:
        gather_targets = []
        if task is not None:
            gather_targets.append(task)
        gather_targets.extend(pending)
        if gather_targets:
            loop.run_until_complete(asyncio.gather(*gather_targets, return_exceptions=True))
        loop.run_until_complete(asyncio.sleep(0))
    except RuntimeError:
        pass


def _make_confirmation(
    step: tuple[int, ...],
    *,
    displayed: tuple[int, ...] | None = (0,),
    axis_labels: tuple[str, ...] | None = ("z",),
    order: tuple[int, ...] | None = (0,),
    labels: tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
    current_level: int = 0,
    levels: tuple[Mapping[str, Any], ...] | None = None,
    level_shapes: tuple[tuple[int, ...], ...] | None = None,
    downgraded: bool | None = None,
) -> WorkerStateUpdateConfirmation:
    if levels is None:
        levels = ({"index": int(current_level), "shape": [1]},)
    if level_shapes is None:
        level_shapes = ((1,),)
    return WorkerStateUpdateConfirmation(
        scope="dims",
        target="main",
        key="snapshot",
        step=tuple(int(v) for v in step),
        ndisplay=2,
        mode="plane",
        displayed=tuple(int(v) for v in displayed) if displayed is not None else None,
        order=tuple(int(v) for v in order) if order is not None else None,
        axis_labels=tuple(str(v) for v in axis_labels) if axis_labels is not None else None,
        labels=tuple(str(v) for v in labels) if labels is not None else None,
        current_level=int(current_level),
        levels=tuple(dict(level) for level in levels),
        level_shapes=tuple(tuple(int(dim) for dim in shape) for shape in level_shapes),
        downgraded=downgraded,
        timestamp=time.time(),
        metadata=metadata,
    )


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
        self._level_downgraded = None
        self._zarr_axes = kwargs.get('zarr_axes')
        self._zarr_path = kwargs.get('zarr_path')
        self._ledger = kwargs.get('ledger', ServerStateLedger())
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

    def _notify_scene_refresh(self, step: Sequence[int] | None = None) -> None:
        callback = getattr(self, "_scene_refresh_cb", None)
        if callback is not None:
            callback(step)

    @property
    def is_ready(self) -> bool:
        if hasattr(self, "_is_ready"):
            return bool(self._is_ready)
        return self._ready

    def set_scene_refresh_callback(self, callback) -> None:  # noqa: ANN001 - test stub
        self._scene_refresh_cb = callback

    def attach_ledger(self, ledger) -> None:
        self._ledger = ledger

    def _record_step_in_ledger(self, step, *, origin: str = "test.stub") -> None:
        ledger = getattr(self, "_ledger", None)
        if ledger is None:
            return
        ledger.record_confirmed(
            "dims",
            "main",
            "current_step",
            tuple(int(v) for v in step),
            origin=str(origin),
        )

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
        return timings, payload, 0, 0

    def cleanup(self) -> None:
        self._ready = False
        self._is_ready = False


def test_dims_confirmation_prefers_scene_source_shape():
    worker = SimpleNamespace()
    worker._canonical_axes = CanonicalAxes(
        ndim=3,
        axis_labels=("z", "y", "x"),
        order=(0, 1, 2),
        ndisplay=2,
        current_step=(5, 2, 1),
        ranges=((0.0, 59.0, 1.0), (0.0, 255.0, 1.0), (0.0, 255.0, 1.0)),
        sizes=(60, 256, 256),
    )
    worker._last_step = (5, 2, 1)
    worker._active_ms_level = 0
    worker._level_downgraded = None
    worker._plane_restore_state = None
    worker._scene_source = SimpleNamespace(
        level_descriptors=[
            SimpleNamespace(shape=(60, 256, 256), index=0),
            SimpleNamespace(shape=(8, 128, 128), index=1),
        ]
    )

    confirmation = _build_dims_confirmation(worker, None)

    assert confirmation.level_shapes == ((60, 256, 256), (8, 128, 128))
    assert confirmation.levels[0]["shape"] == [60, 256, 256]
    assert confirmation.step == (5, 2, 1)
    assert confirmation.axis_labels == ("z", "y", "x")


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


def test_scene_refresh_pushes_dims_notification(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    worker = _FakeWorkerBase(scene_refresh_cb=lambda _: None)
    worker._last_step = (4, 5)
    confirmation = _build_dims_confirmation(worker, (4, 5))
    server._apply_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    assert server.broadcasted_dims, "expected dims notification"
    note = server.broadcasted_dims[-1]
    assert isinstance(note, NotifyDimsPayload)
    assert list(note.current_step) == [4, 5]

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_pushes_meta_notification(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    worker = _FakeWorkerBase(scene_refresh_cb=lambda _: None)
    worker._last_step = (7,)
    confirmation = _build_dims_confirmation(worker, (7,))
    server._apply_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    assert server.broadcasted_dims, "expected dims notification"
    note = server.broadcasted_dims[-1]
    assert isinstance(note, NotifyDimsPayload)
    assert note.current_step[0] == 7

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_skips_duplicate_payload(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    worker = _FakeWorkerBase(scene_refresh_cb=lambda _: None)
    worker._last_step = (3, 4)
    confirmation = _build_dims_confirmation(worker, (3, 4))
    server._apply_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)
    assert server.broadcasted_dims
    first_len = len(server.broadcasted_dims)
    assert server.broadcasted_dims[-1].current_step == (3, 4)

    server.broadcasted_dims.clear()
    server._apply_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    for _ in range(2):
        _run_loop_once(loop)
    assert not server.broadcasted_dims

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_scene_refresh_remaps_z_axis_only(tmp_path):
    class _DummySource:
        def __init__(self) -> None:
            self.axes = ('z', 'y', 'x')
            self.level_descriptors = [
                SimpleNamespace(shape=(60, 256, 256), index=0, path="level-0", downsample=None),
                SimpleNamespace(shape=(8, 128, 128), index=1, path="level-1", downsample=None),
            ]

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
    server.broadcasted_dims.clear()

    worker = _FakeMultiscaleWorker(scene_refresh_cb=lambda _: None)
    worker._scene_source = _DummySource()
    worker._scene_source.level_descriptors[0].shape = (8, 256, 256)
    worker._last_step = (60, 0, 0)

    confirmation = _build_dims_confirmation(worker, worker._last_step)
    server._apply_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    assert server.broadcasted_dims, 'expected dims notification'
    dims_note = server.broadcasted_dims[-1]
    assert isinstance(dims_note, NotifyDimsPayload)
    assert dims_note.level_shapes[0][0] == 8
    assert dims_note.current_step[0] == worker._last_step[0]

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_worker_confirmation_queue_guard_drops_stale(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    fresh = _make_confirmation((1,))
    stale = _make_confirmation((0,))

    server._submit_worker_confirmation(fresh)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    entry = server._state_ledger.get("dims", "main", "current_step")
    assert entry is not None
    assert tuple(entry.value) == (1,)
    assert len(server.broadcasted_dims) == 1
    assert server.broadcasted_dims[-1].current_step == (1,)

    server._submit_worker_confirmation(stale)  # type: ignore[attr-defined]
    for _ in range(2):
        _run_loop_once(loop)

    entry_after = server._state_ledger.get("dims", "main", "current_step")
    assert entry_after is not None
    assert tuple(entry_after.value) == (1,), "stale confirmation should be ignored"
    assert len(server.broadcasted_dims) == 1, "stale confirmation should not broadcast"

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_worker_confirmation_handles_missing_optional_fields(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    confirmation = _make_confirmation(
        (2,),
        displayed=None,
        axis_labels=None,
        order=None,
        labels=None,
    )

    server._submit_worker_confirmation(confirmation)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    displayed_entry = server._state_ledger.get("view", "main", "displayed")
    axis_entry = server._state_ledger.get("dims", "main", "axis_labels")
    assert displayed_entry is not None and displayed_entry.value is None
    assert axis_entry is not None and axis_entry.value is None

    assert server.broadcasted_dims, "mirror should broadcast worker confirmation"
    payload = server.broadcasted_dims[-1]
    assert payload.displayed is None
    assert payload.axis_labels is None
    assert payload.current_step == (2,)

    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)


def test_worker_confirmation_allows_step_regression_on_level_change(tmp_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = make_fake_server(loop, tmp_path)
    server.broadcasted_dims.clear()

    coarse_levels = ({"index": 1, "shape": [1000]},)
    coarse_shapes = ((1000,),)
    confirm_high = _make_confirmation(
        (322,),
        current_level=1,
        levels=coarse_levels,
        level_shapes=coarse_shapes,
    )
    server._submit_worker_confirmation(confirm_high)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    assert server.broadcasted_dims
    assert server._state_ledger.get("dims", "main", "current_step").value == (322,)

    finer_levels = ({"index": 2, "shape": [200]},)
    finer_shapes = ((200,),)
    confirm_low = _make_confirmation(
        (129,),
        current_level=2,
        levels=finer_levels,
        level_shapes=finer_shapes,
    )
    server.broadcasted_dims.clear()
    server._submit_worker_confirmation(confirm_low)  # type: ignore[attr-defined]
    _await_condition(loop, lambda: server.broadcasted_dims)

    entry = server._state_ledger.get("dims", "main", "current_step")
    assert entry is not None
    assert tuple(entry.value) == (129,)
    assert server.broadcasted_dims[-1].current_step == (129,)

    _shutdown_fake_server(server, loop)
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
    _shutdown_fake_server(server, loop)
    loop.close()
    asyncio.set_event_loop(None)
