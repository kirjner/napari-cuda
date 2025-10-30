"""Async harness utilities for exercising the state-channel server.

These helpers stand up a lightweight server façade plus a fake websocket so
tests can drive the full `ingest_state` coroutine without spinning up the
entire `EGLHeadlessServer`.  The harness mirrors the behaviour expected by
`control_channel_server`—including resumable history, dims mirrors, and the
pixel channel bookkeeping—while keeping external side effects contained in
memory for assertions.
"""

from __future__ import annotations

import asyncio
import logging
import json
import threading
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Iterable, Mapping, MutableSequence, Optional

import numpy as np
from websockets.exceptions import ConnectionClosedOK
from websockets.protocol import State

from napari_cuda.protocol import NotifyStreamPayload
from napari_cuda.protocol.messages import NotifyDimsPayload, SessionHello
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.control import control_channel_server as state_channel_handler
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.mirrors.layer_mirror import ServerLayerMirror
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
    ResumableRetention,
)
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_models import ServerLedgerUpdate
from napari_cuda.server.control.state_reducers import reduce_bootstrap_state, reduce_level_update
from napari_cuda.server.rendering import pixel_broadcaster
from napari_cuda.server.scene import (
    snapshot_render_state,
    snapshot_layer_controls,
    snapshot_multiscale_state,
    snapshot_scene,
    snapshot_volume_state,
)
from napari_cuda.server.runtime.core.snapshot_build import RenderLedgerSnapshot
from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdate
from napari_cuda.server.runtime.data import (
    SliceROI,
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)
from napari_cuda.server.data.level_logging import LayerAssignmentLogger
from napari.components import viewer_model
from napari_cuda.server.runtime.viewport import RenderMode, ViewportState

from napari_cuda.server.runtime.worker.interfaces.snapshot_interface import (
    SnapshotInterface,
)
from napari_cuda.server.runtime.worker.snapshots import apply as snapshot_mod
from napari_cuda.server.runtime.camera import CameraCommandQueue
from napari_cuda.server.data.roi import plane_wh_for_level


_SENTINEL = object()


class DummyMetrics:
    """Collect metrics mutations without requiring the real metrics stack."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: list[tuple[str, float]] = []

    def inc(self, name: str, value: float = 1.0, *_: Any, **__: Any) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def set(self, name: str, value: float, *_: Any, **__: Any) -> None:
        self.gauges[name] = float(value)

    def observe_ms(self, name: str, value: float, *_: Any, **__: Any) -> None:
        self.histograms.append((name, float(value)))


class FakeStateWebSocket:
    """Minimal websocket implementation consumed by the state-channel server."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        remote_address: tuple[str, int] = ("127.0.0.1", 4242),
    ) -> None:
        self._loop = loop
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[str] = []
        self.state: State = State.OPEN
        self.transport = SimpleNamespace(get_extra_info=lambda *_: None)
        self.remote_address = remote_address
        self._closed_event = loop.create_future()

    def __hash__(self) -> int:  # pragma: no cover - matches websockets protocol
        return id(self)

    def push_message(self, payload: Any) -> None:
        """Queue an inbound payload (str/bytes) for the server to `recv()`."""

        self._incoming.put_nowait(payload)

    async def recv(self) -> Any:
        payload = await self._incoming.get()
        if payload is _SENTINEL:
            self.state = State.CLOSED
            if not self._closed_event.done():
                self._closed_event.set_result(None)
            raise ConnectionClosedOK(None, None)
        return payload

    def __aiter__(self) -> FakeStateWebSocket:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.recv()
        except ConnectionClosedOK as exc:
            raise StopAsyncIteration from exc

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self, *_: Any, **__: Any) -> None:
        if self.state is State.CLOSED:
            return
        self.state = State.CLOSING
        self._incoming.put_nowait(_SENTINEL)
        if not self._closed_event.done():
            self._closed_event.set_result(None)
        self.state = State.CLOSED

    async def wait_closed(self) -> None:
        await self._closed_event


@dataclass(slots=True)
class ScheduledCall:
    label: str
    task: asyncio.Task[Any]


class _HarnessPanZoomCamera:
    def __init__(self) -> None:
        self.rect: tuple[float, float, float, float] | None = None
        self.center: tuple[float, float] = (0.0, 0.0)
        self.zoom: float = 1.0

    def set_range(
        self,
        *,
        x: tuple[float, float],
        y: tuple[float, float],
    ) -> None:
        self._range_x = tuple(float(v) for v in x)
        self._range_y = tuple(float(v) for v in y)


class _HarnessTurntableCamera:
    def __init__(self) -> None:
        self.center: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.azimuth: float = 0.0
        self.elevation: float = 0.0
        self.roll: float = 0.0
        self.distance: float = 1.0
        self.fov: float = 60.0


class _HarnessLayer:
    def __init__(self) -> None:
        self.scale: tuple[float, float] = (1.0, 1.0)
        self.translate: tuple[float, float] = (0.0, 0.0)
        self.visible: bool = True
        self.opacity: float = 1.0
        self.blending: str = "opaque"
        self.contrast_limits: list[float] = [0.0, 1.0]
        self.data = np.zeros((1, 1), dtype=np.float32)


class _HarnessViewTransform:
    def __init__(self) -> None:
        self.matrix = np.identity(4, dtype=float)

    def imap(self, coords: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = coords
        return (float(x), float(y), float(z))


class _HarnessSceneSource:
    def __init__(self) -> None:
        self.axes = ("z", "y", "x")
        self.level_descriptors = [
            SimpleNamespace(shape=(64, 512, 512), path=None),
            SimpleNamespace(shape=(32, 256, 256), path=None),
        ]
        self.dtype = "float32"
        self._current_level = 0
        self._current_step = (0, 0, 0)

    @property
    def current_level(self) -> int:
        return int(self._current_level)

    @current_level.setter
    def current_level(self, value: int) -> None:
        self._current_level = int(value)

    @property
    def current_step(self) -> tuple[int, ...]:
        return tuple(self._current_step)

    def initial_step(self, level: int) -> tuple[int, ...]:
        _ = int(level)
        return (0, 0, 0)

    def level_index_for_path(self, _path: Optional[str]) -> int:
        return 0

    def level_scale(self, level: int) -> tuple[float, float, float]:
        factor = float(2 ** int(level))
        return (1.0, factor, factor)

    def level_shape(self, index: Optional[int] = None) -> tuple[int, ...]:
        idx = self._current_level if index is None else int(index)
        return tuple(self.level_descriptors[idx].shape)

    def slice(
        self,
        level: int,
        z_index: int,
        *,
        compute: bool = True,
        roi: Optional[SliceROI] = None,
    ) -> np.ndarray:
        del compute, z_index  # unused in harness
        desc = self.level_descriptors[int(level)]
        _, height, width = desc.shape
        if roi is not None and not roi.is_empty():
            height = int(roi.height)
            width = int(roi.width)
        return np.zeros((height, width), dtype=np.float32)

    def set_current_slice(self, step: tuple[int, ...], level: int) -> tuple[int, ...]:
        self._current_step = tuple(int(v) for v in step)
        self._current_level = int(level)
        return self._current_step

    def ensure_contrast(self, level: int) -> tuple[float, float]:
        _ = int(level)
        return (0.0, 1.0)

class CaptureWorker:
    """Match the behaviour exercised in state channel tests without a renderer."""

    def __init__(self) -> None:
        self.policy_calls: list[str] = []
        self.level_requests: list[tuple[int, Any]] = []
        self.force_idr_calls = 0
        self.viewport_state = ViewportState(mode=RenderMode.PLANE)
        initial_level = 0
        self.viewport_state.plane.applied_level = initial_level
        self.viewport_state.plane.target_level = initial_level
        self.viewport_state.volume.level = initial_level
        self.viewport_state.volume.scale = (1.0, 1.0, 1.0)
        self.viewport_state.plane.update_pose(
            rect=(0.0, 0.0, float(640), float(480)),
            center=(320.0, 240.0),
            zoom=1.0,
        )
        self._is_ready = True
        self._data_wh = (640, 480)
        self._data_d = None
        self.width = 640
        self.height = 480
        self._viewer = viewer_model.ViewerModel()
        self.view = SimpleNamespace(
            camera=_HarnessPanZoomCamera(),
            scene=SimpleNamespace(transform=_HarnessViewTransform()),
        )
        self._napari_layer = _HarnessLayer()
        self._scene_source = _HarnessSceneSource()
        self._state_lock = threading.RLock()
        self._zarr_axes = "yx"
        self._zarr_shape = (480, 640)
        self._zarr_dtype = "float32"
        self.volume_dtype = "float32"
        self._last_step = (0, 0)
        self._roi_edge_threshold = 0
        self._roi_align_chunks = False
        self._roi_ensure_contains_viewport = False
        self._roi_pad_chunks = 0
        self._idr_on_z = False
        self._sticky_contrast = False
        self._layer_logger = LayerAssignmentLogger(logging.getLogger(__name__))
        self._log_layer_debug = False
        self._render_tick_required = False
        self._emit_current_camera_pose = lambda *_args, **_kwargs: None
        self._hw_limits = SimpleNamespace(volume_max_bytes=None, volume_max_voxels=None)
        self._volume_max_bytes = None
        self._volume_max_voxels = None
        self._budget_error_cls = RuntimeError
        self._z_index = 0
        self._pose_seq = 0
        self._max_camera_command_seq = 0
        self._last_dims_signature = None
        self._axis_labels = ("z", "y", "x")
        self._axis_order = (0, 1, 2)
        self._displayed = (1, 2)
        self._level_shapes = (
            (16, self._zarr_shape[0], self._zarr_shape[1]),
            (8, self._zarr_shape[0] // 2, self._zarr_shape[1] // 2),
        )
        self.render_updates: list[RenderUpdate] = []
        self._last_scene_state: RenderLedgerSnapshot | None = None
        self._plane_visual_node = SimpleNamespace(
            gamma=1.0,
            clim=(0.0, 1.0),
            cmap=None,
            method="mip",
            opacity=1.0,
            visible=False,
        )
        self._volume_visual_node = SimpleNamespace(
            gamma=1.0,
            clim=(0.0, 1.0),
            cmap=None,
            method="mip",
            opacity=1.0,
            relative_step_size=1.0,
            visible=False,
        )

    def set_policy(self, policy: str) -> None:
        self.policy_calls.append(str(policy))

    def request_multiscale_level(self, level: int, path: Any) -> None:
        self.level_requests.append((int(level), path))

    def force_idr(self) -> None:
        self.force_idr_calls += 1

    def viewer_model(self) -> None:  # pragma: no cover - interface stub
        return None

    def _aligned_roi_signature(
        self,
        source: Any,
        level: int,
        roi: SliceROI,
    ) -> tuple[SliceROI, Optional[tuple[int, int]], Optional[tuple[int, int, int, int]]]:
        chunk_shape = chunk_shape_for_level(source, int(level))
        aligned_roi = roi
        if self._roi_align_chunks and chunk_shape is not None:
            full_h, full_w = plane_wh_for_level(source, int(level))
            aligned_roi = align_roi_to_chunk_grid(
                roi,
                chunk_shape,
                int(self._roi_pad_chunks),
                height=full_h,
                width=full_w,
            )
        signature = roi_chunk_signature(aligned_roi, chunk_shape)
        return aligned_roi, chunk_shape, signature

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # Viewport state bridge ---------------------------------------------------
    @property
    def _volume_scale(self) -> tuple[float, float, float]:
        scale = self.viewport_state.volume.scale
        if scale is None:
            return (1.0, 1.0, 1.0)
        return scale

    @_volume_scale.setter
    def _volume_scale(self, value: tuple[float, float, float]) -> None:
        self.viewport_state.volume.scale = tuple(float(component) for component in value)

    def _current_level_index(self) -> int:
        if self.viewport_state.mode is RenderMode.VOLUME:
            return int(self.viewport_state.volume.level)
        return int(self.viewport_state.plane.applied_level)

    def _set_current_level_index(self, value: int) -> None:
        level = int(value)
        self.viewport_state.plane.applied_level = level
        self.viewport_state.plane.target_level = level
        self.viewport_state.volume.level = level

    def _ledger_axis_labels(self) -> tuple[str, ...]:
        return self._axis_labels

    def _ledger_order(self) -> tuple[int, ...]:
        return self._axis_order

    def _ledger_step(self) -> tuple[int, ...]:
        return (0, 0, 0)

    def _ledger_ndisplay(self) -> int:
        return 2

    def _ledger_level_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._level_shapes

    def _ledger_displayed(self) -> tuple[int, ...]:
        return self._displayed

    # Worker helpers required by render_snapshot ---------------------------------
    def _mark_render_tick_needed(self) -> None:
        self._render_tick_required = True

    def _dims_signature(self, snapshot: RenderLedgerSnapshot) -> tuple:
        return (
            int(snapshot.ndisplay) if snapshot.ndisplay is not None else None,
            tuple(int(v) for v in snapshot.order) if snapshot.order is not None else None,
            tuple(int(v) for v in snapshot.displayed) if snapshot.displayed is not None else None,
            tuple(int(v) for v in snapshot.current_step) if snapshot.current_step is not None else None,
            int(snapshot.current_level) if snapshot.current_level is not None else None,
            tuple(str(v) for v in snapshot.axis_labels) if snapshot.axis_labels is not None else None,
        )

    def _apply_dims_from_snapshot(self, snapshot: RenderLedgerSnapshot, *, signature: tuple) -> None:
        dims = self._viewer.dims
        self._last_dims_signature = signature

        if snapshot.ndisplay is not None:
            dims.ndisplay = max(1, int(snapshot.ndisplay))

        if snapshot.axis_labels is not None:
            dims.axis_labels = tuple(str(v) for v in snapshot.axis_labels)  # type: ignore[assignment]

        if snapshot.order is not None:
            dims.order = tuple(int(v) for v in snapshot.order)  # type: ignore[assignment]

        # napari derives `displayed` from order/ndisplay; no direct setter is available.

        if snapshot.current_step is not None:
            step_tuple = tuple(int(v) for v in snapshot.current_step)
            dims.current_step = step_tuple  # type: ignore[assignment]

        if snapshot.current_level is not None:
            level = int(snapshot.current_level)
            self.viewport_state.plane.applied_level = level
            self.viewport_state.plane.target_level = level
            self.viewport_state.volume.level = level

        self._update_z_index_from_snapshot(snapshot)

    def _update_z_index_from_snapshot(self, snapshot: RenderLedgerSnapshot) -> None:
        if snapshot.axis_labels is None or snapshot.current_step is None:
            return
        labels = [str(label).lower() for label in snapshot.axis_labels]
        if "z" not in labels:
            return
        idx = labels.index("z")
        steps = tuple(int(v) for v in snapshot.current_step)
        if idx < len(steps):
            self._z_index = int(steps[idx])

    def _ensure_scene_source(self) -> _HarnessSceneSource:
        if self._scene_source is None:
            self._scene_source = _HarnessSceneSource()
        return self._scene_source

    def _configure_camera_for_mode(self) -> None:
        if self.viewport_state.mode is RenderMode.VOLUME:
            camera = _HarnessTurntableCamera()
        else:
            camera = _HarnessPanZoomCamera()
        self.view = SimpleNamespace(camera=camera)

    def _register_plane_visual(self, node: Any) -> None:
        self._plane_visual_node = node

    def _register_volume_visual(self, node: Any) -> None:
        self._volume_visual_node = node

    def _ensure_plane_visual(self) -> Any:
        self._volume_visual_node.visible = False
        self._plane_visual_node.visible = True
        return self._plane_visual_node

    def _ensure_volume_visual(self) -> Any:
        self._plane_visual_node.visible = False
        self._volume_visual_node.visible = True
        return self._volume_visual_node

    def _load_slice(self, source: _HarnessSceneSource, level: int, z_index: int):
        return source.slice(level, z_index, compute=True)

    def _get_level_volume(self, source: _HarnessSceneSource, level: int) -> np.ndarray:
        shape = source.level_shape(level)
        return np.zeros(shape, dtype=np.float32)

    def _load_volume(self, source: _HarnessSceneSource, level: int) -> np.ndarray:
        return self._get_level_volume(source, level)

    def _request_encoder_idr(self) -> None:
        return None

    def _emit_current_camera_pose(self, _reason: str) -> None:
        self._pose_seq += 1

    def enqueue_update(self, delta: RenderUpdate) -> None:
        """Capture controller render updates for assertions without a render loop."""

        self.render_updates.append(delta)

        if delta.mode is not None:
            self.viewport_state.mode = delta.mode

        if delta.plane_state is not None:
            self.viewport_state.plane = replace(delta.plane_state)

        if delta.volume_state is not None:
            self.viewport_state.volume = replace(delta.volume_state)

        if delta.scene_state is not None:
            self._last_scene_state = delta.scene_state
            self._mark_render_tick_needed()

    def _update_level_metadata(self, descriptor: SimpleNamespace, applied: Any) -> None:
        level = int(applied.level)
        self.viewport_state.plane.applied_level = level
        self.viewport_state.plane.target_level = level
        self.viewport_state.volume.level = level
        self._z_index = int(applied.z_index) if applied.z_index is not None else 0
        self._zarr_level = getattr(descriptor, "path", None)
        self._zarr_shape = tuple(int(v) for v in getattr(descriptor, "shape", ()))
        self._zarr_axes = getattr(applied, "axes", "".join(self._axis_labels))
        self._zarr_dtype = str(getattr(applied, "dtype", self._zarr_dtype))
        source = self._ensure_scene_source()
        source.current_level = int(applied.level)
        source.set_current_slice(tuple(int(v) for v in applied.step), int(applied.level))



class StateServerHarness:
    """Stand up a control server façade plus fake websocket for protocol tests."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, width: int = 640, height: int = 480) -> None:
        self.loop = loop
        self.width = width
        self.height = height
        self.metrics = DummyMetrics()
        self.sent_raw: list[str] = []
        self.sent_frames: list[dict[str, Any]] = []
        self.scheduled: MutableSequence[ScheduledCall] = []
        self._task_failures: list[tuple[str, BaseException]] = []
        self._send_event = asyncio.Event()

        self.server = self._build_server()
        self.websocket = FakeStateWebSocket(loop)
        self._ingest_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------ public API

    async def start(self) -> None:
        """Launch the ingest loop if not already running."""

        if self._ingest_task is not None:
            return
        self._ingest_task = self.loop.create_task(
            state_channel_handler.ingest_state(self.server, self.websocket)
        )

    async def stop(self) -> None:
        """Tear down the ingest loop and flush scheduled tasks."""

        await self.websocket.close()
        if self._ingest_task is not None:
            await self._ingest_task
            self._ingest_task = None
        await self.drain_scheduled()

    def queue_client_payload(self, payload: Mapping[str, Any]) -> None:
        """Serialise and queue a JSON payload from the fake client."""

        self.websocket.push_message(json.dumps(payload, separators=(",", ":")))

    def queue_raw(self, raw: str | bytes) -> None:
        """Push raw data into the websocket (used for malformed handshakes)."""

        self.websocket.push_message(raw)

    async def perform_handshake(
        self,
        hello: SessionHello,
        *,
        timeout: float = 1.0,
    ) -> dict[str, Any]:
        """Send a `session.hello` frame and await the server response."""

        self.queue_client_payload(hello.to_dict())
        await self.start()

        def _is_handshake(frame: dict[str, Any]) -> bool:
            return frame.get("type") in {"session.welcome", "session.reject"}

        return await self.wait_for_frame(_is_handshake, timeout=timeout)

    async def drain_scheduled(self) -> None:
        """Await and clear all scheduled tasks."""

        while self.scheduled:
            scheduled = self.scheduled.pop(0)
            try:
                await scheduled.task
            except asyncio.CancelledError:  # pragma: no cover - cancellation guard
                continue

    async def wait_for_frame(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 1.0,
    ) -> dict[str, Any]:
        """Wait until a captured frame satisfies *predicate*."""

        deadline = self.loop.time() + timeout
        probe_index = 0
        while True:
            while probe_index < len(self.sent_frames):
                frame = self.sent_frames[probe_index]
                probe_index += 1
                if predicate(frame):
                    return frame
            remaining = deadline - self.loop.time()
            if remaining <= 0:
                raise TimeoutError("expected frame not observed before timeout")
            self._send_event.clear()
            try:
                await asyncio.wait_for(self._send_event.wait(), remaining)
            except asyncio.TimeoutError as exc:  # pragma: no cover - defensive
                raise TimeoutError("expected frame not observed before timeout") from exc

    # ------------------------------------------------------------------ helpers

    def _build_server(self) -> SimpleNamespace:
        """Create the stub server object expected by control_channel_server."""

        worker = CaptureWorker()

        server = SimpleNamespace()
        server._state_lock = threading.RLock()
        server._worker = worker
        server._log_state_traces = False
        server._log_dims_info = False
        server._log_cam_info = False
        server._log_cam_debug = False
        server._allowed_render_modes = {"mip"}
        server.metrics = self.metrics
        server.width = self.width
        server.height = self.height
        server.cfg = SimpleNamespace(fps=60.0)
        server._initial_mode = RenderMode.PLANE
        server._scene_snapshot: Optional[SceneSnapshot] = None
        server._latest_render_snapshot: Optional[RenderLedgerSnapshot] = None
        server._camera_seq: dict[str, int] = {}

        def _next_camera_command_seq(target: str) -> int:
            current = server._camera_seq.get(target, 0) + 1
            server._camera_seq[target] = current
            return current

        server._next_camera_command_seq = _next_camera_command_seq

        server._state_clients: set[Any] = set()
        server._schedule_coro = self._schedule_coro
        server._state_ledger = ServerStateLedger()
        server._update_client_gauges = lambda: None
        server._camera_queue = CameraCommandQueue()

        base_metrics = NotifyDimsPayload.from_dict(_default_dims_snapshot())

        def _current_ndisplay() -> int:
            entry = server._state_ledger.get("view", "main", "ndisplay")
            if entry is None or entry.value is None:
                return int(base_metrics.ndisplay)
            value = int(entry.value)
            return 3 if value >= 3 else 2

        server._current_ndisplay = _current_ndisplay

        def _refresh_scene_snapshot(render_state: Optional[RenderLedgerSnapshot] = None) -> None:
            render_snapshot = render_state or snapshot_render_state(server._state_ledger)
            ledger_snapshot = server._state_ledger.snapshot()
            scene_snapshot = snapshot_scene(
                render_state=render_snapshot,
                ledger_snapshot=ledger_snapshot,
                canvas_size=(self.width, self.height),
                fps_target=float(server.cfg.fps),
                default_layer_id="layer-0",
                default_layer_name="napari-cuda",
                ndisplay=_current_ndisplay(),
                zarr_path=None,
                scene_source=worker._ensure_scene_source(),
                layer_controls=snapshot_layer_controls(ledger_snapshot),
                multiscale_state=snapshot_multiscale_state(ledger_snapshot),
                volume_state=snapshot_volume_state(ledger_snapshot),
            )
            server._scene_snapshot = scene_snapshot
            server._latest_render_snapshot = render_snapshot

        server._refresh_scene_snapshot = _refresh_scene_snapshot
        server._update_scene_manager = _refresh_scene_snapshot

        def _mirror_apply(payload: NotifyDimsPayload) -> None:
            _ = payload
            server._refresh_scene_snapshot()

        server._resumable_store = ResumableHistoryStore(
            {
                state_channel_handler.NOTIFY_SCENE_TYPE: ResumableRetention(),
                state_channel_handler.NOTIFY_LAYERS_TYPE: ResumableRetention(
                    min_deltas=512,
                    max_deltas=2048,
                    max_age_s=300.0,
                ),
                state_channel_handler.NOTIFY_STREAM_TYPE: ResumableRetention(
                    min_deltas=1,
                    max_deltas=32,
                ),
            }
        )

        async def _mirror_broadcast(payload: NotifyDimsPayload) -> None:
            await state_channel_handler._broadcast_dims_state(server, payload=payload)

        server._dims_mirror = ServerDimsMirror(
            ledger=server._state_ledger,
            broadcaster=_mirror_broadcast,
            schedule=lambda coro, label: self._schedule_coro(coro, label),
            on_payload=_mirror_apply,
        )
        async def _layer_mirror_broadcast(
            layer_id: str,
            changes: Mapping[str, object],
            intent_id: Optional[str],
            timestamp: float,
        ) -> None:
            await state_channel_handler._broadcast_layers_delta(
                server,
                layer_id=layer_id,
                changes=changes,
                intent_id=intent_id,
                timestamp=timestamp,
            )

        def _default_layer_id() -> Optional[str]:
            snapshot = server._scene_snapshot
            if snapshot is None or not snapshot.layers:
                return None
            return snapshot.layers[0].layer_id

        server._layer_mirror = ServerLayerMirror(
            ledger=server._state_ledger,
            broadcaster=_layer_mirror_broadcast,
            schedule=lambda coro, label: self._schedule_coro(coro, label),
            default_layer=_default_layer_id,
        )

        self._record_dims_to_ledger(server, base_metrics, origin="bootstrap")
        server._dims_mirror.start()
        server._layer_mirror.start()

        axis_labels = tuple(str(lbl) for lbl in (base_metrics.axis_labels or ("z", "y", "x")))
        order = tuple(int(idx) for idx in (base_metrics.order or (0, 1, 2)))
        level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in base_metrics.level_shapes)
        levels_payload = tuple(dict(level) for level in base_metrics.levels)
        reduce_bootstrap_state(
            server._state_ledger,
            step=tuple(int(v) for v in base_metrics.current_step),
            axis_labels=axis_labels,
            order=order,
            level_shapes=level_shapes,
            levels=levels_payload,
            current_level=int(base_metrics.current_level),
            ndisplay=int(base_metrics.ndisplay),
            origin="harness.bootstrap",
        )

        server._refresh_scene_snapshot()

        server._ndisplay_calls: list[int] = []

        async def _ingest_set_ndisplay(
            ndisplay: int,
            *,
            intent_id: str | None = None,
            timestamp: float | None = None,
            origin: str = "control.view.ndisplay",
        ) -> ServerLedgerUpdate:
            value = 3 if int(ndisplay) >= 3 else 2
            server._ndisplay_calls.append(value)
            server._state_ledger.record_confirmed(
                "view",
                "main",
                "ndisplay",
                value,
                origin=origin,
                timestamp=timestamp,
            )
            server._refresh_scene_snapshot()
            return ServerLedgerUpdate(
                scope="view",
                target="main",
                key="ndisplay",
                value=value,
                intent_id=intent_id,
                timestamp=timestamp,
                origin=origin,
            )

        server._ingest_set_ndisplay = _ingest_set_ndisplay
        server._ensure_keyframe_calls = 0

        async def _ensure_keyframe() -> None:
            server._ensure_keyframe_calls += 1

        server._ensure_keyframe = _ensure_keyframe
        server._idr_on_reset = True

        def _enqueue_camera_delta(cmd: Any) -> None:
            server._camera_queue.append(cmd)

        server._enqueue_camera_delta = _enqueue_camera_delta

        frame_queue: asyncio.Queue[pixel_broadcaster.FramePacket] = asyncio.Queue(maxsize=1)
        broadcast_state = pixel_broadcaster.PixelBroadcastState(
            frame_queue=frame_queue,
            clients=set(),
            log_sends=False,
        )
        from napari_cuda.server.control.pixel_channel import PixelChannelConfig, PixelChannelState

        server._pixel_channel = PixelChannelState(
            broadcast=broadcast_state,
            needs_stream_config=True,
        )
        server._pixel = SimpleNamespace(bypass_until_key=False)
        server._pixel_config = PixelChannelConfig(
            width=self.width,
            height=self.height,
            fps=60.0,
            codec_id=1,
            codec_name="h264",
            kf_watchdog_cooldown_s=0.30,
        )
        server._pixel_channel.last_avcc = b"\x01\x64\x00\x1f"

        def _try_reset_encoder() -> bool:
            # Pretend we requested a reset; important to keep the marker for tests.
            server._pixel_channel.broadcast.kf_last_reset_ts = time.time()
            return True

        server._try_reset_encoder = _try_reset_encoder

        async def _broadcast_stream_config(payload: NotifyStreamPayload) -> None:
            await state_channel_handler.broadcast_stream_config(server, payload=payload)

        server._broadcast_stream_config = _broadcast_stream_config

        server._state_send = self._state_send  # type: ignore[assignment]
        server._scene_source = worker._ensure_scene_source()

        def _commit_level(applied, downgraded):
            reduce_level_update(
                server._state_ledger,
                applied=applied,
                downgraded=bool(downgraded),
                origin="harness.level",
            )
            server._refresh_scene_snapshot()
            snapshot = server._latest_render_snapshot
            assert snapshot is not None, "harness scene snapshot unavailable"
            assert server._worker._viewer is not None, "harness worker viewer not initialised"
            snapshot_iface = SnapshotInterface(server._worker)
            snapshot_mod.apply_render_snapshot(snapshot_iface, snapshot)

        server._commit_level = _commit_level
        server._pending_tasks = self.scheduled
        server._refresh_scene_snapshot()
        return server

    def _state_send(self, _ws: Any, text: str) -> None:
        """Capture outbound frames for later assertions."""

        self.sent_raw.append(text)
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return
        self.sent_frames.append(frame)
        self._send_event.set()

    def _schedule_coro(self, coro: Coroutine[Any, Any, Any], label: str) -> None:
        task = self.loop.create_task(coro)
        scheduled = ScheduledCall(label=label, task=task)
        self.scheduled.append(scheduled)

        def _log_task_result(t: asyncio.Task[Any]) -> None:  # pragma: no cover - debug aid
            try:
                t.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._task_failures.append((label, exc))

        task.add_done_callback(_log_task_result)

    @staticmethod
    def _record_dims_to_ledger(
        server: SimpleNamespace,
        payload: NotifyDimsPayload,
        *,
        origin: str = "worker.state.dims",
    ) -> None:
        entries = [
            ("view", "main", "ndisplay", int(payload.ndisplay)),
            (
                "view",
                "main",
                "displayed",
                tuple(int(idx) for idx in payload.displayed) if payload.displayed is not None else None,
            ),
            ("dims", "main", "current_step", tuple(int(v) for v in payload.current_step)),
            ("dims", "main", "mode", str(payload.mode)),
            (
                "dims",
                "main",
                "order",
                tuple(int(idx) for idx in payload.order) if payload.order is not None else None,
            ),
            (
                "dims",
                "main",
                "axis_labels",
                tuple(str(label) for label in payload.axis_labels) if payload.axis_labels is not None else None,
            ),
            (
                "dims",
                "main",
                "labels",
                tuple(str(label) for label in payload.labels) if getattr(payload, "labels", None) is not None else None,
            ),
            ("multiscale", "main", "level", int(payload.current_level)),
            ("multiscale", "main", "levels", tuple(dict(level) for level in payload.levels)),
            (
                "multiscale",
                "main",
                "level_shapes",
                tuple(tuple(int(dim) for dim in shape) for shape in payload.level_shapes),
            ),
            ("multiscale", "main", "downgraded", payload.downgraded),
        ]
        server._state_ledger.batch_record_confirmed(entries, origin=origin, dedupe=False)


def frames_of_type(frames: Iterable[dict[str, Any]], frame_type: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == frame_type]


def _default_dims_snapshot() -> dict[str, Any]:
    return {
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
