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
import json
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Iterable, Mapping, MutableSequence

from websockets.exceptions import ConnectionClosedOK
from websockets.protocol import State

from napari_cuda.protocol import NotifyStreamPayload
from napari_cuda.protocol.messages import NotifyDimsPayload, SessionHello
from napari_cuda.server.control import control_channel_server as state_channel_handler
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
    ResumableRetention,
)
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_models import ServerLedgerUpdate
from napari_cuda.server.control.state_reducers import reduce_bootstrap_state, reduce_level_update
from napari_cuda.server.rendering import pixel_broadcaster
from napari_cuda.server.scene import create_server_scene_data, build_render_scene_state
from napari_cuda.server.scene.layer_manager import ViewerSceneManager
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot

from napari_cuda.server.runtime import worker_runtime


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


class CaptureWorker:
    """Match the behaviour exercised in state channel tests without a renderer."""

    def __init__(self) -> None:
        self.policy_calls: list[str] = []
        self.level_requests: list[tuple[int, Any]] = []
        self.force_idr_calls = 0
        self.use_volume = False
        self._is_ready = True
        self._data_wh = (640, 480)
        self._zarr_axes = "yx"
        self._zarr_shape = (480, 640)
        self._zarr_dtype = "float32"
        self.volume_dtype = "float32"
        self._active_ms_level = 0
        self._last_step = (0, 0)
        self._scene_source = None
        from napari_cuda.server.rendering.viewer_builder import canonical_axes_from_source

        self._canonical_axes = canonical_axes_from_source(
            axes=("y", "x"),
            shape=(self._zarr_shape[0], self._zarr_shape[1]),
            step=(0, 0),
            use_volume=False,
        )
        self._axis_labels = ("z", "y", "x")
        self._axis_order = (0, 1, 2)
        self._displayed = (1, 2)
        self._level_shapes = (
            (16, self._zarr_shape[0], self._zarr_shape[1]),
            (8, self._zarr_shape[0] // 2, self._zarr_shape[1] // 2),
        )

    def set_policy(self, policy: str) -> None:
        self.policy_calls.append(str(policy))

    def request_multiscale_level(self, level: int, path: Any) -> None:
        self.level_requests.append((int(level), path))

    def force_idr(self) -> None:
        self.force_idr_calls += 1

    def viewer_model(self) -> None:  # pragma: no cover - interface stub
        return None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ViewerSceneManager ledger adapters --------------------------------------------------
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

        from napari_cuda.server.control import latest_intent

        latest_intent.clear_all()

        scene = create_server_scene_data()
        manager = ViewerSceneManager((self.width, self.height))
        worker = CaptureWorker()
        manager.update_from_sources(
            worker=worker,
            scene_state=None,
            multiscale_state=None,
            volume_state=None,
            current_step=None,
            ndisplay=2,
            zarr_path=None,
            scene_source=None,
            layer_controls=None,
        )

        server = SimpleNamespace()
        server._scene = scene
        server._scene_manager = manager
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
        server.use_volume = False
        server._applied_seqs = {"view": 0, "dims": 0, "multiscale": 0}
        server._camera_seq: dict[str, int] = {}
        server._stage_layer_controls_from_ledger = lambda: None

        def _next_camera_command_seq(target: str) -> int:
            current = server._camera_seq.get(target, 0) + 1
            server._camera_seq[target] = current
            return current

        server._next_camera_command_seq = _next_camera_command_seq

        server._state_clients: set[Any] = set()
        server._schedule_coro = self._schedule_coro
        server._state_ledger = ServerStateLedger()
        server._update_client_gauges = lambda: None

        base_metrics = NotifyDimsPayload.from_dict(_default_dims_snapshot())
        scene.multiscale_state["current_level"] = 0
        scene.multiscale_state["levels"] = [{"index": 0, "shape": [10, 10, 10]}]

        def _mirror_apply(payload: NotifyDimsPayload) -> None:
            with server._state_lock:
                multiscale_state = scene.multiscale_state
                multiscale_state["current_level"] = payload.current_level
                multiscale_state["levels"] = [dict(level) for level in payload.levels]
                if payload.downgraded is not None:
                    multiscale_state["downgraded"] = bool(payload.downgraded)
                else:
                    multiscale_state.pop("downgraded", None)

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

        self._record_dims_to_ledger(server, base_metrics, origin="bootstrap")
        server._dims_mirror.start()

        axis_labels = tuple(str(lbl) for lbl in (base_metrics.axis_labels or ("z", "y", "x")))
        order = tuple(int(idx) for idx in (base_metrics.order or (0, 1, 2)))
        level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in base_metrics.level_shapes)
        levels_payload = tuple(dict(level) for level in base_metrics.levels)
        reduce_bootstrap_state(
            scene,
            server._state_ledger,
            server._state_lock,
            step=tuple(int(v) for v in base_metrics.current_step),
            axis_labels=axis_labels,
            order=order,
            level_shapes=level_shapes,
            levels=levels_payload,
            current_level=int(base_metrics.current_level),
            ndisplay=int(base_metrics.ndisplay),
            origin="harness.bootstrap",
        )

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
            server.use_volume = bool(value == 3)
            scene.use_volume = bool(value == 3)
            return ServerLedgerUpdate(
                scope="view",
                target="main",
                key="ndisplay",
                value=value,
                server_seq=scene.next_server_seq,
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
        scene.camera_deltas: list[Any] = []

        def _enqueue_camera_delta(cmd: Any) -> None:
            scene.camera_deltas.append(cmd)

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
        server._scene_source = None

        def _commit_level(applied, downgraded):
            reduce_level_update(
            server._scene,
            server._state_ledger,
            server._state_lock,
            applied=applied,
            downgraded=bool(downgraded),
            origin="harness.level",
        )
            try:
                worker_runtime.apply_worker_slice_level(server._worker, server._scene_source, applied)
            except Exception:
                pass
            server._scene.latest_state = build_render_scene_state(
                server._state_ledger,
                server._scene,
            )

        server._commit_level = _commit_level
        return server
        server._pending_tasks = self.scheduled
        server._scene.pending_layer_updates = {}
        server._scene.latest_state = RenderLedgerSnapshot(layer_updates={})

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
        server._state_ledger.batch_record_confirmed(entries, origin=origin)


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
