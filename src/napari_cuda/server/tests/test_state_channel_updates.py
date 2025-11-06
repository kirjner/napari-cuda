from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import Coroutine, Mapping
from types import SimpleNamespace
from typing import Any, Optional

from dataclasses import replace

import pytest

from napari_cuda.protocol import (
    PROTO_VERSION,
    FeatureToggle,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NotifyStreamPayload,
    build_call_command,
    build_state_update,
)
from napari_cuda.protocol.envelopes import build_session_hello
from napari_cuda.protocol.messages import HelloClientInfo, NotifyDimsPayload
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.control import (
    control_channel_server as state_channel_handler,
)
from napari_cuda.server.control.command_registry import COMMAND_REGISTRY
from napari_cuda.server.control.control_payload_builder import (
    build_notify_layers_payload,
    build_notify_scene_payload,
)
from napari_cuda.server.control.protocol.handshake import perform_state_handshake
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.mirrors.layer_mirror import ServerLayerMirror
from napari_cuda.server.control.resumable_history_store import (
    EnvelopeSnapshot,
    ResumableHistoryStore,
    ResumableRetention,
    ResumeDecision,
    ResumePlan,
)
from napari_cuda.server.control.state_models import ServerLedgerUpdate
from napari_cuda.server.control.state_reducers import (
    reduce_bootstrap_state,
    reduce_layer_property,
    reduce_level_update,
    reduce_plane_restore,
)
from napari_cuda.server.control.topics.notify.baseline import orchestrate_connect
from napari_cuda.server.control.topics.notify.dims import broadcast_dims_state
from napari_cuda.server.control.topics.notify.layers import broadcast_layers_delta
from napari_cuda.server.runtime.api import RuntimeHandle
from napari_cuda.server.runtime.camera import CameraCommandQueue
from napari_cuda.shared.dims_spec import dims_spec_from_payload
from napari_cuda.server.scene.viewport import (
    PlaneState,
    RenderMode,
    ViewportState,
    VolumeState,
)
from napari_cuda.server.scene import (
    LayerVisualState,
    RenderLedgerSnapshot,
    snapshot_layer_controls,
    snapshot_multiscale_state,
    snapshot_render_state,
    snapshot_scene,
    snapshot_volume_state,
)
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_to_payload,
)


class _CaptureWorker:
    def __init__(self) -> None:
        self.policy_calls: list[str] = []
        self.level_requests: list[tuple[int, Any]] = []
        self.force_idr_calls = 0
        self._is_ready = True
        self._data_wh = (640, 480)
        self._zarr_axes = "yx"
        self._zarr_shape = (480, 640)
        self._zarr_dtype = "float32"
        self.volume_dtype = "float32"
        self._last_step = (0, 0)
        self._scene_source = None
        self.viewport_state = ViewportState(mode=RenderMode.PLANE)
        initial_level = 0
        self._axis_labels = ("z", "y", "x")
        self._axis_order = (0, 1, 2)
        self._displayed = (1, 2)
        self._level_shapes = ((16, self._zarr_shape[0], self._zarr_shape[1]), (8, self._zarr_shape[0] // 2, self._zarr_shape[1] // 2))
        self.viewport_state.plane.applied_level = initial_level
        self.viewport_state.plane.applied_step = (0, 0)
        self.viewport_state.volume.level = initial_level
        self.enqueued_updates: list[Any] = []

    def set_policy(self, policy: str) -> None:
        self.policy_calls.append(str(policy))

    def request_multiscale_level(self, level: int, path: Any) -> None:
        self.level_requests.append((int(level), path))

    def force_idr(self) -> None:
        self.force_idr_calls += 1

    def viewer_model(self) -> None:
        return None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def enqueue_update(self, update: Any) -> None:
        self.enqueued_updates.append(update)


    def _ledger_axis_labels(self) -> tuple[str, ...]:
        return self._axis_labels

    def _ledger_order(self) -> tuple[int, ...]:
        return self._axis_order

    def _ledger_displayed(self) -> tuple[int, ...]:
        return self._displayed

    def _ledger_ndisplay(self) -> int:
        return 2

    def _ledger_step(self) -> tuple[int, ...]:
        return (0, 0, 0)

    def _ledger_level_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._level_shapes

    def _ledger_level(self) -> int:
        return int(self.viewport_state.plane.applied_level)


class _FakeWS(SimpleNamespace):
    __hash__ = object.__hash__


def _make_server() -> tuple[SimpleNamespace, list[Coroutine[Any, Any, None]], list[dict[str, Any]]]:
    worker = _CaptureWorker()

    server = SimpleNamespace()
    server._state_lock = threading.RLock()
    server._worker = worker
    server.runtime = RuntimeHandle(lambda: server._worker)
    server._log_state_traces = False
    server._log_dims_info = False
    server._log_cam_info = False
    server._log_cam_debug = False
    server._allowed_render_modes = {"mip"}
    server.metrics = SimpleNamespace(inc=lambda *a, **k: None)
    server.width = 640
    server.height = 480
    server.cfg = SimpleNamespace(fps=60.0)
    server._initial_mode = RenderMode.PLANE
    server._scene_snapshot: Optional[SceneSnapshot] = None
    server._latest_render_snapshot: Optional[RenderLedgerSnapshot] = None
    scheduled: list[Coroutine[Any, Any, None]] = []
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
        "call.command": FeatureToggle(
            enabled=True,
            version=1,
            resume=False,
            commands=COMMAND_REGISTRY.command_names(),
        ),
    }

    fake_ws = _FakeWS(
        _napari_cuda_session="test-session",
        _napari_cuda_features=features,
        _napari_cuda_sequencers={},
        _napari_cuda_resume_plan={}
    )

    async def _fake_send(text: str) -> None:
        await server._state_send(fake_ws, text)

    fake_ws.send = _fake_send  # type: ignore[attr-defined]

    server._state_clients = {fake_ws}
    baseline_axes = []
    axis_labels = ("z", "y", "x")
    for idx, label in enumerate(axis_labels):
        baseline_axes.append(
            DimsSpecAxis(
                index=idx,
                label=label,
                role=label.lower(),
                displayed=idx >= 1,
                order_position=idx,
                current_step=0,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=(10,),
                per_level_world=(AxisExtent(0.0, 9.0, 1.0),),
            )
        )

    baseline_spec = DimsSpec(
        version=1,
        ndim=3,
        ndisplay=2,
        order=(0, 1, 2),
        displayed=(1, 2),
        current_level=0,
        current_step=(0, 0, 0),
        level_shapes=((10, 10, 10),),
        plane_mode=True,
        axes=tuple(baseline_axes),
        levels=({"index": 0, "shape": [10, 10, 10]},),
        downgraded=False,
        labels=None,
    )

    baseline_dims = NotifyDimsPayload(
        current_step=(0, 0, 0),
        level_shapes=((10, 10, 10),),
        levels=({"index": 0, "shape": [10, 10, 10]},),
        current_level=0,
        downgraded=None,
        mode="plane",
        ndisplay=2,
        dims_spec=baseline_spec,
    )
    server._pixel = SimpleNamespace(bypass_until_key=False)

    server._schedule_coro = lambda coro, _label: scheduled.append(coro)
    server._state_ledger = ServerStateLedger()

    def _current_ndisplay() -> int:
        entry = server._state_ledger.get("dims", "main", "dims_spec")
        if entry is None or entry.value is None:
            return int(baseline_dims.ndisplay)
        spec = dims_spec_from_payload(entry.value)
        if spec is None:
            return int(baseline_dims.ndisplay)
        return 3 if int(spec.ndisplay) >= 3 else 2

    server._current_ndisplay = _current_ndisplay

    def _refresh_scene_snapshot(render_state: Optional[RenderLedgerSnapshot] = None) -> None:
        render_snapshot = render_state or snapshot_render_state(server._state_ledger)
        ledger_snapshot = server._state_ledger.snapshot()
        scene_snapshot = snapshot_scene(
            render_state=render_snapshot,
            ledger_snapshot=ledger_snapshot,
            canvas_size=(server.width, server.height),
            fps_target=float(server.cfg.fps),
            default_layer_id="layer-0",
            default_layer_name="napari-cuda",
            ndisplay=_current_ndisplay(),
            zarr_path=None,
            scene_source=None,
            layer_controls=snapshot_layer_controls(ledger_snapshot),
            multiscale_state=snapshot_multiscale_state(ledger_snapshot),
            volume_state=snapshot_volume_state(ledger_snapshot),
        )
        server._scene_snapshot = scene_snapshot
        server._latest_render_snapshot = render_snapshot

    server._refresh_scene_snapshot = _refresh_scene_snapshot
    server._update_scene_manager = _refresh_scene_snapshot

    def _mirror_schedule(coro: Coroutine[Any, Any, None], _label: str) -> None:
        scheduled.append(coro)

    async def _mirror_broadcast(payload: NotifyDimsPayload) -> None:
        await broadcast_dims_state(server, payload=payload)

    def _mirror_apply(payload: NotifyDimsPayload) -> None:
        _ = payload
        server._refresh_scene_snapshot()

    server._dims_mirror = ServerDimsMirror(
        ledger=server._state_ledger,
        broadcaster=_mirror_broadcast,
        schedule=_mirror_schedule,
        on_payload=_mirror_apply,
    )

    _record_dims_to_ledger(server, baseline_dims, origin="bootstrap")
    server._refresh_scene_snapshot()
    server._dims_mirror.start()

    async def _layer_broadcast(
        layer_id: str,
        state: LayerVisualState,
        intent_id: Optional[str],
        timestamp: float,
    ) -> None:
        await broadcast_layers_delta(
            server,
            layer_id=layer_id,
            state=state,
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
        broadcaster=_layer_broadcast,
        schedule=_mirror_schedule,
        default_layer=_default_layer_id,
    )
    server._default_layer_id = _default_layer_id
    server._layer_mirror.start()

    axis_labels = tuple(str(lbl) for lbl in (baseline_dims.axis_labels or ("z", "y", "x")))
    order = tuple(int(idx) for idx in (baseline_dims.order or (0, 1, 2)))
    level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in baseline_dims.level_shapes)
    levels_payload = tuple(dict(level) for level in baseline_dims.levels)
    reduce_bootstrap_state(
        server._state_ledger,
        step=tuple(int(v) for v in baseline_dims.current_step),
        axis_labels=axis_labels,
        order=order,
        level_shapes=level_shapes,
        levels=levels_payload,
        current_level=int(baseline_dims.current_level),
        ndisplay=int(baseline_dims.ndisplay),
        origin="test.bootstrap",
    )
    server._refresh_scene_snapshot()

    server._state_ledger.record_confirmed(
        "scene",
        "main",
        "op_state",
        "applied",
        origin="test.bootstrap",
        dedupe=False,
    )
    while scheduled:
        asyncio.run(scheduled.pop(0))

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

    server._resumable_store = ResumableHistoryStore(
        {
            "notify.scene": ResumableRetention(),
            "notify.layers": ResumableRetention(min_deltas=512, max_deltas=2048, max_age_s=300.0),
            "notify.stream": ResumableRetention(min_deltas=1, max_deltas=32),
        }
    )
    server._ensure_keyframe_calls = 0

    async def _ensure_keyframe() -> None:
        server._ensure_keyframe_calls += 1

    server._ensure_keyframe = _ensure_keyframe
    server._thumbnail_requests: list[str] = []

    def _queue_thumbnail_refresh(layer_id: Optional[str]) -> None:
        if layer_id:
            server._thumbnail_requests.append(layer_id)

    server._queue_thumbnail_refresh = _queue_thumbnail_refresh
    server._idr_on_reset = True
    server._pixel_channel = SimpleNamespace(
        broadcast=SimpleNamespace(bypass_until_key=False, waiting_for_keyframe=False)
    )
    server._pixel_config = SimpleNamespace()
    def _mark_stream_config_dirty() -> None:
        server._pixel_channel.needs_stream_config = True

    def _build_stream_payload(avcc: bytes) -> NotifyStreamPayload:
        data = avcc
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            data = bytes(data)
        encoded = base64.b64encode(data).decode("ascii")
        return NotifyStreamPayload(
            codec="stub",
            format="avcc",
            fps=float(server.cfg.fps),
            frame_size=(server.width, server.height),
            nal_length_size=4,
            avcc=encoded,
            latency_policy={},
            vt_hint=None,
        )

    server.mark_stream_config_dirty = _mark_stream_config_dirty
    server.build_stream_payload = _build_stream_payload
    server._camera_seq: dict[str, int] = {}

    def _next_camera_command_seq(target: str) -> int:
        current = server._camera_seq.get(target, 0) + 1
        server._camera_seq[target] = current
        return current

    server._next_camera_command_seq = _next_camera_command_seq

    server._camera_queue = CameraCommandQueue()

    def _enqueue_camera_delta(cmd: Any) -> None:
        server._camera_queue.append(cmd)

    server._enqueue_camera_delta = _enqueue_camera_delta

    return server, scheduled, captured


def _drain_scheduled(tasks: list[Coroutine[Any, Any, None]]) -> None:
    while tasks:
        coro = tasks.pop(0)
        asyncio.run(coro)


def _record_dims_to_ledger(
    server: Any,
    payload: NotifyDimsPayload,
    *,
    origin: str = "worker.state.dims",
) -> None:
    entries = [
        ("dims", "main", "current_step", tuple(int(v) for v in payload.current_step)),
    ]
    server._state_ledger.batch_record_confirmed(entries, origin=origin, dedupe=False)
    spec = _dims_spec_from_notify_payload(payload)
    server._state_ledger.record_confirmed(
        "dims",
        "main",
        "dims_spec",
        dims_spec_to_payload(spec),
        origin=origin,
    )


def _frames_of_type(frames: list[dict[str, Any]], frame_type: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == frame_type]

def _make_dims_snapshot(**overrides: Any) -> dict[str, Any]:
    overrides = dict(overrides)

    base_level_shapes = overrides.pop("level_shapes", [[512, 256, 64], [256, 128, 32], [128, 64, 16]])
    level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in base_level_shapes)
    base_step = overrides.get("current_step", overrides.get("step", [0, 0, 0]))
    current_step = tuple(int(v) for v in base_step)
    ndim = len(level_shapes[0]) if level_shapes else len(current_step)

    default_axis_labels = tuple(("z", "y", "x", "t", "c")[idx] if idx < 5 else f"axis-{idx}" for idx in range(ndim))
    axis_labels = tuple(overrides.pop("axis_labels", default_axis_labels))
    order_values = tuple(overrides.pop("order", tuple(range(ndim))))

    ndisplay = int(overrides.get("ndisplay", 2))
    displayed_default = tuple(range(max(0, ndim - ndisplay), ndim))
    displayed = tuple(overrides.pop("displayed", displayed_default))

    mode_value = overrides.get("mode", "volume" if ndisplay >= 3 else "plane")
    current_level = int(overrides.get("current_level", 0))
    downgraded = overrides.get("downgraded")
    labels_value = overrides.pop("labels", None)

    axes: list[DimsSpecAxis] = []
    order_lookup = {int(idx): pos for pos, idx in enumerate(order_values)}
    displayed_set = {int(idx) for idx in displayed}
    for idx in range(ndim):
        per_steps = tuple(shape[idx] if idx < len(shape) else 1 for shape in level_shapes)
        per_world = tuple(
            AxisExtent(start=0.0, stop=float(max(count - 1, 0)), step=1.0) for count in per_steps
        )
        current_step_val = current_step[idx] if idx < len(current_step) else 0
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=axis_labels[idx],
                role=axis_labels[idx].lower(),
                displayed=idx in displayed_set,
                order_position=int(order_lookup.get(idx, idx)),
                current_step=current_step_val,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=per_steps,
                per_level_world=per_world,
            )
        )

    levels_payload = overrides.get(
        "levels",
        [
            {"index": idx, "shape": list(shape)}
            for idx, shape in enumerate(level_shapes)
        ],
    )

    base_spec = DimsSpec(
        version=1,
        ndim=ndim,
        ndisplay=ndisplay,
        order=order_values,
        displayed=displayed,
        current_level=current_level,
        current_step=current_step,
        level_shapes=level_shapes,
        plane_mode=(mode_value != "volume"),
        axes=tuple(axes),
        levels=tuple(dict(level) for level in levels_payload),
        downgraded=downgraded,
        labels=None if labels_value is None else tuple(str(v) for v in labels_value),
    )

    base = {
        "step": [0, 0, 0],
        "current_step": [0, 0, 0],
        "ndisplay": ndisplay,
        "mode": mode_value,
        "levels": levels_payload,
        "current_level": current_level,
        "level_shapes": [list(shape) for shape in level_shapes],
        "downgraded": downgraded,
        "labels": labels_value,
        "dims_spec": dims_spec_to_payload(base_spec),
    }
    base.update(overrides)
    return base


def _dims_spec_from_notify_payload(payload: NotifyDimsPayload) -> DimsSpec:
    level_shapes = tuple(tuple(int(dim) for dim in shape) for shape in payload.level_shapes)
    current_step = tuple(int(v) for v in payload.current_step)
    ndim = len(current_step)
    axis_labels = tuple(payload.axis_labels or [f"axis-{idx}" for idx in range(ndim)])
    displayed = tuple(payload.displayed or tuple(range(max(0, ndim - int(payload.ndisplay)), ndim)))
    axes: list[DimsSpecAxis] = []
    for idx in range(ndim):
        per_steps = tuple(shape[idx] if idx < len(shape) else 1 for shape in level_shapes)
        per_world = tuple(
            AxisExtent(start=0.0, stop=float(max(count - 1, 0)), step=1.0) for count in per_steps
        )
        current_step_val = current_step[idx] if idx < len(current_step) else 0
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=axis_labels[idx],
                role=axis_labels[idx],
                displayed=idx in displayed,
                order_position=idx,
                current_step=current_step_val,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=per_steps,
                per_level_world=per_world,
            )
        )
    levels = tuple(dict(level) for level in payload.levels)
    return DimsSpec(
        version=1,
        ndim=ndim,
        ndisplay=int(payload.ndisplay),
        order=tuple(payload.order or tuple(range(ndim))),
        displayed=displayed,
        current_level=int(payload.current_level),
        current_step=current_step,
        level_shapes=level_shapes,
        plane_mode=int(payload.ndisplay) < 3,
        axes=tuple(axes),
        levels=levels,
        downgraded=bool(payload.downgraded) if payload.downgraded is not None else False,
        labels=tuple(payload.labels) if payload.labels is not None else None,
    )



def _build_state_update(payload: dict[str, Any], *, intent_id: str, frame_id: str) -> dict[str, Any]:
    frame = build_state_update(
        session_id="test-session",
        intent_id=intent_id,
        frame_id=frame_id,
        payload=payload,
    )
    return frame.to_dict()


def test_layer_update_emits_ack_and_notify() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "layer",
        "target": "layer-0",
        "key": "colormap",
        "value": "red",
    }

    frame = _build_state_update(payload, intent_id="layer-intent", frame_id="state-layer-1")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    entry = server._state_ledger.get("layer", "layer-0", "colormap")
    assert entry is not None
    assert entry.value == "red"
    assert entry.version is not None

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "layer-intent"
    assert ack_payload["in_reply_to"] == "state-layer-1"
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == "red"
    assert ack_payload["version"] == entry.version

    layer_frames = _frames_of_type(captured, "notify.layers")
    assert layer_frames, "expected notify.layers frame"
    notify_payload = layer_frames[-1]["payload"]
    assert notify_payload["layer_id"] == "layer-0"
    assert notify_payload["controls"]["colormap"] == "red"


def test_layer_update_dedupe_preserves_version() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "layer",
        "target": "layer-0",
        "key": "opacity",
        "value": 0.6,
    }

    frame = _build_state_update(payload, intent_id="layer-dedupe", frame_id="state-layer-dup")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)
    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == pytest.approx(0.6)
    base_version = ack_payload["version"]

    entry = server._state_ledger.get("layer", "layer-0", "opacity")
    assert entry is not None
    assert entry.value == pytest.approx(0.6)
    assert entry.version == base_version

    captured.clear()

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    entry_after = server._state_ledger.get("layer", "layer-0", "opacity")
    assert entry_after is not None
    assert entry_after.version == base_version

    replay_acks = _frames_of_type(captured, "ack.state")
    assert len(replay_acks) == 1
    replay_payload = replay_acks[0]["payload"]
    assert replay_payload["status"] == "accepted"
    assert replay_payload["version"] == base_version


def test_call_command_requests_keyframe() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-test",
        payload={"command": "napari.pixel.request_keyframe"},
    )

    asyncio.run(state_channel_handler._ingest_call_command(server, frame.to_dict(), fake_ws))

    assert server._ensure_keyframe_calls == 1
    replies = _frames_of_type(captured, "reply.command")
    assert replies, "expected reply.command frame"
    payload = replies[-1]["payload"]
    assert payload["status"] == "ok"
    assert payload["in_reply_to"] == "cmd-test"


def test_call_command_unknown_command_errors() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-missing",
        payload={"command": "napari.pixel.unknown"},
    )

    asyncio.run(state_channel_handler._ingest_call_command(server, frame.to_dict(), fake_ws))

    errors = _frames_of_type(captured, "error.command")
    assert errors, "expected error.command frame"
    payload = errors[-1]["payload"]
    assert payload["status"] == "error"
    assert payload["in_reply_to"] == "cmd-missing"
    assert payload["code"] == "command.not_found"


def test_call_command_rejection_propagates_error() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    async def _reject_keyframe() -> None:
        raise state_channel_handler.CommandRejected(
            code="command.busy",
            message="encoder busy",
            details={"retry_after_ms": 250},
        )

    server._ensure_keyframe = _reject_keyframe

    frame = build_call_command(
        session_id="test-session",
        frame_id="cmd-reject",
        payload={"command": "napari.pixel.request_keyframe"},
    )

    asyncio.run(state_channel_handler._ingest_call_command(server, frame.to_dict(), fake_ws))

    errors = _frames_of_type(captured, "error.command")
    assert errors, "expected error.command frame"
    payload = errors[-1]["payload"]
    assert payload["status"] == "error"
    assert payload["code"] == "command.busy"
    assert payload["message"] == "encoder busy"
    assert payload["details"] == {"retry_after_ms": 250}


def test_layer_update_rejects_unknown_key() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "layer",
        "target": "layer-0",
        "key": "unknown",
        "value": "noop",
    }

    frame = _build_state_update(payload, intent_id="bad-layer", frame_id="state-layer-err")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["status"] == "rejected"
    assert ack_payload["intent_id"] == "bad-layer"
    assert ack_payload["in_reply_to"] == "state-layer-err"
    assert ack_payload["error"]["code"] == "state.invalid"


def test_dims_update_emits_ack_and_notify() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "dims",
        "target": "z",
        "key": "step",
        "value": 5,
    }

    frame = _build_state_update(payload, intent_id="dims-intent", frame_id="state-dims-1")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
    assert spec_entry is not None
    spec_payload = dims_spec_from_payload(spec_entry.value)
    assert spec_payload is not None
    assert tuple(int(v) for v in spec_payload.current_step) == (5, 0, 0)
    assert spec_entry.version is not None

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "dims-intent"
    assert ack_payload["in_reply_to"] == "state-dims-1"
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == 5
    assert ack_payload["version"] == spec_entry.version

    notify_payload = NotifyDimsPayload.from_dict(
        _make_dims_snapshot(
            step=[5, 0, 0],
            current_step=[5, 0, 0],
        )
    )
    _record_dims_to_ledger(server, notify_payload)
    _drain_scheduled(scheduled)

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames
    dims_payload = dims_frames[-1]["payload"]
    assert dims_payload["current_step"][0] == 5


def test_view_ndisplay_update_ack_is_immediate() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    captured.clear()

    payload = {
        "scope": "view",
        "target": "main",
        "key": "ndisplay",
        "value": 3,
    }

    frame = _build_state_update(payload, intent_id="view-intent", frame_id="state-view-1")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
    assert spec_entry is not None
    spec_payload = dims_spec_from_payload(spec_entry.value)
    assert spec_payload is not None
    assert int(spec_payload.ndisplay) == 3
    assert spec_entry.version is not None
    assert server._state_ledger.get("view", "main", "ndisplay") is None

    acks = _frames_of_type(captured, "ack.state")
    assert acks, "expected immediate ack"
    ack_payload = acks[-1]["payload"]
    assert ack_payload["intent_id"] == "view-intent"
    assert ack_payload["in_reply_to"] == "state-view-1"
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == 3
    assert ack_payload["version"] == spec_entry.version

    interim_dims = _frames_of_type(captured, "notify.dims")
    if interim_dims:
        assert interim_dims[-1]["payload"]["ndisplay"] == 2

    notify_payload = NotifyDimsPayload.from_dict(
        _make_dims_snapshot(
            step=[0, 0, 0],
            current_step=[0, 0, 0],
            level_shapes=[[512, 256, 64], [256, 128, 32]],
            ndisplay=3,
            mode="volume",
            displayed=[0, 1, 2],
            current_level=1,
        )
    )
    _record_dims_to_ledger(server, notify_payload)
    _drain_scheduled(scheduled)

    spec_entry = server._state_ledger.get("dims", "main", "dims_spec")
    assert spec_entry is not None
    spec_payload = dims_spec_from_payload(spec_entry.value)
    assert spec_payload is not None
    assert int(spec_payload.ndisplay) == 3
    assert int(spec_payload.current_level) == 1





def test_worker_dims_snapshot_updates_multiscale_state() -> None:
    server, scheduled, captured = _make_server()
    captured.clear()

    snapshot_payload = _make_dims_snapshot(
        level_shapes=[[512, 256, 64], [256, 128, 32]],
        displayed=[0, 1, 2],
        current_level=1,
        downgraded=True,
    )
    snapshot_payload["levels"] = [
        {"index": 0, "shape": [512, 256, 64]},
        {"index": 1, "shape": [256, 128, 32]},
    ]
    notify_payload = NotifyDimsPayload.from_dict(snapshot_payload)

    _record_dims_to_ledger(server, notify_payload)
    _drain_scheduled(scheduled)

    dims_frames = _frames_of_type(captured, "notify.dims")
    assert dims_frames, "expected dims broadcast"
    payload = dims_frames[-1]["payload"]
    assert payload["current_level"] == 1
    assert payload.get("downgraded") is True

    ms_state = snapshot_multiscale_state(server._state_ledger.snapshot())
    assert ms_state.get("current_level") == 1
    levels_snapshot = ms_state.get("levels") or []
    assert levels_snapshot and levels_snapshot[1]["shape"] == [256, 128, 32]
    assert ms_state.get("downgraded") is True



def test_volume_rendering_update() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))
    server._allowed_render_modes = {"mip", "iso"}

    payload = {
        "scope": "volume",
        "target": "main",
        "key": "rendering",
        "value": "iso",
    }

    frame = _build_state_update(payload, intent_id="volume-intent", frame_id="state-volume-1")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    entry = server._state_ledger.get("volume", "main", "rendering")
    assert entry is not None and entry.value == "iso"
    volume_entry = server._state_ledger.get("volume", "main", "rendering")
    assert volume_entry is not None
    assert volume_entry.version is not None

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "volume-intent"
    assert ack_payload["in_reply_to"] == "state-volume-1"
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == "iso"
    assert ack_payload["version"] == volume_entry.version

    layer_frames = _frames_of_type(captured, "notify.layers")
    assert layer_frames
    latest_layer_payload = layer_frames[-1]["payload"]
    assert latest_layer_payload["layer_id"] == "layer-0"
    assert latest_layer_payload["controls"]["rendering"] == "iso"


def test_camera_zoom_update_emits_notify() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "zoom",
        "value": {"factor": 1.2, "anchor_px": [12, 24]},
    }

    frame = _build_state_update(payload, intent_id="cam-zoom", frame_id="state-cam-1")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    deltas = server._camera_queue.pop_all()
    assert deltas, "expected camera delta queued"
    cmd = deltas[-1]
    assert cmd.kind == 'zoom'
    assert cmd.factor == 1.2
    assert cmd.anchor_px == (12.0, 24.0)
    seq = server._camera_seq.get("main")
    assert seq is not None

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "cam-zoom"
    assert ack_payload["in_reply_to"] == "state-cam-1"
    assert ack_payload["status"] == "accepted"
    assert ack_payload["applied_value"] == {"factor": 1.2, "anchor_px": [12.0, 24.0]}
    assert ack_payload["version"] == seq

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames, "expected notify.camera frame"
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "zoom"
    assert camera_payload["delta"]["factor"] == 1.2
    assert camera_payload["delta"]["anchor_px"] == [12.0, 24.0]
    assert camera_payload["origin"] == "state.update"


def test_camera_reset_triggers_keyframe() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "reset",
        "value": {"reason": "ui"},
    }

    frame = _build_state_update(payload, intent_id="cam-reset", frame_id="state-cam-2")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    deltas = server._camera_queue.pop_all()
    assert deltas and deltas[-1].kind == 'reset'
    assert server._ensure_keyframe_calls == 1
    seq = server._camera_seq.get("main")
    assert seq is not None

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["intent_id"] == "cam-reset"
    assert ack_payload["applied_value"] == {"reason": "ui"}
    assert ack_payload["version"] == seq

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "reset"
    assert camera_payload["delta"] == {"reason": "ui"}


def test_camera_set_updates_latest_state() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    payload = {
        "scope": "camera",
        "target": "main",
        "key": "set",
        "value": {
            "center": [1, 2, 3],
            "zoom": 2.5,
        },
    }

    frame = _build_state_update(payload, intent_id="cam-set", frame_id="state-cam-3")

    asyncio.run(state_channel_handler._ingest_state_update(server, frame, fake_ws))
    _drain_scheduled(scheduled)

    center_entry = server._state_ledger.get("camera_plane", "main", "center")
    zoom_entry = server._state_ledger.get("camera_plane", "main", "zoom")

    assert center_entry is not None and center_entry.value == (1.0, 2.0)
    assert zoom_entry is not None and zoom_entry.value == 2.5
    assert center_entry.version is not None
    assert zoom_entry.version is not None
    expected_version = max(int(center_entry.version), int(zoom_entry.version))

    acks = _frames_of_type(captured, "ack.state")
    assert acks
    payload = acks[-1]["payload"]
    applied = payload["applied_value"]
    assert applied == {
        "center": [1.0, 2.0],
        "zoom": 2.5,
    }
    assert payload["version"] == expected_version

    notify_frames = _frames_of_type(captured, "notify.camera")
    assert notify_frames
    camera_payload = notify_frames[-1]["payload"]
    assert camera_payload["mode"] == "set"
    assert camera_payload["delta"] == applied


def test_parse_failure_emits_rejection_ack() -> None:
    server, scheduled, captured = _make_server()
    fake_ws = next(iter(server._state_clients))

    bad_frame = {
        "type": "state.update",
        "version": PROTO_VERSION,
        "session": "test-session",
        "frame_id": "state-bad-1",
        "timestamp": 0.0,
        "intent_id": "bad-intent",
        "payload": {"scope": "layer", "key": "opacity"},
    }

    asyncio.run(state_channel_handler._ingest_state_update(server, bad_frame, fake_ws))
    _drain_scheduled(scheduled)

    acks = _frames_of_type(captured, "ack.state")
    assert len(acks) == 1
    ack_payload = acks[0]["payload"]
    assert ack_payload["status"] == "rejected"
    assert ack_payload["intent_id"] == "bad-intent"
    assert ack_payload["in_reply_to"] == "state-bad-1"
    assert ack_payload["error"]["code"] == "state.invalid"


def test_send_state_baseline_emits_notifications(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        server, scheduled, captured = _make_server()
        server._state_ledger.record_confirmed(
            "view",
            "main",
            "ndisplay",
            2,
            origin="test.reset",
        )
        reduce_layer_property(
            server._state_ledger,
            layer_id="layer-0",
            prop="opacity",
            value=0.25,
        )
        reduce_layer_property(
            server._state_ledger,
            layer_id="layer-0",
            prop="colormap",
            value="viridis",
        )

        snapshot_state = snapshot_render_state(server._state_ledger)
        _drain_scheduled(scheduled)
        captured.clear()

        server._update_scene_manager()

        server._await_worker_bootstrap = lambda _timeout: asyncio.sleep(0)
        async def _state_send(_ws: Any, text: str) -> None:
            captured.append(json.loads(text))

        server._state_send = _state_send
        server._update_scene_manager = lambda: None
        server._schedule_coro = lambda coro, _label: scheduled.append(coro)
        server._pixel_channel = SimpleNamespace(last_avcc=None)
        server._pixel_config = SimpleNamespace()

        server.mark_stream_config_dirty = lambda: None

        class _CaptureWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, payload: str) -> None:
                self.sent.append(payload)
                captured.append(json.loads(payload))

        ws = _CaptureWS()
        ws._napari_cuda_session = "baseline-session"
        ws._napari_cuda_features = {
            "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
        }
        ws._napari_cuda_sequencers = {}
        ws._napari_cuda_resume_plan = {}

        resume_map = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
        loop.run_until_complete(orchestrate_connect(server, ws, resume_map))
        _drain_scheduled(scheduled)

        scene_frames = _frames_of_type(captured, "notify.scene")
        assert scene_frames, "expected notify.scene snapshot"

        layer_frames = _frames_of_type(captured, "notify.layers")
        assert layer_frames, "expected notify.layers baseline"
        baseline_frame = next(
            frame
            for frame in reversed(layer_frames)
            if frame.get("session") == "baseline-session" and frame["payload"].get("controls")
        )
        controls = baseline_frame["payload"].get("controls", {})
        assert controls["opacity"] == 0.25
        assert controls["colormap"] == "viridis"

        dims_frames = _frames_of_type(captured, "notify.dims")
        assert dims_frames, "expected notify.dims baseline"
        dims_payload = dims_frames[-1]["payload"]
        entry = server._state_ledger.get("dims", "main", "current_step")
        assert entry is not None
        assert dims_payload["step"] == list(entry.value)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_layer_baseline_replay_without_deltas_sends_defaults(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        server, scheduled, captured = _make_server()
        server._state_ledger.record_confirmed(
            "view",
            "main",
            "ndisplay",
            2,
            origin="test.reset",
        )
        _drain_scheduled(scheduled)
        captured.clear()

        server._update_scene_manager()

        server._await_worker_bootstrap = lambda _timeout: asyncio.sleep(0)
        async def _state_send(_ws: Any, text: str) -> None:
            captured.append(json.loads(text))

        server._state_send = _state_send
        server._update_scene_manager = lambda: None
        server._schedule_coro = lambda coro, _label: scheduled.append(coro)
        server._pixel_channel = SimpleNamespace(last_avcc=None)
        server._pixel_config = SimpleNamespace()

        server.mark_stream_config_dirty = lambda: None

        class _CaptureWS:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send(self, payload: str) -> None:
                self.sent.append(payload)
                captured.append(json.loads(payload))

        ws = _CaptureWS()
        ws._napari_cuda_session = "baseline-session"
        ws._napari_cuda_features = {
            "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
            "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
        }
        ws._napari_cuda_sequencers = {}
        ws._napari_cuda_resume_plan = {
            NOTIFY_SCENE_TYPE: ResumePlan(
                topic=NOTIFY_SCENE_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[],
            ),
            NOTIFY_LAYERS_TYPE: ResumePlan(
                topic=NOTIFY_LAYERS_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[],
            ),
            NOTIFY_STREAM_TYPE: ResumePlan(
                topic=NOTIFY_STREAM_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[],
            ),
        }

        resume_map = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
        loop.run_until_complete(orchestrate_connect(server, ws, resume_map))
        _drain_scheduled(scheduled)

        layer_frames = _frames_of_type(captured, "notify.layers")
        assert layer_frames, "expected notify.layers snapshot even with replay resume plan"
        baseline_frame = next(
            frame
            for frame in reversed(layer_frames)
            if frame.get("session") == "baseline-session" and frame["payload"].get("controls")
        )
        controls = baseline_frame["payload"].get("controls", {})
        assert controls["colormap"] == "gray"
        entry = server._state_ledger.get("layer", "layer-0", "colormap")
        assert entry is not None and entry.value == "gray"
        volume_entry = server._state_ledger.get("volume", "main", "colormap")
        assert volume_entry is not None and volume_entry.value == "gray"
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _prepare_resumable_payloads(server: Any) -> tuple[EnvelopeSnapshot, EnvelopeSnapshot, EnvelopeSnapshot]:
    store: ResumableHistoryStore = server._resumable_store
    server._refresh_scene_snapshot()
    scene_snapshot = server._scene_snapshot
    assert scene_snapshot is not None
    scene_payload = build_notify_scene_payload(
        scene_snapshot=scene_snapshot,
        ledger_snapshot=server._state_ledger.snapshot(),
        viewer_settings={"fps_target": 60.0},
    )
    scene_snapshot = store.snapshot_envelope(
        NOTIFY_SCENE_TYPE,
        payload=scene_payload.to_dict(),
        timestamp=time.time(),
    )

    layer_payload = build_notify_layers_payload(
        layer_id="layer-0",
        controls={"opacity": 0.42},
    )
    layer_snapshot = store.delta_envelope(
        NOTIFY_LAYERS_TYPE,
        payload=layer_payload.to_dict(),
        timestamp=time.time(),
    )

    stream_payload = NotifyStreamPayload(
        codec="h264",
        format="annexb",
        fps=60.0,
        frame_size=(640, 480),
        nal_length_size=4,
        avcc="AAAA",
        latency_policy={"mode": "low"},
    )
    stream_snapshot = store.snapshot_envelope(
        NOTIFY_STREAM_TYPE,
        payload=stream_payload.to_dict(),
        timestamp=time.time(),
    )

    return scene_snapshot, layer_snapshot, stream_snapshot


def test_handshake_stashes_resume_plan(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        server, scheduled, captured = _make_server()
        scene_snap, layer_snap, stream_snap = _prepare_resumable_payloads(server)
        resume_tokens = {
            NOTIFY_SCENE_TYPE: scene_snap.delta_token,
            NOTIFY_LAYERS_TYPE: layer_snap.delta_token,
            NOTIFY_STREAM_TYPE: stream_snap.delta_token,
        }

        hello = build_session_hello(
            client=HelloClientInfo(name="tests", version="1.0", platform="test"),
            features={
                NOTIFY_SCENE_TYPE: True,
                NOTIFY_LAYERS_TYPE: True,
                NOTIFY_STREAM_TYPE: True,
            },
            resume_tokens=resume_tokens,
        )

        hello_json = json.dumps(hello.to_dict(), separators=(",", ":"))

        class _WS:
            def __init__(self) -> None:
                self.sent: list[str] = []
                self.transport = SimpleNamespace(get_extra_info=lambda *_: None)

            async def recv(self) -> str:
                return hello_json

            async def send(self, payload: str) -> None:
                self.sent.append(payload)
                captured.append(json.loads(payload))

            async def close(self) -> None:  # pragma: no cover - defensive
                pass

        ws = _WS()

        async def _state_send(_ws: Any, text: str) -> None:
            captured.append(json.loads(text))

        server._state_send = _state_send

        loop.run_until_complete(perform_state_handshake(server, ws))
        _drain_scheduled(scheduled)

        plan = ws._napari_cuda_resume_plan
        assert plan[NOTIFY_SCENE_TYPE].decision == ResumeDecision.REPLAY
        assert plan[NOTIFY_LAYERS_TYPE].decision == ResumeDecision.REPLAY
        assert plan[NOTIFY_STREAM_TYPE].decision == ResumeDecision.REPLAY

        welcome = _frames_of_type(captured, "session.welcome")
        assert welcome, "expected handshake welcome frame"
        command_feature = welcome[0]["payload"]["features"].get("call.command")
        assert command_feature is not None
        commands = command_feature.get("commands")
        assert commands is not None
        assert "napari.pixel.request_keyframe" in commands
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_send_state_baseline_replays_store(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        server, scheduled, captured = _make_server()
        scene_snap, layer_snap, stream_snap = _prepare_resumable_payloads(server)

        server._await_worker_bootstrap = lambda _timeout: asyncio.sleep(0)
        async def _state_send(_ws: Any, text: str) -> None:
            captured.append(json.loads(text))

        server._state_send = _state_send
        server._pixel_channel = SimpleNamespace(last_avcc=b"avcc")
        server._pixel_config = SimpleNamespace()

        ws = _FakeWS(
            _napari_cuda_session="resume-session",
            _napari_cuda_features={
                "notify.scene": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.layers": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.stream": FeatureToggle(enabled=True, version=1, resume=True),
                "notify.dims": FeatureToggle(enabled=True, version=1, resume=False),
            },
            _napari_cuda_sequencers={},
        )

        async def _ws_send(payload: str) -> None:
            captured.append(json.loads(payload))

        ws.send = _ws_send  # type: ignore[attr-defined]
        ws._napari_cuda_resume_plan = {
            NOTIFY_SCENE_TYPE: ResumePlan(
                topic=NOTIFY_SCENE_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[scene_snap],
            ),
            NOTIFY_LAYERS_TYPE: ResumePlan(
                topic=NOTIFY_LAYERS_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[layer_snap],
            ),
            NOTIFY_STREAM_TYPE: ResumePlan(
                topic=NOTIFY_STREAM_TYPE,
                decision=ResumeDecision.REPLAY,
                deltas=[stream_snap],
            ),
        }

        resume_map = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
        loop.run_until_complete(orchestrate_connect(server, ws, resume_map))
        _drain_scheduled(scheduled)

        scene_frames = _frames_of_type(captured, "notify.scene")
        assert scene_frames and scene_frames[-1]["seq"] == scene_snap.seq

        layer_frames = _frames_of_type(captured, "notify.layers")
        assert layer_frames and layer_frames[-1]["seq"] == layer_snap.seq

        stream_frames = _frames_of_type(captured, "notify.stream")
        assert stream_frames and stream_frames[-1]["seq"] == stream_snap.seq
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_notify_scene_payload_includes_viewport_state() -> None:
    server, scheduled, _captured = _make_server()
    try:
        ledger = server._state_ledger
        plane_state = PlaneState()
        plane_state.target_level = 0
        plane_state.target_ndisplay = 2
        plane_state.target_step = (2, 0, 0)
        plane_state.applied_level = 0
        plane_state.applied_step = (2, 0, 0)
        plane_state.update_pose(rect=(0.0, 0.0, 20.0, 20.0), center=(10.0, 10.0), zoom=2.0)
        volume_state = VolumeState(level=0, downgraded=False)
        volume_state.update_pose(center=(0.0, 0.0, 0.0))

        reduce_level_update(
            ledger,
            level=0,
            level_shape=(32, 512, 512),
            downgraded=False,
            origin="test.level",
            mode=RenderMode.PLANE,
            plane_state=plane_state,
            volume_state=volume_state,
        )

        server._update_scene_manager()
        while scheduled:
            asyncio.run(scheduled.pop(0))

        server._refresh_scene_snapshot()
        snapshot = server._scene_snapshot
        assert snapshot is not None
        payload = build_notify_scene_payload(
            scene_snapshot=snapshot,
            ledger_snapshot=ledger.snapshot(),
            viewer_settings=None,
        )

        metadata = payload.metadata or {}
        viewport_meta = metadata.get("viewport_state")
        assert viewport_meta is not None
        assert viewport_meta["mode"] == "PLANE"
        assert tuple(viewport_meta["plane"]["applied"]["step"]) == (2, 0, 0)
        assert viewport_meta["volume"]["level"] == 0
    finally:
        while scheduled:
            asyncio.run(scheduled.pop(0))


def test_reduce_plane_restore_updates_viewport_state() -> None:
    server, scheduled, _captured = _make_server()
    try:
        ledger = server._state_ledger
        reduce_plane_restore(
            ledger,
            level=2,
            step=(6, 0, 0),
            center=(12.0, 24.0, 0.0),
            zoom=1.1,
            rect=(0.0, 0.0, 40.0, 40.0),
            intent_id="restore-test",
            origin="test.restore",
        )

        while scheduled:
            asyncio.run(scheduled.pop(0))

        plane_entry = ledger.get("viewport", "plane", "state")
        assert plane_entry is not None
        assert plane_entry.value is not None
        plane_snapshot = plane_entry.value
        assert plane_snapshot["applied"]["level"] == 2
        assert tuple(plane_snapshot["applied"]["step"]) == (6, 0, 0)
        assert pytest.approx(float(plane_snapshot["pose"]["zoom"])) == 1.1

        spec_entry = ledger.get("dims", "main", "dims_spec")
        assert spec_entry is not None
        spec_payload = dims_spec_from_payload(spec_entry.value)
        assert spec_payload is not None
        assert int(spec_payload.current_level) == 2
        assert tuple(int(v) for v in spec_payload.current_step) == (6, 0, 0)
    finally:
        while scheduled:
            asyncio.run(scheduled.pop(0))
