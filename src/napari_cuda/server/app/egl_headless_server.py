"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np
import websockets

from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NotifyDimsPayload,
    NotifyScenePayload,
    NotifyStreamPayload,
)
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.app import metrics_server
from napari_cuda.server.app.context import (
    EncodeConfig,
    capture_env,
    configure_bitstream_policy,
    resolve_data_roots,
    resolve_encode_config,
    resolve_server_ctx,
    resolve_volume_caps,
)
from napari_cuda.server.app.dataset_lifecycle import (
    ServerLifecycleHooks,
    apply_dataset_bootstrap,
    enter_idle_state,
)
from napari_cuda.server.app.scene_publisher import (
    broadcast_state_baseline,
    build_scene_payload as build_scene_payload_helper,
    cache_scene_history,
)
from napari_cuda.server.app.metrics_core import Metrics
from napari_cuda.server.config import ServerConfig, ServerCtx
from napari_cuda.server.control.control_channel_server import ingest_state
from napari_cuda.server.control.control_payload_builder import (
    build_notify_scene_payload,
)
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.mirrors.layer_mirror import ServerLayerMirror
from napari_cuda.server.control.protocol.runtime import state_sequencer
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
    ResumableRetention,
)
from napari_cuda.server.control.state_reducers import (
    reduce_bootstrap_state,
    reduce_camera_update,
    reduce_level_update,
)
from napari_cuda.server.control.topics.notify.camera import broadcast_camera_update
from napari_cuda.server.control.topics.notify.dims import broadcast_dims_state
from napari_cuda.server.control.topics.notify.layers import broadcast_layers_delta
from napari_cuda.server.control.topics.notify.stream import broadcast_stream_config
from napari_cuda.server.data.hw_limits import get_hw_limits
from napari_cuda.server.data.zarr_discovery import (
    ZarrDatasetDisambiguationError,
    discover_dataset_root,
    inspect_zarr_directory,
)
from napari_cuda.server.engine.api import (
    ParamCache,
    PixelBroadcastState,
    PixelChannelConfig,
    PixelChannelState,
    build_avcc_config,
    build_notify_stream_payload,
    ensure_keyframe,
    ingest_client,
    mark_stream_config_dirty,
    prepare_client_attach,
    run_channel_loop,
    send_cached_stream_snapshot,
    send_stream_snapshot_if_needed,
)
from napari_cuda.server.runtime.api import RuntimeHandle
from napari_cuda.server.runtime.bootstrap.runtime_driver import (
    probe_scene_bootstrap,
)
from napari_cuda.server.runtime.camera import (
    CameraCommandQueue,
    CameraPoseApplied,
)
from napari_cuda.server.runtime.ipc import (
    WorkerIntentMailbox,
)
from napari_cuda.server.runtime.worker import (
    WorkerLifecycleState,
    start_worker as lifecycle_start_worker,
    stop_worker as lifecycle_stop_worker,
)
from napari_cuda.server.scene import (
    CameraDeltaCommand,
    LayerVisualState,
    RenderLedgerSnapshot,
    RenderMode,
    RenderUpdate,
    pull_render_snapshot,
    snapshot_layer_controls,
    snapshot_multiscale_state,
    snapshot_render_state,
    snapshot_scene,
    snapshot_volume_state,
)
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.server.utils.websocket import safe_send
from napari_cuda.server.control.state_reducers import reduce_thumbnail_capture
from napari_cuda.shared.dims_spec import dims_spec_from_payload
import hashlib
import time

logger = logging.getLogger(__name__)

DEFAULT_LAYER_ID = "layer-0"
DEFAULT_LAYER_NAME = "napari-cuda"

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from napari_cuda.server.runtime.worker import EGLRendererWorker


class EGLHeadlessServer:
    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False,
                 host: str = '0.0.0.0', state_port: int = 8081, pixel_port: int = 8082, fps: int = 60,
                 animate: bool = False, animate_dps: float = 30.0, log_sends: bool = False,
                 zarr_path: str | None = None, zarr_level: str | None = None,
                 zarr_axes: str | None = None, zarr_z: int | None = None,
                 debug: bool = False) -> None:
        # Build once: resolved runtime context (observe-only for now)
        self._ctx_env: dict[str, str] = capture_env()
        self._ctx: ServerCtx = resolve_server_ctx(self._ctx_env)
        self._data_root, self._browse_root = resolve_data_roots(self._ctx_env, logger=logger)
        configure_bitstream_policy(self._ctx, logger=logger)

        hw_limits = get_hw_limits()
        (
            self._volume_max_bytes_cfg,
            self._volume_max_voxels_cfg,
            self._hw_volume_max_bytes,
            self._hw_volume_max_voxels,
        ) = resolve_volume_caps(self._ctx, hw_limits)

        self.width = width
        self.height = height
        self._initial_mode = RenderMode.VOLUME if use_volume else RenderMode.PLANE
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        encode_cfg = getattr(self._ctx.cfg, 'encode', None)
        self._codec_name, self.cfg = resolve_encode_config(encode_cfg, fallback_fps=fps)
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception:
            self._animate_dps = 30.0

        policy_logging = self._ctx.debug_policy.logging
        self.metrics = Metrics()
        self._metrics_runner = None
        self._state_clients: set[websockets.WebSocketServerProtocol] = set()
        self._resumable_store = ResumableHistoryStore(
            {
                NOTIFY_SCENE_TYPE: ResumableRetention(),
                NOTIFY_SCENE_LEVEL_TYPE: ResumableRetention(),
                NOTIFY_LAYERS_TYPE: ResumableRetention(
                    min_deltas=512,
                    max_deltas=2048,
                    max_age_s=300.0,
                ),
                NOTIFY_STREAM_TYPE: ResumableRetention(
                    min_deltas=1,
                    max_deltas=32,
                    max_age_s=None,
                ),
            }
        )
        self._dataset_lock: Optional[asyncio.Lock] = None
        self._mirrors_started = False
        # Keep queue size at 1 for latest-wins, never-block behavior
        qsize = int(self._ctx.frame_queue)
        frame_queue: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        log_sends_flag = bool(log_sends or policy_logging.log_sends_env)
        broadcast_state = PixelBroadcastState(
            frame_queue=frame_queue,
            clients=set(),
            log_sends=log_sends_flag,
        )
        self._pixel_config = PixelChannelConfig(
            width=self.width,
            height=self.height,
            fps=float(self.cfg.fps),
            codec_id=self.cfg.codec,
            codec_name=self._codec_name,
            kf_watchdog_cooldown_s=float(self._ctx.kf_watchdog_cooldown_s),
        )
        self._pixel_channel = PixelChannelState(
            broadcast=broadcast_state,
            needs_stream_config=True,
            last_avcc=None,
            kf_watchdog_cooldown_s=self._pixel_config.kf_watchdog_cooldown_s,
        )
        self._seq = 0
        self._worker_lifecycle = WorkerLifecycleState()
        self._runtime_handle = RuntimeHandle(lambda: self._worker_lifecycle.worker)
        self._worker_intents = WorkerIntentMailbox()
        self._camera_queue = CameraCommandQueue()
        self._scene_snapshot: Optional[SceneSnapshot] = None
        self._state_ledger = ServerStateLedger()
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        # Optional bitstream dump for validation
        self._dump_remaining = int(self._ctx.dump_bitstream)
        self._dump_dir = self._ctx.dump_dir
        self._dump_path: Optional[str] = None
        # State access synchronization for latest-wins camera op coalescing
        self._state_lock = threading.RLock()
        self._control_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bootstrap_snapshot: Optional[RenderLedgerSnapshot] = None
        # Camera sequencing per target
        self._camera_command_seq: dict[str, int] = {}
        # Thumbnail scheduling via worker mailbox; no server state machine

        def _schedule_from_mirror(coro: Awaitable[None], label: str) -> None:
            loop = self._control_loop
            if loop is None:
                raise RuntimeError("control loop not initialized")
            loop.call_soon_threadsafe(self._schedule_coro, coro, label)

        async def _mirror_broadcast(payload: NotifyDimsPayload) -> None:
            await broadcast_dims_state(self, payload=payload)

        def _mirror_apply(payload: NotifyDimsPayload) -> None:
            _ = payload  # dims mirror ensures ledger is already updated
            loop = self._control_loop
            if loop is not None:
                loop.call_soon_threadsafe(self._refresh_scene_snapshot)

        self._dims_mirror = ServerDimsMirror(
            ledger=self._state_ledger,
            broadcaster=_mirror_broadcast,
            schedule=_schedule_from_mirror,
            on_payload=_mirror_apply,
        )
        async def _mirror_layer_broadcast(
            layer_id: str,
            state: LayerVisualState,
            intent_id: Optional[str],
            timestamp: float,
        ) -> None:
            await broadcast_layers_delta(
                self,
                layer_id=layer_id,
                state=state,
                intent_id=intent_id,
                timestamp=timestamp,
            )

        self._layer_mirror = ServerLayerMirror(
            ledger=self._state_ledger,
            broadcaster=_mirror_layer_broadcast,
            schedule=_schedule_from_mirror,
            default_layer=self._default_layer_id,
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info("Server debug policy: %s", self._ctx.debug_policy)
        # Logging controls for camera ops
        self._log_cam_info = bool(policy_logging.log_camera_info)
        self._log_cam_debug = bool(policy_logging.log_camera_debug)
        logger.info(
            "camera logging flags info=%s debug=%s",
            self._log_cam_info,
            self._log_cam_debug,
        )
        # State trace toggles (per-message start/end logs)
        self._log_state_traces = bool(policy_logging.log_state_traces)
        # Logging controls for volume/multiscale intents
        self._log_volume_info = bool(policy_logging.log_volume_info)
        # Force IDR on reset (default True)
        self._idr_on_reset = bool(self._ctx.cfg.idr_on_reset)

        # Data configuration (optional OME-Zarr dataset for real data)
        self._zarr_path = zarr_path or self._ctx.cfg.zarr_path or None
        self._zarr_level = zarr_level or self._ctx.cfg.zarr_level or None
        self._zarr_axes = zarr_axes or self._ctx.cfg.zarr_axes or None
        _z_fallback = self._ctx.cfg.zarr_z
        _z_resolved = zarr_z if zarr_z is not None else _z_fallback
        self._zarr_z = (_z_resolved if (_z_resolved is not None and int(_z_resolved) >= 0) else None)
        # Verbose dims logging control: default debug, upgrade to info with flag
        self._log_dims_info = bool(policy_logging.log_dims_info)
        # Dedicated debug control for this module only (no dependency loggers)
        self._debug_only_this_logger = bool(debug) or bool(self._ctx.debug_policy.enabled)
        self._allowed_render_modes = {'mip', 'translucent', 'iso'}
        # Populate multiscale description from NGFF metadata if available
        # In-memory last-accepted content tokens for thumbnails (per layer)
        self._last_thumb_token: dict[str, tuple] = {}

    @property
    def _worker(self) -> Optional[EGLRendererWorker]:
        return self._worker_lifecycle.worker

    @_worker.setter
    def _worker(self, worker: Optional[EGLRendererWorker]) -> None:
        self._worker_lifecycle.worker = worker

    @property
    def runtime(self) -> RuntimeHandle:
        return self._runtime_handle

    # --- Logging + Broadcast helpers --------------------------------------------
    def _next_camera_command_seq(self, target: str) -> int:
        current = self._camera_command_seq.get(target, 0) + 1
        self._camera_command_seq[target] = current
        return current

    def _enqueue_camera_delta(self, cmd: CameraDeltaCommand) -> None:
        self._camera_queue.append(cmd)
        queue_len = len(self._camera_queue)
        if self._log_cam_info or self._log_cam_debug:
            logger.info(
                "enqueue camera command kind=%s queue_len=%d payload=%s",
                cmd.kind,
                queue_len,
                cmd,
            )

    def _volume_budget_caps(self) -> tuple[Optional[int], Optional[int]]:
        cfg_bytes = self._volume_max_bytes_cfg
        cfg_voxels = self._volume_max_voxels_cfg
        max_bytes = cfg_bytes or self._hw_volume_max_bytes
        max_voxels = cfg_voxels or self._hw_volume_max_voxels
        return (
            int(max_bytes) if max_bytes else None,
            int(max_voxels) if max_voxels else None,
        )

    def _handle_worker_level_intents(self) -> None:
        intent = self._worker_intents.pop_level_switch()
        if intent is None:
            return

        worker = self._worker
        if worker is None:
            logger.debug("level intent dropped (no active worker)")
            return

        intent_mode = intent.mode if intent.mode is not None else RenderMode.PLANE
        if not isinstance(intent_mode, RenderMode):
            try:
                intent_mode = RenderMode[str(intent_mode).upper()]
            except Exception:
                intent_mode = RenderMode.PLANE

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "apply.level_intent: prev=%d target=%d reason=%s",
                int(intent.previous_level),
                int(intent.selected_level),
                intent.reason,
            )
        reduce_level_update(
            self._state_ledger,
            level=int(intent.selected_level),
            level_shape=tuple(int(dim) for dim in intent.level_shape) if intent.level_shape is not None else None,
            intent_id=None,
            timestamp=None,
            origin="worker.state.level",
            mode=intent_mode,
            plane_state=intent.plane_state,
            volume_state=intent.volume_state,
        )

        snapshot = snapshot_render_state(self._state_ledger)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "level.intent snapshot: level=%d step=%s",
                int(snapshot.current_level) if snapshot.current_level is not None else -1,
                snapshot.current_step,
            )

        worker.enqueue_update(
            RenderUpdate(
                scene_state=snapshot,
                mode=intent_mode,
                plane_state=intent.plane_state,
                volume_state=intent.volume_state,
            )
        )
        # No explicit thumbnail queue; post-frame logic will emit

    def _log_volume_update(self, fmt: str, *args) -> None:
        try:
            if self._log_volume_info:
                logger.info(fmt, *args)
            else:
                logger.debug(fmt, *args)
        except Exception:
            logger.debug("volume update log failed", exc_info=True)

    def _apply_worker_camera_pose(
        self,
        pose: CameraPoseApplied,
    ) -> None:
        target = pose.target or "main"
        seq = int(pose.command_seq)

        with self._state_lock:
            if self._log_cam_debug:
                logger.debug(
                    "camera pose applied target=%s seq=%d center=%s zoom=%s angles=%s distance=%s fov=%s rect=%s",
                    target,
                    seq,
                    pose.center,
                    pose.zoom,
                    pose.angles,
                    pose.distance,
                    pose.fov,
                    pose.rect,
                )
            ack_state, _ = reduce_camera_update(
                self._state_ledger,
                center=pose.center,
                zoom=pose.zoom,
                angles=pose.angles,
                distance=pose.distance,
                fov=pose.fov,
                rect=pose.rect,
                origin="worker.state.camera",
                metadata={"pose_seq": seq},
            )

        # reducer call persists the pose for both plane and volume scopes

        self._schedule_coro(
            broadcast_camera_update(
                self,
                mode="pose",
                state=ack_state,
                intent_id=None,
                origin="worker.state.camera",
            ),
            "worker-camera-pose",
        )
        # No explicit thumbnail queue; post-frame logic will emit

    def _try_reset_encoder(self, *, reason: str = "unspecified") -> bool:
        worker = self._worker
        if worker is None:
            return False
        logger.info("Encoder reset requested (reason=%s)", reason)
        worker.reset_encoder()
        mark_stream_config_dirty(self._pixel_channel)
        return True

    def _try_force_idr(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        worker.force_idr()
        return True

    def _current_avcc_bytes(self) -> Optional[bytes]:
        avcc_blob = self._pixel_channel.last_avcc
        if avcc_blob is not None:
            assert isinstance(avcc_blob, bytes), "pixel channel cached avcc must be bytes"
            return avcc_blob

        avcc_from_cache = build_avcc_config(self._param_cache)
        if avcc_from_cache is None:
            return None
        assert isinstance(avcc_from_cache, bytes), "param cache returned invalid avcc payload"
        return avcc_from_cache

    def _schedule_coro(self, coro: Awaitable[None], label: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        task = loop.create_task(coro)

        def _log_task_result(t: asyncio.Task) -> None:
            try:
                t.result()
            except Exception:
                logger.exception("Scheduled task '%s' failed", label)

        task.add_done_callback(_log_task_result)

    def _reset_thumbnail_state(self) -> None:
        # Deprecated; no-op
        return

    def _reset_mirrors(self) -> None:
        self._layer_mirror.reset()
        self._dims_mirror.reset()
        # Clear last-accepted tokens so dataset resets don't hold stale dedupe
        self._last_thumb_token.clear()

    def _ack_scene_op_if_open(
        self,
        *,
        frame_state: RenderLedgerSnapshot,
        origin: str,
    ) -> None:
        op_entry = self._state_ledger.get("scene", "main", "op_state")
        if op_entry is None or str(op_entry.value) != "open":
            return
        latest_seq_entry = self._state_ledger.get("scene", "main", "op_seq")
        latest_seq = None
        if latest_seq_entry is not None and isinstance(latest_seq_entry.value, int):
            latest_seq = int(latest_seq_entry.value)
        frame_seq = None
        if isinstance(frame_state.op_seq, int):
            frame_seq = int(frame_state.op_seq)
        if latest_seq is not None and frame_seq is not None and frame_seq != latest_seq:
            return
        self._state_ledger.record_confirmed(
            "scene",
            "main",
            "op_state",
            "applied",
            origin=origin,
        )

    def _set_dataset_metadata(
        self,
        path: Optional[str],
        level: Optional[str],
        axes: Optional[str],
        z_index: Optional[int],
    ) -> None:
        self._zarr_path = path
        self._zarr_level = level
        self._zarr_axes = axes
        self._zarr_z = z_index

    def _set_bootstrap_snapshot(self, snapshot: RenderLedgerSnapshot) -> None:
        self._bootstrap_snapshot = snapshot

    def _build_lifecycle_hooks(self) -> ServerLifecycleHooks:
        return ServerLifecycleHooks(
            stop_worker=self._stop_worker,
            clear_frame_queue=self._clear_frame_queue,
            reset_mirrors=self._reset_mirrors,
            refresh_scene_snapshot=self._refresh_scene_snapshot,
            start_mirrors_if_needed=self._start_mirrors_if_needed,
            pull_render_snapshot=lambda: pull_render_snapshot(self),
            set_dataset_metadata=self._set_dataset_metadata,
            set_bootstrap_snapshot=self._set_bootstrap_snapshot,
            pixel_channel=self._pixel_channel,
        )

    def _queue_thumbnail_refresh(self, layer_id: Optional[str]) -> None:
        # Deprecated: no-op
        return

    def _on_render_tick(self, snapshot: RenderLedgerSnapshot) -> None:
        # Deprecated: thumbnails are emitted via worker mailbox
        return

    def _handle_worker_thumbnails(self) -> None:
        while True:
            payload = self._worker_intents.pop_thumbnail_capture()
            if payload is None:
                break
            self._ingest_worker_thumbnail(payload)

    def _ingest_worker_thumbnail(self, payload) -> None:
        layer_id = str(payload.layer_id)
        # Use the worker-provided frame token (inputs used for that frame)
        token = tuple(payload.frame_token)
        last = self._last_thumb_token.get(layer_id)
        if last is not None and last == token:
            return
        arr = np.asarray(payload.array)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size > 0:
                max_val = float(np.nanmax(arr))
                if max_val > 0.0:
                    arr = arr / max_val
        np.clip(arr, 0.0, 1.0, out=arr)
        arr = np.flip(arr, axis=0)
        ts = time.time()
        payload_dict = {
            "array": arr.tolist(),
            "shape": list(arr.shape),
            "dtype": "float32",
            "generated_at": float(ts),
        }
        reduce_thumbnail_capture(
            self._state_ledger,
            layer_id=layer_id,
            payload=payload_dict,
            origin="server.thumbnail",
            timestamp=ts,
        )
        self._last_thumb_token[layer_id] = token
        self._refresh_scene_snapshot()

    async def _ensure_keyframe(self) -> None:
        """Request a clean keyframe and arrange for watchdog + config resend."""

        await ensure_keyframe(
            self._pixel_channel,
            config=self._pixel_config,
            metrics=self.metrics,
            reset_encoder=self._try_reset_encoder,
            send_stream=self._broadcast_stream_config,
            capture_avcc=self._current_avcc_bytes,
        )
        worker = self._worker
        if worker is not None and hasattr(worker, "_mark_render_tick_needed"):
            try:
                worker._mark_render_tick_needed()
            except Exception:
                logger.debug("ensure_keyframe: mark_render_tick_needed failed", exc_info=True)
        worker = self._worker
        if worker is not None and hasattr(worker, "_request_encoder_idr"):
            try:
                worker._request_encoder_idr()
            except Exception:
                logger.debug("ensure_keyframe: request_idr failed", exc_info=True)

    def _start_kf_watchdog(self) -> None:
        return

    def mark_stream_config_dirty(self) -> None:
        """Mark the pixel channel so the next notify.stream carries fresh config."""

        mark_stream_config_dirty(self._pixel_channel)

    def build_stream_payload(self, avcc: bytes) -> NotifyStreamPayload:
        """Build a notify.stream payload for the current pixel configuration."""

        return build_notify_stream_payload(self._pixel_config, avcc)

    def _maybe_enable_debug_logger(self) -> None:
        """Enable DEBUG logs for this module only, leaving root/others unchanged.

        - Attaches a dedicated StreamHandler at DEBUG to our module logger.
        - Disables propagation to avoid duplicate INFO/WARNING via root handlers.
        - Does nothing unless NAPARI_CUDA_DEBUG or --debug is provided.
        """
        if not getattr(self, '_debug_only_this_logger', False):
            return
        try:
            # Avoid stacking handlers on repeated calls
            has_ours = any(isinstance(h, logging.StreamHandler) and getattr(h, '_napari_cuda_local', False)
                           for h in logger.handlers)
            if has_ours:
                return
            h = logging.StreamHandler()
            fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            h.setFormatter(logging.Formatter(fmt))
            h.setLevel(logging.DEBUG)
            # Tag so we don't add duplicates later
            h._napari_cuda_local = True
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            # Ensure our records don't also bubble to root (which may be INFO)
            logger.propagate = False
        except Exception:
            # Fail silent; we don't want logging issues to break the server
            pass

    def _refresh_scene_snapshot(self, render_state: Optional[RenderLedgerSnapshot] = None) -> None:
        worker = self._worker
        worker_ready = worker is not None and worker.is_ready

        ledger_snapshot = self._state_ledger.snapshot()

        render_snapshot = render_state
        if render_snapshot is None:
            if worker_ready:
                render_snapshot = snapshot_render_state(self._state_ledger)
            elif self._bootstrap_snapshot is not None:
                render_snapshot = self._bootstrap_snapshot
            else:
                render_snapshot = snapshot_render_state(self._state_ledger)

        scene_source = getattr(worker, "_scene_source", None) if worker_ready else None
        multiscale_state = snapshot_multiscale_state(ledger_snapshot)
        volume_state = snapshot_volume_state(ledger_snapshot)
        layer_controls = snapshot_layer_controls(ledger_snapshot)

        thumbnail_provider = self._layer_thumbnail

        self._scene_snapshot = snapshot_scene(
            render_state=render_snapshot,
            ledger_snapshot=ledger_snapshot,
            canvas_size=(self.width, self.height),
            fps_target=float(self.cfg.fps),
            default_layer_id=DEFAULT_LAYER_ID if self._zarr_path else None,
            default_layer_name=DEFAULT_LAYER_NAME,
            ndisplay=self._current_ndisplay(),
            zarr_path=self._zarr_path,
            scene_source=scene_source,
            layer_controls=layer_controls,
            multiscale_state=multiscale_state,
            volume_state=volume_state,
            thumbnail_provider=thumbnail_provider,
        )

    def _default_layer_id(self) -> Optional[str]:
        snapshot = self._scene_snapshot
        if snapshot is None or not snapshot.layers:
            return None
        return snapshot.layers[0].layer_id

    def _layer_thumbnail(self, layer_id: str) -> Optional[np.ndarray]:
        worker = self._worker
        if worker is None:
            return None
        viewer = worker.viewer_model()
        if viewer is None or not viewer.layers:
            return None
        layer = None
        if not viewer.layers:
            return None
        if len(viewer.layers) == 1:
            layer = viewer.layers[0]
        else:
            for candidate in viewer.layers:
                if candidate.name == layer_id:
                    layer = candidate
                    break
            if layer is None:
                layer = viewer.layers[0]
        if layer is None:
            return None
        layer._update_thumbnail()
        thumbnail = layer.thumbnail
        if thumbnail is None:
            return None
        if isinstance(thumbnail, np.ndarray):
            arr = thumbnail
        elif isinstance(thumbnail, (list, tuple)):
            arr = np.asarray(thumbnail)
        else:
            return None
        if arr.size == 0:
            return None
        return arr

    def _current_ndisplay(self) -> int:
        ledger = self._state_ledger
        assert ledger is not None, "state ledger unavailable"

        spec_entry = ledger.get("dims", "main", "dims_spec")
        assert spec_entry is not None, "ledger missing dims_spec entry"
        spec = dims_spec_from_payload(getattr(spec_entry, "value", None))
        assert spec is not None, "dims_spec payload missing"
        return 3 if int(spec.ndisplay) >= 3 else 2

    def _start_mirrors_if_needed(self) -> None:
        if not self._mirrors_started:
            self._dims_mirror.start()
            self._layer_mirror.start()
            self._mirrors_started = True

    def _viewer_settings(self) -> dict[str, Any]:
        spec_entry = self._state_ledger.get("dims", "main", "dims_spec")
        assert spec_entry is not None, "ndisplay not initialised"
        spec = dims_spec_from_payload(getattr(spec_entry, "value", None))
        assert spec is not None, "dims_spec payload missing"
        use_volume = int(spec.ndisplay) >= 3
        return {
            "fps_target": float(self.cfg.fps),
            "canvas_size": [int(self.width), int(self.height)],
            "volume_enabled": use_volume,
        }

    def _build_scene_payload(self) -> NotifyScenePayload:
        snapshot = self._scene_snapshot
        assert snapshot is not None, "scene snapshot unavailable"
        return build_scene_payload_helper(
            snapshot,
            self._state_ledger.snapshot(),
            self._viewer_settings(),
        )

    def _clear_frame_queue(self) -> None:
        queue = self._pixel_channel.broadcast.frame_queue
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _enter_idle_state(self) -> None:
        logger.info("Entering idle state (no dataset)")
        hooks = self._build_lifecycle_hooks()
        enter_idle_state(hooks, self._state_ledger)

    async def _switch_dataset(
        self,
        path: Path,
        *,
        preferred_level: Optional[str],
        axes_override: Optional[str],
        z_override: Optional[int],
        notify_clients: bool,
        initial: bool,
    ) -> None:
        if self._dataset_lock is None:
            self._dataset_lock = asyncio.Lock()
        async with self._dataset_lock:
            logger.info("Activating dataset: %s", path)
            if self._data_root and not initial:
                resolved = self._resolve_dataset_path(str(path))
            else:
                resolved = path.expanduser().resolve(strict=True)
                if not resolved.is_dir():
                    raise NotADirectoryError(str(resolved))
                if not (resolved / ".zattrs").exists():
                    raise ValueError(f"Path is not an OME-Zarr root: {resolved}")
            bootstrap_meta = probe_scene_bootstrap(
                path=str(resolved),
                use_volume=self._initial_mode is RenderMode.VOLUME,
                preferred_level=preferred_level,
                axes_override=tuple(axes_override) if axes_override is not None else None,
                z_override=z_override,
                canvas_size=(self.width, self.height),
                oversampling_thresholds=self._ctx.policy.oversampling_thresholds,
                oversampling_hysteresis=self._ctx.policy.oversampling_hysteresis,
                threshold_in=self._ctx.policy.threshold_in,
                threshold_out=self._ctx.policy.threshold_out,
                fine_threshold=self._ctx.policy.fine_threshold,
                policy_hysteresis=self._ctx.policy.hysteresis,
                cooldown_ms=self._ctx.policy.cooldown_ms,
            )

            hooks = self._build_lifecycle_hooks()
            apply_dataset_bootstrap(
                hooks,
                self._state_ledger,
                bootstrap_meta,
                resolved_path=str(resolved),
                preferred_level=preferred_level,
                z_override=z_override,
            )

            loop = asyncio.get_running_loop()
            self._start_worker(loop)
            self._refresh_scene_snapshot()

            scene_payload = self._build_scene_payload()
            cache_scene_history(self._resumable_store, self._state_clients, scene_payload)

            if notify_clients:
                broadcast_state_baseline(
                    self,
                    self._state_clients,
                    schedule_coro=self._schedule_coro,
                    reason="dataset-load",
                )
            else:
                self._schedule_coro(self._ensure_keyframe(), "dataset-startup-keyframe")

    async def _send_layer_thumbnail(
        self,
        layer_id: str,
        *,
        retries: int = 5,
        delay_s: float = 0.25,
    ) -> None:
        # Deprecated: thumbnails are emitted post-frame automatically
        _ = (layer_id, retries, delay_s)
        return

    def _require_data_root(self) -> Path:
        root_candidate = self._data_root or str(self._browse_root)
        root = Path(root_candidate).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise RuntimeError(f"Data root is not a directory: {root}")
        return root

    def _resolve_data_path(self, path: Optional[str]) -> Path:
        base = self._browse_root
        if not path:
            return base
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve(strict=False)
        if self._data_root:
            resolved.relative_to(Path(self._data_root))
        return resolved

    def _resolve_dataset_path(self, path: str) -> Path:
        resolved = self._resolve_data_path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))

        dataset = discover_dataset_root(resolved)

        root = self._require_data_root()
        if dataset != root and root not in dataset.parents:
            raise RuntimeError(f"Resolved dataset escaped data root: {dataset}")
        return dataset

    def _list_directory(
        self,
        path: Optional[str],
        *,
        only: Optional[Sequence[str]],
        show_hidden: bool,
        limit: int = 1000,
    ) -> dict[str, Any]:
        resolved = self._resolve_data_path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))

        suffix_filters: Optional[list[str]] = None
        if only:
            suffix_filters = [s.lower() for s in only if isinstance(s, str) and s]
            if not suffix_filters:
                suffix_filters = None

        filtered: list[dict[str, Any]] = []
        children = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
        for child in children:
            name = child.name
            if not show_hidden and name.startswith("."):
                continue
            stat_result = child.stat()
            is_dir = child.is_dir()
            if not is_dir and suffix_filters is not None:
                lowered = name.lower()
                if not any(lowered.endswith(suffix) for suffix in suffix_filters):
                    continue
            zarr_meta = inspect_zarr_directory(child) if is_dir else None
            filtered.append(
                {
                    "name": name,
                    "path": str(child),
                    "is_dir": is_dir,
                    "size": int(stat_result.st_size),
                    "mtime": float(stat_result.st_mtime),
                    "zarr_metadata": {
                        "is_dataset": bool(zarr_meta.is_dataset),
                        "has_multiscales": bool(zarr_meta.has_multiscales),
                        "dataset_children": list(zarr_meta.dataset_children),
                    } if zarr_meta is not None else None,
                }
            )

        has_more = len(filtered) > limit
        entries = filtered[:limit]
        return {
            "path": str(resolved),
            "entries": entries,
            "has_more": bool(has_more),
        }

    async def _handle_zarr_load(self, path: str) -> None:
        if self._data_root:
            resolved = self._resolve_dataset_path(path)
        else:
            resolved = Path(path).expanduser().resolve(strict=True)
            if not resolved.exists():
                raise FileNotFoundError(str(resolved))
            if not resolved.is_dir():
                raise NotADirectoryError(str(resolved))
            resolved = discover_dataset_root(resolved)
        await self._switch_dataset(
            resolved,
            preferred_level=None,
            axes_override=None,
            z_override=None,
            notify_clients=True,
            initial=False,
        )

    # --- Validation helpers -------------------------------------------------------
    async def start(self) -> None:
        # Keep global logging at INFO; optionally enable module-only DEBUG logging
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
        logger.info(
            "camera logging flags info=%s debug=%s",
            self._log_cam_info,
            self._log_cam_debug,
        )
        # Apply module-only debug after basicConfig so our handler takes precedence
        self._maybe_enable_debug_logger()
        # Observe-only: resolve and log ServerConfig with visibility
        try:
            _cfg: ServerConfig = self._ctx.cfg
            if self._debug_only_this_logger:
                logger.info("Resolved ServerConfig: %s", _cfg)
            else:
                logger.debug("Resolved ServerConfig: %s", _cfg)
        except Exception:
            logger.debug("ServerConfig log failed", exc_info=True)
        logger.info("Starting EGLHeadlessServer %dx%d @ %dfps", self.width, self.height, self.cfg.fps)
        loop = asyncio.get_running_loop()
        self._control_loop = loop
        self._dataset_lock = asyncio.Lock()

        if self._zarr_path:
            await self._switch_dataset(
                Path(self._zarr_path),
                preferred_level=self._zarr_level,
                axes_override=self._zarr_axes,
                z_override=self._zarr_z,
                notify_clients=False,
                initial=True,
            )
        else:
            self._enter_idle_state()
            self._start_mirrors_if_needed()
            self._refresh_scene_snapshot()
            idle_payload = self._build_scene_payload()
            cache_scene_history(self._resumable_store, self._state_clients, idle_payload)

        # Start websocket servers; disable permessage-deflate to avoid CPU and latency on large frames
        async def state_handler(ws: websockets.WebSocketServerProtocol) -> None:
            await ingest_state(self, ws)

        state_server = await websockets.serve(
            state_handler,
            self.host,
            self.state_port,
            compression=None,
            max_size=None,
        )
        pixel_server = await websockets.serve(
            self._ingest_pixel,
            self.host,
            self.pixel_port,
            compression=None,
            max_size=None,
        )
        self._metrics_runner = metrics_server.start_metrics_dashboard(
            self.host,
            int(self._ctx.metrics_port),
            self.metrics,
            int(self._ctx.metrics_refresh_ms),
        )
        logger.info(
            "WS listening on %s:%d (state), %s:%d (pixel) | Dashboard: http://%s:%s/dash/ JSON: http://%s:%s/metrics.json",
            self.host,
            self.state_port,
            self.host,
            self.pixel_port,
            self.host,
            str(self._ctx.metrics_port),
            self.host,
            str(self._ctx.metrics_port),
        )
        broadcaster = asyncio.create_task(self._broadcast_loop())
        try:
            await asyncio.Future()
        finally:
            broadcaster.cancel()
            state_server.close()
            await state_server.wait_closed()
            pixel_server.close()
            await pixel_server.wait_closed()
            metrics_server.stop_metrics_dashboard(self._metrics_runner)
            self._stop_worker()

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        lifecycle_start_worker(self, loop, self._worker_lifecycle)

    def _stop_worker(self) -> None:
        lifecycle_stop_worker(self._worker_lifecycle)

    async def _ingest_pixel(self, ws: websockets.WebSocketServerProtocol):
        prepare_client_attach(self._pixel_channel)
        await ingest_client(
            self._pixel_channel,
            ws,
            config=self._pixel_config,
            metrics=self.metrics,
            reset_encoder=self._try_reset_encoder,
            send_stream=self._broadcast_stream_config,
            on_clients_change=self._update_client_gauges,
            on_client_join=lambda: self._schedule_coro(
                self._ensure_keyframe(), "pixel-connect-keyframe"
            ),
        )

    async def _broadcast_loop(self) -> None:
        await run_channel_loop(
            self._pixel_channel,
            config=self._pixel_config,
            metrics=self.metrics,
        )

    def _apply_worker_camera_pose(
        self,
        pose: CameraPoseApplied,
    ) -> None:
        target = pose.target or "main"
        seq = int(pose.command_seq)

        with self._state_lock:
            if self._log_cam_debug:
                logger.debug(
                    "camera pose applied target=%s seq=%d center=%s zoom=%s angles=%s distance=%s fov=%s rect=%s",
                    target,
                    seq,
                    pose.center,
                    pose.zoom,
                pose.angles,
                pose.distance,
                pose.fov,
                pose.rect,
            )
            ack_state, _ = reduce_camera_update(
                self._state_ledger,
                center=pose.center,
                zoom=pose.zoom,
                angles=pose.angles,
                distance=pose.distance,
                fov=pose.fov,
                rect=pose.rect,
                origin="worker.state.camera",
                metadata={"pose_seq": seq},
            )

        # Close any open op after camera commit so dims+level+camera are atomic for mirrors
        op_state = self._state_ledger.get("scene", "main", "op_state")
        if op_state is not None:
            val = str(op_state.value)
            if val == "open":
                self._state_ledger.record_confirmed(
                    "scene",
                    "main",
                    "op_state",
                    "applied",
                    origin="worker.state.camera",
                )

        self._schedule_coro(
            broadcast_camera_update(
                self,
                mode="pose",
                state=ack_state,
                intent_id=None,
                origin="worker.state.camera",
            ),
            "worker-camera-pose",
        )

    def _scene_snapshot_json(self) -> Optional[str]:
        self._refresh_scene_snapshot()
        snapshot = self._scene_snapshot
        if snapshot is None:
            return None

        payload = build_notify_scene_payload(
            scene_snapshot=snapshot,
            ledger_snapshot=self._state_ledger.snapshot(),
        )
        json_payload = json.dumps(payload.to_dict())

        if self._log_dims_info:
            dims = snapshot.viewer.dims
            logger.info(
                "notify.scene dims: level_shapes=%s current_level=%s",
                dims.get("level_shapes"),
                dims.get("current_level"),
            )

        return json_payload

    async def _broadcast_stream_config(self, payload: NotifyStreamPayload) -> None:
        await broadcast_stream_config(self, payload=payload)

    async def _state_send(self, ws: websockets.WebSocketServerProtocol, text: str) -> None:
        if not await safe_send(ws, text):
            self._state_clients.discard(ws)

    def _update_client_gauges(self) -> None:
        count = len(self._pixel_channel.broadcast.clients)
        self.metrics.set('napari_cuda_pixel_clients', float(count))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='napari-cuda EGL headless server')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '0.0.0.0'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--animate', action='store_true', help='Enable simple turntable camera animation')
    parser.add_argument('--animate-dps', type=float, default=float(os.getenv('NAPARI_CUDA_TURNTABLE_DPS', '30.0')),
                        help='Turntable speed in degrees per second (default 30)')
    parser.add_argument('--volume', action='store_true', help='Use 3D volume visual')
    parser.add_argument('--zarr', dest='zarr_path', default=os.getenv('NAPARI_CUDA_ZARR_PATH'), help='Path to OME-Zarr root (enables 2D slice from ZYX volume)')
    parser.add_argument('--zarr-level', dest='zarr_level', default=os.getenv('NAPARI_CUDA_ZARR_LEVEL'), help='Dataset level path inside OME-Zarr (e.g., level_02). If omitted, inferred from multiscales.')
    parser.add_argument('--zarr-axes', dest='zarr_axes', default=os.getenv('NAPARI_CUDA_ZARR_AXES', 'zyx'), help='Axes order of the dataset (default: zyx)')
    parser.add_argument('--zarr-z', dest='zarr_z', type=int, default=int(os.getenv('NAPARI_CUDA_ZARR_Z', '-1')), help='Initial Z index for 2D slice (default: mid-slice)')
    parser.add_argument('--log-sends', action='store_true', help='Log per-send timing (seq, send_ts, stamp_ts, delta)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG for this server module only')
    args = parser.parse_args()

    async def run():
        srv = EGLHeadlessServer(width=args.width, height=args.height, use_volume=args.volume,
                                host=args.host, state_port=args.state_port, pixel_port=args.pixel_port, fps=args.fps,
                                animate=args.animate, animate_dps=args.animate_dps, log_sends=bool(args.log_sends),
                                zarr_path=args.zarr_path, zarr_level=args.zarr_level,
                                zarr_axes=args.zarr_axes, zarr_z=(None if int(args.zarr_z) < 0 else int(args.zarr_z)),
                                debug=bool(args.debug))
        await srv.start()

    asyncio.run(run())


if __name__ == '__main__':
    main()
