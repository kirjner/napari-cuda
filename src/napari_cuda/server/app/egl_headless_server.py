"""
EGLHeadlessServer - Async server harness for headless EGL rendering + NVENC streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import struct
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Dict, Optional, Set, Mapping, TYPE_CHECKING, Any, Sequence, List

import numpy as np
import websockets
import importlib.resources as ilr
import socket
from websockets.exceptions import ConnectionClosed

from napari_cuda.server.engine.api import (
    ParamCache,
    PixelBroadcastConfig,
    PixelBroadcastState,
    PixelChannelConfig,
    PixelChannelState,
    broadcast_loop,
    build_notify_stream_payload,
    build_avcc_config,
    configure_bitstream,
    ensure_keyframe,
    ingest_client,
    mark_stream_config_dirty,
    prepare_client_attach,
    run_channel_loop,
)
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.runtime.bootstrap.runtime_driver import probe_scene_bootstrap
from napari_cuda.server.scene import (
    CameraDeltaCommand,
    RenderLedgerSnapshot,
    RenderMode,
    snapshot_dims_metadata,
    snapshot_layer_controls,
    snapshot_multiscale_state,
    snapshot_render_state,
    snapshot_scene,
    snapshot_viewport_state,
    snapshot_volume_state,
)
from napari_cuda.server.scene import pull_render_snapshot
from napari_cuda.server.control.state_reducers import reduce_level_update
from napari_cuda.server.app.metrics_core import Metrics
from napari_cuda.utils.env import env_bool
from napari_cuda.server.data.zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.config import ServerConfig, ServerCtx
from napari_cuda.server.app.config import load_server_ctx
from napari_cuda.protocol import (
    NotifyStreamPayload,
    NotifyDimsPayload,
    NotifyScenePayload,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
)
from napari_cuda.server.control.control_payload_builder import build_notify_scene_payload
from napari_cuda.server.app import metrics_server
from napari_cuda.server.control.control_channel_server import (
    _broadcast_camera_update,
    ingest_state,
    _send_state_baseline,
    CommandRejected,
)
from napari_cuda.server.control.topics.stream import broadcast_stream_config
from napari_cuda.server.control.protocol_runtime import state_sequencer
from napari_cuda.server.control.topics.dims import broadcast_dims_state
from napari_cuda.server.control.topics.layers import broadcast_layers_delta
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.server.scene import BootstrapSceneMetadata, RenderUpdate
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.mirrors.layer_mirror import ServerLayerMirror
from napari_cuda.server.control.state_reducers import (
    reduce_bootstrap_state,
    reduce_dims_update,
    reduce_camera_update,
    reduce_volume_colormap,
    reduce_volume_contrast_limits,
    reduce_volume_opacity,
)
from napari_cuda.server.data.hw_limits import get_hw_limits
from napari_cuda.server.runtime.api import RuntimeHandle
from napari_cuda.server.runtime.ipc import LevelSwitchIntent, WorkerIntentMailbox
from napari_cuda.server.runtime.camera import CameraCommandQueue
from napari_cuda.server.runtime.worker import (
    WorkerLifecycleState,
    start_worker as lifecycle_start_worker,
    stop_worker as lifecycle_stop_worker,
)
from napari_cuda.server.runtime.camera import CameraPoseApplied
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
    ResumableRetention,
)
logger = logging.getLogger(__name__)

DEFAULT_LAYER_ID = "layer-0"
DEFAULT_LAYER_NAME = "napari-cuda"

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from napari_cuda.server.runtime.worker import EGLRendererWorker


@dataclass
class EncodeConfig:
    fps: int = 60
    codec: int = 1  # 1=h264, 2=hevc, 3=av1
    bitrate: int = 12_000_000
    keyint: int = 120


class EGLHeadlessServer:
    def __init__(self, width: int = 1920, height: int = 1080, use_volume: bool = False,
                 host: str = '0.0.0.0', state_port: int = 8081, pixel_port: int = 8082, fps: int = 60,
                 animate: bool = False, animate_dps: float = 30.0, log_sends: bool = False,
                 zarr_path: str | None = None, zarr_level: str | None = None,
                 zarr_axes: str | None = None, zarr_z: int | None = None,
                 debug: bool = False) -> None:
        # Build once: resolved runtime context (observe-only for now)
        self._ctx_env: dict[str, str] = dict(os.environ)
        try:
            self._ctx: ServerCtx = load_server_ctx(self._ctx_env)
        except Exception:
            # Fallback to minimal defaults if ctx load fails for any reason
            self._ctx = load_server_ctx({})  # type: ignore[arg-type]
        self._data_root = self._resolve_env_path(self._ctx_env.get("NAPARI_CUDA_DATA_ROOT"))
        try:
            configure_bitstream(self._ctx.bitstream)
        except Exception:
            logger.debug("Bitstream configuration failed", exc_info=True)

        hw_limits = get_hw_limits()
        cfg_limits = self._ctx.cfg
        self._volume_max_bytes_cfg = int(max(0, getattr(cfg_limits, "max_volume_bytes", 0)))
        self._volume_max_voxels_cfg = int(max(0, getattr(cfg_limits, "max_volume_voxels", 0)))
        self._hw_volume_max_bytes = int(getattr(hw_limits, "volume_max_bytes", 0))
        self._hw_volume_max_voxels = int(getattr(hw_limits, "volume_max_voxels", 0))

        self.width = width
        self.height = height
        self._initial_mode = RenderMode.VOLUME if use_volume else RenderMode.PLANE
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        encode_cfg = getattr(self._ctx.cfg, 'encode', None)
        codec_map = {'h264': 1, 'hevc': 2, 'av1': 3}
        if encode_cfg is None:
            self._codec_name = 'h264'
            self.cfg = EncodeConfig(fps=fps)
        else:
            codec_name = str(getattr(encode_cfg, 'codec', 'h264')).lower()
            self._codec_name = codec_name if codec_name in codec_map else 'h264'
            self.cfg = EncodeConfig(
                fps=int(getattr(encode_cfg, 'fps', fps)),
                codec=codec_map.get(codec_name, 1),
                bitrate=int(getattr(encode_cfg, 'bitrate', 12_000_000)),
                keyint=int(getattr(encode_cfg, 'keyint', 120)),
            )
        self._animate = bool(animate)
        try:
            self._animate_dps = float(animate_dps)
        except Exception:
            self._animate_dps = 30.0

        policy_logging = self._ctx.debug_policy.logging
        self.metrics = Metrics()
        self._metrics_runner = None
        self._state_clients: Set[websockets.WebSocketServerProtocol] = set()
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
        self._camera_command_seq: Dict[str, int] = {}

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
            changes: Mapping[str, object],
            intent_id: Optional[str],
            timestamp: float,
        ) -> None:
            await broadcast_layers_delta(
                self,
                layer_id=layer_id,
                changes=changes,
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

    @staticmethod
    def _resolve_env_path(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return str(Path(value).expanduser().resolve())
    @property
    def _worker(self) -> Optional["EGLRendererWorker"]:
        return self._worker_lifecycle.worker

    @_worker.setter
    def _worker(self, worker: Optional["EGLRendererWorker"]) -> None:
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

        context = intent.context
        if context is None:
            logger.warning("level intent dropped (missing context)")
            return

        intent_mode = intent.mode if intent.mode is not None else RenderMode.PLANE
        if not isinstance(intent_mode, RenderMode):
            try:
                intent_mode = RenderMode[str(intent_mode).upper()]
            except Exception:
                intent_mode = RenderMode.PLANE

        if logger.isEnabledFor(logging.INFO):
            context_level = int(context.level)
            logger.info(
                "apply.level_intent: prev=%d target=%d reason=%s downgraded=%s",
                int(intent.previous_level),
                context_level,
                intent.reason,
                bool(intent.downgraded),
            )
        reduce_level_update(
            self._state_ledger,
            applied=context,
            downgraded=bool(intent.downgraded),
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

            # Persist per-mode camera caches for deterministic restores.
            # Volume cache: camera_volume.*
            if pose.angles is not None and pose.distance is not None and pose.fov is not None:
                assert pose.center is not None, "volume pose requires center"

                vol_entries = [
                    ("camera_volume", "main", "center", tuple(float(c) for c in pose.center)),
                    ("camera_volume", "main", "angles", tuple(float(a) for a in pose.angles)),
                    ("camera_volume", "main", "distance", float(pose.distance)),
                    ("camera_volume", "main", "fov", float(pose.fov)),
                ]
                self._state_ledger.batch_record_confirmed(
                    vol_entries,
                    origin="worker.state.camera_volume",
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
            _broadcast_camera_update(
                self,
                mode="pose",
                state=ack_state,
                intent_id=None,
                origin="worker.state.camera",
            ),
            "worker-camera-pose",
        )

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
            setattr(h, '_napari_cuda_local', True)
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG)
            # Ensure our records don't also bubble to root (which may be INFO)
            logger.propagate = False
        except Exception:
            # Fail silent; we don't want logging issues to break the server
            pass

    # --- Helper to compute dims metadata for piggyback on dims_update ---------
    def _dims_metadata(self) -> dict:
        self._refresh_scene_snapshot()
        snapshot = self._scene_snapshot
        if snapshot is None:
            return {}
        return snapshot_dims_metadata(snapshot)

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
            default_layer_id=DEFAULT_LAYER_ID,
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
            return DEFAULT_LAYER_ID
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

        entry = ledger.get("view", "main", "ndisplay")
        assert entry is not None, "ledger missing view/ndisplay"

        value = entry.value
        assert isinstance(value, int), "ledger ndisplay must be int"
        return 3 if value >= 3 else 2

    def _start_mirrors_if_needed(self) -> None:
        if not self._mirrors_started:
            self._dims_mirror.start()
            self._layer_mirror.start()
            self._mirrors_started = True

    def _viewer_settings(self) -> Dict[str, Any]:
        entry = self._state_ledger.get("view", "main", "ndisplay")
        assert entry is not None and isinstance(entry.value, int), "ndisplay not initialised"
        use_volume = int(entry.value) >= 3
        return {
            "fps_target": float(self.cfg.fps),
            "canvas_size": [int(self.width), int(self.height)],
            "volume_enabled": use_volume,
        }

    def _build_scene_payload(self) -> NotifyScenePayload:
        snapshot = self._scene_snapshot
        assert snapshot is not None, "scene snapshot unavailable"
        return build_notify_scene_payload(
            scene_snapshot=snapshot,
            ledger_snapshot=self._state_ledger.snapshot(),
            viewer_settings=self._viewer_settings(),
        )

    def _cache_scene_history(self, payload: NotifyScenePayload) -> None:
        store = self._resumable_store
        if store is None:
            return
        now = time.time()
        store.snapshot_envelope(
            NOTIFY_SCENE_TYPE,
            payload=payload.to_dict(),
            timestamp=now,
        )
        store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=now)
        store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=now)
        for ws in list(self._state_clients):
            state_sequencer(ws, NOTIFY_SCENE_TYPE).clear()
            state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
            state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()

    def _clear_frame_queue(self) -> None:
        queue = self._pixel_channel.broadcast.frame_queue
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _enter_idle_state(self) -> None:
        logger.info("Entering idle state (no dataset)")
        self._stop_worker()
        self._clear_frame_queue()
        self._layer_mirror.reset()
        self._dims_mirror.reset()
        step = (0, 0)
        axis_labels = ("y", "x")
        order = (0, 1)
        level_shapes = ((1, 1),)
        levels = (
            {"index": 0, "shape": [1, 1], "downsample": [1.0, 1.0], "path": ""},
        )
        reduce_bootstrap_state(
            self._state_ledger,
            step=step,
            axis_labels=axis_labels,
            order=order,
            level_shapes=level_shapes,
            levels=levels,
            current_level=0,
            ndisplay=2,
            origin="server.idle-bootstrap",
        )
        self._zarr_path = None
        self._zarr_level = None
        self._zarr_axes = None
        self._zarr_z = None
        self._bootstrap_snapshot = pull_render_snapshot(self)
        self._refresh_scene_snapshot(self._bootstrap_snapshot)
        mark_stream_config_dirty(self._pixel_channel)
        self._pixel_channel.last_avcc = None
        broadcast = self._pixel_channel.broadcast
        broadcast.last_key_seq = None
        broadcast.last_key_ts = None
        broadcast.waiting_for_keyframe = True

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

            self._stop_worker()
            self._clear_frame_queue()
            self._layer_mirror.reset()
            self._dims_mirror.reset()

            reduce_bootstrap_state(
                self._state_ledger,
                step=bootstrap_meta.step,
                axis_labels=bootstrap_meta.axis_labels,
                order=bootstrap_meta.order,
                level_shapes=bootstrap_meta.level_shapes,
                levels=bootstrap_meta.levels,
                current_level=bootstrap_meta.current_level,
                ndisplay=bootstrap_meta.ndisplay,
                origin="server.bootstrap",
            )

            self._zarr_path = str(resolved)
            self._zarr_level = preferred_level
            self._zarr_axes = "".join(bootstrap_meta.axis_labels)
            self._zarr_z = z_override

            self._bootstrap_snapshot = pull_render_snapshot(self)
            self._start_mirrors_if_needed()
            self._refresh_scene_snapshot(self._bootstrap_snapshot)

            loop = asyncio.get_running_loop()
            self._start_worker(loop)
            self._refresh_scene_snapshot()

            scene_payload = self._build_scene_payload()
            self._cache_scene_history(scene_payload)

            mark_stream_config_dirty(self._pixel_channel)
            self._pixel_channel.last_avcc = None
            broadcast = self._pixel_channel.broadcast
            broadcast.last_key_seq = None
            broadcast.last_key_ts = None
            broadcast.waiting_for_keyframe = True

            if notify_clients:
                self._broadcast_state_baseline(reason="dataset-load")
            else:
                self._schedule_coro(self._ensure_keyframe(), "dataset-startup-keyframe")

    def _broadcast_state_baseline(self, *, reason: str) -> None:
        for ws in list(self._state_clients):
            self._schedule_coro(
                _send_state_baseline(self, ws),
                f"baseline-{reason}",
            )

    async def _emit_layer_thumbnail(
        self,
        layer_id: str,
        *,
        retries: int = 5,
        delay_s: float = 0.25,
    ) -> None:
        for attempt in range(retries):
            worker = self._worker
            if worker is None or not worker.is_ready:
                await asyncio.sleep(delay_s)
                continue
            array = self._layer_thumbnail(layer_id)
            if array is None or array.size == 0:
                await asyncio.sleep(delay_s)
                continue
            arr = np.asarray(array, dtype=np.float32)
            np.clip(arr, 0.0, 1.0, out=arr)
            metadata = {
                "thumbnail": arr.tolist(),
                "thumbnail_status": "ready",
                "thumbnail_shape": list(arr.shape),
                "thumbnail_dtype": str(arr.dtype),
                "thumbnail_version": float(time.time()),
            }
            self._state_ledger.record_confirmed(
                "layer",
                layer_id,
                "metadata",
                metadata,
                origin="server.thumbnail",
            )
            self._refresh_scene_snapshot()
            return
            
        logger.debug("thumbnail emission skipped for layer %s (no data)", layer_id)

    def _require_data_root(self) -> Path:
        if not self._data_root:
            raise RuntimeError("NAPARI_CUDA_DATA_ROOT not configured")
        root = Path(self._data_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise RuntimeError(f"Data root is not a directory: {root}")
        return root

    def _resolve_data_path(self, path: Optional[str]) -> Path:
        root = self._require_data_root()
        if not path:
            return root
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
        return resolved

    def _resolve_dataset_path(self, path: str) -> Path:
        resolved = self._resolve_data_path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        if not (resolved / ".zattrs").exists():
            raise ValueError(f"Path is not an OME-Zarr root: {resolved}")
        return resolved

    def _list_directory(
        self,
        path: Optional[str],
        *,
        only: Optional[Sequence[str]],
        show_hidden: bool,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        resolved = self._resolve_data_path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))

        suffix_filters: Optional[List[str]] = None
        if only:
            suffix_filters = [s.lower() for s in only if isinstance(s, str) and s]
            if not suffix_filters:
                suffix_filters = None

        filtered: List[Dict[str, Any]] = []
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
            filtered.append(
                {
                    "name": name,
                    "path": str(child),
                    "is_dir": is_dir,
                    "size": int(stat_result.st_size),
                    "mtime": float(stat_result.st_mtime),
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
            if not (resolved / ".zattrs").exists():
                raise ValueError(f"Path is not an OME-Zarr root: {resolved}")
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
            self._cache_scene_history(idle_payload)

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

            # Volume cache: camera_volume.*
            if pose.angles is not None and pose.distance is not None and pose.fov is not None:
                assert pose.center is not None, "volume pose requires center"

                vol_entries = [
                    ("camera_volume", "main", "center", tuple(float(c) for c in pose.center)),
                    ("camera_volume", "main", "angles", tuple(float(a) for a in pose.angles)),
                    ("camera_volume", "main", "distance", float(pose.distance)),
                    ("camera_volume", "main", "fov", float(pose.fov)),
                ]
                self._state_ledger.batch_record_confirmed(
                    vol_entries,
                    origin="worker.state.camera_volume",
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
            _broadcast_camera_update(
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
        try:
            await ws.send(text)
        except Exception as e:
            logger.debug("State send error: %s", e)
            try:
                await ws.close()
            except Exception as e2:
                logger.debug("State WS close error: %s", e2)
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
