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
from typing import Awaitable, Dict, Optional, Sequence, Set, Mapping, TYPE_CHECKING, Any

import websockets
import importlib.resources as ilr
import socket
from websockets.exceptions import ConnectionClosed

from napari_cuda.server.rendering.bitstream import build_avcc_config
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.scene import (
    CameraDeltaCommand,
    ServerSceneData,
    create_server_scene_data,
    layer_controls_from_ledger,
    prune_control_metadata,
    build_render_scene_state,
)
from napari_cuda.server.scene.layer_manager import ViewerSceneManager
from napari_cuda.server.rendering.bitstream import ParamCache, configure_bitstream
from napari_cuda.server.app.metrics_core import Metrics
from napari_cuda.utils.env import env_bool
from napari_cuda.server.data.zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.app.config import (
    ServerConfig,
    ServerCtx,
    load_server_ctx,
)
from napari_cuda.protocol import (
    NotifyStreamPayload,
    NotifyDimsPayload,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_SCENE_LEVEL_TYPE,
    NOTIFY_STREAM_TYPE,
)
from napari_cuda.server.control.control_payload_builder import build_notify_scene_payload
from napari_cuda.server.rendering import pixel_broadcaster
from napari_cuda.server.control import pixel_channel
from napari_cuda.server.app import metrics_server
from napari_cuda.server.control.control_channel_server import (
    _broadcast_camera_update,
    _broadcast_dims_state,
    broadcast_stream_config,
    ingest_state,
)
from napari_cuda.server.control.state_ledger import ServerStateLedger
from napari_cuda.server.control.state_models import BootstrapSceneMetadata
from napari_cuda.server.control.mirrors.dims_mirror import ServerDimsMirror
from napari_cuda.server.control.state_reducers import (
    reduce_bootstrap_state,
    reduce_dims_update,
    reduce_camera_update,
)
from napari_cuda.server.control.transactions import apply_level_switch_transaction
from napari_cuda.server.data.lod import LevelContext
from napari_cuda.server.runtime.bootstrap_probe import probe_scene_bootstrap
from napari_cuda.server.runtime.render_update_mailbox import RenderUpdate
from napari_cuda.server.runtime.intents import LevelSwitchIntent
from napari_cuda.server.runtime.worker_intent_mailbox import WorkerIntentMailbox
from napari_cuda.server.runtime.worker_lifecycle import (
    WorkerLifecycleState,
    start_worker as lifecycle_start_worker,
    stop_worker as lifecycle_stop_worker,
)
from napari_cuda.server.runtime.camera_pose import CameraPoseApplied
from napari_cuda.server.control.resumable_history_store import (
    ResumableHistoryStore,
    ResumableRetention,
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from napari_cuda.server.runtime.egl_worker import EGLRendererWorker


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
        try:
            configure_bitstream(self._ctx.bitstream)
        except Exception:
            logger.debug("Bitstream configuration failed", exc_info=True)

        self.width = width
        self.height = height
        self.use_volume = use_volume
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
        # Keep queue size at 1 for latest-wins, never-block behavior
        qsize = int(self._ctx.frame_queue)
        frame_queue: asyncio.Queue[tuple[bytes, int, int, float]] = asyncio.Queue(maxsize=max(1, qsize))
        log_sends_flag = bool(log_sends or policy_logging.log_sends_env)
        broadcast_state = pixel_broadcaster.PixelBroadcastState(
            frame_queue=frame_queue,
            clients=set(),
            log_sends=log_sends_flag,
        )
        self._pixel_config = pixel_channel.PixelChannelConfig(
            width=self.width,
            height=self.height,
            fps=float(self.cfg.fps),
            codec_id=self.cfg.codec,
            codec_name=self._codec_name,
            kf_watchdog_cooldown_s=float(self._ctx.kf_watchdog_cooldown_s),
        )
        self._pixel_channel = pixel_channel.PixelChannelState(
            broadcast=broadcast_state,
            needs_stream_config=True,
            last_avcc=None,
            kf_watchdog_cooldown_s=self._pixel_config.kf_watchdog_cooldown_s,
        )
        self._seq = 0
        self._worker_lifecycle = WorkerLifecycleState()
        self._worker_intents = WorkerIntentMailbox()
        self._scene_manager = ViewerSceneManager((self.width, self.height))
        self._state_ledger = ServerStateLedger()
        self._scene: ServerSceneData = create_server_scene_data(
            policy_event_path=self._ctx.policy_event_path,
            state_ledger=self._state_ledger,
        )
        self._scene.use_volume = bool(self.use_volume)
        # Bitstream parameter cache and config tracking (server-side)
        self._param_cache = ParamCache()
        # Optional bitstream dump for validation
        self._dump_remaining = int(self._ctx.dump_bitstream)
        self._dump_dir = self._ctx.dump_dir
        self._dump_path: Optional[str] = None
        # State access synchronization for latest-wins camera op coalescing
        self._state_lock = threading.RLock()
        self._control_loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-scope last applied seqs to help the worker skip idle ticks
        self._applied_seqs: Dict[str, int] = {}
        # Camera sequencing and pose tracking (per target)
        self._camera_command_seq: Dict[str, int] = {}
        self._last_camera_pose_seq: Dict[str, int] = {}

        def _schedule_from_mirror(coro: Awaitable[None], label: str) -> None:
            loop = self._control_loop
            if loop is None:
                raise RuntimeError("control loop not initialized")
            loop.call_soon_threadsafe(self._schedule_coro, coro, label)

        async def _mirror_broadcast(payload: NotifyDimsPayload) -> None:
            await _broadcast_dims_state(self, payload=payload)

        def _mirror_apply(payload: NotifyDimsPayload) -> None:
            with self._state_lock:
                multiscale_state = self._scene.multiscale_state
                multiscale_state["current_level"] = payload.current_level
                multiscale_state["levels"] = [dict(level) for level in payload.levels]
                if payload.downgraded is not None:
                    multiscale_state["downgraded"] = bool(payload.downgraded)
                else:
                    multiscale_state.pop("downgraded", None)
            loop = self._control_loop
            if loop is not None:
                loop.call_soon_threadsafe(self._update_scene_manager)

        self._dims_mirror = ServerDimsMirror(
            ledger=self._state_ledger,
            broadcaster=_mirror_broadcast,
            schedule=_schedule_from_mirror,
            on_payload=_mirror_apply,
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
        try:
            self._populate_multiscale_state()
        except Exception:
            logger.debug("populate multiscale state failed", exc_info=True)

    @property
    def _worker(self) -> Optional["EGLRendererWorker"]:
        return self._worker_lifecycle.worker

    @_worker.setter
    def _worker(self, worker: Optional["EGLRendererWorker"]) -> None:
        self._worker_lifecycle.worker = worker

    # --- Logging + Broadcast helpers --------------------------------------------
    def _next_camera_command_seq(self, target: str) -> int:
        current = self._camera_command_seq.get(target, 0) + 1
        self._camera_command_seq[target] = current
        return current

    def _enqueue_camera_delta(self, cmd: CameraDeltaCommand) -> None:
        with self._state_lock:
            self._scene.camera_deltas.append(cmd)
            queue_len = len(self._scene.camera_deltas)
        if self._log_cam_info or self._log_cam_debug:
            logger.info(
                "enqueue camera command kind=%s queue_len=%d payload=%s",
                cmd.kind,
                queue_len,
                cmd,
            )

    def _handle_worker_level_intents(self) -> None:
        intent = self._worker_intents.pop_level_switch()
        if intent is None:
            return

        worker = self._worker
        if worker is None:
            logger.debug("level intent dropped (no active worker)")
            return

        self._apply_level_switch_transaction(worker, intent)

    def _apply_level_switch_transaction(
        self,
        worker: "EGLRendererWorker",
        intent: LevelSwitchIntent,
    ) -> None:
        context = intent.context
        downgraded = bool(intent.downgraded)

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "apply.level_intent: prev=%d target=%d reason=%s downgraded=%s",
                int(intent.previous_level),
                int(context.level),
                intent.reason,
                downgraded,
            )

        self._commit_level_snapshot(context, downgraded)

        snapshot = build_render_scene_state(self._state_ledger, self._scene)
        self._scene.latest_state = snapshot
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "level.intent snapshot: level=%d step=%s",
                int(snapshot.current_level) if snapshot.current_level is not None else -1,
                snapshot.current_step,
            )
        worker.enqueue_update(RenderUpdate(multiscale=None, scene_state=snapshot))

    def _log_volume_update(self, fmt: str, *args) -> None:
        try:
            if self._log_volume_info:
                logger.info(fmt, *args)
            else:
                logger.debug(fmt, *args)
        except Exception:
            logger.debug("volume update log failed", exc_info=True)

    def _try_reset_encoder(self, *, reason: str = "unspecified") -> bool:
        worker = self._worker
        if worker is None:
            return False
        logger.info("Encoder reset requested (reason=%s)", reason)
        worker.reset_encoder()
        pixel_channel.mark_stream_config_dirty(self._pixel_channel)
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

        await pixel_channel.ensure_keyframe(
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

    def _stage_layer_controls_from_ledger(self) -> None:
        controls = layer_controls_from_ledger(self._state_ledger.snapshot())
        if not controls:
            return

        ledger_sync: list[tuple[str, Any]] = []
        with self._state_lock:
            pending = self._scene.pending_layer_updates
            for layer_id, props in controls.items():
                layer_pending = pending.setdefault(layer_id, {})
                for key, value in props.items():
                    layer_pending[str(key)] = value

            primary_props = next((props for props in controls.values() if props), None)
            if primary_props:
                volume_state = self._scene.volume_state

                colormap = primary_props.get("colormap")
                if colormap is not None:
                    name = str(colormap)
                    if volume_state.get("colormap") != name:
                        volume_state["colormap"] = name
                        ledger_sync.append(("colormap", name))

                clim = primary_props.get("contrast_limits")
                if clim is not None:
                    lo, hi = (float(clim[0]), float(clim[1]))
                    if volume_state.get("clim") != [lo, hi]:
                        volume_state["clim"] = [lo, hi]
                        ledger_sync.append(("contrast_limits", (lo, hi)))

                opacity = primary_props.get("opacity")
                if opacity is not None:
                    alpha = float(opacity)
                    if volume_state.get("opacity") != alpha:
                        volume_state["opacity"] = alpha
                        ledger_sync.append(("opacity", alpha))

        for key, value in ledger_sync:
            self._state_ledger.record_confirmed(
                "volume",
                "main",
                key,
                value,
                origin="control.layer_sync",
            )

    def _start_kf_watchdog(self) -> None:
        return

    # --- Meta builders ------------------------------------------------------------
    def _populate_multiscale_state(self) -> None:
        """Populate self._scene.multiscale_state['levels'] from NGFF multiscales if available.

        Stores minimal fields needed by clients and worker switching:
        - path: dataset subpath (e.g., 'level_03')
        - downsample: per-axis scale factors ordered as z,y,x when axes present; else [1,1,1]
        - shape: optional, if inexpensive to obtain (omitted here for speed)
        """
        root = self._zarr_path
        if not root:
            return
        axes_override = tuple(self._zarr_axes) if isinstance(self._zarr_axes, str) else None
        try:
            source = ZarrSceneSource(root, preferred_level=self._zarr_level, axis_override=axes_override)
        except ZarrSceneSourceError:
            logger.debug("multiscale state: invalid Zarr scene source", exc_info=True)
            return
        except Exception:
            logger.debug("multiscale state: unexpected error building Zarr scene source", exc_info=True)
            return

        levels: list[dict] = []
        for desc in source.level_descriptors:
            levels.append(
                {
                    'path': desc.path,
                    'downsample': list(desc.downsample),
                    'shape': [int(x) for x in desc.shape],
                }
            )

        if levels:
            self._scene.multiscale_state['levels'] = levels
            self._scene.multiscale_state['current_level'] = int(source.current_level)
            self._zarr_axes = ''.join(source.axes)
            current_desc = source.level_descriptors[source.current_level]
            if current_desc.path:
                self._zarr_level = current_desc.path

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
        self._update_scene_manager()
        return self._scene_manager.dims_metadata()

    def _update_scene_manager(self) -> None:
        worker = self._worker
        if worker is None or not worker.is_ready:
            return

        current_step = None
        with self._state_lock:
            if self._scene.latest_state.current_step is not None:
                current_step = list(self._scene.latest_state.current_step)  # type: ignore[arg-type]

        source = getattr(worker, '_scene_source', None)
        if source is not None:
            current_level = int(source.current_level)
            self._scene.multiscale_state['policy'] = 'auto'
            self._scene.multiscale_state['current_level'] = current_level
            descriptors = source.level_descriptors
            self._scene.multiscale_state['levels'] = [
                {
                    'path': desc.path,
                    'shape': [int(x) for x in desc.shape],
                    'downsample': list(desc.downsample),
                    'scale': [float(x) for x in desc.scale],
                }
                for desc in descriptors
            ]
            self._scene.multiscale_state['index_space'] = 'base'
        else:
            self._scene.multiscale_state.pop('levels', None)

        viewer_model = None
        viewer_model = worker.viewer_model()
        self._scene_manager.update_from_sources(
            worker=worker,
            scene_state=self._scene.latest_state,
            multiscale_state=dict(self._scene.multiscale_state),
            volume_state=dict(self._scene.volume_state),
            current_step=current_step,
            ndisplay=self._current_ndisplay(),
            zarr_path=self._zarr_path,
            scene_source=source,
            viewer_model=viewer_model,
            layer_controls=dict(self._scene.layer_controls),
        )

        snapshot = self._scene_manager.scene_snapshot()
        layer_ids = [layer.layer_id for layer in snapshot.layers] if snapshot else []
        dims_meta = self._scene_manager.dims_metadata()

        prune_control_metadata(
            self._scene,
            layer_ids=layer_ids,
            dims_meta=dims_meta,
            current_step=current_step,
        )

    def _current_ndisplay(self) -> int:
        ledger = self._state_ledger
        assert ledger is not None, "state ledger unavailable"

        entry = ledger.get("view", "main", "ndisplay")
        assert entry is not None, "ledger missing view/ndisplay"

        value = entry.value
        assert isinstance(value, int), "ledger ndisplay must be int"
        return 3 if value >= 3 else 2

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
        if self._zarr_path is None:
            raise RuntimeError("bootstrap requires zarr_path")
        bootstrap_meta = probe_scene_bootstrap(
            path=self._zarr_path,
            use_volume=self.use_volume,
            preferred_level=self._zarr_level,
            axes_override=tuple(self._zarr_axes) if self._zarr_axes is not None else None,
            z_override=self._zarr_z,
            canvas_size=(self.width, self.height),
            oversampling_thresholds=self._ctx.policy.oversampling_thresholds,
            oversampling_hysteresis=self._ctx.policy.oversampling_hysteresis,
            threshold_in=self._ctx.policy.threshold_in,
            threshold_out=self._ctx.policy.threshold_out,
            fine_threshold=self._ctx.policy.fine_threshold,
            policy_hysteresis=self._ctx.policy.hysteresis,
            cooldown_ms=self._ctx.policy.cooldown_ms,
        )
        with self._state_lock:
            reduce_bootstrap_state(
                self._scene,
                self._state_ledger,
                self._state_lock,
                step=bootstrap_meta.step,
                axis_labels=bootstrap_meta.axis_labels,
                order=bootstrap_meta.order,
                level_shapes=bootstrap_meta.level_shapes,
                levels=bootstrap_meta.levels,
                current_level=bootstrap_meta.current_level,
                ndisplay=bootstrap_meta.ndisplay,
                origin="server.bootstrap",
            )
        self._dims_mirror.start()
        self._start_worker(loop)
        try:
            self._update_scene_manager()
        except Exception:
            logger.debug("Initial scene manager sync failed", exc_info=True)
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
        pixel_channel.prepare_client_attach(self._pixel_channel)
        await pixel_channel.ingest_client(
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
        await pixel_channel.run_channel_loop(
            self._pixel_channel,
            config=self._pixel_config,
            metrics=self.metrics,
        )

    def _commit_level_snapshot(
        self,
        applied: LevelContext,
        downgraded: bool,
    ) -> None:
        level_update = apply_level_switch_transaction(
            store=self._scene,
            ledger=self._state_ledger,
            lock=self._state_lock,
            applied=applied,
            downgraded=bool(downgraded),
            origin="worker.state.level",
        )
        seq_value = int(level_update.server_seq)
        self._applied_seqs["multiscale"] = seq_value
        self._applied_seqs["dims"] = seq_value
        # Keep the plane view cache in sync with the latest applied level/step
        # when we are in 2D mode. This ensures 3D->2D restores pick the most
        # recent plane level even if no camera interaction occurred since the
        # level change.
        with self._state_lock:
            nd_entry = self._state_ledger.get("view", "main", "ndisplay")
            if nd_entry is not None and int(nd_entry.value) < 3:
                step_entry = self._state_ledger.get("dims", "main", "current_step")
                if step_entry is not None:
                    step_tuple = tuple(int(v) for v in step_entry.value)
                    cache_entries = [
                        ("view_cache", "plane", "level", int(applied.level)),
                        ("view_cache", "plane", "step", step_tuple),
                    ]
                    self._state_ledger.batch_record_confirmed(
                        cache_entries,
                        origin="worker.state.level",
                    )

    def _commit_applied_camera(
        self,
        pose: CameraPoseApplied,
    ) -> None:
        target = pose.target or "main"
        seq = int(pose.command_seq)
        last_seq = int(self._last_camera_pose_seq.get(target, 0))
        if seq <= last_seq:
            logger.debug(
                "camera pose ignored (stale) target=%s seq=%d last_seq=%d",
                target,
                seq,
                last_seq,
            )
            return
        self._last_camera_pose_seq[target] = seq

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
            ack = reduce_camera_update(
                self._state_ledger,
                center=pose.center,
                zoom=pose.zoom,
                angles=pose.angles,
                distance=pose.distance,
                fov=pose.fov,
                rect=pose.rect,
                origin="worker.state.camera",
            )
            self._scene.latest_state = build_render_scene_state(
                self._state_ledger,
                self._scene,
            )

            # Persist per-mode camera caches for deterministic restores.
            # Plane cache: camera_plane.* and view_cache.plane.*
            # Volume cache: camera_volume.*
            # Decide plane vs volume from presence of angles/distance/fov in applied pose.
            if pose.angles is None and pose.distance is None and pose.fov is None:
                # 2D plane pose expected: center, zoom, rect must be present
                assert pose.center is not None, "plane pose requires center"
                assert pose.zoom is not None, "plane pose requires zoom"
                assert pose.rect is not None, "plane pose requires rect"

                lvl_entry = self._state_ledger.get("multiscale", "main", "level")
                assert lvl_entry is not None, "ledger missing multiscale/level"
                lvl_value = int(lvl_entry.value)
                step_entry = self._state_ledger.get("dims", "main", "current_step")
                assert step_entry is not None, "ledger missing dims/current_step"
                step_tuple = tuple(int(v) for v in step_entry.value)

                plane_entries = [
                    ("camera_plane", "main", "center", tuple(float(c) for c in pose.center)),
                    ("camera_plane", "main", "zoom", float(pose.zoom)),
                    ("camera_plane", "main", "rect", tuple(float(v) for v in pose.rect)),
                ]
                self._state_ledger.batch_record_confirmed(
                    plane_entries,
                    origin="worker.state.camera_plane",
                )

                view_cache_entries = [
                    ("view_cache", "plane", "level", lvl_value),
                    ("view_cache", "plane", "step", step_tuple),
                ]
                self._state_ledger.batch_record_confirmed(
                    view_cache_entries,
                    origin="worker.state.camera_plane",
                )
            else:
                # 3D volume pose expected: center, angles, distance, fov must be present
                assert pose.center is not None, "volume pose requires center"
                assert pose.angles is not None, "volume pose requires angles"
                assert pose.distance is not None, "volume pose requires distance"
                assert pose.fov is not None, "volume pose requires fov"

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
                state=ack,
                intent_id=None,
                origin="worker.state.camera",
            ),
            "worker-camera-pose",
        )

    def _scene_snapshot_json(self) -> Optional[str]:
        try:
            self._update_scene_manager()
        except Exception:
            logger.debug("scene manager update failed before notify.scene", exc_info=True)

        try:
            payload = build_notify_scene_payload(
                self._scene,
                self._scene_manager,
                ledger_snapshot=self._state_ledger.snapshot(),
                timestamp=time.time(),
            )
            json_payload = json.dumps(payload.to_dict())
        except Exception:
            logger.debug("scene notify snapshot build failed", exc_info=True)
            return None

        if self._log_dims_info:
            snapshot = self._scene_manager.scene_snapshot()
            dims = snapshot.viewer.dims if snapshot is not None else {}
            logger.info(
                "notify.scene dims: level_shapes=%s current_level=%s",
                dims.get('level_shapes'),
                dims.get('current_level'),
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
        # We could track state clients separately if desired; here we reuse pixel_clients for demo
        self._publish_policy_metrics()

    def _publish_policy_metrics(self) -> None:
        worker = self._worker
        if worker is None:
            return
        snapshot = worker.policy_metrics_snapshot()
        if not isinstance(snapshot, Mapping):
            raise TypeError("policy metrics snapshot must be a mapping")

        metrics_server.update_policy_metrics(
            self._scene,
            self.metrics,
            snapshot,
        )


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
