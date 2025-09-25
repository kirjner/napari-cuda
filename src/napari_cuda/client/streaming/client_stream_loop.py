from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from threading import Thread
from typing import Callable, Dict, Optional
import weakref
from dataclasses import dataclass, field
from contextlib import ExitStack

import numpy as np
from qtpy import QtCore
import uuid

from napari_cuda.client.streaming.presenter import FixedLatencyPresenter, SourceMux
from napari_cuda.client.streaming.receiver import PixelReceiver, Packet
from napari_cuda.client.streaming.state import StateChannel
from napari_cuda.client.streaming.controllers import StateController, ReceiveController
from napari_cuda.client.streaming.types import Source, SubmittedFrame
from napari_cuda.client.streaming.renderer import GLRenderer
from napari_cuda.client.streaming.decoders.pyav import PyAVDecoder
from napari_cuda.client.streaming.decoders.vt import VTLiveDecoder
from napari_cuda.client.streaming.eventloop_monitor import EventLoopMonitor
from napari_cuda.client.streaming.input import InputSender
from napari_cuda.codec.avcc import (
    annexb_to_avcc,
    is_annexb,
    split_annexb,
    split_avcc_by_len,
    build_avcc,
    find_sps_pps,
)
from napari_cuda.codec.h264 import contains_idr_annexb, contains_idr_avcc
from napari_cuda.codec.h264_encoder import H264Encoder, EncoderConfig
from napari_cuda.client.streaming.client_loop.scheduler import CallProxy, WakeProxy
from napari_cuda.client.streaming.client_loop.pipelines import (
    build_pyav_pipeline,
    build_vt_pipeline,
)
from napari_cuda.client.streaming.client_loop.renderer_fallbacks import RendererFallbacks
from napari_cuda.client.streaming.client_loop.smoke_helpers import start_smoke_mode
from napari_cuda.client.streaming.client_loop.input_helpers import (
    attach_input_sender,
    bind_shortcuts,
)
from napari_cuda.client.streaming.client_loop.scheduler_helpers import (
    init_wake_scheduler,
)
from napari_cuda.client.streaming.client_loop.telemetry import (
    build_telemetry_config,
    create_metrics,
    start_metrics_timer,
    start_stats_timer,
)
from napari_cuda.client.streaming.client_loop.client_loop_config import load_client_loop_config
from napari_cuda.client.streaming.config import extract_video_config
from napari_cuda.client.layers import RemoteLayerRegistry, RegistrySnapshot
from napari_cuda.protocol.messages import (
    LayerRemoveMessage,
    LayerSpec,
    LayerUpdateMessage,
    SceneSpecMessage,
)

logger = logging.getLogger(__name__)


def _maybe_enable_debug_logger() -> None:
    """Enable DEBUG logs for this module only when env is set.

    - Attaches a dedicated StreamHandler at DEBUG.
    - Disables propagation so other libraries don't flood the console.
    - Triggered by NAPARI_CUDA_CLIENT_DEBUG or NAPARI_CUDA_DEBUG.
    """
    flag = (os.getenv('NAPARI_CUDA_CLIENT_DEBUG') or os.getenv('NAPARI_CUDA_DEBUG') or '').lower()
    if flag not in ('1', 'true', 'yes', 'on', 'dbg', 'debug'):
        return
    has_local = any(hasattr(h, '_napari_cuda_local') and h._napari_cuda_local for h in logger.handlers)
    if has_local:
        return
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(logging.DEBUG)
    setattr(handler, '_napari_cuda_local', True)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def _int_or_none(value: object) -> Optional[int]:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _float_or_none(value: object) -> Optional[float]:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _bool_or_none(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


@dataclass
class _SyncState:
    last_seq: Optional[int] = None
    last_disco_log: float = 0.0
    keyframes_seen: int = 0

    def update_and_check(self, cur: int) -> bool:
        """Return True if sequence discontinuity detected."""
        if self.last_seq is None:
            self.last_seq = int(cur)
            return False
        expected = (int(self.last_seq) + 1) & 0xFFFFFFFF
        if int(cur) != expected:
            now = time.time()
            if (now - self.last_disco_log) > 0.2:
                logger.warning(
                    "Pixel stream discontinuity: expected=%d got=%d; gating until keyframe",
                    expected,
                    int(cur),
                )
                self.last_disco_log = now
            self.last_seq = int(cur)
            return True
        self.last_seq = int(cur)
        return False


@dataclass
class LoopState:
    """Aggregate threads, channel, and cached state for the client loop."""

    threads: list[Thread] = field(default_factory=list)
    state_channel: 'StateChannel | None' = None
    state_thread: Thread | None = None
    pixel_receiver: 'PixelReceiver | None' = None
    pixel_thread: Thread | None = None
    pending_intents: dict[int, dict[str, object]] = field(default_factory=dict)
    last_dims_seq: int | None = None
    last_dims_payload: dict[str, object] | None = None


class ClientStreamLoop:
    """Orchestrates receiver, state, decoders, presenter, and renderer.

    This is a slim re-hosting of the logic previously embedded in
    StreamingCanvas, with behavior unchanged.
    """

    def __init__(
        self,
        scene_canvas,
        server_host: str,
        server_port: int,
        state_port: int,
        vt_latency_s: float,
        vt_buffer_limit: int,
        pyav_latency_s: Optional[float] = None,
        stream_format: str = 'avcc',
        vt_backlog_trigger: int = 16,
        pyav_backlog_trigger: int = 16,
        vt_smoke: bool = False,
        client_cfg: object | None = None,
        *,
        on_first_dims_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        self._scene_canvas = scene_canvas
        self._canvas_native = scene_canvas.native if hasattr(scene_canvas, 'native') else None
        _maybe_enable_debug_logger()
        # Set smoke mode early so start() can rely on it even if later init changes
        self._vt_smoke = bool(vt_smoke)
        # Phase 0: stash client config for future phases (no behavior change)
        self._client_cfg = client_cfg
        self._env_cfg = load_client_loop_config()
        self._telemetry_cfg = build_telemetry_config(
            stats_mode=self._env_cfg.vt_stats_mode,
            metrics_enabled=self._env_cfg.metrics_enabled,
            metrics_interval_ms=self._env_cfg.metrics_interval_ms,
        )
        self._first_dims_ready_cb = on_first_dims_ready
        self._first_dims_notified = False
        self._loop_state = LoopState()
        self.server_host = server_host
        self.server_port = int(server_port)
        self.state_port = int(state_port)
        self._stream_format = stream_format
        self._stream_format_set = False
        # Thread/state handles
        # Optional weakref to a ViewerModel mirror (e.g., ProxyViewer)
        self._viewer_mirror = None  # type: ignore[var-annotated]
        # UI call proxy to marshal updates to the GUI thread
        self._ui_call: CallProxy | None
        self._ui_call = CallProxy(self._canvas_native) if self._canvas_native is not None else None

        # Presenter + source mux (server timestamps only)
        # Single preview guard (ms)
        preview_guard_attr = self._client_cfg.preview_guard_ms if (self._client_cfg is not None and hasattr(self._client_cfg, 'preview_guard_ms')) else 0.0
        preview_guard_ms = float(preview_guard_attr)
        # No env overrides; preview_guard_ms comes from ClientConfig only
        self._presenter = FixedLatencyPresenter(
            latency_s=float(vt_latency_s),
            buffer_limit=int(vt_buffer_limit),
            preview_guard_s=float(preview_guard_ms) / 1000.0,
        )
        logger.info(
            "Presenter init: preview_guard=%.1fms latency=%.0fms",
            float(preview_guard_ms),
            float(vt_latency_s) * 1000.0,
        )
        self._source_mux = SourceMux(Source.PYAV)
        self._vt_latency_s = float(vt_latency_s)
        self._pyav_latency_s = float(pyav_latency_s) if pyav_latency_s is not None else max(0.06, float(vt_latency_s))
        # Default to PyAV latency until VT is proven ready
        self._presenter.set_latency(self._pyav_latency_s)
        # Scene specification cache for client-side mirroring work
        self._scene_lock = threading.Lock()
        self._latest_scene_spec: Optional[SceneSpecMessage] = None
        self._layer_registry = RemoteLayerRegistry()
        self._layer_registry.add_listener(self._on_registry_snapshot)
        # Keep-last-frame fallback default enabled for smoother presentation
        self._keep_last_frame_fallback = True
        # Monotonic scheduling marker for next due
        self._next_due_pending_until: float = 0.0
        # Startup warmup (arrival mode): temporarily increase latency then ramp down
        # Auto-sized to roughly exceed one frame interval (assume 60 Hz if FPS unknown)
        self._warmup_ms_override = self._env_cfg.warmup_ms_override
        self._warmup_window_s = self._env_cfg.warmup_window_s
        self._warmup_margin_ms = self._env_cfg.warmup_margin_ms
        self._warmup_max_ms = self._env_cfg.warmup_max_ms
        self._warmup_until: float = 0.0
        self._warmup_extra_active_s: float = 0.0
        self._fps: Optional[float] = None

        # Renderer
        self._renderer = GLRenderer(self._scene_canvas)

        # Decoders
        self.decoder: Optional[PyAVDecoder] = None
        self._vt_decoder: Optional[VTLiveDecoder] = None
        self._vt_backend: Optional[str] = None
        self._vt_cfg_key = None
        self._vt_wait_keyframe: bool = False
        self._vt_gate_lift_time: float = 0.0  # wall time at VT gate lift
        self._mono_at_gate: float = 0.0       # perf_counter() at VT gate lift
        self._wall_to_mono: Optional[float] = None  # server_wall -> client_mono offset
        self._vt_ts_offset: Optional[float] = None  # legacy wall->wall offset (unused once monotonic is active)
        self._vt_errors = 0
        # Bias for server timestamp alignment; default 0 for exact due times
        self._server_bias_s = self._env_cfg.server_bias_s
        # Wake scheduling fudge (ms) to compensate for timer quantization; added to due delay
        self._wake_fudge_ms = self._env_cfg.wake_fudge_ms
        # avcC nal length size (default 4 if unknown)
        self._nal_length_size: int = 4

        # VT pipeline replaces inline queues/workers
        self._vt_backlog_trigger = int(vt_backlog_trigger)
        self._vt_enqueued = 0
        # Client metrics (optional, env-controlled)
        self._metrics = create_metrics(self._telemetry_cfg)

        # Presenter-owned wake scheduling helper
        self._wake_timer = None
        self._in_present = False
        self._use_display_loop = self._env_cfg.use_display_loop
        self._wake_proxy: WakeProxy | None = None
        self._gui_thread = None
        self._schedule_next_wake = init_wake_scheduler(self)

        # Build pipelines after scheduler hooks are ready
        # Create a wake proxy so pipelines can nudge scheduling from any thread
        if not self._use_display_loop and self._wake_proxy is not None:
            wake_cb = self._wake_proxy.trigger.emit
        else:
            wake_cb = (lambda: None)
        self._vt_pipeline = build_vt_pipeline(self, schedule_next_wake=wake_cb, logger=logger)
        self._pyav_backlog_trigger = int(pyav_backlog_trigger)
        self._pyav_enqueued = 0
        self._pyav_pipeline = build_pyav_pipeline(self, schedule_next_wake=wake_cb)
        self._pyav_wait_keyframe: bool = False

        # Renderer fallback manager (VT cache + PyAV reuse)
        self._renderer_fallbacks = RendererFallbacks()

        # Smoke harness (optional)
        self._smoke = None

        # Frame queue for renderer (latest-wins)
        # Holds either numpy arrays or (capsule, release_cb) tuples for VT
        self._frame_q: "queue.Queue[object]" = queue.Queue(maxsize=3)

        # Receiver/state flags
        self._stream_seen_keyframe = False

        # Stats/logging and diagnostics
        self._stats_level = self._telemetry_cfg.stats_level
        self._last_stats_time: float = 0.0
        self._stats_timer = None
        self._metrics_timer = None
        self._warmup_reset_timer = None
        self._relearn_logged: bool = False
        self._last_relearn_log_ts: float = 0.0
        # Debounce duplicate video_config
        self._last_vcfg_key = None
        # Stream continuity and gate tracking
        self._sync = _SyncState()
        self._disco_gated: bool = False
        self._last_key_logged: Optional[int] = None
        # Draw pacing diagnostics
        self._last_draw_pc: float = 0.0
        self._last_present_mono: float = 0.0
        self._in_draw: bool = False
        # Draw watchdog
        self._watchdog_timer = None
        self._watchdog_ms = self._env_cfg.watchdog_ms
        # Event loop stall monitor (disabled by default)
        self._evloop_stall_ms = self._env_cfg.evloop_stall_ms
        self._evloop_sample_ms = self._env_cfg.evloop_sample_ms
        self._evloop_mon: Optional[EventLoopMonitor] = None
        self._quit_hook_connected = False
        self._stopped = False
        # Video dimensions (from video_config)
        self._vid_w: Optional[int] = None
        self._vid_h: Optional[int] = None
        # View HUD diagnostics (for tuning 3D volume controls)
        self._last_zoom_factor: Optional[float] = None
        self._last_zoom_widget_px: Optional[tuple[float, float]] = None
        self._last_zoom_video_px: Optional[tuple[float, float]] = None
        self._last_zoom_anchor_px: Optional[tuple[float, float]] = None
        self._last_pan_dx_sent: float = 0.0
        self._last_pan_dy_sent: float = 0.0
        self._zoom_base: float = self._env_cfg.zoom_base
        # Dims/Z control (client-initiated)
        self._dims_z = self._env_cfg.dims_z
        self._dims_z_min = self._env_cfg.dims_z_min
        self._dims_z_max = self._env_cfg.dims_z_max
        # Clamp initial Z if provided
        if self._dims_z is not None:
            if self._dims_z_min is not None and self._dims_z < self._dims_z_min:
                self._dims_z = self._dims_z_min
            if self._dims_z_max is not None and self._dims_z > self._dims_z_max:
                self._dims_z = self._dims_z_max
        self._wheel_px_accum: float = 0.0
        self._wheel_step = self._env_cfg.wheel_step
        # dims intents rate limiting (coalesce)
        rate = self._env_cfg.dims_rate_hz
        self._dims_min_dt = 1.0 / max(1.0, rate)
        self._last_dims_send: float = 0.0
        # Camera ops coalescing
        cam_rate = self._env_cfg.camera_rate_hz
        self._cam_min_dt = 1.0 / max(1.0, cam_rate)
        self._last_cam_send: float = 0.0
        self._dragging: bool = False
        self._last_wx: float = 0.0
        self._last_wy: float = 0.0
        self._pan_dx_accum: float = 0.0
        self._pan_dy_accum: float = 0.0
        # Orbit (Alt-drag) accumulation (3D volume mode)
        self._orbit_daz_accum: float = 0.0
        self._orbit_del_accum: float = 0.0
        self._orbit_dragging: bool = False
        self._orbit_deg_per_px_x = self._env_cfg.orbit_deg_per_px_x
        self._orbit_deg_per_px_y = self._env_cfg.orbit_deg_per_px_y
        # --- Dims/intent state (intent-only client) ----------------------------
        self._dims_ready: bool = False
        self._dims_meta: dict = {
            'ndim': None,
            'order': None,
            'axis_labels': None,
            'range': None,
            'sizes': None,
            'ndisplay': None,
            'volume': None,
            'render': None,
            'multiscale': None,
        }
        self._primary_axis_index: int | None = None
        # Client identity for intents
        self._client_id: str = uuid.uuid4().hex
        self._client_seq: int = 0
        # Settings/mode intents rate limiting (coalesce like dims intents)
        settings_rate = self._env_cfg.settings_rate_hz
        self._settings_min_dt = 1.0 / max(1.0, settings_rate)
        self._last_settings_send: float = 0.0

    # --- Thread-safe scheduling helpers --------------------------------------
    def _on_gui_thread(self) -> bool:
        gui_thr = self._gui_thread
        if gui_thr is None:
            return True
        return QtCore.QThread.currentThread() == gui_thr

    def schedule_next_wake_threadsafe(self) -> None:
        """Ensure wake scheduling runs on the GUI thread.

        Prefer the wake proxy; fallback to UI call proxy; otherwise no-op.
        """
        if self._on_gui_thread():
            self._schedule_next_wake()
            return

        if self._wake_proxy is not None:
            self._wake_proxy.trigger.emit()
            return

        if self._ui_call is not None:
            self._ui_call.call.emit(self._schedule_next_wake)
            return

        logger.debug("threadsafe schedule: no proxy available; skipping")

    def _scene_canvas_update(self) -> None:
        if self._canvas_native is not None:
            self._canvas_native.update()
            return
        if hasattr(self._scene_canvas, 'update'):
            update = self._scene_canvas.update  # type: ignore[attr-defined]
            assert callable(update), "Streaming canvas must expose update()"
            update()
            return
        raise AssertionError("Streaming canvas must expose update()")

    def _on_present_timer(self) -> None:
        with ExitStack() as stack:
            # On wake, request a paint; the actual GL draw happens inside the canvas draw event
            self._in_present = True
            stack.callback(self._present_reset)
            # Clear pending marker to allow next wake to be scheduled freely
            self._next_due_pending_until = 0.0
            self._scene_canvas_update()

    def _present_reset(self) -> None:
        self._in_present = False

    def start(self) -> None:
        self._stopped = False
        # State channel thread (disabled in offline VT smoke mode)
        if not self._vt_smoke:
            st = StateController(
                self.server_host,
                self.state_port,
                handle_video_config=self._handle_video_config,
                handle_dims_update=self._handle_dims_update,
                handle_scene_spec=self._handle_scene_spec,
                handle_layer_update=self._handle_layer_update,
                handle_layer_remove=self._handle_layer_remove,
                handle_connected=self._on_state_connected,
                handle_disconnect=self._on_state_disconnect,
            )
            state_channel, t_state = st.start()
            self._loop_state.state_channel = state_channel
            self._loop_state.state_thread = t_state
            self._loop_state.threads.append(t_state)
            # Attach input forwarding (wheel + resize) now that state channel exists
            # Use same flag for dims logging (INFO when enabled, DEBUG otherwise)
            self._log_dims_info = bool(self._env_cfg.input_log)  # type: ignore[attr-defined]
            attach_input_sender(self)
            bind_shortcuts(self)

        # Start VT pipeline workers
        self._vt_pipeline.start()

        # Start PyAV pipeline worker
        self._pyav_pipeline.start()

        # Receiver thread or smoke mode
        if self._vt_smoke:
            self._smoke = start_smoke_mode(self)
        else:
            rc = ReceiveController(
                self.server_host,
                self.server_port,
                on_connected=self._handle_connected,
                on_frame=self._on_frame,
                on_disconnect=self._handle_disconnect,
            )
            receiver, t_rx = rc.start()
            self._loop_state.pixel_receiver = receiver
            self._loop_state.pixel_thread = t_rx
            self._loop_state.threads.append(t_rx)

        # Dedicated 1 Hz presenter stats timer (decoupled from draw loop)
        self._stats_timer = start_stats_timer(
            self._scene_canvas.native,
            stats_level=self._stats_level,
            callback=self._log_stats,
            logger=logger,
        )

        # Client metrics CSV dump timer (independent of stats log level)
        self._metrics_timer = start_metrics_timer(
            self._scene_canvas.native,
            config=self._telemetry_cfg,
            metrics=self._metrics,
            logger=logger,
        )
        # Optional draw watchdog: if no draw observed for threshold, kick an update
        if self._watchdog_ms > 0:
            native_canvas = self._scene_canvas.native
            assert native_canvas is not None, "Draw watchdog requires a native canvas"
            timer = QtCore.QTimer(native_canvas)
            timer.setTimerType(QtCore.Qt.PreciseTimer)
            timer.setInterval(max(100, int(self._watchdog_ms // 2) or 100))

            def _wd_tick() -> None:
                last = float(self._last_draw_pc or 0.0)
                if last <= 0.0:
                    return
                now = time.perf_counter()
                if (now - last) * 1000.0 < float(self._watchdog_ms):
                    return
                if self._presenter.peek_next_due(self._source_mux.active) is None:
                    return
                self._scene_canvas_update()
                if self._metrics is not None:
                    self._metrics.inc('napari_cuda_client_draw_watchdog_kicks', 1.0)

            timer.timeout.connect(_wd_tick)
            timer.start()
            self._watchdog_timer = timer
            logger.info("Draw watchdog enabled: threshold=%d ms", self._watchdog_ms)
        # Optional event loop monitor (diagnostic): logs and metrics on stalls
        if self._evloop_stall_ms > 0:
            def _kick() -> None:
                self._scene_canvas_update()

            self._evloop_mon = EventLoopMonitor(
                parent=self._scene_canvas.native,
                metrics=self._metrics,
                stall_threshold_ms=int(self._evloop_stall_ms),
                sample_interval_ms=int(self._evloop_sample_ms),
                on_stall_kick=_kick,
            )

        if not self._quit_hook_connected:
            app = QtCore.QCoreApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self.stop)
                self._quit_hook_connected = True

    def _enqueue_frame(self, frame: object) -> None:
        if self._frame_q.full():
            self._frame_q.get_nowait()
        self._frame_q.put(frame)

    def draw(self) -> None:
        # Guard to avoid scheduling immediate repaints from within draw
        in_draw_prev = self._in_draw
        self._in_draw = True
        # Draw-loop pacing metric (perf_counter for monotonic interval)
        now_pc = time.perf_counter()
        last_pc = float(self._last_draw_pc or 0.0)
        if last_pc > 0.0 and self._metrics is not None:
            self._metrics.observe_ms('napari_cuda_client_draw_interval_ms', (now_pc - last_pc) * 1000.0)
        self._last_draw_pc = now_pc
        # Apply warmup ramp/restore on GUI thread (timer-less)
        self._apply_warmup(now_pc)
        # VT output is drained continuously by worker; draw focuses on presenting
        # If no offset learned yet (common in smoke/offline), derive from buffer samples.
        clock_offset = self._presenter.clock.offset
        if clock_offset is None:
            learned = self._presenter.relearn_offset(Source.VT)
            if learned is not None and math.isfinite(learned):
                logger.info("Presenter offset learned from buffer: %.3fs", float(learned))

        ready = self._presenter.pop_due(time.perf_counter(), self._source_mux.active)
        if ready is not None:
            src_val = ready.source.value
            # Record presentation lateness relative to due time for non-preview frames
            if not ready.preview and self._metrics is not None and self._metrics.enabled:
                late_ms = (time.perf_counter() - float(ready.due_ts)) * 1000.0
                self._metrics.observe_ms('napari_cuda_client_present_lateness_ms', float(late_ms))
            # Optional: log a one-time relearn attempt on early preview streaks (lightweight)
            if src_val == 'vt':
                if ready.preview and not self._relearn_logged:
                    # Rate-limit relearn logs to 1/sec
                    now = time.time()
                    if (now - float(self._last_relearn_log_ts or 0.0)) >= 1.0:
                        # Only log preview/relearn when explicitly enabled
                        import os as _os
                        if (_os.getenv('NAPARI_CUDA_PREVIEW_DEBUG') or '').lower() in ('1','true','yes','on','dbg','debug'):
                            off = self._presenter.relearn_offset(Source.VT)
                            if off is not None:
                                logger.debug(
                                    "VT preview detected; relearned offset=%s",
                                    f"{off:.3f}s",
                                )
                            else:
                                logger.debug("VT preview detected; offset relearn not available yet")
                        self._last_relearn_log_ts = now
                    self._relearn_logged = True
                elif not ready.preview:
                    self._relearn_logged = False
            if src_val == 'vt':
                # Always pass renderer release callback; leases manage refcounts internally.
                self._enqueue_frame((ready.payload, ready.release_cb))
            else:
                # Cache for last-frame fallback and enqueue
                self._renderer_fallbacks.store_pyav_frame(ready.payload)
                self._enqueue_frame(ready.payload)

        frame = None
        if ready is None:
            if self._source_mux.active == Source.VT:
                self._renderer_fallbacks.try_enqueue_cached_vt(self._enqueue_frame)
            elif self._source_mux.active == Source.PYAV:
                self._renderer_fallbacks.try_enqueue_pyav(self._enqueue_frame)
        # Schedule next wake based on earliest due; avoids relying on a 60 Hz loop
        # Safe: draw() runs on GUI thread
        self._schedule_next_wake()
        while not self._frame_q.empty():
            frame = self._frame_q.get_nowait()
        # Time the render operation based on frame type
        if frame is not None:
            is_vt = isinstance(frame, tuple) and len(frame) == 2
            t_render0 = time.perf_counter()
            self._renderer.draw(frame)
            t_render1 = time.perf_counter()
            if self._metrics is not None and self._metrics.enabled:
                metric_name = 'napari_cuda_client_render_vt_ms' if is_vt else 'napari_cuda_client_render_pyav_ms'
                self._metrics.observe_ms(metric_name, (t_render1 - t_render0) * 1000.0)
            # Count a presented frame for client-side FPS derivation (only when a frame was actually drawn)
            if self._metrics is not None:
                self._metrics.inc('napari_cuda_client_presented_total', 1.0)
            now = time.perf_counter()
            last = self._last_present_mono
            if last:
                # Only log PRESENT timing when explicitly enabled to avoid spam
                import os as _os
                if (_os.getenv('NAPARI_CUDA_PRESENT_DEBUG') or '').lower() in ('1','true','yes','on','dbg','debug'):
                    inter_ms = (now - float(last)) * 1000.0
                    logger.debug("PRESENT inter_ms=%.3f", float(inter_ms))
            self._last_present_mono = now
        else:
            self._renderer.draw(frame)
        # Clear draw guard
        self._in_draw = in_draw_prev

    def _log_stats(self) -> None:
        if self._stats_level is None:
            return
        pres_stats = self._presenter.stats()
        vt_counts = self._vt_pipeline.counts()
        if self._metrics is not None:
            self._metrics.set('napari_cuda_client_present_buf_vt', float(pres_stats['buf'].get('vt', 0)))
            self._metrics.set('napari_cuda_client_present_buf_pyav', float(pres_stats['buf'].get('pyav', 0)))
            self._metrics.set('napari_cuda_client_present_latency_ms', float(pres_stats.get('latency_ms', 0.0)))
        logger.log(
            self._stats_level,
            "presenter=%s vt_counts=%s keyframes_seen=%d",
            pres_stats,
            vt_counts,
            self._sync.keyframes_seen,
        )

    def _init_decoder(self) -> None:
        import os as _os
        swap_rb = (_os.getenv('NAPARI_CUDA_CLIENT_SWAP_RB', '0') or '0') in ('1',)
        pf = (_os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24') or 'rgb24').lower()
        self.decoder = PyAVDecoder(self._stream_format, pixfmt=pf, swap_rb=swap_rb)

    def _init_vt_from_avcc(self, avcc_b64: str, width: int, height: int) -> None:
        import base64
        import sys
        avcc = base64.b64decode(avcc_b64)
        cfg_key = (int(width), int(height), avcc)
        # Ignore exact duplicate configs
        if self._last_vcfg_key == cfg_key:
            logger.debug("Duplicate video_config ignored (%dx%d)", width, height)
            return
        if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
            logger.debug("VT already initialized; ignoring duplicate video_config")
            return
        # Respect NAPARI_CUDA_VT_BACKEND env: off/0/false disables VT
        backend_env = (os.getenv('NAPARI_CUDA_VT_BACKEND', 'shim') or 'shim').lower()
        vt_disabled = backend_env in ("off", "0", "false", "no")
        if sys.platform == 'darwin' and not vt_disabled:
            try:
                self._vt_decoder = VTLiveDecoder(avcc, width, height)
                self._vt_backend = 'shim'
                self._vt_pipeline.set_decoder(self._vt_decoder)
            except Exception as exc:
                logger.warning("VT shim unavailable: %s; falling back to PyAV", exc)
                self._vt_decoder = None
        else:
            self._vt_decoder = None
        self._vt_cfg_key = cfg_key
        self._last_vcfg_key = cfg_key
        self._vt_wait_keyframe = True
        # Parse nal length size from avcC if available (5th byte low 2 bits + 1)
        if len(avcc) >= 5:
            nsz = int((avcc[4] & 0x03) + 1)
            if nsz in (1, 2, 3, 4):
                self._nal_length_size = nsz
            else:
                logger.warning("Invalid avcC nal_length_size=%s; defaulting to 4", nsz)
                self._nal_length_size = 4
        else:
            logger.warning("avcC too short (%d); defaulting nal_length_size=4", len(avcc))
            self._nal_length_size = 4
        self._vt_pipeline.update_nal_length_size(self._nal_length_size)
        logger.info("VideoToolbox live decoder initialized: %dx%d", width, height)

    def _request_keyframe_once(self) -> None:
        ch = self._loop_state.state_channel
        if ch is not None:
            ch.request_keyframe_once()

    # Extracted helpers
    def _handle_video_config(self, data: dict) -> None:
        w, h, fps, fmt, avcc_b64 = extract_video_config(data)
        if fps > 0:
            self._fps = fps
        self._stream_format = fmt
        self._stream_format_set = True
        if w > 0 and h > 0 and avcc_b64:
            # cache video dimensions for input mapping
            self._vid_w = int(w)
            self._vid_h = int(h)
            self._init_vt_from_avcc(avcc_b64, w, h)

    def _handle_dims_update(self, data: dict) -> None:
        seq_val = _int_or_none(data.get('seq'))
        if seq_val is not None:
            self._loop_state.last_dims_seq = seq_val

        cur = data.get('current_step')
        ndisp = data.get('ndisplay')
        ndim = data.get('ndim')
        dims_range = data.get('range')
        order = data.get('order')
        axis_labels = data.get('axis_labels')
        sizes = data.get('sizes')
        volume = data.get('volume')
        render = data.get('render')
        multiscale = data.get('multiscale')
        displayed = data.get('displayed')
        level = data.get('level')
        level_shape = data.get('level_shape')
        dtype = data.get('dtype')
        normalized = data.get('normalized')
        ack_val = data.get('ack') if isinstance(data.get('ack'), bool) else None
        intent_seq = _int_or_none(data.get('intent_seq'))

        if ndim is not None:
            self._dims_meta['ndim'] = int(ndim)
        if order is not None:
            self._dims_meta['order'] = order
        if axis_labels is not None:
            self._dims_meta['axis_labels'] = axis_labels
        if dims_range is not None:
            self._dims_meta['range'] = dims_range
        if sizes is not None:
            self._dims_meta['sizes'] = sizes
        if displayed is not None:
            self._dims_meta['displayed'] = displayed
        if level is not None:
            self._dims_meta['level'] = level
        if level_shape is not None:
            self._dims_meta['level_shape'] = level_shape
        if dtype is not None:
            self._dims_meta['dtype'] = dtype
        if normalized is not None:
            self._dims_meta['normalized'] = normalized
        if volume is not None:
            self._dims_meta['volume'] = bool(volume)
        if render is not None:
            self._dims_meta['render'] = render
        if multiscale is not None:
            self._dims_meta['multiscale'] = multiscale
        if ndisp is not None:
            self._dims_meta['ndisplay'] = int(ndisp)

        if not self._dims_ready and (ndim is not None or order is not None):
            self._dims_ready = True
            logger.info("dims.update: metadata received; client intents enabled")
            self._notify_first_dims_ready()

        self._primary_axis_index = self._compute_primary_axis_index()

        if isinstance(cur, (list, tuple)) and cur and self._log_dims_info:
            logger.info(
                "dims_update: step=%s ndisp=%s order=%s labels=%s",
                list(cur),
                self._dims_meta.get('ndisplay'),
                self._dims_meta.get('order'),
                self._dims_meta.get('axis_labels'),
            )

        vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        if vm_ref is not None:
            self._mirror_dims_to_viewer(
                vm_ref,
                cur,
                ndisp,
                ndim,
                dims_range,
                order,
                axis_labels,
                sizes,
                displayed,
            )

        if intent_seq is not None:
            info = self._loop_state.pending_intents.pop(intent_seq, None)
            if info is not None:
                if ack_val is False:
                    logger.warning("dims_update ack=false for intent_seq=%s info=%s", intent_seq, info)
                elif self._log_dims_info:
                    logger.debug("dims_update ack: intent_seq=%s info=%s", intent_seq, info)

        self._loop_state.last_dims_payload = {
            'current_step': cur,
            'ndisplay': ndisp,
            'ndim': ndim,
            'dims_range': dims_range,
            'order': order,
            'axis_labels': axis_labels,
            'sizes': sizes,
            'displayed': displayed,
        }

    def _notify_first_dims_ready(self) -> None:
        if self._first_dims_notified:
            return
        cb = self._first_dims_ready_cb
        if not callable(cb):
            self._first_dims_notified = True
            return

        def _invoke() -> None:
            cb()

        self._first_dims_notified = True
        if self._ui_call is not None:
            self._ui_call.call.emit(_invoke)
            return
        _invoke()

    def _record_pending_intent(self, seq: int, info: dict[str, object]) -> None:
        """Track in-flight intents so we can reconcile on server ACKs."""
        self._loop_state.pending_intents[seq] = info

    def _replay_last_dims_payload(self) -> None:
        payload = self._loop_state.last_dims_payload
        if not payload:
            return
        vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        if vm_ref is None:
            return
        self._mirror_dims_to_viewer(
            vm_ref,
            payload.get('current_step'),
            payload.get('ndisplay'),
            payload.get('ndim'),
            payload.get('dims_range'),
            payload.get('order'),
            payload.get('axis_labels'),
            payload.get('sizes'),
            payload.get('displayed'),
        )

    def _handle_scene_spec(self, msg: SceneSpecMessage) -> None:
        """Cache latest scene specification and forward to registry."""
        with self._scene_lock:
            self._latest_scene_spec = msg
        self._layer_registry.apply_scene(msg)
        logger.debug(
            "scene.spec received: %d layers, capabilities=%s",
            len(msg.scene.layers),
            msg.scene.capabilities,
        )

    def _handle_layer_update(self, msg: LayerUpdateMessage) -> None:
        self._layer_registry.apply_update(msg)
        layer_id = msg.layer.layer_id if msg.layer else None
        logger.debug("layer.update: id=%s partial=%s", layer_id, msg.partial)

    def _handle_layer_remove(self, msg: LayerRemoveMessage) -> None:
        self._layer_registry.remove_layer(msg)
        logger.debug("layer.remove: id=%s reason=%s", msg.layer_id, msg.reason)

    def _on_registry_snapshot(self, snapshot: RegistrySnapshot) -> None:
        def _apply() -> None:
            self._sync_remote_layers(snapshot)
            self._replay_last_dims_payload()
            self._replay_last_dims_payload()
        if self._ui_call is not None:
            self._ui_call.call.emit(_apply)
            return
        _apply()

    def _sync_remote_layers(self, snapshot: RegistrySnapshot) -> None:
        vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        if vm_ref is None or not hasattr(vm_ref, '_sync_remote_layers'):
            return
        vm_ref._sync_remote_layers(snapshot)  # type: ignore[attr-defined]

    def _handle_connected(self) -> None:
        self._init_decoder()
        dec = self.decoder.decode if self.decoder else None
        self._pyav_pipeline.set_decoder(dec)

    def _handle_disconnect(self, exc: Exception | None) -> None:
        logger.info("PixelReceiver disconnected: %s", exc)

    # State channel lifecycle: gate dims intents safely across reconnects
    def _on_state_connected(self) -> None:
        self._dims_ready = False
        self._primary_axis_index = None
        logger.info("StateChannel connected; gating dims intents until dims.update meta arrives")

    def _on_state_disconnect(self, exc: Exception | None) -> None:
        self._dims_ready = False
        self._primary_axis_index = None
        self._loop_state.pending_intents.clear()
        self._loop_state.last_dims_seq = None
        self._loop_state.last_dims_payload = None
        logger.info("StateChannel disconnected: %s; dims intents gated", exc)

    # --- Input mapping: unified wheel handler -------------------------------------
    def _on_wheel(self, data: dict) -> None:
        # Decide between dims.set (plain) and zoom (modifier)
        mods = int(data.get('mods') or 0)
        ctrl = int(QtCore.Qt.ControlModifier)
        meta = int(QtCore.Qt.MetaModifier)
        alt = int(QtCore.Qt.AltModifier)
        use_zoom = (mods & (ctrl | meta | alt)) != 0
        if use_zoom:
            self._on_wheel_for_zoom(data)
        else:
            self._on_wheel_for_dims(data)

    # --- Input mapping: wheel -> dims.intent.step (primary axis) ---------------------
    def _on_wheel_for_dims(self, data: dict) -> None:
        ay = int(data.get('angle_y') or 0)
        py = int(data.get('pixel_y') or 0)
        mods = int(data.get('mods') or 0)
        # For now, ignore modifiers (server may treat ctrl as zoom based on input.wheel)
        step = 0
        if ay != 0:
            step = (1 if ay > 0 else -1) * int(self._wheel_step or 1)
        elif py != 0:
            # Accumulate pixel delta to synthesize steps (assume ~30 px per notch)
            self._wheel_px_accum += float(py)
            thr = 30.0
            while self._wheel_px_accum >= thr:
                step += int(self._wheel_step or 1)
                self._wheel_px_accum -= thr
            while self._wheel_px_accum <= -thr:
                step -= int(self._wheel_step or 1)
                self._wheel_px_accum += thr
        if step == 0:
            return
        # Send intent to server on primary axis
        sent = self.dims_step('primary', int(step), origin='wheel')
        if self._log_dims_info:
            logger.info(
                "wheel->dims.intent.step d=%+d sent=%s", int(step), bool(sent)
            )
        else:
            logger.debug("wheel->dims.intent.step d=%+d sent=%s", int(step), bool(sent))

    # Shortcut-driven stepping via intents (arrows/page)
    def _step_primary(self, delta: int, origin: str = 'keys') -> None:
        dz = int(delta)
        if dz == 0:
            return
        _ = self.dims_step('primary', dz, origin=origin)

    # --- Camera ops: zoom/pan/reset -----------------------------------------------
    def _widget_to_video(self, xw: float, yw: float) -> tuple[float, float]:
        # Map widget pixel coordinates to video pixel coordinates assuming
        # the video is stretched to fill the widget (no letterboxing).
        vw = float(self._vid_w or 0)
        vh = float(self._vid_h or 0)
        ww = float(self._scene_canvas.native.width())
        wh = float(self._scene_canvas.native.height())
        if vw <= 0 or vh <= 0 or ww <= 0 or wh <= 0:
            return (xw, yw)
        sx = vw / ww
        sy = vh / wh
        xv = float(xw) * sx
        yv = float(yw) * sy
        xv = max(0.0, min(vw, xv))
        yv = max(0.0, min(vh, yv))
        return (xv, yv)

    def _video_delta_to_canvas(self, dx_v: float, dy_v: float) -> tuple[float, float]:
        # Map video delta to canvas delta; make vertical drag direction feel natural
        return (float(dx_v), float(dy_v))

    def _server_anchor_from_video(self, xv: float, yv: float) -> tuple[float, float]:
        # Flip Y for bottom-left canvas origin expected by server
        vh = float(self._vid_h or 0)
        clamped = float(max(0.0, min(vh, vh - yv))) if vh > 0 else float(yv)
        return (float(xv), clamped)

    def _zoom_steps_at_center(self, steps: int) -> None:
        ch = self._loop_state.state_channel
        if ch is None:
            return
        base = float(self._zoom_base)
        s = int(steps)
        if s == 0:
            return
        # '+' zooms in, '-' zooms out
        f = base ** (-s)
        # Use cursor position if available; otherwise fall back to view center.
        cursor_xw = self._cursor_wx if hasattr(self, '_cursor_wx') else None
        cursor_yw = self._cursor_wy if hasattr(self, '_cursor_wy') else None
        if isinstance(cursor_xw, (int, float)) and isinstance(cursor_yw, (int, float)):
            xv, yv = self._widget_to_video(float(cursor_xw), float(cursor_yw))
        else:
            xv = float(self._vid_w or 0) / 2.0
            yv = float(self._vid_h or 0) / 2.0
        ax_s, ay_s = self._server_anchor_from_video(xv, yv)
        ok = ch.post({'type': 'camera.zoom_at', 'factor': float(f), 'anchor_px': [float(ax_s), float(ay_s)]})
        if self._log_dims_info:
            logger.info(
                "key->camera.zoom_at f=%.4f at(%.1f,%.1f) sent=%s",
                float(f), float(ax_s), float(ay_s), bool(ok)
            )

    def _reset_camera(self) -> None:
        ch = self._loop_state.state_channel
        if ch is None:
            return
        logger.info("key->camera.reset (sending)")
        ok = ch.post({'type': 'camera.reset'})
        logger.info("key->camera.reset sent=%s", bool(ok))

    def _on_key_event(self, data: dict) -> bool:
        """Optional app-level key handling for bindings that struggle as QShortcut.

        Triggers camera reset on plain '0' with no modifiers. Also accepts
        keypad 0 when the only modifier is the keypad flag.
        """
        key_raw = data.get('key')
        key = int(key_raw) if key_raw is not None else -1
        mods = int(data.get('mods') or 0)
        txt = str(data.get('text') or '')
        # Accept if explicitly '0' text with no modifiers
        if txt == '0' and mods == 0:
            logger.info("keycb: '0' -> camera.reset")
            self._reset_camera()
            return True
        # Or if the key is Key_0 and modifiers are none (or just keypad)
        keypad_mask = int(QtCore.Qt.KeypadModifier)
        keypad_only = (mods & ~keypad_mask) == 0 and (mods & keypad_mask) != 0
        if key == int(QtCore.Qt.Key_0) and (mods == 0 or keypad_only):
            logger.info("keycb: Key_0 -> camera.reset")
            self._reset_camera()
            return True
        # Map arrows and page keys to primary-axis stepping
        k_left = int(QtCore.Qt.Key_Left)
        k_right = int(QtCore.Qt.Key_Right)
        k_up = int(QtCore.Qt.Key_Up)
        k_down = int(QtCore.Qt.Key_Down)
        k_pgup = int(QtCore.Qt.Key_PageUp)
        k_pgdn = int(QtCore.Qt.Key_PageDown)
        if key in (k_left, k_right, k_up, k_down, k_pgup, k_pgdn):
            # Coarse step for PageUp/Down
            coarse = 10
            if key == k_left or key == k_down:
                self._step_primary(-1, origin='keys')
            elif key == k_right or key == k_up:
                self._step_primary(+1, origin='keys')
            elif key == k_pgup:
                self._step_primary(+coarse, origin='keys')
            elif key == k_pgdn:
                self._step_primary(-coarse, origin='keys')
            return True
        return False

    # --- Public UI/state bridge methods -------------------------------------------
    def attach_viewer_mirror(self, viewer: object) -> None:
        """Attach a viewer to mirror server dims updates into local UI.

        Stores a weak reference to avoid lifetime coupling.
        """
        self._viewer_mirror = weakref.ref(viewer)  # type: ignore[attr-defined]
        self._replay_last_dims_payload()

    def post(self, obj: dict) -> bool:
        ch = self._loop_state.state_channel
        return bool(ch.post(obj)) if ch is not None else False

    def reset_camera(self, origin: str = 'ui') -> bool:
        """Send a camera.reset to the server (used by UI bindings)."""
        ch = self._loop_state.state_channel
        if ch is None:
            return False
        logger.info("%s->camera.reset (sending)", origin)
        ok = ch.post({'type': 'camera.reset'})
        logger.info("%s->camera.reset sent=%s", origin, bool(ok))
        return bool(ok)

    def set_camera(self, *, center=None, zoom=None, angles=None, origin: str = 'ui') -> bool:
        """Send absolute camera fields when provided.

        Prefer using zoom_at/pan_px/reset ops for interactions; this is
        intended for explicit UI actions that set a known state.
        """
        ch = self._loop_state.state_channel
        if ch is None:
            return False
        payload: dict = {'type': 'set_camera'}
        if center is not None:
            payload['center'] = list(center)
        if zoom is not None:
            payload['zoom'] = float(zoom)
        if angles is not None:
            payload['angles'] = list(angles)
        logger.info("%s->set_camera %s", origin, {k: v for k, v in payload.items() if k != 'type'})
        ok = ch.post(payload)
        return bool(ok)

    # --- Intents API --------------------------------------------------------------
    def _axis_to_index(self, axis: int | str) -> Optional[int]:
        if axis == 'primary':
            return int(self._primary_axis_index) if self._primary_axis_index is not None else 0
        if isinstance(axis, (int, float)) or (isinstance(axis, str) and str(axis).isdigit()):
            return int(axis)
        labels = self._dims_meta.get('axis_labels')
        if isinstance(labels, (list, tuple)):
            label_map = {str(lbl): i for i, lbl in enumerate(labels)}
            match = label_map.get(str(axis))
            return int(match) if match is not None else None
        return None

    def _compute_primary_axis_index(self) -> Optional[int]:
        order = self._dims_meta.get('order')
        ndisplay = self._dims_meta.get('ndisplay')
        labels = self._dims_meta.get('axis_labels')
        nd = int(ndisplay) if ndisplay is not None else 2
        # Normalize order to indices
        idx_order: list[int] | None = None
        if isinstance(order, (list, tuple)) and len(order) > 0:
            if all(isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) for x in order):
                idx_order = [int(x) for x in order]
            elif isinstance(labels, (list, tuple)) and all(isinstance(x, str) for x in order):
                l2i = {str(lbl): i for i, lbl in enumerate(labels)}
                idx_order = [int(l2i.get(str(lbl), i)) for i, lbl in enumerate(order)]
        # Primary axis = first non-displayed axis (front of order excluding last ndisplay)
        if idx_order and len(idx_order) > nd:
            return int(idx_order[0])
        # Fallback: 0
        return 0

    def _next_client_seq(self) -> int:
        self._client_seq = (int(self._client_seq) + 1) & 0x7FFFFFFF
        return int(self._client_seq)

    # --- Mode helpers -----------------------------------------------------------
    def _is_volume_mode(self) -> bool:
        vol = bool(self._dims_meta.get('volume'))
        nd = int(self._dims_meta.get('ndisplay') or 2)
        return bool(vol) and int(nd) == 3

    # --- Small utilities (no behavior change) ----------------------------------
    def _clamp01(self, a: float) -> float:
        a = float(a)
        if a < 0.0:
            return 0.0
        if a > 1.0:
            return 1.0
        return a

    def _clamp_sample_step(self, r: float) -> float:
        r = float(r)
        if r < 0.1:
            return 0.1
        if r > 4.0:
            return 4.0
        return r

    def _ensure_lo_hi(self, lo: float, hi: float) -> tuple[float, float]:
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f <= lo_f:
            lo_f, hi_f = hi_f, lo_f
        return lo_f, hi_f

    def _clamp_level(self, level: int) -> int:
        ms = self._dims_meta.get('multiscale') if isinstance(self._dims_meta.get('multiscale'), dict) else None
        if isinstance(ms, dict):
            levels = ms.get('levels')
            if isinstance(levels, (list, tuple)) and levels:
                lo, hi = 0, len(levels) - 1
                lv = int(level)
                if lv < lo:
                    return lo
                if lv > hi:
                    return hi
                return lv
        return int(level)

    def _send_intent(self, type_str: str, fields: dict, origin: str) -> bool:
        ch = self._loop_state.state_channel
        if ch is None:
            return False
        payload = {'type': type_str}
        payload.update(fields)
        payload['client_id'] = self._client_id
        payload['client_seq'] = self._next_client_seq()
        payload['origin'] = str(origin)
        ok = ch.post(payload)
        fields_to_log = {k: v for k, v in payload.items() if k not in ('type', 'client_id', 'client_seq', 'origin')}
        logger.info("%s->%s %s sent=%s", origin, type_str, fields_to_log, bool(ok))
        return bool(ok)

    # --- View HUD snapshot (for overlay) ----------------------------------------
    def view_hud_snapshot(self) -> dict:
        """Return a compact snapshot of 3D view/volume tuning state.

        Safe to call from GUI timer; avoids raising on missing fields.
        """
        meta = self._dims_meta
        snap: dict[str, object] = {}
        snap['ndisplay'] = _int_or_none(meta.get('ndisplay'))
        snap['volume'] = _bool_or_none(meta.get('volume'))
        snap['vol_mode'] = bool(self._is_volume_mode())

        render = meta.get('render') if isinstance(meta.get('render'), dict) else None
        if isinstance(render, dict):
            snap['render_mode'] = render.get('mode')
            clim = render.get('clim')
            if isinstance(clim, (list, tuple)):
                snap['clim_lo'] = _float_or_none(clim[0] if len(clim) > 0 else None)
                snap['clim_hi'] = _float_or_none(clim[1] if len(clim) > 1 else None)
            else:
                snap['clim_lo'] = None
                snap['clim_hi'] = None
            snap['colormap'] = render.get('colormap')
            snap['opacity'] = _float_or_none(render.get('opacity'))
            snap['sample_step'] = _float_or_none(render.get('sample_step'))

        ms = meta.get('multiscale') if isinstance(meta.get('multiscale'), dict) else None
        if isinstance(ms, dict):
            snap['ms_policy'] = ms.get('policy')
            current_level = ms.get('current_level')
            level_value = _int_or_none(current_level)
            snap['ms_level'] = level_value
            levels_obj = ms.get('levels')
            if isinstance(levels_obj, (list, tuple)):
                snap['ms_levels'] = len(levels_obj)
                if level_value is not None and 0 <= level_value < len(levels_obj):
                    entry = levels_obj[level_value]
                    snap['ms_path'] = entry.get('path') if isinstance(entry, dict) else None
                else:
                    snap['ms_path'] = None
            else:
                snap['ms_levels'] = None
                snap['ms_path'] = None

        snap['primary_axis'] = _int_or_none(self._primary_axis_index)
        snap['last_zoom_factor'] = self._last_zoom_factor
        snap['last_zoom_widget_px'] = self._last_zoom_widget_px
        snap['last_zoom_video_px'] = self._last_zoom_video_px
        snap['last_zoom_anchor_px'] = self._last_zoom_anchor_px
        snap['last_pan_dx'] = self._last_pan_dx_sent
        snap['last_pan_dy'] = self._last_pan_dy_sent
        snap['video_w'] = _int_or_none(self._vid_w)
        snap['video_h'] = _int_or_none(self._vid_h)
        snap['zoom_base'] = float(self._zoom_base)
        return snap

    def _mirror_dims_to_viewer(
        self,
        vm_ref,
        cur,
        ndisplay,
        ndim,
        dims_range,
        order,
        axis_labels,
        sizes,
        displayed,
    ) -> None:
        """Mirror dims metadata/step into an attached viewer on the GUI thread.

        Keeps exception handling contained and avoids deep nesting at callsite.
        """
        if vm_ref is None or not hasattr(vm_ref, '_apply_remote_dims_update'):
            return
        # Build a callable to execute on the GUI thread (or inline fallback)
        apply_remote = vm_ref._apply_remote_dims_update  # type: ignore[attr-defined]

        def _apply() -> None:
            apply_remote(
                current_step=cur,
                ndisplay=ndisplay,
                ndim=ndim,
                dims_range=dims_range,
                order=order,
                axis_labels=axis_labels,
                sizes=sizes,
                displayed=displayed,
            )
        if self._ui_call is not None:
            self._ui_call.call.emit(_apply)
            return
        _apply()

    def dims_step(self, axis: int | str, delta: int, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        idx = self._axis_to_index(axis)
        if idx is None:
            return False
        # Suppress local inputs on the playing axis while napari is animating it
        vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        if vm_ref is not None:
            is_playing = bool(vm_ref._is_playing) if hasattr(vm_ref, '_is_playing') else False
            play_axis = vm_ref._play_axis if hasattr(vm_ref, '_play_axis') else None
            if is_playing and play_axis is not None and int(play_axis) == int(idx) and origin != 'play':
                return False
        now = time.perf_counter()
        if (now - float(self._last_dims_send or 0.0)) < self._dims_min_dt:
            logger.debug("dims.intent.step gated by rate limiter (%s)", origin)
            return False
        ch = self._loop_state.state_channel
        if ch is None:
            return False
        payload = {
            'type': 'dims.intent.step',
            'axis': int(idx),
            'delta': int(delta),
            'client_id': self._client_id,
            'client_seq': self._next_client_seq(),
            'origin': str(origin),
        }
        seq_val = int(payload['client_seq'])
        ok = ch.post(payload)
        if ok:
            self._record_pending_intent(
                seq_val,
                {
                    'kind': 'step',
                    'axis': int(idx),
                    'origin': str(origin),
                },
            )
        self._last_dims_send = now
        return bool(ok)

    def dims_set_index(self, axis: int | str, value: int, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        idx = self._axis_to_index(axis)
        if idx is None:
            return False
        # Suppress local inputs on the playing axis while napari is animating it
        vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        if vm_ref is not None:
            is_playing = bool(vm_ref._is_playing) if hasattr(vm_ref, '_is_playing') else False
            play_axis = vm_ref._play_axis if hasattr(vm_ref, '_play_axis') else None
            if is_playing and play_axis is not None and int(play_axis) == int(idx) and origin != 'play':
                return False
        now = time.perf_counter()
        if (now - float(self._last_dims_send or 0.0)) < self._dims_min_dt:
            # Allow coalescing on caller side; treat as not sent
            logger.debug("dims.intent.set_index gated by rate limiter (%s)", origin)
            return False
        ch = self._loop_state.state_channel
        if ch is None:
            return False
        payload = {
            'type': 'dims.intent.set_index',
            'axis': int(idx),
            'value': int(value),
            'client_id': self._client_id,
            'client_seq': self._next_client_seq(),
            'origin': str(origin),
        }
        seq_val = int(payload['client_seq'])
        ok = ch.post(payload)
        if ok:
            self._record_pending_intent(
                seq_val,
                {
                    'kind': 'set_index',
                    'axis': int(idx),
                    'value': int(value),
                    'origin': str(origin),
                },
            )
        self._last_dims_send = now
        return bool(ok)

    # --- Volume/multiscale intent senders --------------------------------------
    def _rate_gate_settings(self, origin: str) -> bool:
        now = time.perf_counter()
        if (now - float(self._last_settings_send or 0.0)) < self._settings_min_dt:
            logger.debug("settings intent gated by rate limiter (%s)", origin)
            return True
        self._last_settings_send = now
        return False

    def volume_set_render_mode(self, mode: str, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if not self._is_volume_mode():
            logger.debug("volume_set_render_mode gated: not in volume mode")
            return False
        if self._rate_gate_settings(origin):
            return False
        return self._send_intent('volume.intent.set_render_mode', {'mode': str(mode)}, origin)

    def volume_set_clim(self, lo: float, hi: float, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if not self._is_volume_mode():
            logger.debug("volume_set_clim gated: not in volume mode")
            return False
        if self._rate_gate_settings(origin):
            return False
        lo_f, hi_f = self._ensure_lo_hi(lo, hi)
        return self._send_intent('volume.intent.set_clim', {'lo': float(lo_f), 'hi': float(hi_f)}, origin)

    def volume_set_colormap(self, name: str, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if not self._is_volume_mode():
            logger.debug("volume_set_colormap gated: not in volume mode")
            return False
        if self._rate_gate_settings(origin):
            return False
        return self._send_intent('volume.intent.set_colormap', {'name': str(name)}, origin)

    def volume_set_opacity(self, alpha: float, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if not self._is_volume_mode():
            logger.debug("volume_set_opacity gated: not in volume mode")
            return False
        if self._rate_gate_settings(origin):
            return False
        a = self._clamp01(alpha)
        return self._send_intent('volume.intent.set_opacity', {'alpha': float(a)}, origin)

    def volume_set_sample_step(self, relative: float, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if not self._is_volume_mode():
            logger.debug("volume_set_sample_step gated: not in volume mode")
            return False
        if self._rate_gate_settings(origin):
            return False
        r = self._clamp_sample_step(relative)
        return self._send_intent('volume.intent.set_sample_step', {'relative': float(r)}, origin)

    def multiscale_set_policy(self, policy: str, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if self._rate_gate_settings(origin):
            return False
        pol = str(policy).lower().strip()
        if pol not in {'oversampling', 'thresholds', 'ratio'}:
            logger.debug("multiscale_set_policy rejected: policy=%s", pol)
            return False
        return self._send_intent('multiscale.intent.set_policy', {'policy': pol}, origin)

    def multiscale_set_level(self, level: int, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if self._rate_gate_settings(origin):
            return False
        lv = self._clamp_level(level)
        return self._send_intent('multiscale.intent.set_level', {'level': int(lv)}, origin)

    def view_set_ndisplay(self, ndisplay: int, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        if self._rate_gate_settings(origin):
            return False
        nd_value = int(ndisplay)
        nd_target = 3 if nd_value >= 3 else 2
        cur = self._dims_meta.get('ndisplay')
        if cur is not None and int(cur) == nd_target:
            return True
        return self._send_intent('view.intent.set_ndisplay', {'ndisplay': nd_target}, origin)

    def current_ndisplay(self) -> Optional[int]:
        """Return the last known ndisplay value from dims metadata."""

        try:
            return _int_or_none(self._dims_meta.get('ndisplay'))
        except Exception:
            logger.debug('current_ndisplay lookup failed', exc_info=True)
            return None

    def toggle_ndisplay(self, *, origin: str = 'ui') -> bool:
        """Toggle between 2D and 3D display modes if dims metadata is ready."""

        if not self._dims_ready:
            return False
        current = self.current_ndisplay()
        target = 2 if current == 3 else 3
        return self.view_set_ndisplay(target, origin=origin)

    def _on_wheel_for_zoom(self, data: dict) -> None:
        ch = self._loop_state.state_channel
        if ch is None:
            return
        ay = float(data.get('angle_y') or 0.0)
        py = float(data.get('pixel_y') or 0.0)
        xw = float(data.get('x_px') or 0.0)
        yw = float(data.get('y_px') or 0.0)
        base = float(self._zoom_base)
        if ay != 0.0:
            s = 1.0 if ay > 0 else -1.0
            factor = base ** s
        elif py != 0.0:
            factor = base ** (py / 30.0)
        else:
            return
        xv, yv = self._widget_to_video(xw, yw)
        ax, ay = self._server_anchor_from_video(xv, yv)
        # Stash for HUD
        self._last_zoom_factor = float(factor)
        self._last_zoom_widget_px = (float(xw), float(yw))
        self._last_zoom_video_px = (float(xv), float(yv))
        self._last_zoom_anchor_px = (float(ax), float(ay))
        ok = ch.post({'type': 'camera.zoom_at', 'factor': float(factor), 'anchor_px': [float(ax), float(ay)]})
        if self._log_dims_info:
            logger.info("wheel+mod->camera.zoom_at f=%.4f at(%.1f,%.1f) sent=%s", float(factor), float(ax), float(ay), bool(ok))

    def _on_pointer(self, data: dict) -> None:
        phase = (data.get('phase') or '').lower()
        xw_raw = data.get('x_px')
        yw_raw = data.get('y_px')
        xw = float(xw_raw) if xw_raw is not None else 0.0
        yw = float(yw_raw) if yw_raw is not None else 0.0
        # Track latest cursor position regardless of drag
        self._cursor_wx = xw
        self._cursor_wy = yw
        # Detect Alt modifier (for orbit)
        mods = int(data.get('mods') or 0)
        alt_mask = int(QtCore.Qt.AltModifier)
        alt = (mods & alt_mask) != 0
        in_vol3d = self._is_volume_mode() and int(self._dims_meta.get('ndisplay') or 2) == 3

        if phase == 'down':
            self._dragging = True
            self._last_wx = xw
            self._last_wy = yw
            self._pan_dx_accum = 0.0
            self._pan_dy_accum = 0.0
            # Begin orbit drag when Alt held in 3D volume mode
            if alt and in_vol3d:
                self._orbit_dragging = True
                self._orbit_daz_accum = 0.0
                self._orbit_del_accum = 0.0
            return
        if phase == 'move' and self._dragging:
            # Accumulate pan in video px, then convert to canvas delta
            xv0, yv0 = self._widget_to_video(self._last_wx, self._last_wy)
            xv1, yv1 = self._widget_to_video(xw, yw)
            dx_v = (xv1 - xv0)
            dy_v = (yv1 - yv0)
            dx_c, dy_c = self._video_delta_to_canvas(dx_v, dy_v)
            if self._log_dims_info:
                logger.info(
                    "pointer move: mods=%d alt=%s vol3d=%s dx_c=%.2f dy_c=%.2f",
                    int(mods), bool(alt), bool(in_vol3d), float(dx_c), float(dy_c)
                )
            # If Alt held in 3D volume mode: orbit; suppress pan
            if alt and in_vol3d:
                # If orbit just started mid-drag (Alt pressed), reset accumulators
                if not self._orbit_dragging:
                    self._orbit_dragging = True
                    self._orbit_daz_accum = 0.0
                    self._orbit_del_accum = 0.0
                self._orbit_daz_accum += float(dx_c) * float(self._orbit_deg_per_px_x)
                self._orbit_del_accum += float(-dy_c) * float(self._orbit_deg_per_px_y)
                # Do not accumulate pan while orbiting
            else:
                # If we were orbiting but Alt released, flush residual orbit and stop orbit mode
                if self._orbit_dragging:
                    self._flush_orbit(force=True)
                    self._orbit_dragging = False
                self._pan_dx_accum += dx_c
                self._pan_dy_accum += dy_c
            self._last_wx = xw
            self._last_wy = yw
            # Flush orbit or pan at camera cadence
            if self._orbit_dragging:
                self._flush_orbit_if_due()
            else:
                self._flush_pan_if_due()
            return
        if phase == 'up':
            self._dragging = False
            # Flush any residual orbit first if active; otherwise flush pan
            if self._orbit_dragging:
                self._flush_orbit(force=True)
                self._orbit_dragging = False
            else:
                # Flush any residual pan
                self._flush_pan(force=True)

    def _flush_pan_if_due(self) -> None:
        now = time.perf_counter()
        if (now - float(self._last_cam_send or 0.0)) >= self._cam_min_dt:
            self._flush_pan()

    def _flush_pan(self, force: bool = False) -> None:
        dx = float(self._pan_dx_accum or 0.0)
        dy = float(self._pan_dy_accum or 0.0)
        if not force and abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return
        ch = self._loop_state.state_channel
        if ch is None:
            self._pan_dx_accum = 0.0
            self._pan_dy_accum = 0.0
            return
        ok = ch.post({'type': 'camera.pan_px', 'dx_px': float(dx), 'dy_px': float(dy)})
        # Stash for HUD
        self._last_pan_dx_sent = float(dx)
        self._last_pan_dy_sent = float(dy)
        self._last_cam_send = time.perf_counter()
        self._pan_dx_accum = 0.0
        self._pan_dy_accum = 0.0
        if self._log_dims_info:
            logger.info("drag->camera.pan_px dx=%.1f dy=%.1f sent=%s", float(dx), float(dy), bool(ok))

    def _flush_orbit_if_due(self) -> None:
        now = time.perf_counter()
        if (now - float(self._last_cam_send or 0.0)) >= self._cam_min_dt:
            self._flush_orbit()

    def _flush_orbit(self, force: bool = False) -> None:
        # Coalesced orbit send using same cadence as pan
        daz = float(self._orbit_daz_accum or 0.0)
        delv = float(self._orbit_del_accum or 0.0)
        if not force and abs(daz) < 1e-2 and abs(delv) < 1e-2:
            return
        ch = self._loop_state.state_channel
        if ch is None:
            self._orbit_daz_accum = 0.0
            self._orbit_del_accum = 0.0
            return
        ok = ch.post({'type': 'camera.orbit', 'd_az_deg': float(daz), 'd_el_deg': float(delv)})
        self._last_cam_send = time.perf_counter()
        self._orbit_daz_accum = 0.0
        self._orbit_del_accum = 0.0
        if self._log_dims_info:
            logger.info(
                "alt-drag->camera.orbit daz=%.2f del=%.2f sent=%s",
                float(daz), float(delv), bool(ok)
            )

    # (no key→dims mapping)

    def _on_frame(self, pkt: Packet) -> None:
        cur = int(pkt.seq)
        if not (self._vt_wait_keyframe or self._pyav_wait_keyframe):
            if self._sync.update_and_check(cur):
                if self._vt_decoder is not None:
                    self._vt_wait_keyframe = True
                    self._vt_pipeline.clear(preserve_cache=True)
                    self._presenter.clear(Source.VT)
                    self._request_keyframe_once()
                self._pyav_wait_keyframe = True
                self._pyav_pipeline.clear()
                self._presenter.clear(Source.PYAV)
                self._init_decoder()
                dec = self.decoder.decode if self.decoder else None
                self._pyav_pipeline.set_decoder(dec)
                self._disco_gated = True
        if not self._stream_seen_keyframe:
            if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                self._stream_seen_keyframe = True
            else:
                return
        if self._vt_decoder is not None and self._vt_wait_keyframe:
            if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                self._vt_wait_keyframe = False
                self._vt_gate_lift_time = time.time()
                self._mono_at_gate = time.perf_counter()
                # Compute wall->mono offset once at gate lift
                self._wall_to_mono = float(self._mono_at_gate - float(self._vt_gate_lift_time) - float(self._server_bias_s))
                self._vt_ts_offset = float(self._vt_gate_lift_time - float(pkt.ts)) - float(self._server_bias_s)
                self._source_mux.set_active(Source.VT)
                # Use wall->mono offset for monotonic scheduling
                self._presenter.set_offset(self._wall_to_mono)
                self._presenter.set_latency(self._vt_latency_s)
                self._presenter.clear(Source.PYAV)
                logger.info("VT gate lifted on keyframe (seq=%d); presenter=VT", cur)
                self._disco_gated = False
                if self._warmup_window_s > 0:
                    if self._warmup_ms_override is not None:
                        extra_ms = max(0.0, float(self._warmup_ms_override))
                    else:
                        frame_ms = 1000.0 / (self._fps if (self._fps and self._fps > 0) else 60.0)
                        target_ms = frame_ms + float(self._warmup_margin_ms)
                        base_ms = float(self._vt_latency_s) * 1000.0
                        extra_ms = max(0.0, min(float(self._warmup_max_ms), target_ms - base_ms))
                    extra_s = extra_ms / 1000.0
                    if extra_s > 0.0:
                        # Timer-less warmup: set extra latency and let draw() ramp it down
                        self._presenter.set_latency(self._vt_latency_s + extra_s)
                        self._warmup_extra_active_s = extra_s
                        self._warmup_until = time.perf_counter() + float(self._warmup_window_s)
                # Schedule first wake after VT becomes active (thread-safe)
                self.schedule_next_wake_threadsafe()
            else:
                self._request_keyframe_once()
                return
        if self._vt_decoder is not None and not self._vt_wait_keyframe:
            ts_float = float(pkt.ts)
            b = pkt.payload
            self._vt_pipeline.enqueue(b, ts_float)
            self._vt_enqueued += 1
        else:
            ts_val = pkt.ts
            ts_float = float(ts_val) if ts_val is not None else None
            b = pkt.payload
            if self._pyav_pipeline.qsize() >= max(2, self._pyav_backlog_trigger - 1):
                self._pyav_wait_keyframe = True
                self._pyav_pipeline.clear()
                self._presenter.clear(Source.PYAV)
                self._init_decoder()
                dec = self.decoder.decode if self.decoder else None
                self._pyav_pipeline.set_decoder(dec)
            if self._pyav_wait_keyframe:
                if not (self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01)):
                    return
                self._pyav_wait_keyframe = False
                self._init_decoder()
                self._disco_gated = False
            self._pyav_pipeline.enqueue(b, ts_float)
            self._pyav_enqueued += 1

    def _apply_warmup(self, now: float) -> None:
        if self._warmup_until > 0:
            if now >= self._warmup_until:
                self._presenter.set_latency(self._vt_latency_s)
                self._warmup_until = 0.0
            else:
                remain = max(0.0, self._warmup_until - now)
                frac = remain / max(1e-6, self._warmup_window_s)
                cur = self._vt_latency_s + self._warmup_extra_active_s * frac
                self._presenter.set_latency(cur)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self._stats_timer is not None:
            self._stats_timer.stop()
            self._stats_timer = None
        if self._metrics_timer is not None:
            self._metrics_timer.stop()
            self._metrics_timer = None
        if self._warmup_reset_timer is not None:
            self._warmup_reset_timer.stop()
            self._warmup_reset_timer = None
        if self._watchdog_timer is not None:
            self._watchdog_timer.stop()
            self._watchdog_timer = None
        if self._evloop_mon is not None:
            self._evloop_mon.stop()
        self._evloop_mon = None

        self._presenter.clear()

        if self._smoke is not None:
            try:
                self._smoke.stop()
            except Exception:
                logger.debug("ClientStreamLoop.stop: smoke stop failed", exc_info=True)
            self._smoke = None

        while True:
            try:
                frame = self._frame_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(frame, tuple) and len(frame) == 2:
                payload, release_cb = frame  # type: ignore[assignment]
                if release_cb is not None:
                    release_cb(payload)  # type: ignore[misc]

        lease = self._renderer_fallbacks.pop_vt_cache()
        if lease is not None:
            lease.close()
        self._renderer_fallbacks.clear_pyav()

        self._vt_pipeline.stop()
        self._pyav_pipeline.stop()

        if self._vt_decoder is not None:
            self._vt_decoder.flush()
            close_decoder = getattr(self._vt_decoder, 'close', None)
            if callable(close_decoder):
                close_decoder()
            self._vt_decoder = None

    # VT decode/submit is handled by VTPipeline

    # Keyframe detection via shared helpers (AnnexB/AVCC)
    def _is_keyframe(self, payload: bytes | memoryview, codec: int) -> bool:
        hevc = int(codec) == 2
        if is_annexb(payload):
            return contains_idr_annexb(payload, hevc=hevc)
        # Use parsed nal_length_size when available (defaults to 4)
        nsz = int(self._nal_length_size or 4)
        return contains_idr_avcc(payload, nal_len_size=nsz, hevc=hevc)

# Coordinator end
