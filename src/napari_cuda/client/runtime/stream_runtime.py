from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from threading import Thread
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TYPE_CHECKING
from concurrent.futures import Future
import weakref
from contextlib import ExitStack

import numpy as np
from qtpy import QtCore

from napari_cuda.client.rendering.presenter import FixedLatencyPresenter, SourceMux
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.client.runtime.receiver import PixelReceiver, Packet
from napari_cuda.client.rendering.types import Source, SubmittedFrame
from napari_cuda.client.rendering.renderer import GLRenderer
from napari_cuda.client.rendering.decoders.pyav import PyAVDecoder
from napari_cuda.client.rendering.decoders.vt import VTLiveDecoder
from napari_cuda.client.runtime.input import InputSender
from napari_cuda.codec.avcc import (
    annexb_to_avcc,
    is_annexb,
    split_annexb,
    split_avcc_by_len,
    build_avcc,
    find_sps_pps,
)
from napari_cuda.codec.h264 import contains_idr_annexb, contains_idr_avcc
from napari_cuda.client.runtime.client_loop.scheduler import CallProxy, WakeProxy
from napari_cuda.client.runtime.client_loop.pipelines import (
    build_pyav_pipeline,
    build_vt_pipeline,
)
from napari_cuda.client.runtime.client_loop.renderer_fallbacks import RendererFallbacks
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.runtime.client_loop import warmup, camera, loop_lifecycle
from napari_cuda.client.control import state_update_actions as control_actions
from napari_cuda.client.control.emitters import NapariDimsIntentEmitter
from napari_cuda.client.control.mirrors import napari_dims_mirror
from napari_cuda.client.runtime.client_loop.scheduler_helpers import (
    init_wake_scheduler,
)
from napari_cuda.client.runtime.client_loop.telemetry import (
    build_telemetry_config,
    create_metrics,
)
from napari_cuda.client.runtime.client_loop.client_loop_config import load_client_loop_config
from napari_cuda.client.data import RemoteLayerRegistry, RegistrySnapshot
from napari_cuda.protocol.messages import (
    NotifyDimsFrame,
    NotifyLayersFrame,
    NotifySceneFrame,
    NotifySceneLevelPayload,
    NotifyStreamFrame,
)
from napari_cuda.protocol import AckState, build_call_command, build_state_update
from napari_cuda.protocol.snapshots import (
    layer_delta_from_payload,
    scene_snapshot_from_payload,
)
from napari_cuda.client.state import LayerStateBridge
from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.control_channel_client import HeartbeatAckError

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.client.control.control_channel_client import StateChannel, SessionMetadata
    from napari_cuda.client.control.client_state_ledger import IntentRecord, AckReconciliation
    from napari_cuda.client.runtime.config import ClientConfig
    from napari_cuda.protocol import ReplyCommand, ErrorCommand

logger = logging.getLogger(__name__)

_KEYFRAME_COMMAND = "napari.pixel.request_keyframe"


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



class CommandError(RuntimeError):
    """Raised when a command lane invocation returns an error frame."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details) if details else None
        self.idempotency_key = idempotency_key




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
        vt_backlog_trigger: int = 32,
        pyav_backlog_trigger: int = 16,
        client_cfg: 'ClientConfig | None' = None,
        *,
        on_first_dims_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        self._scene_canvas = scene_canvas
        self._canvas_native = scene_canvas.native if hasattr(scene_canvas, 'native') else None
        _maybe_enable_debug_logger()
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
        self._loop_state = ClientLoopState()
        self.server_host = server_host
        self.server_port = int(server_port)
        self.state_port = int(state_port)
        self._stream_format = stream_format
        self._stream_format_set = False
        self._command_lock = threading.Lock()
        self._pending_commands: Dict[str, Future] = {}
        self._command_catalog: tuple[str, ...] = ()
        self._log_dims_info: bool = False
        self._slider_tx_interval_ms = int(getattr(self._env_cfg, 'slider_tx_ms', 0))
        backoff_env = os.getenv('NAPARI_CUDA_KEYFRAME_BACKOFF_S', '1.0') or '1.0'
        try:
            self._keyframe_backoff_s = max(0.0, float(backoff_env))
        except ValueError:
            self._keyframe_backoff_s = 1.0
        self._last_keyframe_request_ts = 0.0
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
        self._loop_state.presenter = self._presenter
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
        self._latest_scene_frame: Optional[NotifySceneFrame] = None
        self._control_state = control_actions.ControlStateContext.from_env(self._env_cfg)
        self._loop_state.control_state = self._control_state
        self._state_ledger = ClientStateLedger(clock=time.time)
        self._layer_registry = RemoteLayerRegistry()
        self._layer_registry.add_listener(self._on_registry_snapshot)
        self._dims_mirror: napari_dims_mirror.NapariDimsMirror | None = None
        self._dims_emitter: NapariDimsIntentEmitter | None = None
        # Keep-last-frame fallback default enabled for smoother presentation
        self._keep_last_frame_fallback = True
        self._state_session_metadata: "SessionMetadata | None" = None
        # Monotonic scheduling marker for next due
        # Startup warmup policy: temporarily boost VT latency and ramp down
        self._warmup_policy = warmup.WarmupPolicy(
            ms_override=(self._env_cfg.warmup_ms_override if self._env_cfg.warmup_ms_override is not None else None),
            window_s=float(self._env_cfg.warmup_window_s or 0.0),
            margin_ms=float(self._env_cfg.warmup_margin_ms or 0.0),
            max_ms=float(self._env_cfg.warmup_max_ms or 0.0),
        )
        self._loop_state.warmup_policy = self._warmup_policy
        self._fps: Optional[float] = None

        # Renderer
        self._renderer = GLRenderer(self._scene_canvas)
        self._presenter_facade = PresenterFacade()

        self._layer_bridge = LayerStateBridge(
            self,
            self._presenter_facade,
            self._layer_registry,
            control_state=self._control_state,
            loop_state=self._loop_state,
            state_ledger=self._state_ledger,
        )

        # Decoders
        self.decoder: Optional[PyAVDecoder] = None
        self._vt_decoder: Optional[VTLiveDecoder] = None
        self._vt_backend: Optional[str] = None
        self._vt_cfg_key = None
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
        self._loop_state.metrics = create_metrics(self._telemetry_cfg)

        # Presenter-owned wake scheduling helper
        self._use_display_loop = self._env_cfg.use_display_loop
        self._schedule_next_wake = init_wake_scheduler(self)

        self._presenter_facade.start_presenting(
            scene_canvas=self._scene_canvas,
            loop=self,
            presenter=self._presenter,
            renderer=self._renderer,
            client_cfg=self._client_cfg,
            use_display_loop=self._use_display_loop,
        )

        # Build pipelines after scheduler hooks are ready
        # Create a wake proxy so pipelines can nudge scheduling from any thread
        if not self._use_display_loop and self._loop_state.wake_proxy is not None:
            wake_cb = self._loop_state.wake_proxy.trigger.emit
        else:
            wake_cb = (lambda: None)
        self._loop_state.vt_pipeline = build_vt_pipeline(self, schedule_next_wake=wake_cb, logger=logger)
        self._pyav_backlog_trigger = int(pyav_backlog_trigger)
        self._pyav_enqueued = 0
        self._loop_state.pyav_pipeline = build_pyav_pipeline(self, schedule_next_wake=wake_cb)

        # Renderer fallback manager (VT cache + PyAV reuse)
        self._loop_state.fallbacks = RendererFallbacks()

        # Frame queue for renderer (latest-wins)
        # Holds either numpy arrays or (capsule, release_cb) tuples for VT
        self._loop_state.frame_queue = queue.Queue(maxsize=3)

        # Receiver/state flags
        self._stream_seen_keyframe = False
        self._keyframe_skip_log_ts: float = 0.0

        # Stats/logging and diagnostics
        self._stats_level = self._telemetry_cfg.stats_level
        self._last_stats_time: float = 0.0
        self._relearn_logged: bool = False
        self._last_relearn_log_ts: float = 0.0
        # Debounce duplicate notify.stream payloads
        self._last_vcfg_key = None
        # Stream continuity and gate tracking
        # Draw pacing diagnostics
        self._in_draw: bool = False
        # Draw watchdog
        self._watchdog_ms = self._env_cfg.watchdog_ms
        # Event loop stall monitor (disabled by default)
        self._evloop_stall_ms = self._env_cfg.evloop_stall_ms
        self._evloop_sample_ms = self._env_cfg.evloop_sample_ms
        self._quit_hook_connected = False
        self._stopped = False
        # Video dimensions (from notify.stream)
        self._vid_w: Optional[int] = None
        self._vid_h: Optional[int] = None
        # View HUD diagnostics (for tuning 3D volume controls)
        self._camera_state = camera.CameraState.from_env(self._env_cfg)
        self._loop_state.camera = self._camera_state
        # Camera ops coalescing
        cam_rate = self._env_cfg.camera_rate_hz
        self._camera_state.cam_min_dt = 1.0 / max(1.0, cam_rate)
        self._camera_state.orbit_deg_per_px_x = float(self._env_cfg.orbit_deg_per_px_x)
        self._camera_state.orbit_deg_per_px_y = float(self._env_cfg.orbit_deg_per_px_y)
        self._camera_state.zoom_base = float(self._env_cfg.zoom_base)
    # --- Thread-safe scheduling helpers --------------------------------------
    def _on_gui_thread(self) -> bool:
        gui_thr = self._loop_state.gui_thread
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

        if self._loop_state.wake_proxy is not None:
            self._loop_state.wake_proxy.trigger.emit()
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
            self._loop_state.in_present = True
            stack.callback(self._present_reset)
            # Clear pending marker to allow next wake to be scheduled freely
            self._loop_state.next_due_pending_until = 0.0
            self._scene_canvas_update()

    def _present_reset(self) -> None:
        self._loop_state.in_present = False

    def start(self) -> None:
        self._initialize_mirrors_and_emitters()
        loop_lifecycle.start_loop(self)

    def _enqueue_frame(self, frame: object) -> None:
        if self._loop_state.frame_queue.full():
            self._loop_state.frame_queue.get_nowait()
        self._loop_state.frame_queue.put(frame)

    def draw(self) -> None:
        # Guard to avoid scheduling immediate repaints from within draw
        in_draw_prev = self._in_draw
        self._in_draw = True
        # Draw-loop pacing metric (perf_counter for monotonic interval)
        now_pc = time.perf_counter()
        last_pc = float(self._loop_state.last_draw_pc or 0.0)
        if last_pc > 0.0 and self._loop_state.metrics is not None:
            self._loop_state.metrics.observe_ms('napari_cuda_client_draw_interval_ms', (now_pc - last_pc) * 1000.0)
        self._loop_state.last_draw_pc = now_pc
        # Apply warmup ramp/restore on GUI thread (timer-less)
        if self._warmup_policy is not None:
            warmup.apply_ramp(
                self._warmup_policy,
                self._loop_state,
                self._presenter,
                self._vt_latency_s,
                now_pc,
            )
        # VT output is drained continuously by worker; draw focuses on presenting
        # If no offset learned yet (e.g., during early startup), derive from buffer samples.
        clock_offset = self._presenter.clock.offset
        if clock_offset is None:
            learned = self._presenter.relearn_offset(Source.VT)
            if learned is not None and math.isfinite(learned):
                logger.info("Presenter offset learned from buffer: %.3fs", float(learned))

        ready = self._presenter.pop_due(time.perf_counter(), self._source_mux.active)
        if ready is not None:
            src_val = ready.source.value
            # Record presentation lateness relative to due time for non-preview frames
            if not ready.preview and self._loop_state.metrics is not None and self._loop_state.metrics.enabled:
                late_ms = (time.perf_counter() - float(ready.due_ts)) * 1000.0
                self._loop_state.metrics.observe_ms('napari_cuda_client_present_lateness_ms', float(late_ms))
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
                self._loop_state.fallbacks.store_pyav_frame(ready.payload)
                self._enqueue_frame(ready.payload)

        frame = None
        if ready is None:
            if self._source_mux.active == Source.VT:
                self._loop_state.fallbacks.try_enqueue_cached_vt(self._enqueue_frame)
            elif self._source_mux.active == Source.PYAV:
                self._loop_state.fallbacks.try_enqueue_pyav(self._enqueue_frame)
        # Schedule next wake based on earliest due; avoids relying on a 60 Hz loop
        # Safe: draw() runs on GUI thread
        self._schedule_next_wake()
        while not self._loop_state.frame_queue.empty():
            frame = self._loop_state.frame_queue.get_nowait()
        # Time the render operation based on frame type
        if frame is not None:
            is_vt = isinstance(frame, tuple) and len(frame) == 2
            t_render0 = time.perf_counter()
            self._renderer.draw(frame)
            t_render1 = time.perf_counter()
            if self._loop_state.metrics is not None and self._loop_state.metrics.enabled:
                metric_name = 'napari_cuda_client_render_vt_ms' if is_vt else 'napari_cuda_client_render_pyav_ms'
                self._loop_state.metrics.observe_ms(metric_name, (t_render1 - t_render0) * 1000.0)
            # Count a presented frame for client-side FPS derivation (only when a frame was actually drawn)
            if self._loop_state.metrics is not None:
                self._loop_state.metrics.inc('napari_cuda_client_presented_total', 1.0)
            now = time.perf_counter()
            last = self._loop_state.last_present_mono
            if last:
                # Only log PRESENT timing when explicitly enabled to avoid spam
                import os as _os
                if (_os.getenv('NAPARI_CUDA_PRESENT_DEBUG') or '').lower() in ('1','true','yes','on','dbg','debug'):
                    inter_ms = (now - float(last)) * 1000.0
                    logger.debug("PRESENT inter_ms=%.3f", float(inter_ms))
            self._loop_state.last_present_mono = now
        else:
            self._renderer.draw(frame)
        # Clear draw guard
        self._in_draw = in_draw_prev

    def _log_stats(self) -> None:
        if self._stats_level is None:
            return
        pres_stats = self._presenter.stats()
        vt_counts = self._loop_state.vt_pipeline.counts()
        if self._loop_state.metrics is not None:
            self._loop_state.metrics.set('napari_cuda_client_present_buf_vt', float(pres_stats['buf'].get('vt', 0)))
            self._loop_state.metrics.set('napari_cuda_client_present_buf_pyav', float(pres_stats['buf'].get('pyav', 0)))
            self._loop_state.metrics.set('napari_cuda_client_present_latency_ms', float(pres_stats.get('latency_ms', 0.0)))
        logger.log(
            self._stats_level,
            "presenter=%s vt_counts=%s keyframes_seen=%d",
            pres_stats,
            vt_counts,
            self._loop_state.sync.keyframes_seen,
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
            logger.debug("Duplicate notify.stream payload ignored (%dx%d)", width, height)
            return
        if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
            logger.debug("VT already initialized; ignoring duplicate notify.stream payload")
            return
        # Respect NAPARI_CUDA_VT_BACKEND env: off/0/false disables VT
        backend_env = (os.getenv('NAPARI_CUDA_VT_BACKEND', 'shim') or 'shim').lower()
        vt_disabled = backend_env in ("off", "0", "false", "no")
        if sys.platform == 'darwin' and not vt_disabled:
            try:
                self._vt_decoder = VTLiveDecoder(avcc, width, height)
                self._vt_backend = 'shim'
                self._loop_state.vt_pipeline.set_decoder(self._vt_decoder)
            except Exception as exc:
                logger.warning("VT shim unavailable: %s; falling back to PyAV", exc)
                self._vt_decoder = None
        else:
            self._vt_decoder = None
        self._vt_cfg_key = cfg_key
        self._last_vcfg_key = cfg_key
        self._loop_state.vt_wait_keyframe = True
        self._loop_state.sync.reset_sequence()
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
        self._loop_state.vt_pipeline.update_nal_length_size(self._nal_length_size)
        logger.info("VideoToolbox live decoder initialized: %dx%d", width, height)

    # Extracted helpers
    def _handle_notify_stream(self, frame: NotifyStreamFrame) -> None:
        payload = frame.payload
        fps = float(payload.fps)
        if fps > 0:
            self._fps = fps
        fmt = str(payload.format)
        self._stream_format = fmt
        self._stream_format_set = True
        self._nal_length_size = int(payload.nal_length_size)
        width, height = payload.frame_size
        if width > 0 and height > 0:
            self._vid_w = int(width)
            self._vid_h = int(height)
        avcc_b64 = payload.avcc
        if width > 0 and height > 0 and avcc_b64:
            # Cache video dimensions for input mapping and prime VT decode
            self._init_vt_from_avcc(avcc_b64, int(width), int(height))

    def _handle_notify_camera(self, frame: Any) -> None:
        result = control_actions.handle_notify_camera(
            self._control_state,
            self._state_ledger,
            frame,
            log_debug=logger.isEnabledFor(logging.DEBUG),
        )
        if result is None:
            return
        mode, delta = result
        try:
            self._presenter_facade.apply_camera_update(mode=mode, delta=delta)
        except Exception:
            logger.debug('apply_camera_update failed', exc_info=True)

    def _log_keyframe_skip(self, reason: str) -> None:
        now = time.time()
        if (now - self._keyframe_skip_log_ts) >= 1.0:
            logger.info("Keyframe request skipped (%s); awaiting server keyframe", reason)
            self._keyframe_skip_log_ts = now

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

    def _replay_last_dims_payload(self) -> None:
        mirror = self._dims_mirror
        if mirror is None:
            return
        mirror.replay_last_payload()

    def _handle_scene_snapshot(self, frame: NotifySceneFrame) -> None:
        """Cache latest notify.scene frame and forward to registry."""
        with self._scene_lock:
            self._latest_scene_frame = frame
        snapshot = scene_snapshot_from_payload(frame.payload)
        with self._layer_bridge.remote_sync():
            self._layer_registry.apply_snapshot(snapshot)
            for layer_snapshot in snapshot.layers:
                if isinstance(layer_snapshot.block, Mapping):
                    self._layer_bridge.seed_snapshot_block(layer_snapshot.layer_id, layer_snapshot.block)
        logger.debug(
            "notify.scene received: layers=%d policies=%s",
            len(snapshot.layers),
            tuple(snapshot.policies.keys()) if snapshot.policies else (),
        )

    def _handle_scene_policies(self, policies: Mapping[str, object]) -> None:
        try:
            control_actions.apply_scene_policies(self._control_state, policies)
        except Exception:
            logger.debug("apply_scene_policies failed", exc_info=True)
            return
        if logger.isEnabledFor(logging.DEBUG):
            multiscale = policies.get('multiscale') if isinstance(policies, Mapping) else None
            if multiscale is not None:
                logger.debug("scene policies updated: multiscale keys=%s", list(multiscale.keys()) if isinstance(multiscale, Mapping) else type(multiscale))

    def _handle_scene_level(self, payload: NotifySceneLevelPayload) -> None:
        multiscale: Dict[str, Any] = {
            'current_level': int(payload.current_level),
            'active_level': int(payload.current_level),
        }
        if payload.downgraded is not None:
            multiscale['downgraded'] = bool(payload.downgraded)
        if payload.levels:
            multiscale['levels'] = [dict(entry) for entry in payload.levels]
        try:
            control_actions.apply_scene_policies(
                self._control_state,
                {'multiscale': multiscale},
            )
        except Exception:
            logger.debug("apply_scene_policies (scene level) failed", exc_info=True)
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "scene level update: level=%s downgraded=%s",
                multiscale['current_level'],
                multiscale.get('downgraded'),
            )

    def _handle_layer_delta(self, frame: NotifyLayersFrame) -> None:
        delta = layer_delta_from_payload(frame.payload)
        with self._layer_bridge.remote_sync():
            self._layer_registry.apply_delta(delta)
        self._layer_bridge.seed_remote_values(delta.layer_id, delta.changes)
        logger.debug(
            "notify.layers: id=%s keys=%s",
            delta.layer_id,
            tuple(delta.changes.keys()),
        )
        self._presenter_facade.apply_layer_delta(delta)

    def request_keyframe(self, origin: str = "ui") -> Future | None:
        """Expose a best-effort keyframe command for external callers."""

        return self._issue_command(_KEYFRAME_COMMAND, origin=origin)

    def _handle_session_ready(self, metadata: "SessionMetadata") -> None:
        self._state_session_metadata = metadata
        self._control_state.session_id = metadata.session_id
        self._control_state.ack_timeout_ms = metadata.ack_timeout_ms
        self._control_state.intent_counter = 0
        self._loop_state.state_session_metadata = metadata
        toggle = metadata.features.get("call.command")
        if toggle and getattr(toggle, "enabled", False):
            commands = tuple(toggle.commands or ())
        else:
            commands = ()
        self._command_catalog = commands
        logger.info(
            "State session ready: session_id=%s ack_timeout_ms=%s",
            metadata.session_id,
            metadata.ack_timeout_ms,
        )
        if commands:
            logger.info("State session command catalogue: %s", ", ".join(commands))

    def _handle_ack_state(self, frame: AckState) -> None:
        outcome = self._state_ledger.apply_ack(frame)
        logger.debug(
            "ack.state outcome: status=%s intent=%s in_reply_to=%s pending=%d was_pending=%s",
            outcome.status,
            outcome.intent_id,
            outcome.in_reply_to,
            outcome.pending_len,
            outcome.was_pending,
        )
        scope = outcome.scope
        if scope == "layer":
            self._layer_bridge.handle_ack(outcome)
            return
        if scope == "dims":
            control_actions.handle_dims_ack(
                self._control_state,
                self._loop_state,
                outcome,
                presenter=self._presenter_facade,
                viewer_ref=self._viewer_mirror,
                ui_call=self._ui_call,
                log_dims_info=self._log_dims_info,
            )
            return
        control_actions.handle_generic_ack(self._control_state, self._loop_state, outcome)

    def _handle_reply_command(self, frame: "ReplyCommand") -> None:
        payload = frame.payload
        logger.info(
            "reply.command received in_reply_to=%s result=%s",
            payload.in_reply_to,
            payload.result,
        )
        future = self._pop_command_future(payload.in_reply_to)
        if future is not None and not future.done():
            future.set_result(payload)
        elif future is None:
            logger.debug(
                "reply.command without pending future: in_reply_to=%s",
                payload.in_reply_to,
            )

    def _handle_error_command(self, frame: "ErrorCommand") -> None:
        payload = frame.payload
        logger.warning(
            "error.command received in_reply_to=%s code=%s message=%s",
            payload.in_reply_to,
            payload.code,
            payload.message,
        )
        future = self._pop_command_future(payload.in_reply_to)
        if future is not None and not future.done():
            future.set_exception(
                CommandError(
                    code=payload.code,
                    message=payload.message,
                    details=payload.details,
                    idempotency_key=payload.idempotency_key,
                )
            )
        elif future is None:
            logger.debug(
                "error.command without pending future: in_reply_to=%s",
                payload.in_reply_to,
            )

    def _pop_command_future(self, frame_id: str | None) -> Future | None:
        if not frame_id:
            return None
        with self._command_lock:
            return self._pending_commands.pop(frame_id, None)

    def _abort_pending_commands(self, *, code: str, message: str) -> None:
        with self._command_lock:
            pending = list(self._pending_commands.values())
            self._pending_commands.clear()
        for future in pending:
            if future.done():
                continue
            future.set_exception(CommandError(code=code, message=message))

    def _issue_command(
        self,
        command: str,
        *,
        args: Optional[Sequence[object]] = None,
        kwargs: Optional[Mapping[str, object]] = None,
        origin: str = "ui",
    ) -> Future | None:
        catalog = self._command_catalog
        if catalog and command not in catalog:
            logger.debug("Command %s not advertised; skipping", command)
            return None
        state_channel = self._loop_state.state_channel
        if state_channel is None:
            logger.debug("Command %s skipped: state channel unavailable", command)
            return None
        if not state_channel.feature_enabled("call.command"):
            logger.debug("Command %s skipped: call.command feature disabled", command)
            return None
        session_id = getattr(self._control_state, "session_id", None)
        if not session_id:
            logger.debug("Command %s skipped: missing session id", command)
            return None
        payload_dict: Dict[str, object] = {"command": command}
        if args:
            payload_dict["args"] = list(args)
        if kwargs:
            payload_dict["kwargs"] = dict(kwargs)
        if origin:
            payload_dict["origin"] = origin
        frame = build_call_command(session_id=session_id, frame_id=None, payload=payload_dict)
        frame_id = frame.envelope.frame_id or ""
        future: Future = Future()
        with self._command_lock:
            self._pending_commands[frame_id] = future
        if not state_channel.send_frame(frame):
            with self._command_lock:
                self._pending_commands.pop(frame_id, None)
            future.set_exception(
                CommandError(code="command.send_failed", message="failed to enqueue command frame"),
            )
            return future
        logger.debug("Command %s emitted frame=%s", command, frame_id)
        return future

    def _send_keyframe_command(
        self,
        *,
        origin: str,
        enforce_backoff: bool,
    ) -> Future | None:
        now = time.time()
        cooldown = self._keyframe_backoff_s if enforce_backoff else 0.0
        if enforce_backoff and self._last_keyframe_request_ts > 0.0:
            elapsed = now - self._last_keyframe_request_ts
            if elapsed < cooldown:
                remaining = max(0.0, cooldown - elapsed)
                logger.debug(
                    "Keyframe request skipped: backoff active (remaining=%.0fms origin=%s)",
                    remaining * 1000.0,
                    origin,
                )
                return None

        future = self._issue_command(_KEYFRAME_COMMAND, origin=origin)
        if future is None:
            return None
        if future.done() and future.exception() is not None:
            return future

        self._last_keyframe_request_ts = now
        self._loop_state.sync.reset_sequence()
        return future

    def _request_keyframe_command(self, *, origin: str = "auto.keyframe") -> None:
        future = self._send_keyframe_command(origin=origin, enforce_backoff=True)
        if future is None or (future.done() and future.exception() is not None):
            return

        def _log_result(fut: Future) -> None:
            try:
                payload = fut.result()
                logger.debug(
                    "Keyframe command acknowledged in_reply_to=%s",
                    getattr(payload, "in_reply_to", None),
                )
            except CommandError as exc:
                logger.warning(
                    "Keyframe command rejected code=%s message=%s",
                    exc.code,
                    exc,
                )
            except Exception:
                logger.debug("Keyframe command future failed", exc_info=True)

        future.add_done_callback(_log_result)

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
        if vm_ref is None:
            return
        self._layer_bridge.sync_viewer_layers(vm_ref, snapshot)

    def _handle_connected(self) -> None:
        self._loop_state.sync.reset_sequence()
        self._init_decoder()
        dec = self.decoder.decode if self.decoder else None
        self._loop_state.pyav_pipeline.set_decoder(dec)

    def _handle_disconnect(self, exc: Exception | None) -> None:
        logger.info("PixelReceiver disconnected: %s", exc)
        self._loop_state.sync.reset_sequence()

    # State channel lifecycle: gate dims state.update traffic safely across reconnects
    def _on_state_connected(self) -> None:
        control_actions.on_state_connected(self._control_state)
        logger.info("StateChannel connected; gating dims intents until notify.dims metadata arrives")

    def _on_state_disconnect(self, exc: Exception | None) -> None:
        control_actions.on_state_disconnected(self._loop_state, self._control_state)
        self._layer_bridge.clear_pending_on_reconnect()
        self._state_ledger.clear_pending_on_reconnect()
        self._loop_state.state_session_metadata = None
        self._command_catalog = ()
        self._abort_pending_commands(
            code="command.session_closed",
            message="state channel disconnected",
        )
        if isinstance(exc, HeartbeatAckError):
            logger.warning("StateChannel disconnect triggered by heartbeat timeout; dims intents gated")
        else:
            logger.info("StateChannel disconnected: %s; dims state.update traffic gated", exc)

    def _dispatch_state_update(self, pending_update: "IntentRecord", origin: str) -> bool:
        channel = self._loop_state.state_channel
        if channel is None:
            logger.debug("state.update emit skipped: channel unavailable")
            self._state_ledger.discard_pending(pending_update.frame_id)
            return False

        session_id = self._control_state.session_id
        if not session_id:
            logger.warning(
                "state.update emit blocked: session_id missing for intent=%s frame=%s",
                pending_update.intent_id,
                pending_update.frame_id,
            )
            self._state_ledger.discard_pending(pending_update.frame_id)
            return False

        frame = build_state_update(
            session_id=session_id,
            intent_id=pending_update.intent_id,
            frame_id=pending_update.frame_id,
            payload=pending_update.payload_dict(),
        )
        frame_id = frame.envelope.frame_id

        ok = channel.send_frame(frame)
        logger.debug(
            "state.update emit: origin=%s scope=%s target=%s key=%s intent=%s frame=%s sent=%s metadata=%s",
            origin,
            pending_update.scope,
            pending_update.target,
            pending_update.key,
            pending_update.intent_id,
            frame_id,
            bool(ok),
            pending_update.metadata,
        )
        if not ok:
            self._state_ledger.discard_pending(pending_update.frame_id)
            return False
        return True

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
        emitter = self._dims_emitter
        if emitter is None:
            logger.debug("wheel event skipped: emitter unavailable")
            return
        try:
            emitter.handle_wheel(data)
        except Exception:
            logger.debug("wheel handler failed", exc_info=True)

    # Shortcut-driven stepping via intents (arrows/page)
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
        camera.zoom_steps_at_center(
            self._control_state,
            self._camera_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            steps,
            widget_to_video=self._widget_to_video,
            server_anchor_from_video=self._server_anchor_from_video,
            log_dims_info=self._log_dims_info,
            vid_size=(self._vid_w, self._vid_h),
        )

    def _reset_camera(self) -> None:
        camera.reset_camera(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            origin='keys',
        )

    def _on_key_event(self, data: dict) -> bool:
        return control_actions.handle_key_event(
            data,
            reset_camera=self._reset_camera,
            step_primary=lambda delta: self.dims_step('primary', delta, origin='keys'),
        )

    def _current_viewer(self) -> object | None:
        return self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]

    def _initialize_mirrors_and_emitters(self) -> None:
        if self._dims_mirror is None:
            self._dims_mirror = napari_dims_mirror.NapariDimsMirror(
                ledger=self._state_ledger,
                state=self._control_state,
                loop_state=self._loop_state,
                viewer_ref=self._current_viewer,
                ui_call=self._ui_call,
                presenter=self._presenter_facade,
                log_dims_info=self._log_dims_info,
                notify_first_ready=self._notify_first_dims_ready,
            )

        tx_interval = int(getattr(self, '_slider_tx_interval_ms', 0))
        if self._dims_emitter is None:
            self._dims_emitter = NapariDimsIntentEmitter(
                ledger=self._state_ledger,
                state=self._control_state,
                loop_state=self._loop_state,
                dispatch_state_update=self._dispatch_state_update,
                ui_call=self._ui_call,
                log_dims_info=self._log_dims_info,
                tx_interval_ms=tx_interval,
            )
        else:
            self._dims_emitter.set_tx_interval_ms(tx_interval)

        self._dims_mirror.set_logging(self._log_dims_info)
        self._dims_emitter.set_logging(self._log_dims_info)
        self._dims_mirror.attach_emitter(self._dims_emitter)

    # --- Public UI/state bridge methods -------------------------------------------
    def attach_viewer_proxy(self, viewer: object) -> None:
        """Attach the viewer proxy the runtime mirrors state into.

        We keep only a weak reference so the viewer can drop without waiting
        on the runtime, and immediately replay the last confirmed payload to
        seed both dims and presenter state.
        """
        self._viewer_mirror = weakref.ref(viewer)  # type: ignore[attr-defined]
        self._presenter_facade.set_viewer_mirror(viewer)
        self._initialize_mirrors_and_emitters()
        if self._dims_emitter is not None:
            self._dims_emitter.attach_viewer(viewer)
        self._replay_last_dims_payload()

    @property
    def presenter_facade(self) -> PresenterFacade:
        """Expose the client presenter façade for auxiliary wiring."""

        return self._presenter_facade

    @property
    def state_ledger(self) -> ClientStateLedger:
        """Expose the shared reducer for auxiliary bridges/tests."""

        return self._state_ledger

    def post(self, obj: dict) -> bool:
        ch = self._loop_state.state_channel
        return bool(ch.post(obj)) if ch is not None else False

    def reset_camera(self, origin: str = 'ui') -> bool:
        """Send a camera.reset to the server (used by UI bindings)."""
        return camera.reset_camera(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            origin=origin,
        )

    def set_camera(self, *, center=None, zoom=None, angles=None, origin: str = 'ui') -> bool:
        """Send absolute camera fields when provided.

        Prefer using zoom_at/pan_px/reset ops for interactions; this is
        intended for explicit UI actions that set a known state.
        """
        return camera.set_camera(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            center=center,
            zoom=zoom,
            angles=angles,
            origin=origin,
        )

    # --- Intents API --------------------------------------------------------------
    # --- Mode helpers -----------------------------------------------------------
    def _is_volume_mode(self) -> bool:
        return control_actions._is_volume_mode(self._control_state)  # type: ignore[attr-defined]

    # --- Small utilities (no behavior change) ----------------------------------
    # --- View HUD snapshot (for overlay) ----------------------------------------
    def view_hud_snapshot(self) -> dict:
        """Return a compact snapshot of 3D view/volume tuning state.

        Safe to call from GUI timer; avoids raising on missing fields.
        """
        cam_state = self._camera_state
        zoom_state = {
            'last_zoom_factor': cam_state.last_zoom_factor,
            'last_zoom_widget_px': cam_state.last_zoom_widget_px,
            'last_zoom_video_px': cam_state.last_zoom_video_px,
            'last_zoom_anchor_px': cam_state.last_zoom_anchor_px,
            'last_pan_dx': cam_state.last_pan_dx_sent,
            'last_pan_dy': cam_state.last_pan_dy_sent,
            'zoom_base': float(cam_state.zoom_base),
        }
        return control_actions.hud_snapshot(
            self._control_state,
            video_size=(self._vid_w, self._vid_h),
            zoom_state=zoom_state,
        )

    def dims_step(self, axis: int | str, delta: int, *, origin: str = 'ui') -> bool:
        emitter = self._dims_emitter
        if emitter is None:
            logger.debug("dims_step skipped: emitter unavailable")
            return False
        return emitter.dims_step(axis, delta, origin=origin)

    def dims_set_index(self, axis: int | str, value: int, *, origin: str = 'ui') -> bool:
        emitter = self._dims_emitter
        if emitter is None:
            logger.debug("dims_set_index skipped: emitter unavailable")
            return False
        return emitter.dims_set_index(axis, value, origin=origin)

    # --- Volume/multiscale intent senders --------------------------------------
    def volume_set_render_mode(self, mode: str, *, origin: str = 'ui') -> bool:
        return control_actions.volume_set_render_mode(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            mode,
            origin=origin,
        )

    def volume_set_clim(self, lo: float, hi: float, *, origin: str = 'ui') -> bool:
        return control_actions.volume_set_clim(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            lo,
            hi,
            origin=origin,
        )

    def volume_set_colormap(self, name: str, *, origin: str = 'ui') -> bool:
        return control_actions.volume_set_colormap(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            name,
            origin=origin,
        )

    def volume_set_opacity(self, alpha: float, *, origin: str = 'ui') -> bool:
        return control_actions.volume_set_opacity(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            alpha,
            origin=origin,
        )

    def volume_set_sample_step(self, relative: float, *, origin: str = 'ui') -> bool:
        return control_actions.volume_set_sample_step(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            relative,
            origin=origin,
        )

    def multiscale_set_policy(self, policy: str, *, origin: str = 'ui') -> bool:
        return control_actions.multiscale_set_policy(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            policy,
            origin=origin,
        )

    def multiscale_set_level(self, level: int, *, origin: str = 'ui') -> bool:
        return control_actions.multiscale_set_level(
            self._control_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            level,
            origin=origin,
        )

    def view_set_ndisplay(self, ndisplay: int, *, origin: str = 'ui') -> bool:
        emitter = self._dims_emitter
        if emitter is None:
            logger.debug("view_set_ndisplay skipped: emitter unavailable")
            return False
        return emitter.view_set_ndisplay(ndisplay, origin=origin)

    def current_ndisplay(self) -> Optional[int]:
        return control_actions.current_ndisplay(self._control_state)

    def toggle_ndisplay(self, *, origin: str = 'ui') -> bool:
        emitter = self._dims_emitter
        if emitter is None:
            logger.debug("toggle_ndisplay skipped: emitter unavailable")
            return False
        return emitter.toggle_ndisplay(origin=origin)

    def _on_wheel_for_zoom(self, data: dict) -> None:
        camera.handle_wheel_zoom(
            self._control_state,
            self._camera_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            data,
            widget_to_video=self._widget_to_video,
            server_anchor_from_video=self._server_anchor_from_video,
            log_dims_info=self._log_dims_info,
        )

    def _on_pointer(self, data: dict) -> None:
        if not self._control_state.dims_ready:
            return
        dims_meta = self._control_state.dims_meta
        mode_obj = dims_meta.get('mode')
        ndisplay_obj = dims_meta.get('ndisplay')
        if isinstance(mode_obj, str) and isinstance(ndisplay_obj, int):
            in_vol3d = (mode_obj.lower() == 'volume') and ndisplay_obj == 3
        else:
            in_vol3d = False
        camera.handle_pointer(
            self._control_state,
            self._camera_state,
            self._loop_state,
            self._state_ledger,
            self._dispatch_state_update,
            data,
            widget_to_video=self._widget_to_video,
            video_delta_to_canvas=self._video_delta_to_canvas,
            log_dims_info=self._log_dims_info,
            in_vol3d=in_vol3d,
            alt_mask=int(QtCore.Qt.AltModifier),
        )

    # (no key→dims mapping)

    def _on_frame(self, pkt: Packet) -> None:
        cur = int(pkt.seq)
        if not (self._loop_state.vt_wait_keyframe or self._loop_state.pyav_wait_keyframe):
            if self._loop_state.sync.update_and_check(cur):
                if self._vt_decoder is not None:
                    self._loop_state.vt_wait_keyframe = True
                    self._loop_state.vt_pipeline.clear(preserve_cache=True)
                    self._presenter.clear(Source.VT)
                self._loop_state.pyav_wait_keyframe = True
                self._loop_state.pyav_pipeline.clear()
                future = self._send_keyframe_command(
                    origin="auto.discontinuity",
                    enforce_backoff=False,
                )
                if future is None or (future.done() and future.exception() is not None):
                    self._log_keyframe_skip("stream discontinuity")
                else:
                    logger.debug(
                        "Keyframe request queued for stream discontinuity (seq=%d)",
                        cur,
                    )
            self._presenter.clear(Source.PYAV)
            self._init_decoder()
            dec = self.decoder.decode if self.decoder else None
            self._loop_state.pyav_pipeline.set_decoder(dec)
            self._loop_state.disco_gated = True
        if not self._stream_seen_keyframe:
            if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                self._stream_seen_keyframe = True
            else:
                return
        if self._vt_decoder is not None and self._loop_state.vt_wait_keyframe:
            if self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01):
                self._loop_state.vt_wait_keyframe = False
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
                self._loop_state.disco_gated = False
                if self._warmup_policy is not None:
                    warmup.on_gate_lift(
                        self._warmup_policy,
                        self._loop_state,
                        self._presenter,
                        self._vt_latency_s,
                        self._fps,
                    )
                # Schedule first wake after VT becomes active (thread-safe)
                self.schedule_next_wake_threadsafe()
            else:
                self._log_keyframe_skip("vt gate pending")
                return
        if self._vt_decoder is not None and not self._loop_state.vt_wait_keyframe:
            ts_float = float(pkt.ts)
            b = pkt.payload
            self._loop_state.vt_pipeline.enqueue(b, ts_float)
            self._vt_enqueued += 1
        else:
            ts_val = pkt.ts
            ts_float = float(ts_val) if ts_val is not None else None
            b = pkt.payload
            if self._loop_state.pyav_pipeline.qsize() >= max(2, self._pyav_backlog_trigger - 1):
                self._loop_state.pyav_wait_keyframe = True
                self._loop_state.pyav_pipeline.clear()
                self._presenter.clear(Source.PYAV)
                self._init_decoder()
                dec = self.decoder.decode if self.decoder else None
                self._loop_state.pyav_pipeline.set_decoder(dec)
            if self._loop_state.pyav_wait_keyframe:
                if not (self._is_keyframe(pkt.payload, pkt.codec) or (pkt.flags & 0x01)):
                    return
                self._loop_state.pyav_wait_keyframe = False
                self._init_decoder()
                self._loop_state.disco_gated = False
            self._loop_state.pyav_pipeline.enqueue(b, ts_float)
            self._pyav_enqueued += 1

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        loop_lifecycle.stop_loop(self)

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
