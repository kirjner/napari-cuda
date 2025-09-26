from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from threading import Thread
from typing import Callable, Dict, Optional, TYPE_CHECKING
import weakref
from contextlib import ExitStack

import numpy as np
from qtpy import QtCore

from napari_cuda.client.streaming.presenter import FixedLatencyPresenter, SourceMux
from napari_cuda.client.streaming.presenter_facade import PresenterFacade
from napari_cuda.client.streaming.receiver import PixelReceiver, Packet
from napari_cuda.client.streaming.state import StateChannel
from napari_cuda.client.streaming.types import Source, SubmittedFrame
from napari_cuda.client.streaming.renderer import GLRenderer
from napari_cuda.client.streaming.decoders.pyav import PyAVDecoder
from napari_cuda.client.streaming.decoders.vt import VTLiveDecoder
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
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState
from napari_cuda.client.streaming.client_loop import warmup, intents, camera, loop_lifecycle
from napari_cuda.client.streaming.client_loop.scheduler_helpers import (
    init_wake_scheduler,
)
from napari_cuda.client.streaming.client_loop.telemetry import (
    build_telemetry_config,
    create_metrics,
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

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from napari_cuda.client.streaming.config import ClientConfig

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
        self._latest_scene_spec: Optional[SceneSpecMessage] = None
        self._layer_registry = RemoteLayerRegistry()
        self._layer_registry.add_listener(self._on_registry_snapshot)
        # Keep-last-frame fallback default enabled for smoother presentation
        self._keep_last_frame_fallback = True
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

        # Stats/logging and diagnostics
        self._stats_level = self._telemetry_cfg.stats_level
        self._last_stats_time: float = 0.0
        self._relearn_logged: bool = False
        self._last_relearn_log_ts: float = 0.0
        # Debounce duplicate video_config
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
        # Video dimensions (from video_config)
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
        # --- Dims/intent state (intent-only client) ----------------------------
        self._intent_state = intents.IntentState.from_env(self._env_cfg)
        self._loop_state.intents = self._intent_state

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
                self._loop_state.vt_pipeline.set_decoder(self._vt_decoder)
            except Exception as exc:
                logger.warning("VT shim unavailable: %s; falling back to PyAV", exc)
                self._vt_decoder = None
        else:
            self._vt_decoder = None
        self._vt_cfg_key = cfg_key
        self._last_vcfg_key = cfg_key
        self._loop_state.vt_wait_keyframe = True
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
        intents.handle_dims_update(
            self._intent_state,
            self._loop_state,
            data,
            presenter=self._presenter_facade,
            viewer_ref=self._viewer_mirror,
            ui_call=self._ui_call,
            notify_first_dims_ready=self._notify_first_dims_ready,
            log_dims_info=self._log_dims_info,
        )

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
        intents.replay_last_dims_payload(
            self._intent_state,
            self._loop_state,
            self._viewer_mirror,
            self._ui_call,
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
        self._loop_state.pyav_pipeline.set_decoder(dec)

    def _handle_disconnect(self, exc: Exception | None) -> None:
        logger.info("PixelReceiver disconnected: %s", exc)

    # State channel lifecycle: gate dims intents safely across reconnects
    def _on_state_connected(self) -> None:
        intents.on_state_connected(self._intent_state)
        logger.info("StateChannel connected; gating dims intents until dims.update meta arrives")

    def _on_state_disconnect(self, exc: Exception | None) -> None:
        intents.on_state_disconnected(self._loop_state, self._intent_state)
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
        intents.handle_wheel_for_dims(
            self._intent_state,
            self._loop_state,
            data,
            viewer_ref=self._viewer_mirror,
            ui_call=self._ui_call,
            log_dims_info=self._log_dims_info,
        )

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
            self._camera_state,
            self._loop_state,
            steps,
            widget_to_video=self._widget_to_video,
            server_anchor_from_video=self._server_anchor_from_video,
            log_dims_info=self._log_dims_info,
            vid_size=(self._vid_w, self._vid_h),
        )

    def _reset_camera(self) -> None:
        camera.reset_camera(self._loop_state, origin='keys')

    def _on_key_event(self, data: dict) -> bool:
        return intents.handle_key_event(
            data,
            reset_camera=self._reset_camera,
            step_primary=lambda delta: self.dims_step('primary', delta, origin='keys'),
        )

    # --- Public UI/state bridge methods -------------------------------------------
    def attach_viewer_mirror(self, viewer: object) -> None:
        """Attach a viewer to mirror server dims updates into local UI.

        Stores a weak reference to avoid lifetime coupling.
        """
        self._viewer_mirror = weakref.ref(viewer)  # type: ignore[attr-defined]
        self._presenter_facade.set_viewer_mirror(viewer)
        self._replay_last_dims_payload()

    @property
    def presenter_facade(self) -> PresenterFacade:
        """Expose the client presenter façade for auxiliary wiring."""

        return self._presenter_facade

    def post(self, obj: dict) -> bool:
        ch = self._loop_state.state_channel
        return bool(ch.post(obj)) if ch is not None else False

    def reset_camera(self, origin: str = 'ui') -> bool:
        """Send a camera.reset to the server (used by UI bindings)."""
        return camera.reset_camera(self._loop_state, origin=origin)

    def set_camera(self, *, center=None, zoom=None, angles=None, origin: str = 'ui') -> bool:
        """Send absolute camera fields when provided.

        Prefer using zoom_at/pan_px/reset ops for interactions; this is
        intended for explicit UI actions that set a known state.
        """
        return camera.set_camera(
            self._loop_state,
            center=center,
            zoom=zoom,
            angles=angles,
            origin=origin,
        )

    # --- Intents API --------------------------------------------------------------
    # --- Mode helpers -----------------------------------------------------------
    def _is_volume_mode(self) -> bool:
        return intents._is_volume_mode(self._intent_state)  # type: ignore[attr-defined]

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
        return intents.hud_snapshot(
            self._intent_state,
            video_size=(self._vid_w, self._vid_h),
            zoom_state=zoom_state,
        )

    def dims_step(self, axis: int | str, delta: int, *, origin: str = 'ui') -> bool:
        return intents.dims_step(
            self._intent_state,
            self._loop_state,
            axis,
            delta,
            origin=origin,
            viewer_ref=self._viewer_mirror,
            ui_call=self._ui_call,
        )

    def dims_set_index(self, axis: int | str, value: int, *, origin: str = 'ui') -> bool:
        return intents.dims_set_index(
            self._intent_state,
            self._loop_state,
            axis,
            value,
            origin=origin,
            viewer_ref=self._viewer_mirror,
            ui_call=self._ui_call,
        )

    # --- Volume/multiscale intent senders --------------------------------------
    def volume_set_render_mode(self, mode: str, *, origin: str = 'ui') -> bool:
        return intents.volume_set_render_mode(self._intent_state, self._loop_state, mode, origin=origin)

    def volume_set_clim(self, lo: float, hi: float, *, origin: str = 'ui') -> bool:
        return intents.volume_set_clim(self._intent_state, self._loop_state, lo, hi, origin=origin)

    def volume_set_colormap(self, name: str, *, origin: str = 'ui') -> bool:
        return intents.volume_set_colormap(self._intent_state, self._loop_state, name, origin=origin)

    def volume_set_opacity(self, alpha: float, *, origin: str = 'ui') -> bool:
        return intents.volume_set_opacity(self._intent_state, self._loop_state, alpha, origin=origin)

    def volume_set_sample_step(self, relative: float, *, origin: str = 'ui') -> bool:
        return intents.volume_set_sample_step(self._intent_state, self._loop_state, relative, origin=origin)

    def multiscale_set_policy(self, policy: str, *, origin: str = 'ui') -> bool:
        return intents.multiscale_set_policy(self._intent_state, self._loop_state, policy, origin=origin)

    def multiscale_set_level(self, level: int, *, origin: str = 'ui') -> bool:
        return intents.multiscale_set_level(self._intent_state, self._loop_state, level, origin=origin)

    def view_set_ndisplay(self, ndisplay: int, *, origin: str = 'ui') -> bool:
        return intents.view_set_ndisplay(self._intent_state, self._loop_state, ndisplay, origin=origin)

    def current_ndisplay(self) -> Optional[int]:
        return intents.current_ndisplay(self._intent_state)

    def toggle_ndisplay(self, *, origin: str = 'ui') -> bool:
        return intents.toggle_ndisplay(self._intent_state, self._loop_state, origin=origin)

    def _on_wheel_for_zoom(self, data: dict) -> None:
        camera.handle_wheel_zoom(
            self._camera_state,
            self._loop_state,
            data,
            widget_to_video=self._widget_to_video,
            server_anchor_from_video=self._server_anchor_from_video,
            log_dims_info=self._log_dims_info,
        )

    def _on_pointer(self, data: dict) -> None:
        dims_meta = self._intent_state.dims_meta
        in_vol3d = self._is_volume_mode() and int(dims_meta.get('ndisplay') or 2) == 3
        camera.handle_pointer(
            self._camera_state,
            self._loop_state,
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
                    self._request_keyframe_once()
                self._loop_state.pyav_wait_keyframe = True
                self._loop_state.pyav_pipeline.clear()
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
                self._request_keyframe_once()
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
