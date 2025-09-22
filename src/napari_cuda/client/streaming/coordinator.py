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
from dataclasses import dataclass

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
from napari_cuda.client.streaming.pipelines.pyav_pipeline import PyAVPipeline
from napari_cuda.client.streaming.pipelines.vt_pipeline import VTPipeline
from napari_cuda.client.streaming.pipelines.smoke_pipeline import SmokePipeline, SmokeConfig
from napari_cuda.client.streaming.metrics import ClientMetrics
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
from napari_cuda.utils.env import env_float, env_str
from napari_cuda.client.streaming.smoke.generators import make_generator
from napari_cuda.client.streaming.smoke.submit import submit_vt, submit_pyav
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
    try:
        import os as _os
        flag = (_os.getenv('NAPARI_CUDA_CLIENT_DEBUG') or _os.getenv('NAPARI_CUDA_DEBUG') or '').lower()
        if flag not in ('1', 'true', 'yes', 'on', 'dbg', 'debug'):
            return
        has_local = any(getattr(h, '_napari_cuda_local', False) for h in logger.handlers)
        if has_local:
            return
        h = logging.StreamHandler()
        fmt = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        h.setFormatter(logging.Formatter(fmt))
        h.setLevel(logging.DEBUG)
        setattr(h, '_napari_cuda_local', True)
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    except Exception:
        # Don't let logging issues affect runtime
        pass


class _WakeProxy(QtCore.QObject):
    """Qt signal proxy to safely schedule wakes from any thread.

    Emitting `trigger` from worker threads posts a queued call to the GUI
    thread where `_schedule_next_wake` runs and arms the QTimer.
    """

    trigger = QtCore.Signal()

    def __init__(self, slot, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.trigger.connect(slot)


class _CallProxy(QtCore.QObject):
    """Post callables to the GUI thread via queued signal delivery."""

    call = QtCore.Signal(object)

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.call.connect(self._on_call)

    def _on_call(self, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            if callable(fn):
                fn()
        except Exception:
            logger.debug("_CallProxy execution failed", exc_info=True)

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


class StreamCoordinator:
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
        _maybe_enable_debug_logger()
        # Set smoke mode early so start() can rely on it even if later init changes
        self._vt_smoke = bool(vt_smoke)
        # Phase 0: stash client config for future phases (no behavior change)
        self._client_cfg = client_cfg
        self._first_dims_ready_cb = on_first_dims_ready
        self._first_dims_notified = False
        self._pending_intents: dict[int, dict[str, object]] = {}
        self._last_dims_seq: Optional[int] = None
        self._last_dims_payload: Optional[dict[str, object]] = None
        self.server_host = server_host
        self.server_port = int(server_port)
        self.state_port = int(state_port)
        self._stream_format = stream_format
        self._stream_format_set = False
        # Thread/state handles
        self._threads: list[Thread] = []
        self._state_channel: Optional[StateChannel] = None
        # Optional weakref to a ViewerModel mirror (e.g., ProxyViewer)
        self._viewer_mirror = None  # type: ignore[var-annotated]
        # UI call proxy to marshal updates to the GUI thread
        try:
            self._ui_call = _CallProxy(getattr(self._scene_canvas, 'native', None))
        except Exception:
            self._ui_call = None

        # Presenter + source mux (server timestamps only)
        # Single preview guard (ms)
        try:
            preview_guard_ms = float(getattr(self._client_cfg, 'preview_guard_ms', 0.0))
        except Exception:
            preview_guard_ms = 0.0
        # No env overrides; preview_guard_ms comes from ClientConfig only
        self._presenter = FixedLatencyPresenter(
            latency_s=float(vt_latency_s),
            buffer_limit=int(vt_buffer_limit),
            preview_guard_s=float(preview_guard_ms) / 1000.0,
        )
        try:
            logger.info("Presenter init: preview_guard=%.1fms latency=%.0fms",
                        float(preview_guard_ms),
                        float(vt_latency_s) * 1000.0)
        except Exception:
            pass
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
        self._warmup_ms_override = env_str('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS', None)
        self._warmup_window_s = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_WINDOW_S', 0.75)
        self._warmup_margin_ms = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MARGIN_MS', 2.0)
        self._warmup_max_ms = env_float('NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MAX_MS', 24.0)
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
        self._server_bias_s = env_float('NAPARI_CUDA_SERVER_TS_BIAS_MS', 0.0) / 1000.0
        # Wake scheduling fudge (ms) to compensate for timer quantization; added to due delay
        self._wake_fudge_ms = env_float('NAPARI_CUDA_WAKE_FUDGE_MS', 1.0)
        # avcC nal length size (default 4 if unknown)
        self._nal_length_size: int = 4

        # VT pipeline replaces inline queues/workers
        self._vt_backlog_trigger = int(vt_backlog_trigger)
        self._vt_enqueued = 0
        # Callbacks for VT pipeline coordination
        def _is_vt_gated() -> bool:
            return bool(self._vt_wait_keyframe)
        def _on_vt_backlog_gate() -> None:
            self._vt_wait_keyframe = True
        def _req_keyframe() -> None:
            self._request_keyframe_once()
        def _on_cache_last(pb: object, persistent: bool) -> None:
            try:
                # Keep local cache for draw fallback
                self._last_vt_payload = pb
                self._last_vt_persistent = bool(persistent)
            except Exception:
                logger.debug("cache last VT payload callback failed", exc_info=True)
        # Client metrics (optional, env-controlled)
        try:
            metrics_enabled = (os.getenv('NAPARI_CUDA_CLIENT_METRICS', '0') or '0').lower() in ('1', 'true', 'yes')
        except Exception:
            metrics_enabled = False
        self._metrics = ClientMetrics(enabled=metrics_enabled)

        # Presenter-owned wake scheduling: single-shot QTimer owned by the GUI thread,
        # optionally disabled when an external fixed-cadence display loop is active.
        # Opt-in via NAPARI_CUDA_USE_DISPLAY_LOOP=1.
        self._wake_timer = None
        self._in_present: bool = False
        try:
            use_disp_loop_env = (os.getenv('NAPARI_CUDA_USE_DISPLAY_LOOP', '0') or '0').lower()
            self._use_display_loop = use_disp_loop_env in ('1', 'true', 'yes', 'on')
        except Exception:
            self._use_display_loop = False
        if not self._use_display_loop:
            try:
                self._wake_timer = QtCore.QTimer(self._scene_canvas.native)
                self._wake_timer.setTimerType(QtCore.Qt.PreciseTimer)
                self._wake_timer.setSingleShot(True)
                # On wake, request a canvas update and immediately schedule the next wake
                self._wake_timer.timeout.connect(self._on_present_timer)
            except Exception:
                logger.debug("Wake timer init failed", exc_info=True)
        # Record GUI thread affinity for safe scheduling from worker threads
        try:
            self._gui_thread = self._scene_canvas.native.thread()  # type: ignore[attr-defined]
        except Exception:
            self._gui_thread = None  # type: ignore[attr-defined]

        def _schedule_next_wake() -> None:
            if self._use_display_loop:
                # External display loop drives cadence; no per-frame wake scheduling
                return
            try:
                earliest = self._presenter.peek_next_due(self._source_mux.active)
                if earliest is None:
                    return
                now_mono = time.perf_counter()
                delta_ms = (float(earliest) - now_mono) * 1000.0 + float(self._wake_fudge_ms or 0.0)
                # If a wake is already scheduled earlier (within a small margin), keep it to avoid thrash
                if self._next_due_pending_until > now_mono:
                    pending_ms = (self._next_due_pending_until - now_mono) * 1000.0
                    if delta_ms >= (pending_ms - 0.5):
                        return
                else:
                    # Clear stale marker
                    self._next_due_pending_until = 0.0
                # (Re)schedule
                try:
                    if self._wake_timer is not None and self._wake_timer.isActive():
                        # Only stop if we're moving the wake earlier by a meaningful amount
                        try:
                            self._wake_timer.stop()
                        except Exception:
                            pass
                except Exception:
                    logger.debug("schedule_next_wake: timer state check failed", exc_info=True)
                self._next_due_pending_until = now_mono + max(0.0, delta_ms) / 1000.0
                try:
                    if self._wake_timer is not None:
                        self._wake_timer.start(max(0, int(delta_ms)))
                    else:
                        # Fallback: poke the canvas to ensure progress
                        self._scene_canvas.native.update()
                except Exception:
                    # As a last resort, poke the canvas to drive a draw
                    try:
                        self._scene_canvas.native.update()
                    except Exception:
                        getattr(self._scene_canvas, 'update', lambda: None)()
            except Exception:
                logger.debug("schedule_next_wake failed", exc_info=True)

        # Expose to pipelines via closure
        self._schedule_next_wake = _schedule_next_wake

        # Build pipelines after scheduler hooks are ready
        # Create a wake proxy so pipelines can nudge scheduling from any thread
        if not self._use_display_loop:
            try:
                self._wake_proxy = _WakeProxy(_schedule_next_wake, self._scene_canvas.native)
                wake_cb = self._wake_proxy.trigger.emit
            except Exception:
                logger.debug("WakeProxy init failed; using thread-safe scheduler wrapper", exc_info=True)
                # Fallback to a wrapper that routes scheduling to GUI thread
                wake_cb = (lambda: self.schedule_next_wake_threadsafe())
        else:
            # External loop owns draw cadence; pipelines don't need to schedule wakes
            wake_cb = (lambda: None)
        self._vt_pipeline = VTPipeline(
            presenter=self._presenter,
            source_mux=self._source_mux,
            scene_canvas=self._scene_canvas,
            backlog_trigger=self._vt_backlog_trigger,
            is_gated=_is_vt_gated,
            on_backlog_gate=_on_vt_backlog_gate,
            request_keyframe=_req_keyframe,
            on_cache_last=_on_cache_last,
            metrics=self._metrics,
            schedule_next_wake=wake_cb,
        )
        self._pyav_backlog_trigger = int(pyav_backlog_trigger)
        self._pyav_enqueued = 0
        self._pyav_pipeline = PyAVPipeline(
            presenter=self._presenter,
            source_mux=self._source_mux,
            scene_canvas=self._scene_canvas,
            backlog_trigger=self._pyav_backlog_trigger,
            latency_s=self._pyav_latency_s,
            metrics=self._metrics,
            schedule_next_wake=wake_cb,
        )
        self._pyav_wait_keyframe: bool = False

        # Last-frame caches for redraw fallback
        self._last_vt_payload = None  # VT payload capsule
        self._last_vt_persistent = False
        self._last_pyav_frame = None

        # Frame queue for renderer (latest-wins)
        # Holds either numpy arrays or (capsule, release_cb) tuples for VT
        self._frame_q: "queue.Queue[object]" = queue.Queue(maxsize=3)

        # Receiver/state flags
        self._stream_seen_keyframe = False

        # Stats/logging and diagnostics
        lvl_env = (env_str('NAPARI_CUDA_VT_STATS', '') or '').lower()
        if lvl_env in ('1', 'true', 'yes', 'info'):
            self._stats_level = logging.INFO
        elif lvl_env in ('debug', 'dbg'):
            self._stats_level = logging.DEBUG
        else:
            self._stats_level = None
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
        # Draw watchdog
        self._watchdog_timer = None
        try:
            self._watchdog_ms = max(0, int(env_float('NAPARI_CUDA_CLIENT_DRAW_WATCHDOG_MS', 0.0)))
        except Exception:
            self._watchdog_ms = 0
        # Event loop stall monitor (disabled by default)
        try:
            self._evloop_stall_ms = max(0, int(env_float('NAPARI_CUDA_CLIENT_EVENTLOOP_STALL_MS', 0.0)))
            self._evloop_sample_ms = max(50, int(env_float('NAPARI_CUDA_CLIENT_EVENTLOOP_SAMPLE_MS', 100.0)))
        except Exception:
            self._evloop_stall_ms = 0
            self._evloop_sample_ms = 100
        self._evloop_mon: Optional[EventLoopMonitor] = None
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
        # Dims/Z control (client-initiated)
        try:
            import os as _os
            z0 = _os.getenv('NAPARI_CUDA_ZARR_Z')
            self._dims_z: Optional[int] = int(z0) if z0 is not None and z0.strip() != '' else None
        except Exception:
            self._dims_z = None
        try:
            import os as _os
            zmin = _os.getenv('NAPARI_CUDA_ZARR_Z_MIN')
            zmax = _os.getenv('NAPARI_CUDA_ZARR_Z_MAX')
            self._dims_z_min: Optional[int] = int(zmin) if zmin else None
            self._dims_z_max: Optional[int] = int(zmax) if zmax else None
        except Exception:
            self._dims_z_min = None
            self._dims_z_max = None
        # Clamp initial Z if provided
        if self._dims_z is not None:
            if self._dims_z_min is not None and self._dims_z < self._dims_z_min:
                self._dims_z = self._dims_z_min
            if self._dims_z_max is not None and self._dims_z > self._dims_z_max:
                self._dims_z = self._dims_z_max
        self._wheel_px_accum: float = 0.0
        try:
            import os as _os
            self._wheel_step: int = max(1, int(_os.getenv('NAPARI_CUDA_WHEEL_Z_STEP', '1') or '1'))
        except Exception:
            self._wheel_step = 1
        # dims intents rate limiting (coalesce)
        try:
            import os as _os
            rate = float(_os.getenv('NAPARI_CUDA_DIMS_SET_RATE', '60') or '60')
        except Exception:
            rate = 60.0
        self._dims_min_dt: float = 1.0 / max(1.0, rate)
        self._last_dims_send: float = 0.0
        # Camera ops coalescing
        try:
            import os as _os
            cam_rate = float(_os.getenv('NAPARI_CUDA_CAMERA_SET_RATE', '60') or '60')
        except Exception:
            cam_rate = 60.0
        self._cam_min_dt: float = 1.0 / max(1.0, cam_rate)
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
        try:
            self._orbit_deg_per_px_x: float = float(env_float('NAPARI_CUDA_ORBIT_DEG_PER_PX_X', 0.2))
        except Exception:
            self._orbit_deg_per_px_x = 0.2
        try:
            self._orbit_deg_per_px_y: float = float(env_float('NAPARI_CUDA_ORBIT_DEG_PER_PX_Y', 0.2))
        except Exception:
            self._orbit_deg_per_px_y = 0.2
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
        try:
            import os as _os
            settings_rate = float(_os.getenv('NAPARI_CUDA_SETTINGS_SET_RATE', '60') or '60')
        except Exception:
            settings_rate = 60.0
        self._settings_min_dt: float = 1.0 / max(1.0, settings_rate)
        self._last_settings_send: float = 0.0

    # --- Thread-safe scheduling helpers --------------------------------------
    def _on_gui_thread(self) -> bool:
        try:
            cur = QtCore.QThread.currentThread()
            gui_thr = getattr(self, '_gui_thread', None)
            return (gui_thr is None) or (cur == gui_thr)
        except Exception:
            return True

    def schedule_next_wake_threadsafe(self) -> None:
        """Ensure wake scheduling runs on the GUI thread.

        Prefer the wake proxy; fallback to UI call proxy; otherwise no-op.
        """
        try:
            if self._on_gui_thread():
                self._schedule_next_wake()
                return
        except Exception:
            pass
        # Otherwise forward via proxies
        try:
            wp = getattr(self, '_wake_proxy', None)
            if wp is not None and hasattr(wp, 'trigger'):
                wp.trigger.emit()  # type: ignore[attr-defined]
                return
        except Exception:
            logger.debug("threadsafe schedule: wake proxy failed", exc_info=True)
        try:
            ui_call = getattr(self, '_ui_call', None)
            if ui_call is not None and hasattr(ui_call, 'call'):
                ui_call.call.emit(self._schedule_next_wake)  # type: ignore[attr-defined]
                return
        except Exception:
            logger.debug("threadsafe schedule: ui_call failed", exc_info=True)
        logger.debug("threadsafe schedule: no proxy available; skipping")

    def _on_present_timer(self) -> None:
        # On wake, request a paint; the actual GL draw happens inside the canvas draw event
        try:
            self._in_present = True
            # Clear pending marker to allow next wake to be scheduled freely
            self._next_due_pending_until = 0.0
            # Prefer non-blocking update() over repaint()
            try:
                self._scene_canvas.native.update()
            except Exception:
                getattr(self._scene_canvas, 'update', lambda: None)()
        except Exception:
            logger.debug("present timer: repaint/update failed", exc_info=True)
        # Do not schedule next wake here; draw() re-arms after consumption to reduce early wakes
        finally:
            try:
                self._in_present = False
            except Exception:
                pass

    def start(self) -> None:
        # State channel thread (disabled in offline VT smoke mode)
        if not self._vt_smoke:
            st = StateController(
                self.server_host,
                self.state_port,
                self._on_video_config,
                self._on_dims_update,
                on_scene_spec=self._on_scene_spec,
                on_layer_update=self._on_layer_update,
                on_layer_remove=self._on_layer_remove,
                on_connected=self._on_state_connected,
                on_disconnect=self._on_state_disconnect,
            )
            self._state_channel, t_state = st.start()
            self._threads.append(t_state)
            # Attach input forwarding (wheel + resize) now that state channel exists
            try:
                # Env knobs
                import os as _os
                try:
                    max_rate_hz = float(_os.getenv('NAPARI_CUDA_INPUT_MAX_RATE', '120'))
                except Exception:
                    max_rate_hz = 120.0
                try:
                    resize_debounce_ms = int(_os.getenv('NAPARI_CUDA_RESIZE_DEBOUNCE_MS', '80'))
                except Exception:
                    resize_debounce_ms = 80
                try:
                    enable_wheel_set_dims = (_os.getenv('NAPARI_CUDA_CLIENT_WHEEL_SET_DIMS', '1') or '1').lower() in ('1','true','yes','on')
                except Exception:
                    enable_wheel_set_dims = True
                try:
                    log_input_info = (_os.getenv('NAPARI_CUDA_INPUT_LOG', '0') or '0').lower() in ('1','true','yes','on')
                except Exception:
                    log_input_info = False
                # Use same flag for dims logging (INFO when enabled, DEBUG otherwise)
                self._log_dims_info = bool(log_input_info)  # type: ignore[attr-defined]
                # Unified wheel handler: dims intent step for plain wheel; zoom for modifiers
                wheel_cb = self._on_wheel
                if self._state_channel is not None:
                    sender = InputSender(
                        widget=self._scene_canvas.native,
                        send_json=self._state_channel.send_json,
                        max_rate_hz=max_rate_hz,
                        resize_debounce_ms=resize_debounce_ms,
                        on_wheel=wheel_cb,
                        on_pointer=self._on_pointer,
                        on_key=self._on_key_event,
                        log_info=log_input_info,
                    )
                    sender.start()
                    self._input_sender = sender  # type: ignore[attr-defined]
                    logger.info("InputSender attached (wheel+resize+pointer)")
                    # Bind zoom/reset shortcuts; arrow keys handled via key callback
                    try:
                        from qtpy import QtWidgets, QtGui, QtCore  # type: ignore
                        # Bind shortcuts on the top-level window rather than the
                        # canvas widget to ensure they receive key events across
                        # platforms (notably macOS) with ApplicationShortcut.
                        parent = getattr(self._scene_canvas, 'native', None)
                        try:
                            if parent is not None and hasattr(parent, 'window'):
                                w = parent.window()  # type: ignore[attr-defined]
                                if w is not None:
                                    parent = w
                        except Exception:
                            pass
                        if parent is None:
                            # Fallback: use the active window if available
                            try:
                                from qtpy import QtWidgets as _QtWidgets  # type: ignore
                                aw = _QtWidgets.QApplication.activeWindow()
                                if aw is not None:
                                    parent = aw
                            except Exception:
                                pass
                        # Up/Down/Left/Right are handled in app-level key callback to avoid focus issues
                        plus_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Plus), parent)  # type: ignore
                        eq_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Equal), parent)  # type: ignore
                        minus_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Minus), parent)  # type: ignore
                        home_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Home), parent)  # type: ignore
                        # 0-key shortcuts disabled for investigation; rely on Home for reset
                        # zero_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0), parent)  # type: ignore
                        # zero_str_sc = QtWidgets.QShortcut(QtGui.QKeySequence("0"), parent)  # type: ignore
                        # num0_sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0), parent)  # type: ignore
                        # Ensure shortcuts fire regardless of widget focus
                        for sc in (plus_sc, eq_sc, minus_sc, home_sc):
                            try:
                                sc.setContext(QtCore.Qt.ApplicationShortcut)  # type: ignore[attr-defined]
                            except Exception as e:
                                logger.info("Shortcut context=ApplicationShortcut failed: %s", e, exc_info=True)
                        try:
                            plus_sc.setAutoRepeat(True)
                            eq_sc.setAutoRepeat(True)
                            minus_sc.setAutoRepeat(True)
                        except Exception:
                            pass
                        plus_sc.activated.connect(lambda: self._zoom_steps_at_center(+1))  # type: ignore
                        eq_sc.activated.connect(lambda: self._zoom_steps_at_center(+1))  # type: ignore
                        minus_sc.activated.connect(lambda: self._zoom_steps_at_center(-1))  # type: ignore
                        home_sc.activated.connect(lambda: (logger.info("shortcut: Home -> camera.reset"), self._reset_camera()))  # type: ignore
                        # Note: all 0-related bindings commented out for debugging
                        # Keep references to avoid GC
                        self._shortcuts = [plus_sc, eq_sc, minus_sc, home_sc]  # type: ignore[attr-defined]
                        logger.info("Shortcuts bound: +/-/=→zoom, Home→reset (arrows via keycb)")
                    except Exception:
                        logger.debug("Failed to bind zoom/reset shortcuts", exc_info=True)
            except Exception:
                logger.debug("Failed to attach InputSender", exc_info=True)

        # Start VT pipeline workers
        self._vt_pipeline.start()

        # Start PyAV pipeline worker
        self._pyav_pipeline.start()

        # Receiver thread or smoke mode
        if self._vt_smoke:
            logger.info("StreamCoordinator in smoke test mode (offline)")
            # Ensure PyAV decoder is ready if targeting pyav
            smoke_source = (os.getenv('NAPARI_CUDA_SMOKE_SOURCE') or 'vt').lower()
            if smoke_source == 'pyav':
                self._init_decoder()
                dec = self.decoder.decode if self.decoder else None
                self._pyav_pipeline.set_decoder(dec)
            # Read config
            try:
                sw = int(os.getenv('NAPARI_CUDA_SMOKE_W', '1280'))
            except Exception:
                sw = 1280
            try:
                sh = int(os.getenv('NAPARI_CUDA_SMOKE_H', '720'))
            except Exception:
                sh = 720
            # Preset handling
            preset = (os.getenv('NAPARI_CUDA_SMOKE_PRESET') or '').lower().strip()
            if preset == '4k60':
                sw = 3840
                sh = 2160
                fps = 60.0
            else:
                try:
                    fps = float(os.getenv('NAPARI_CUDA_SMOKE_FPS', '60'))
                except Exception:
                    fps = 60.0
            smoke_mode = (os.getenv('NAPARI_CUDA_SMOKE_MODE', 'checker') or 'checker').lower()
            preencode = (os.getenv('NAPARI_CUDA_SMOKE_PREENCODE', '0') or '0') in ('1', 'true', 'yes')
            try:
                pre_frames = int(os.getenv('NAPARI_CUDA_SMOKE_PRE_FRAMES', str(int(fps) * 3)))
            except Exception:
                pre_frames = int(fps) * 3
            # Phase 5: memory cap and disk path for preencode cache
            try:
                mem_cap_mb = int(os.getenv('NAPARI_CUDA_SMOKE_PRE_MB', '0') or '0')
            except Exception:
                mem_cap_mb = 0
            pre_path = os.getenv('NAPARI_CUDA_SMOKE_PRE_PATH', None)

            cfg = SmokeConfig(
                width=sw,
                height=sh,
                fps=fps,
                smoke_mode=smoke_mode,
                preencode=preencode,
                pre_frames=pre_frames,
                backlog_trigger=self._vt_backlog_trigger if smoke_source == 'vt' else self._pyav_backlog_trigger,
                target='pyav' if smoke_source == 'pyav' else 'vt',
                vt_latency_s=self._vt_latency_s,
                pyav_latency_s=self._pyav_latency_s,
                mem_cap_mb=mem_cap_mb,
                pre_path=pre_path,
            )
            def _init_and_clear(avcc_b64: str, w: int, h: int) -> None:
                self._init_vt_from_avcc(avcc_b64, w, h)
                # In smoke mode, lift VT gate immediately after init
                self._vt_wait_keyframe = False

            self._smoke = SmokePipeline(
                config=cfg,
                presenter=self._presenter,
                source_mux=self._source_mux,
                pipeline=self._pyav_pipeline if smoke_source == 'pyav' else self._vt_pipeline,
                init_vt_from_avcc=_init_and_clear,
                metrics=self._metrics,
            )
            self._smoke.start()
        else:
            rc = ReceiveController(
                self.server_host,
                self.server_port,
                on_connected=self._on_connected,
                on_frame=self._on_frame,
                on_disconnect=self._on_disconnect,
            )
            self._receiver, t_rx = rc.start()
            self._threads.append(t_rx)

        # Dedicated 1 Hz presenter stats timer (decoupled from draw loop)
        if self._stats_level is not None:
            try:
                self._stats_timer = QtCore.QTimer(self._scene_canvas.native)
                self._stats_timer.setTimerType(QtCore.Qt.PreciseTimer)
                self._stats_timer.setInterval(1000)
                self._stats_timer.timeout.connect(self._log_stats)
                self._stats_timer.start()
            except Exception:
                logger.debug("Failed to start stats timer; falling back to draw-based logging", exc_info=True)

        # Client metrics CSV dump timer (independent of stats log level)
        if getattr(self._metrics, 'enabled', False):
            try:
                interval_s = env_float('NAPARI_CUDA_CLIENT_METRICS_INTERVAL', 1.0)
                interval_ms = max(100, int(round(1000.0 * max(0.1, float(interval_s)))))
                self._metrics_timer = QtCore.QTimer(self._scene_canvas.native)
                self._metrics_timer.setTimerType(QtCore.Qt.PreciseTimer)
                self._metrics_timer.setInterval(interval_ms)
                self._metrics_timer.timeout.connect(self._metrics.dump_csv_row)  # type: ignore[arg-type]
                self._metrics_timer.start()
            except Exception:
                logger.debug("Failed to start client metrics timer", exc_info=True)
        # Optional draw watchdog: if no draw observed for threshold, kick an update
        if self._watchdog_ms > 0:
            try:
                self._watchdog_timer = QtCore.QTimer(self._scene_canvas.native)
                self._watchdog_timer.setTimerType(QtCore.Qt.PreciseTimer)
                self._watchdog_timer.setInterval(max(100, int(self._watchdog_ms // 2) or 100))
                def _wd_tick():
                    try:
                        last = float(self._last_draw_pc or 0.0)
                        if last <= 0.0:
                            return
                        now = time.perf_counter()
                        if (now - last) * 1000.0 >= float(self._watchdog_ms):
                            try:
                                # Only kick if frames are pending
                                if self._presenter.peek_next_due(self._source_mux.active) is not None:
                                    try:
                                        self._scene_canvas.native.update()
                                    except Exception:
                                        # Final fallback to canvas.update
                                        getattr(self._scene_canvas, 'update', lambda: None)()
                            except Exception:
                                logger.debug("watchdog: kick failed", exc_info=True)
                            try:
                                self._metrics.inc('napari_cuda_client_draw_watchdog_kicks', 1.0)
                            except Exception:
                                pass
                    except Exception:
                        logger.debug("watchdog tick failed", exc_info=True)
                self._watchdog_timer.timeout.connect(_wd_tick)
                self._watchdog_timer.start()
                logger.info("Draw watchdog enabled: threshold=%d ms", self._watchdog_ms)
            except Exception:
                logger.debug("Failed to start draw watchdog", exc_info=True)
        # Optional event loop monitor (diagnostic): logs and metrics on stalls
        if self._evloop_stall_ms > 0:
            try:
                def _kick():
                    try:
                        self._scene_canvas.native.update()
                    except Exception:
                        getattr(self._scene_canvas, 'update', lambda: None)()
                self._evloop_mon = EventLoopMonitor(
                    parent=self._scene_canvas.native,
                    metrics=self._metrics,
                    stall_threshold_ms=int(self._evloop_stall_ms),
                    sample_interval_ms=int(self._evloop_sample_ms),
                    on_stall_kick=_kick,
                )
            except Exception:
                logger.debug("Failed to start EventLoopMonitor", exc_info=True)

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                logger.debug("Renderer queue drain race", exc_info=True)
        self._frame_q.put(frame)

    def draw(self) -> None:
        # Guard to avoid scheduling immediate repaints from within draw
        in_draw_prev = getattr(self, '_in_draw', False)
        self._in_draw = True
        # Draw-loop pacing metric (perf_counter for monotonic interval)
        try:
            now_pc = time.perf_counter()
            last_pc = float(self._last_draw_pc or 0.0)
            if last_pc > 0.0 and self._metrics is not None:
                self._metrics.observe_ms('napari_cuda_client_draw_interval_ms', (now_pc - last_pc) * 1000.0)
            self._last_draw_pc = now_pc
        except Exception:
            logger.debug("draw: pacing metric failed", exc_info=True)
        # Apply warmup ramp/restore on GUI thread (timer-less)
        try:
            self._apply_warmup(now_pc)
        except Exception:
            logger.debug("draw: apply_warmup failed", exc_info=True)
        # Stats are reported via a dedicated timer now

        # VT output is drained continuously by worker; draw focuses on presenting
        # If no offset learned yet (common in smoke/offline), derive from buffer samples.
        try:
            off = getattr(self._presenter.clock, 'offset', None)
            if off is None:
                learned = self._presenter.relearn_offset(Source.VT)
                if learned is not None and math.isfinite(learned):
                    logger.info("Presenter offset learned from buffer: %.3fs", float(learned))
        except Exception:
            logger.debug("draw: offset learn attempt failed", exc_info=True)

        ready = self._presenter.pop_due(time.perf_counter(), self._source_mux.active)
        if ready is not None:
            src_val = getattr(ready.source, 'value', str(ready.source))
            # Record presentation lateness relative to due time for non-preview frames
            try:
                if not getattr(ready, 'preview', False) and self._metrics is not None and self._metrics.enabled:
                    late_ms = (time.perf_counter() - float(getattr(ready, 'due_ts', 0.0))) * 1000.0
                    self._metrics.observe_ms('napari_cuda_client_present_lateness_ms', float(late_ms))
            except Exception:
                logger.debug("draw: lateness metric failed", exc_info=True)
            # Optional: log a one-time relearn attempt on early preview streaks (lightweight)
            if src_val == 'vt':
                if getattr(ready, 'preview', False) and not self._relearn_logged:
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
                elif not getattr(ready, 'preview', False):
                    self._relearn_logged = False
            if src_val == 'vt':
                # Zero-copy path: pass CVPixelBuffer + release_cb through to GLRenderer
                try:
                    release_cb = None if getattr(ready, 'preview', False) else ready.release_cb
                    self._enqueue_frame((ready.payload, release_cb))
                except Exception:
                    logger.debug("enqueue VT payload failed", exc_info=True)
            else:
                # Cache for last-frame fallback and enqueue
                try:
                    self._last_pyav_frame = ready.payload
                except Exception:
                    logger.debug("Cache last PyAV frame failed", exc_info=True)
                self._enqueue_frame(ready.payload)

        frame = None
        # Optional legacy last-frame fallback (temporary, gated). When disabled,
        # rely on renderer persistence and presenter preview to avoid flicker.
        if ready is None and self._keep_last_frame_fallback:
            if self._source_mux.active == Source.VT:
                try:
                    pb, persistent = self._vt_pipeline.last_payload_info()
                except Exception:
                    pb, persistent = (None, False)
                if pb is not None and persistent:
                    try:
                        self._enqueue_frame((pb, None))
                    except Exception:
                        logger.debug("enqueue last VT payload failed", exc_info=True)
            elif self._source_mux.active == Source.PYAV and self._last_pyav_frame is not None:
                try:
                    self._enqueue_frame(self._last_pyav_frame)
                except Exception:
                    logger.debug("enqueue last PyAV frame failed", exc_info=True)
        # Schedule next wake based on earliest due; avoids relying on a 60 Hz loop
        try:
            # Safe: draw() runs on GUI thread
            self._schedule_next_wake()
        except Exception:
            logger.debug("draw: schedule_next_wake failed", exc_info=True)
        while not self._frame_q.empty():
            try:
                frame = self._frame_q.get_nowait()
            except queue.Empty:
                break
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
            self._metrics.inc('napari_cuda_client_presented_total', 1.0)
            try:
                now = time.perf_counter()
                last = getattr(self, '_last_present_mono', 0.0)
                if last:
                    # Only log PRESENT timing when explicitly enabled to avoid spam
                    import os as _os
                    if (_os.getenv('NAPARI_CUDA_PRESENT_DEBUG') or '').lower() in ('1','true','yes','on','dbg','debug'):
                        inter_ms = (now - float(last)) * 1000.0
                        logger.debug("PRESENT inter_ms=%.3f", float(inter_ms))
                self._last_present_mono = now
            except Exception:
                pass
        else:
            self._renderer.draw(frame)
        # Clear draw guard
        self._in_draw = in_draw_prev

    def _log_stats(self) -> None:
        if self._stats_level is None:
            return
        try:
            pres_stats = self._presenter.stats()
            vt_counts = None
            try:
                vt_counts = self._vt_pipeline.counts()
            except Exception:
                vt_counts = None
            # Also update a few gauges in the client metrics sink
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
        except Exception:
            logger.debug("stats logging failed", exc_info=True)

    def _init_decoder(self) -> None:
        import os as _os
        swap_rb = (_os.getenv('NAPARI_CUDA_CLIENT_SWAP_RB', '0') or '0') in ('1',)
        pf = (_os.getenv('NAPARI_CUDA_CLIENT_PIXEL_FMT', 'rgb24') or 'rgb24').lower()
        self.decoder = PyAVDecoder(self._stream_format, pixfmt=pf, swap_rb=swap_rb)

    def _init_vt_from_avcc(self, avcc_b64: str, width: int, height: int) -> None:
        import base64
        import sys
        try:
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
                    # Bind decoder to VT pipeline
                    self._vt_pipeline.set_decoder(self._vt_decoder)
                except Exception as e:
                    logger.warning("VT shim unavailable: %s; falling back to PyAV", e)
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
        except Exception as e:
            logger.error("VT live init failed: %s", e)

    def _request_keyframe_once(self) -> None:
        ch = self._state_channel
        if ch is not None:
            ch.request_keyframe_once()

    # Extracted helpers
    def _on_video_config(self, data: dict) -> None:
        w, h, fps, fmt, avcc_b64 = extract_video_config(data)
        if fps > 0:
            self._fps = fps
        self._stream_format = fmt
        self._stream_format_set = True
        if w > 0 and h > 0 and avcc_b64:
            # cache video dimensions for input mapping
            try:
                self._vid_w = int(w)
                self._vid_h = int(h)
            except Exception:
                self._vid_w = w
                self._vid_h = h
            self._init_vt_from_avcc(avcc_b64, w, h)

    def _handle_dims_update(self, data: dict) -> None:
        try:
            seq_raw = data.get('seq')
            try:
                seq_val = int(seq_raw) if seq_raw is not None else None
            except Exception:
                seq_val = None
            if seq_val is not None:
                self._last_dims_seq = seq_val
            cur = data.get('current_step')
            # Cache metadata and compute primary axis index
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
            intent_seq = data.get('intent_seq')
            try:
                self._dims_meta['ndim'] = int(ndim) if ndim is not None else self._dims_meta.get('ndim')
            except Exception:
                pass
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
            # New meta facets (volume/render/multiscale)
            try:
                if volume is not None:
                    self._dims_meta['volume'] = bool(volume)
            except Exception:
                pass
            if render is not None:
                self._dims_meta['render'] = render
            if multiscale is not None:
                self._dims_meta['multiscale'] = multiscale
            try:
                self._dims_meta['ndisplay'] = int(ndisp) if ndisp is not None else self._dims_meta.get('ndisplay')
            except Exception:
                pass
            # Mark ready on first dims.update with metadata
            if not self._dims_ready and (ndim is not None or order is not None):
                self._dims_ready = True
                logger.info("dims.update: metadata received; client intents enabled")
                self._notify_first_dims_ready()
            # Compute primary axis (first non-displayed axis)
            try:
                self._primary_axis_index = self._compute_primary_axis_index()
            except Exception:
                logger.debug("compute primary axis failed", exc_info=True)
            if isinstance(cur, (list, tuple)) and len(cur) > 0 and bool(getattr(self, '_log_dims_info', False)):
                try:
                    logger.info("dims_update: step=%s ndisp=%s order=%s labels=%s", list(cur), self._dims_meta.get('ndisplay'), self._dims_meta.get('order'), self._dims_meta.get('axis_labels'))
                except Exception:
                    pass
            # Mirror to local viewer UI if attached (including optional metadata)
            try:
                vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
            except Exception:
                vm_ref = None
            if vm_ref is not None:
                # Note: viewer mirror currently ignores volume/render/multiscale
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
                try:
                    seq_i = int(intent_seq)
                except Exception:
                    seq_i = None
                if seq_i is not None:
                    info = self._pending_intents.pop(seq_i, None)
                    if info is not None:
                        if ack_val is False:
                            logger.warning("dims_update ack=false for intent_seq=%s info=%s", seq_i, info)
                        elif bool(getattr(self, '_log_dims_info', False)):
                            logger.debug("dims_update ack: intent_seq=%s info=%s", seq_i, info)

            # Remember the latest payload for newly attached viewers.
            self._last_dims_payload = {
                'current_step': cur,
                'ndisplay': ndisp,
                'ndim': ndim,
                'dims_range': dims_range,
                'order': order,
                'axis_labels': axis_labels,
                'sizes': sizes,
                'displayed': displayed,
            }
        except Exception:
            logger.debug("dims_update parse failed", exc_info=True)

    def _on_dims_update(self, data: dict) -> None:
        self._handle_dims_update(data)

    def _notify_first_dims_ready(self) -> None:
        if self._first_dims_notified:
            return
        cb = self._first_dims_ready_cb
        if not callable(cb):
            self._first_dims_notified = True
            return

        def _invoke() -> None:
            try:
                cb()
            except Exception:
                logger.debug("first dims ready callback failed", exc_info=True)

        self._first_dims_notified = True
        proxy = getattr(self, '_ui_call', None)
        if proxy is not None:
            try:
                proxy.call.emit(_invoke)
                return
            except Exception:
                logger.debug("schedule first dims callback failed", exc_info=True)
        _invoke()

    def _record_pending_intent(self, seq: int, info: dict[str, object]) -> None:
        """Track in-flight intents so we can reconcile on server ACKs."""
        self._pending_intents[seq] = info

    def _replay_last_dims_payload(self) -> None:
        payload = self._last_dims_payload
        if not payload:
            return
        try:
            vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        except Exception:
            vm_ref = None
        if vm_ref is None:
            return
        try:
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
        except Exception:
            logger.debug("replay last dims payload failed", exc_info=True)

    def _on_scene_spec(self, msg: SceneSpecMessage) -> None:
        """Cache latest scene specification and forward to registry."""
        try:
            with self._scene_lock:
                self._latest_scene_spec = msg
        except Exception:
            logger.debug("scene.spec bookkeeping failed", exc_info=True)
        try:
            self._layer_registry.apply_scene(msg)
            logger.debug(
                "scene.spec received: %d layers, capabilities=%s",
                len(msg.scene.layers),
                msg.scene.capabilities,
            )
        except Exception:
            logger.debug("scene.spec registry apply failed", exc_info=True)

    def _on_layer_update(self, msg: LayerUpdateMessage) -> None:
        try:
            self._layer_registry.apply_update(msg)
            layer_id = msg.layer.layer_id if msg.layer else None
            logger.debug("layer.update: id=%s partial=%s", layer_id, msg.partial)
        except Exception:
            logger.debug("layer.update handling failed", exc_info=True)

    def _on_layer_remove(self, msg: LayerRemoveMessage) -> None:
        try:
            self._layer_registry.remove_layer(msg)
            logger.debug("layer.remove: id=%s reason=%s", msg.layer_id, msg.reason)
        except Exception:
            logger.debug("layer.remove handling failed", exc_info=True)

    def _on_registry_snapshot(self, snapshot: RegistrySnapshot) -> None:
        def _apply() -> None:
            self._sync_remote_layers(snapshot)
            self._replay_last_dims_payload()
            self._replay_last_dims_payload()

        ui_call = getattr(self, '_ui_call', None)
        if ui_call is not None:
            try:
                ui_call.call.emit(_apply)  # type: ignore[attr-defined]
                return
            except Exception:
                logger.debug("schedule remote layer sync failed", exc_info=True)
        try:
            _apply()
        except Exception:
            logger.debug("remote layer sync failed", exc_info=True)

    def _sync_remote_layers(self, snapshot: RegistrySnapshot) -> None:
        try:
            vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
        except Exception:
            vm_ref = None
        if vm_ref is None or not hasattr(vm_ref, '_sync_remote_layers'):
            return

        def _run() -> None:
            try:
                vm_ref._sync_remote_layers(snapshot)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("ProxyViewer remote layer sync failed", exc_info=True)

        # We are already on the GUI thread when called from _on_registry_snapshot
        _run()

    def _on_connected(self) -> None:
        self._init_decoder()
        dec = self.decoder.decode if self.decoder else None
        self._pyav_pipeline.set_decoder(dec)

    def _on_disconnect(self, exc: Exception | None) -> None:
        logger.info("PixelReceiver disconnected: %s", exc)

    # State channel lifecycle: gate dims intents safely across reconnects
    def _on_state_connected(self) -> None:
        try:
            # Safe option: require fresh dims.update before allowing intents
            self._dims_ready = False
            self._primary_axis_index = None
            logger.info("StateChannel connected; gating dims intents until dims.update meta arrives")
        except Exception:
            logger.debug("_on_state_connected failed", exc_info=True)

    def _on_state_disconnect(self, exc: Exception | None) -> None:
        try:
            self._dims_ready = False
            self._primary_axis_index = None
            self._pending_intents.clear()
            self._last_dims_seq = None
            self._last_dims_payload = None
            logger.info("StateChannel disconnected: %s; dims intents gated", exc)
        except Exception:
            logger.debug("_on_state_disconnect failed", exc_info=True)

    # --- Input mapping: unified wheel handler -------------------------------------
    def _on_wheel(self, data: dict) -> None:
        # Decide between dims.set (plain) and zoom (modifier)
        try:
            mods = int(data.get('mods') or 0)
        except Exception:
            mods = 0
        try:
            # Prefer explicit check for Ctrl/Meta/Alt; fallback to any modifier
            ctrl = int(getattr(QtCore.Qt, 'ControlModifier'))
            meta = int(getattr(QtCore.Qt, 'MetaModifier'))
            alt = int(getattr(QtCore.Qt, 'AltModifier'))
            use_zoom = (mods & (ctrl | meta | alt)) != 0
        except Exception:
            use_zoom = (mods != 0)
        if use_zoom:
            self._on_wheel_for_zoom(data)
        else:
            self._on_wheel_for_dims(data)

    # --- Input mapping: wheel -> dims.intent.step (primary axis) ---------------------
    def _on_wheel_for_dims(self, data: dict) -> None:
        try:
            ay = int(data.get('angle_y') or 0)
            py = int(data.get('pixel_y') or 0)
            mods = int(data.get('mods') or 0)
        except Exception:
            ay = 0; py = 0; mods = 0
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
        if getattr(self, '_log_dims_info', False):
            logger.info(
                "wheel->dims.intent.step d=%+d sent=%s", int(step), bool(sent)
            )
        else:
            logger.debug("wheel->dims.intent.step d=%+d sent=%s", int(step), bool(sent))

    # Shortcut-driven stepping via intents (arrows/page)
    def _step_primary(self, delta: int, origin: str = 'keys') -> None:
        try:
            dz = int(delta)
        except Exception:
            dz = 0
        if dz == 0:
            return
        _ = self.dims_step('primary', dz, origin=origin)

    # --- Camera ops: zoom/pan/reset -----------------------------------------------
    def _widget_to_video(self, xw: float, yw: float) -> tuple[float, float]:
        # Map widget pixel coordinates to video pixel coordinates assuming
        # the video is stretched to fill the widget (no letterboxing).
        try:
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
            # Clamp to [0, video_wh]
            xv = max(0.0, min(vw, xv))
            yv = max(0.0, min(vh, yv))
            return (xv, yv)
        except Exception:
            return (xw, yw)

    def _video_delta_to_canvas(self, dx_v: float, dy_v: float) -> tuple[float, float]:
        # Map video delta to canvas delta; make vertical drag direction feel natural
        try:
            return (float(dx_v), float(dy_v))
        except Exception:
            return (dx_v, dy_v)

    def _server_anchor_from_video(self, xv: float, yv: float) -> tuple[float, float]:
        # Flip Y for bottom-left canvas origin expected by server
        try:
            vh = float(self._vid_h or 0)
            return (float(xv), float(max(0.0, min(vh, vh - yv))))
        except Exception:
            return (xv, yv)

    def _zoom_steps_at_center(self, steps: int) -> None:
        ch = self._state_channel
        if ch is None:
            return
        try:
            base = float(os.getenv('NAPARI_CUDA_ZOOM_BASE', '1.1') or '1.1')
        except Exception:
            base = 1.1
        try:
            s = int(steps)
        except Exception:
            s = 0
        if s == 0:
            return
        # '+' zooms in, '-' zooms out
        f = base ** (-s)
        # Use cursor position if available; otherwise fall back to view center.
        cursor_xw = getattr(self, '_cursor_wx', None)
        cursor_yw = getattr(self, '_cursor_wy', None)
        if isinstance(cursor_xw, (int, float)) and isinstance(cursor_yw, (int, float)):
            xv, yv = self._widget_to_video(float(cursor_xw), float(cursor_yw))
        else:
            cx = float(self._vid_w or 0) / 2.0
            cy = float(self._vid_h or 0) / 2.0
            xv, yv = cx, cy
        ax_s, ay_s = self._server_anchor_from_video(xv, yv)
        ok = ch.send_json({'type': 'camera.zoom_at', 'factor': float(f), 'anchor_px': [float(ax_s), float(ay_s)]})
        if getattr(self, '_log_dims_info', False):
            logger.info(
                "key->camera.zoom_at f=%.4f at(%.1f,%.1f) sent=%s",
                float(f), float(ax_s), float(ay_s), bool(ok)
            )

    def _reset_camera(self) -> None:
        ch = self._state_channel
        if ch is None:
            return
        logger.info("key->camera.reset (sending)")
        ok = ch.send_json({'type': 'camera.reset'})
        logger.info("key->camera.reset sent=%s", bool(ok))

    def _on_key_event(self, data: dict) -> bool:
        """Optional app-level key handling for bindings that struggle as QShortcut.

        Triggers camera reset on plain '0' with no modifiers. Also accepts
        keypad 0 when the only modifier is the keypad flag.
        """
        try:
            from qtpy import QtCore as _QtCore  # type: ignore
        except Exception:
            return
        try:
            key = int(data.get('key'))
        except Exception:
            key = -1
        try:
            mods = int(data.get('mods') or 0)
        except Exception:
            mods = 0
        txt = str(data.get('text') or '')
        # Accept if explicitly '0' text with no modifiers
        if txt == '0' and mods == 0:
            logger.info("keycb: '0' -> camera.reset")
            self._reset_camera()
            return True
        # Or if the key is Key_0 and modifiers are none (or just keypad)
        try:
            keypad_only = (mods & ~int(_QtCore.Qt.KeypadModifier)) == 0 and (mods & int(_QtCore.Qt.KeypadModifier)) != 0  # type: ignore[attr-defined]
        except Exception:
            keypad_only = False
        if key == int(getattr(_QtCore.Qt, 'Key_0', 0)) and (mods == 0 or keypad_only):  # type: ignore[attr-defined]
            logger.info("keycb: Key_0 -> camera.reset")
            self._reset_camera()
            return True
        # Map arrows and page keys to primary-axis stepping
        try:
            k_left = int(getattr(_QtCore.Qt, 'Key_Left'))
            k_right = int(getattr(_QtCore.Qt, 'Key_Right'))
            k_up = int(getattr(_QtCore.Qt, 'Key_Up'))
            k_down = int(getattr(_QtCore.Qt, 'Key_Down'))
            k_pgup = int(getattr(_QtCore.Qt, 'Key_PageUp'))
            k_pgdn = int(getattr(_QtCore.Qt, 'Key_PageDown'))
        except Exception:
            k_left = k_right = k_up = k_down = k_pgup = k_pgdn = -1
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
        try:
            self._viewer_mirror = weakref.ref(viewer)  # type: ignore[attr-defined]
        except Exception:
            self._viewer_mirror = None
        self._replay_last_dims_payload()

    def send_json(self, obj: dict) -> bool:
        ch = self._state_channel
        return bool(ch.send_json(obj)) if ch is not None else False

    def reset_camera(self, origin: str = 'ui') -> bool:
        """Send a camera.reset to the server (used by UI bindings)."""
        ch = self._state_channel
        if ch is None:
            return False
        try:
            logger.info("%s->camera.reset (sending)", origin)
        except Exception:
            pass
        ok = ch.send_json({'type': 'camera.reset'})
        try:
            logger.info("%s->camera.reset sent=%s", origin, bool(ok))
        except Exception:
            pass
        return bool(ok)

    def set_camera(self, *, center=None, zoom=None, angles=None, origin: str = 'ui') -> bool:
        """Send absolute camera fields when provided.

        Prefer using zoom_at/pan_px/reset ops for interactions; this is
        intended for explicit UI actions that set a known state.
        """
        ch = self._state_channel
        if ch is None:
            return False
        payload: dict = {'type': 'set_camera'}
        if center is not None:
            try:
                payload['center'] = list(center)
            except Exception:
                payload['center'] = center
        if zoom is not None:
            payload['zoom'] = float(zoom)
        if angles is not None:
            try:
                payload['angles'] = list(angles)
            except Exception:
                payload['angles'] = angles
        try:
            logger.info("%s->set_camera %s", origin, {k: v for k, v in payload.items() if k != 'type'})
        except Exception:
            pass
        ok = ch.send_json(payload)
        return bool(ok)

    # --- Intents API --------------------------------------------------------------
    def _axis_to_index(self, axis: int | str) -> Optional[int]:
        try:
            if axis == 'primary':
                return int(self._primary_axis_index) if self._primary_axis_index is not None else 0
            if isinstance(axis, (int, float)) or (isinstance(axis, str) and str(axis).isdigit()):
                return int(axis)
            # Map label to index via axis_labels
            labels = self._dims_meta.get('axis_labels') or []
            if isinstance(labels, (list, tuple)):
                try:
                    return int({str(lbl): i for i, lbl in enumerate(labels)}.get(str(axis)))
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def _compute_primary_axis_index(self) -> Optional[int]:
        order = self._dims_meta.get('order')
        ndisplay = self._dims_meta.get('ndisplay')
        labels = self._dims_meta.get('axis_labels')
        try:
            nd = int(ndisplay) if ndisplay is not None else 2
        except Exception:
            nd = 2
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
        try:
            vol = bool(self._dims_meta.get('volume'))
        except Exception:
            vol = False
        try:
            nd = int(self._dims_meta.get('ndisplay') or 2)
        except Exception:
            nd = 2
        return bool(vol) and int(nd) == 3

    # --- Small utilities (no behavior change) ----------------------------------
    def _clamp01(self, a: float) -> float:
        try:
            a = float(a)
        except Exception:
            a = 1.0
        if a < 0.0:
            return 0.0
        if a > 1.0:
            return 1.0
        return a

    def _clamp_sample_step(self, r: float) -> float:
        try:
            r = float(r)
        except Exception:
            r = 1.0
        if r < 0.1:
            return 0.1
        if r > 4.0:
            return 4.0
        return r

    def _ensure_lo_hi(self, lo: float, hi: float) -> tuple[float, float]:
        try:
            lo_f = float(lo)
        except Exception:
            lo_f = 0.0
        try:
            hi_f = float(hi)
        except Exception:
            hi_f = lo_f + 1.0
        if hi_f <= lo_f:
            lo_f, hi_f = hi_f, lo_f
        return lo_f, hi_f

    def _clamp_level(self, level: int) -> int:
        try:
            ms = self._dims_meta.get('multiscale') or {}
            levels = ms.get('levels') if isinstance(ms, dict) else None
            if isinstance(levels, (list, tuple)) and len(levels) > 0:
                lo, hi = 0, len(levels) - 1
                lv = int(level)
                if lv < lo:
                    return lo
                if lv > hi:
                    return hi
                return lv
            return int(level)
        except Exception:
            return int(level) if isinstance(level, (int, float)) else 0

    def _send_intent(self, type_str: str, fields: dict, origin: str) -> bool:
        ch = self._state_channel
        if ch is None:
            return False
        payload = {'type': type_str}
        try:
            payload.update(fields)
        except Exception:
            # Best-effort
            for k, v in (fields or {}).items():
                payload[k] = v
        payload['client_id'] = self._client_id
        payload['client_seq'] = self._next_client_seq()
        payload['origin'] = str(origin)
        ok = ch.send_json(payload)
        try:
            # Log without the 'type' field for brevity
            fields_to_log = {k: v for k, v in payload.items() if k not in ('type', 'client_id', 'client_seq', 'origin')}
            logger.info("%s->%s %s sent=%s", origin, type_str, fields_to_log, bool(ok))
        except Exception:
            pass
        return bool(ok)

    # --- View HUD snapshot (for overlay) ----------------------------------------
    def view_hud_snapshot(self) -> dict:
        """Return a compact snapshot of 3D view/volume tuning state.

        Safe to call from GUI timer; avoids raising on missing fields.
        """
        snap: dict = {}
        try:
            snap['ndisplay'] = int(self._dims_meta.get('ndisplay') or 0)
        except Exception:
            snap['ndisplay'] = None
        try:
            snap['volume'] = bool(self._dims_meta.get('volume'))
        except Exception:
            snap['volume'] = None
        snap['vol_mode'] = bool(self._is_volume_mode())
        # Render state
        r = self._dims_meta.get('render') or {}
        if isinstance(r, dict):
            snap['render_mode'] = r.get('mode')
            try:
                clim = r.get('clim') or []
                snap['clim_lo'] = float(clim[0]) if len(clim) > 0 else None
                snap['clim_hi'] = float(clim[1]) if len(clim) > 1 else None
            except Exception:
                snap['clim_lo'] = snap['clim_hi'] = None
            snap['colormap'] = r.get('colormap')
            try:
                snap['opacity'] = float(r.get('opacity')) if r.get('opacity') is not None else None
            except Exception:
                snap['opacity'] = None
            try:
                snap['sample_step'] = float(r.get('sample_step')) if r.get('sample_step') is not None else None
            except Exception:
                snap['sample_step'] = None
        # Multiscale
        ms = self._dims_meta.get('multiscale') or {}
        if isinstance(ms, dict):
            snap['ms_policy'] = ms.get('policy')
            try:
                snap['ms_level'] = int(ms.get('current_level')) if ms.get('current_level') is not None else None
            except Exception:
                snap['ms_level'] = None
            levels_obj = ms.get('levels')
            total_levels: Optional[int] = None
            if isinstance(levels_obj, (list, tuple)):
                total_levels = len(levels_obj)
            snap['ms_levels'] = int(total_levels) if total_levels is not None else None
            snap['ms_path'] = None
            if (
                isinstance(levels_obj, (list, tuple))
                and snap.get('ms_level') is not None
                and 0 <= int(snap['ms_level']) < len(levels_obj)
            ):
                entry = levels_obj[int(snap['ms_level'])]
                if isinstance(entry, dict):
                    path = entry.get('path')
                    if isinstance(path, str) and path:
                        snap['ms_path'] = path
        # Primary axis
        try:
            snap['primary_axis'] = int(self._primary_axis_index) if self._primary_axis_index is not None else None
        except Exception:
            snap['primary_axis'] = None
        # Zoom/pan last actions
        snap['last_zoom_factor'] = self._last_zoom_factor
        snap['last_zoom_widget_px'] = self._last_zoom_widget_px
        snap['last_zoom_video_px'] = self._last_zoom_video_px
        snap['last_zoom_anchor_px'] = self._last_zoom_anchor_px
        snap['last_pan_dx'] = self._last_pan_dx_sent
        snap['last_pan_dy'] = self._last_pan_dy_sent
        # Video size and zoom base
        try:
            snap['video_w'] = int(self._vid_w) if self._vid_w is not None else None
            snap['video_h'] = int(self._vid_h) if self._vid_h is not None else None
        except Exception:
            snap['video_w'] = self._vid_w; snap['video_h'] = self._vid_h
        try:
            import os as _os
            snap['zoom_base'] = float(_os.getenv('NAPARI_CUDA_ZOOM_BASE', '1.1') or '1.1')
        except Exception:
            snap['zoom_base'] = None
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
        def _apply() -> None:  # type: ignore[no-redef]
            vm_ref._apply_remote_dims_update(  # type: ignore[attr-defined]
                current_step=cur,
                ndisplay=ndisplay,
                ndim=ndim,
                dims_range=dims_range,
                order=order,
                axis_labels=axis_labels,
                sizes=sizes,
                displayed=displayed,
            )
        ui_call = getattr(self, '_ui_call', None)
        if ui_call is not None:
            try:
                ui_call.call.emit(_apply)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("schedule viewer mirror dims update failed", exc_info=True)
        else:
            try:
                _apply()
            except Exception:
                logger.debug("viewer mirror dims update failed", exc_info=True)

    def dims_step(self, axis: int | str, delta: int, *, origin: str = 'ui') -> bool:
        if not self._dims_ready:
            return False
        idx = self._axis_to_index(axis)
        if idx is None:
            return False
        # Suppress local inputs on the playing axis while napari is animating it
        try:
            vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
            if vm_ref is not None:
                is_playing = bool(getattr(vm_ref, '_is_playing', False))
                play_axis = getattr(vm_ref, '_play_axis', None)
                if is_playing and play_axis is not None and int(play_axis) == int(idx) and origin != 'play':
                    return False
        except Exception:
            logger.debug("dims_step: play-axis suppression probe failed", exc_info=True)
        now = time.perf_counter()
        if (now - float(self._last_dims_send or 0.0)) < self._dims_min_dt:
            logger.debug("dims.intent.step gated by rate limiter (%s)", origin)
            return False
        ch = self._state_channel
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
        ok = ch.send_json(payload)
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
        try:
            vm_ref = self._viewer_mirror() if callable(self._viewer_mirror) else None  # type: ignore[misc]
            if vm_ref is not None:
                is_playing = bool(getattr(vm_ref, '_is_playing', False))
                play_axis = getattr(vm_ref, '_play_axis', None)
                if is_playing and play_axis is not None and int(play_axis) == int(idx) and origin != 'play':
                    return False
        except Exception:
            logger.debug("dims_set_index: play-axis suppression probe failed", exc_info=True)
        now = time.perf_counter()
        if (now - float(self._last_dims_send or 0.0)) < self._dims_min_dt:
            # Allow coalescing on caller side; treat as not sent
            logger.debug("dims.intent.set_index gated by rate limiter (%s)", origin)
            return False
        ch = self._state_channel
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
        ok = ch.send_json(payload)
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

    def _on_wheel_for_zoom(self, data: dict) -> None:
        ch = self._state_channel
        if ch is None:
            return
        try:
            ay = float(data.get('angle_y') or 0.0)
            py = float(data.get('pixel_y') or 0.0)
            xw = float(data.get('x_px') or 0.0)
            yw = float(data.get('y_px') or 0.0)
        except Exception:
            ay = 0.0; py = 0.0; xw = yw = 0.0
        try:
            base = float(os.getenv('NAPARI_CUDA_ZOOM_BASE', '1.1') or '1.1')
        except Exception:
            base = 1.1
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
        try:
            self._last_zoom_factor = float(factor)
            self._last_zoom_widget_px = (float(xw), float(yw))
            self._last_zoom_video_px = (float(xv), float(yv))
            self._last_zoom_anchor_px = (float(ax), float(ay))
        except Exception:
            logger.debug("zoom HUD stash failed", exc_info=True)
        ok = ch.send_json({'type': 'camera.zoom_at', 'factor': float(factor), 'anchor_px': [float(ax), float(ay)]})
        if getattr(self, '_log_dims_info', False):
            logger.info("wheel+mod->camera.zoom_at f=%.4f at(%.1f,%.1f) sent=%s", float(factor), float(ax), float(ay), bool(ok))

    def _on_pointer(self, data: dict) -> None:
        phase = (data.get('phase') or '').lower()
        # Parse positions with specific exception handling
        xw_raw = data.get('x_px')
        yw_raw = data.get('y_px')
        try:
            xw = float(xw_raw) if xw_raw is not None else 0.0
        except (TypeError, ValueError):
            logger.debug("pointer x_px parse failed: %r", xw_raw, exc_info=True)
            xw = 0.0
        try:
            yw = float(yw_raw) if yw_raw is not None else 0.0
        except (TypeError, ValueError):
            logger.debug("pointer y_px parse failed: %r", yw_raw, exc_info=True)
            yw = 0.0
        # Track latest cursor position regardless of drag
        self._cursor_wx = xw
        self._cursor_wy = yw
        # Track latest cursor position regardless of drag
        try:
            self._cursor_wx = float(xw)
            self._cursor_wy = float(yw)
        except Exception:
            pass
        # Detect Alt modifier (for orbit)
        try:
            mods = int(data.get('mods') or 0)
        except Exception:
            mods = 0
        try:
            alt_mask = int(getattr(QtCore.Qt, 'AltModifier'))
            alt = (mods & alt_mask) != 0
        except Exception:
            alt = (mods != 0)
        in_vol3d = bool(self._is_volume_mode()) and int(self._dims_meta.get('ndisplay') or 2) == 3

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
            if getattr(self, '_log_dims_info', False):
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
                    try:
                        self._flush_orbit(force=True)
                    except Exception:
                        logger.debug("flush orbit (release) failed", exc_info=True)
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
                try:
                    self._flush_orbit(force=True)
                except Exception:
                    logger.debug("flush orbit (up) failed", exc_info=True)
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
        ch = self._state_channel
        if ch is None:
            self._pan_dx_accum = 0.0
            self._pan_dy_accum = 0.0
            return
        ok = ch.send_json({'type': 'camera.pan_px', 'dx_px': float(dx), 'dy_px': float(dy)})
        # Stash for HUD
        try:
            self._last_pan_dx_sent = float(dx)
            self._last_pan_dy_sent = float(dy)
        except Exception:
            logger.debug("pan HUD stash failed", exc_info=True)
        self._last_cam_send = time.perf_counter()
        self._pan_dx_accum = 0.0
        self._pan_dy_accum = 0.0
        if getattr(self, '_log_dims_info', False):
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
        ch = self._state_channel
        if ch is None:
            self._orbit_daz_accum = 0.0
            self._orbit_del_accum = 0.0
            return
        ok = ch.send_json({'type': 'camera.orbit', 'd_az_deg': float(daz), 'd_el_deg': float(delv)})
        self._last_cam_send = time.perf_counter()
        self._orbit_daz_accum = 0.0
        self._orbit_del_accum = 0.0
        if getattr(self, '_log_dims_info', False):
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
                    self._vt_pipeline.clear()
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
                try:
                    # Compute wall->mono offset once at gate lift
                    self._wall_to_mono = float(self._mono_at_gate - float(self._vt_gate_lift_time) - float(self._server_bias_s))
                    self._vt_ts_offset = float(self._vt_gate_lift_time - float(pkt.ts)) - float(self._server_bias_s)
                except Exception:
                    self._wall_to_mono = None
                self._source_mux.set_active(Source.VT)
                # Use wall->mono offset for monotonic scheduling
                self._presenter.set_offset(self._wall_to_mono)
                try:
                    self._presenter.set_latency(self._vt_latency_s)
                except Exception:
                    logger.debug("Presenter set_latency restore for VT failed", exc_info=True)
                self._presenter.clear(Source.PYAV)
                logger.info("VT gate lifted on keyframe (seq=%d); presenter=VT", cur)
                self._disco_gated = False
                try:
                    if self._warmup_window_s > 0:
                        if self._warmup_ms_override:
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
                except Exception:
                    logger.debug("Warmup latency set failed", exc_info=True)
                # Schedule first wake after VT becomes active (thread-safe)
                try:
                    self.schedule_next_wake_threadsafe()
                except Exception:
                    logger.debug("schedule_next_wake after VT gate failed", exc_info=True)
            else:
                self._request_keyframe_once()
                return
        if self._vt_decoder is not None and not self._vt_wait_keyframe:
            ts_float = float(pkt.ts)
            b = pkt.payload
            try:
                self._vt_pipeline.enqueue(b, ts_float)
                self._vt_enqueued += 1
            except Exception:
                logger.debug("VT pipeline enqueue failed", exc_info=True)
        else:
            try:
                ts_float = float(pkt.ts)
            except Exception:
                ts_float = None
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

    def _start_stats_timer(self) -> None:
        try:
            self._stats_timer = QtCore.QTimer(self._scene_canvas.native)
            self._stats_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self._stats_timer.setInterval(1000)
            self._stats_timer.timeout.connect(self._log_stats)
            self._stats_timer.start()
        except Exception:
            logger.debug("Failed to start stats timer; falling back to draw-based logging", exc_info=True)

    def _start_metrics_timer(self) -> None:
        try:
            interval_s = env_float('NAPARI_CUDA_CLIENT_METRICS_INTERVAL', 1.0)
            interval_ms = max(100, int(round(1000.0 * max(0.1, float(interval_s)))))
            self._metrics_timer = QtCore.QTimer(self._scene_canvas.native)
            self._metrics_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self._metrics_timer.setInterval(interval_ms)
            self._metrics_timer.timeout.connect(self._metrics.dump_csv_row)  # type: ignore[arg-type]
            self._metrics_timer.start()
        except Exception:
            logger.debug("Failed to start client metrics timer", exc_info=True)

    def stop(self) -> None:
        try:
            if self._stats_timer is not None:
                self._stats_timer.stop()
        except Exception:
            logger.debug("stop: stats timer stop failed", exc_info=True)
        try:
            if self._metrics_timer is not None:
                self._metrics_timer.stop()
        except Exception:
            logger.debug("stop: metrics timer stop failed", exc_info=True)
        try:
            if self._warmup_reset_timer is not None:
                self._warmup_reset_timer.stop()
        except Exception:
            logger.debug("stop: warmup reset timer stop failed", exc_info=True)
        try:
            if self._watchdog_timer is not None:
                self._watchdog_timer.stop()
        except Exception:
            logger.debug("stop: watchdog timer stop failed", exc_info=True)
        try:
            if self._evloop_mon is not None:
                self._evloop_mon.stop()
        except Exception:
            logger.debug("stop: event loop monitor stop failed", exc_info=True)

    # VT decode/submit is handled by VTPipeline

    # Keyframe detection via shared helpers (AnnexB/AVCC)
    def _is_keyframe(self, payload: bytes | memoryview, codec: int) -> bool:
        try:
            hevc = int(codec) == 2
        except Exception:
            hevc = False
        if is_annexb(payload):
            return contains_idr_annexb(payload, hevc=hevc)
        # Use parsed nal_length_size when available (defaults to 4)
        nsz = int(self._nal_length_size or 4)
        return contains_idr_avcc(payload, nal_len_size=nsz, hevc=hevc)

# Coordinator end
