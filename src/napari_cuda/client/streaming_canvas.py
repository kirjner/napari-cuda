"""
StreamingCanvas - Displays video stream from remote server.

This replaces the normal VispyCanvas with one that shows decoded video frames
instead of locally rendered content.
"""

import contextlib
import logging
import os
import queue
from threading import Thread
import time

import numpy as np
from qtpy import QtCore, QtWidgets
from vispy import app as vispy_app

from napari._vispy.canvas import VispyCanvas

# Silence VisPy warnings about copying discontiguous data
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# No direct networking or shader setup here; StreamCoordinator + GLRenderer own these


from napari_cuda.utils.env import env_bool, env_int, env_float
from napari_cuda.client.streaming.config import ClientConfig


class StreamingCanvas(VispyCanvas):
    """
    Canvas that displays video stream from remote server instead of
    rendering local content.
    """

    def __init__(
        self,
        viewer,
        server_host='localhost',
        server_port=8082,
        vt_smoke: bool = False,
        key_map_handler=None,
        state_port: int | None = None,
        **kwargs,
    ):
        """
        Initialize streaming canvas.

        Parameters
        ----------
        viewer : ProxyViewer
            The proxy viewer instance
        server_host : str
            Remote server hostname
        server_port : int
            Remote server pixel stream port
        """
        # Optional: control vsync via env before any GL surface is created
        try:
            vs_env = (os.getenv('NAPARI_CUDA_VSYNC') or '').strip().lower()
            if vs_env:
                from qtpy import QtGui  # type: ignore
                fmt = QtGui.QSurfaceFormat()
                if vs_env in ('0', 'off', 'false', 'no'):
                    fmt.setSwapInterval(0)
                    logger.info('GL vsync disabled via NAPARI_CUDA_VSYNC=0')
                else:
                    fmt.setSwapInterval(1)
                    logger.info('GL vsync enabled via NAPARI_CUDA_VSYNC=1')
                QtGui.QSurfaceFormat.setDefaultFormat(fmt)
        except Exception:
            logger.debug('VSync default format setup failed', exc_info=True)

        # Ensure we have a KeymapHandler; create a minimal one if not provided
        if key_map_handler is None:
            try:
                from napari.utils.key_bindings import (
                    KeymapHandler,  # type: ignore
                )

                key_map_handler = KeymapHandler()
                key_map_handler.keymap_providers = [viewer]
            except Exception:
                logger.debug(
                    'KeymapHandler unavailable; using dummy handler',
                    exc_info=True,
                )

                # Fallback to a dummy object with required attributes
                class _DummyKM:
                    def on_key_press(self, *a, **k):
                        pass

                    def on_key_release(self, *a, **k):
                        pass

                key_map_handler = _DummyKM()  # type: ignore
        # Forward to base canvas
        super().__init__(viewer, key_map_handler, **kwargs)

        self.server_host = server_host
        self.server_port = server_port
        self.state_port = int(
            state_port or int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081'))
        )
        # Offline VT smoke test mode (no server required)
        env_smoke = env_bool('NAPARI_CUDA_SMOKE', False)
        self._vt_smoke = bool(vt_smoke or env_smoke)

        # Queue for decoded frames (latest-wins draining in draw)
        buf_n = env_int('NAPARI_CUDA_CLIENT_BUFFER_FRAMES', 3)
        self.frame_queue = queue.Queue(maxsize=max(1, buf_n))

        # Video display resources
        self._video_texture = None
        self._video_program = None
        self._fullscreen_quad = None

        # Decoder (will be initialized based on stream type)
        self.decoder = None
        # VT live-decoder state
        self._vt_decoder = None
        self._vt_cfg_key = None
        self._vt_errors = 0
        # Gate VT decode until we see a keyframe after (re)initialization
        self._vt_wait_keyframe = False
        self._vt_gate_lift_time: float | None = None
        # Active presentation source ('pyav' or 'vt') to prevent cross-talk
        self._active_source: str = 'pyav'
        # VT presenter jitter buffer + latency target
        self._vt_latency_s = max(
            0.0, env_float('NAPARI_CUDA_CLIENT_VT_LATENCY_MS', 0.0) / 1000.0
        )
        # If not explicitly set, derive a sane default from latency and a 60 Hz
        # output cadence: roughly ceil(latency * 60) + 2 frames. This avoids
        # trimming not-yet-due frames in SERVER mode with higher latency.
        try:
            buf_env = os.getenv('NAPARI_CUDA_CLIENT_VT_BUFFER')
            if buf_env is None or buf_env.strip() == '':
                import math
                derived = max(3, int(math.ceil(self._vt_latency_s * 60.0)) + 2)
                self._vt_buffer_limit = derived
            else:
                self._vt_buffer_limit = max(1, int(buf_env))
        except Exception:
            self._vt_buffer_limit = 3
        # Higher latency when falling back to PyAV (smoother on CPU decode)
        self._pyav_latency_s = max(
            0.0,
            env_float(
                'NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS',
                max(50.0, self._vt_latency_s * 1000.0),
            )
            / 1000.0,
        )
        # Presenter + source mux
        from napari_cuda.client.streaming.presenter import (
            FixedLatencyPresenter,
            SourceMux,
        )
        from napari_cuda.client.streaming.types import Source

        self._presenter = FixedLatencyPresenter(
            latency_s=self._vt_latency_s,
            buffer_limit=self._vt_buffer_limit,
            preview_guard_s=0.0,
        )
        self._source_mux = SourceMux(Source.PYAV)
        assert self._presenter is not None
        assert self._source_mux is not None
        # Construct minimal client config (phase 0; no behavior change)
        try:
            # Estimate display FPS early from env to avoid NameError before DisplayLoop init
            fps_guess = env_float('NAPARI_CUDA_CLIENT_DISPLAY_FPS', env_float('NAPARI_CUDA_SMOKE_FPS', 60.0))
            self._client_cfg = ClientConfig.from_env(
                default_latency_ms=self._vt_latency_s * 1000.0,
                default_buffer_limit=self._vt_buffer_limit,
                default_draw_fps=fps_guess,
            )
            logger.info(
                "ClientConfig initialized: latency=%.0fms buf=%d draw_fps=%.1f preview_guard=%.1fms",
                self._client_cfg.base_latency_ms,
                self._client_cfg.buffer_limit,
                self._client_cfg.draw_fps,
                self._client_cfg.preview_guard_ms,
            )
        except Exception:
            logger.debug("ClientConfig init failed", exc_info=True)
        # VT diagnostics
        self._vt_last_stats_log: float = 0.0
        self._vt_last_submit_count: int = 0
        self._vt_last_out_count: int = 0
        # VT submission decoupling: queue + worker to avoid blocking asyncio recv loop
        # Larger input queue to avoid backpressure on websocket when VT is momentarily slow
        self._vt_in_q: queue.Queue[tuple[bytes, float|None]] = queue.Queue(
            maxsize=64
        )
        self._vt_enqueued = 0
        # Backlog handling: if queue builds up, resync on next keyframe to avoid smear
        self._vt_backlog_trigger = env_int(
            'NAPARI_CUDA_CLIENT_VT_BACKLOG_TRIGGER', 16
        )
        # Workers are managed by StreamCoordinator when enabled

        # Expected bitstream format from server ('avcc' or 'annexb'); default to AVCC
        self._stream_format = 'avcc'
        self._stream_format_set = False

        # PyAV decode decoupling
        self._pyav_in_q: queue.Queue[tuple[bytes, float|None]] = queue.Queue(
            maxsize=64
        )
        self._pyav_enqueued = 0
        self._pyav_backlog_trigger = env_int(
            'NAPARI_CUDA_CLIENT_PYAV_BACKLOG_TRIGGER', 16
        )
        # PyAV worker is managed by StreamCoordinator when enabled

        # Orchestrate with StreamCoordinator (it handles vt_smoke internally)
        self._use_manager = True
        if self._use_manager:
            from napari_cuda.client.streaming import StreamCoordinator

            self._manager = StreamCoordinator(
                scene_canvas=self._scene_canvas,
                server_host=self.server_host,
                server_port=self.server_port,
                state_port=self.state_port,
                vt_latency_s=self._vt_latency_s,
                pyav_latency_s=self._pyav_latency_s,
                vt_buffer_limit=self._vt_buffer_limit,
                stream_format=self._stream_format,
                vt_backlog_trigger=self._vt_backlog_trigger,
                pyav_backlog_trigger=env_int(
                    'NAPARI_CUDA_CLIENT_PYAV_BACKLOG_TRIGGER', 16
                ),
                vt_smoke=self._vt_smoke,
                client_cfg=getattr(self, '_client_cfg', None),
            )
            self._manager.start()
            # Bridge coordinator with ProxyViewer for unified state path
            try:
                self._manager.attach_viewer_mirror(viewer)
                if hasattr(viewer, 'attach_state_sender'):
                    viewer.attach_state_sender(self._manager)  # type: ignore[attr-defined]
            except Exception:
                logger.debug('StreamingCanvas: failed to attach viewer/state sender bridge', exc_info=True)
            # Optional fixed-cadence display loop to stabilize draw phase under jitter
            try:
                import os as _os
                use_disp_loop = (_os.getenv('NAPARI_CUDA_USE_DISPLAY_LOOP', '0') or '0').lower() in ('1','true','yes','on')
            except Exception:
                use_disp_loop = False
            if use_disp_loop:
                try:
                    from napari_cuda.client.streaming.display_loop import DisplayLoop as _DisplayLoop
                    fps = None
                    try:
                        # Prefer configured draw_fps from ClientConfig
                        fps = float(getattr(self, '_client_cfg', None).draw_fps)  # type: ignore[union-attr]
                    except Exception:
                        fps = None
                    self._display_loop = _DisplayLoop(scene_canvas=self._scene_canvas, fps=fps, prefer_vispy=True)
                    self._display_loop.start()
                    logger.info("DisplayLoop enabled (vispy timer preferred) for steady 60 Hz cadence")
                except Exception:
                    logger.exception("Failed to start DisplayLoop")
        else:
            pass

        # Override draw to show video instead, but keep play-enable hook intact
        try:
            # Detach only the base on_draw; keep other listeners (e.g., enable_dims_play)
            self._scene_canvas.events.draw.disconnect(self.on_draw)
        except Exception:
            pass
        try:
            # Ensure dims play continues to tick
            self._scene_canvas.events.draw.connect(self.enable_dims_play, position='first')
        except Exception:
            pass
        # Connect our video draw after play-enable so _play_ready toggles properly
        self._scene_canvas.events.draw.connect(self._draw_video_frame, position='last')
        # DisplayLoop removed: presenter-owned wake schedules repaint on demand

        logger.info(
            f'StreamingCanvas initialized for {server_host}:{server_port}'
        )

        # Optional FPS/metrics HUD overlay.
        # Enable when:
        #  - smoke mode is active, or
        #  - NAPARI_CUDA_FPS_HUD=1, or
        #  - client metrics are enabled (via --metrics or NAPARI_CUDA_CLIENT_METRICS=1)
        try:
            hud_env = os.getenv('NAPARI_CUDA_FPS_HUD')
            view_hud_env = (os.getenv('NAPARI_CUDA_VIEW_HUD') or '0').lower() in ('1','true','yes','on')
            metrics_env = (os.getenv('NAPARI_CUDA_CLIENT_METRICS', '0') or '0').lower() in ('1', 'true', 'yes', 'on')
            # Enable the overlay when FPS HUD requested, metrics enabled, smoke mode, or view HUD requested
            self._hud_enabled = (hud_env == '1') or bool(self._vt_smoke) or bool(metrics_env) or bool(view_hud_env)
            if self._hud_enabled:
                self._init_fps_hud()
        except Exception:
            logger.debug('FPS/VIEW HUD init failed', exc_info=True)

        # Timestamp handling for VT scheduling (server timestamps only)
        self._vt_ts_offset = None  # server_ts -> local_now offset (seconds)
        # Keyframe request throttling while VT waits for sync
        self._vt_last_key_req: float | None = None

        # VT stats logging level control (default: disabled)
        stats_env = (os.getenv('NAPARI_CUDA_VT_STATS') or '').lower()
        if stats_env in ('1', 'true', 'yes', 'info'):
            self._vt_stats_level = logging.INFO
        elif stats_env in ('debug', 'dbg'):
            self._vt_stats_level = logging.DEBUG
        else:
            self._vt_stats_level = None

    # No state-channel helpers here; StreamCoordinator owns StateChannel

    def _init_vt_from_avcc(
        self, avcc_b64: str, width: int, height: int
    ) -> None:
        try:
            avcc = base64.b64decode(avcc_b64)
            cfg_key = (int(width), int(height), avcc)
            if self._vt_decoder is not None and self._vt_cfg_key == cfg_key:
                logger.debug(
                    'VT already initialized; ignoring duplicate video_config'
                )
                return
            # Prefer native shim on macOS; fallback to PyAV only (PyObjC VT retired)
            backend = (
                os.getenv('NAPARI_CUDA_VT_BACKEND', 'shim') or 'shim'
            ).lower()
            self._vt_backend = None
            if sys.platform == 'darwin' and backend != 'off':
                try:
                    from napari_cuda.client.streaming.decoders.vt import (
                        VTLiveDecoder,
                    )  # type: ignore

                    self._vt_decoder = VTLiveDecoder(avcc, width, height)
                    self._vt_backend = 'shim'
                except Exception as e:
                    logger.warning(
                        'VT shim unavailable: %s; falling back to PyAV', e
                    )
                    self._vt_decoder = None
            self._vt_cfg_key = cfg_key
            # Require a fresh keyframe before decoding with VT to ensure sync
            self._vt_wait_keyframe = True
            logger.info(
                'VideoToolbox live decoder initialized: %dx%d', width, height
            )
        except Exception as e:
            logger.exception('VT live init failed: %s', e)

    # Legacy VT smoke worker removed; StreamCoordinator owns smoke modes now

    def _stream_worker(self):
        """Background thread to receive and decode video stream via PixelReceiver."""
        # No direct stream worker here; StreamCoordinator owns the receiver

    # No local decoder init; StreamCoordinator owns decoders

    # No local AnnexB→AVCC conversion; handled in StreamCoordinator/decoders

    # No local VT live decoding; handled in StreamCoordinator

    # No local VT pixel buffer mapping here; handled in StreamCoordinator/smoke

    def _decoded_to_queue(self, frame: np.ndarray) -> None:
        """Enqueue a decoded RGB frame with latest-wins behavior."""
        if self.frame_queue.full():
            with contextlib.suppress(queue.Empty):
                self.frame_queue.get_nowait()
        self.frame_queue.put(frame)

    def _draw_video_frame(self, event):
        """Draw via StreamCoordinator when enabled; else legacy path for VT smoke."""
        if getattr(self, '_use_manager', False):
            try:
                self._manager.draw()
            except Exception:
                logger.exception('StreamCoordinator draw failed')
            return
        # Fallback to legacy smoke path
        try:
            frame = None
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._display_frame(frame)
        except Exception as e:
            logger.exception(f'Error drawing video frame: {e}')

    def _display_frame(self, frame):
        """Display frame using OpenGL."""
        # Delegate to GLRenderer
        try:
            if not hasattr(self, '_renderer'):
                from napari_cuda.client.streaming.renderer import GLRenderer

                self._renderer = GLRenderer(self._scene_canvas)
            self._renderer.draw(frame)
        except Exception:
            logger.exception('GLRenderer draw failed')

    # --- FPS HUD helpers ---
    def _init_fps_hud(self) -> None:
        lbl = QtWidgets.QLabel(self._scene_canvas.native)
        lbl.setObjectName('napari_cuda_fps_hud')
        lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        lbl.setStyleSheet(
            "#napari_cuda_fps_hud {"
            "  background: rgba(0,0,0,120);"
            "  color: #fff;"
            "  border-radius: 4px;"
            "  padding: 3px 6px;"
            "  font-family: Menlo, Monaco, Consolas, 'Courier New', monospace;"
            "  font-size: 11px;"
            "}"
        )
        lbl.setText('fps: --')
        lbl.setWordWrap(False)
        # Let the label size to content; keep a sensible minimum width
        lbl.adjustSize()
        lbl.setMinimumWidth(260)
        lbl.move(8, 8)
        lbl.raise_()
        lbl.show()
        self._fps_label = lbl
        # Prime counters
        self._hud_prev_time = 0.0
        self._hud_prev_submit = {'vt': 0, 'pyav': 0}
        self._hud_prev_out = {'vt': 0, 'pyav': 0}
        self._hud_prev_preview = {'vt': 0, 'pyav': 0}
        # Jitter HUD helpers (per-second deltas)
        self._hud_prev_jit_time = 0.0
        self._hud_prev_jitter = {
            'delivered': 0,
            'dropped': 0,
            'reordered': 0,
            'duplicated': 0,
        }
        # 1 Hz HUD updater
        self._fps_timer = QtCore.QTimer(self._scene_canvas.native)
        self._fps_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._update_fps_hud)
        self._fps_timer.start()

    def _update_fps_hud(self) -> None:
        mgr = getattr(self, '_manager', None)
        if mgr is None or not hasattr(mgr, '_presenter'):
            return
        now = time.time()
        last = float(self._hud_prev_time or 0.0)
        # Only guard the stats() call; everything else should be deterministic
        try:
            stats = mgr._presenter.stats()
        except Exception:
            logger.debug('FPS HUD: presenter.stats failed', exc_info=True)
            return
        sub = stats.get('submit', {})
        out = stats.get('out', {})
        prev = stats.get('preview', {})
        vt_sub = int(sub.get('vt', 0)); py_sub = int(sub.get('pyav', 0))
        vt_out = int(out.get('vt', 0)); py_out = int(out.get('pyav', 0))
        vt_prev = int(prev.get('vt', 0)); py_prev = int(prev.get('pyav', 0))
        if last == 0.0:
            self._hud_prev_time = now
            self._hud_prev_submit = {'vt': vt_sub, 'pyav': py_sub}
            self._hud_prev_out = {'vt': vt_out, 'pyav': py_out}
            self._hud_prev_preview = {'vt': vt_prev, 'pyav': py_prev}
            return
        dt = max(1e-3, now - last)
        fps_sub_vt = (vt_sub - self._hud_prev_submit['vt']) / dt
        fps_sub_py = (py_sub - self._hud_prev_submit['pyav']) / dt
        fps_out_vt = (vt_out - self._hud_prev_out['vt']) / dt
        fps_out_py = (py_out - self._hud_prev_out['pyav']) / dt
        fps_prev_vt = (vt_prev - self._hud_prev_preview['vt']) / dt
        fps_prev_py = (py_prev - self._hud_prev_preview['pyav']) / dt
        self._hud_prev_time = now
        self._hud_prev_submit = {'vt': vt_sub, 'pyav': py_sub}
        self._hud_prev_out = {'vt': vt_out, 'pyav': py_out}
        self._hud_prev_preview = {'vt': vt_prev, 'pyav': py_prev}
        active = getattr(mgr._source_mux, 'active', None)
        active_str = getattr(active, 'value', str(active)) if active is not None else '-'
        lat_ms = int(stats.get('latency_ms', 0) or 0)
        sub_fps = fps_sub_vt if active_str == 'vt' else fps_sub_py
        out_fps = fps_out_vt if active_str == 'vt' else fps_out_py
        prev_fps = fps_prev_vt if active_str == 'vt' else fps_prev_py
        # Presenter queue depths by source (buf sizes)
        buf = stats.get('buf', {})
        buf_vt = int(buf.get('vt', 0)); buf_py = int(buf.get('pyav', 0))
        # VT decoder internal queue length (shim counts), if available
        vt_q_len = None
        try:
            vt_counts = getattr(mgr, '_vt_pipeline', None).counts() if hasattr(getattr(mgr, '_vt_pipeline', None), 'counts') else None
            if vt_counts is not None:
                # (submits, outputs, qlen)
                vt_q_len = int(vt_counts[2])
        except Exception:
            logger.debug('FPS HUD: vt counts failed', exc_info=True)
        # Pipelines qsize (ingress)
        try:
            q_vt = getattr(mgr, '_vt_pipeline', None).qsize() if hasattr(getattr(mgr, '_vt_pipeline', None), 'qsize') else 0
        except Exception:
            logger.debug('FPS HUD: vt qsize failed', exc_info=True)
            q_vt = 0
        try:
            q_py = getattr(mgr, '_pyav_pipeline', None).qsize() if hasattr(getattr(mgr, '_pyav_pipeline', None), 'qsize') else 0
        except Exception:
            logger.debug('FPS HUD: pyav qsize failed', exc_info=True)
            q_py = 0
        # Selected client-side timing means (if metrics enabled)
        dec_py_ms = None
        vt_dec_ms = None
        vt_submit_ms = None
        render_vt_ms = None
        render_pyav_ms = None
        draw_mean_ms = None
        draw_last_ms = None
        present_fps = None
        # Metrics snapshot (decode/render/draw + jitter + lateness)
        jit_q = None
        jit_deliv_rate = None
        jit_drop_rate = None
        jit_sched_mean = None
        late_last_ms = None
        late_mean_ms = None
        late_p90_ms = None
        try:
            metrics = getattr(mgr, '_metrics', None)
            if metrics is not None and hasattr(metrics, 'snapshot'):
                snap = metrics.snapshot()
                h = snap.get('histograms', {}) or {}
                # Decode/submit/render
                dec_py_ms = (h.get('napari_cuda_client_pyav_decode_ms') or {}).get('mean_ms')
                vt_dec_ms = (h.get('napari_cuda_client_vt_decode_ms') or {}).get('mean_ms')
                vt_submit_ms = (h.get('napari_cuda_client_vt_submit_ms') or {}).get('mean_ms')
                render_vt_ms = (h.get('napari_cuda_client_render_vt_ms') or {}).get('mean_ms')
                render_pyav_ms = (h.get('napari_cuda_client_render_pyav_ms') or {}).get('mean_ms')
                d_hist = (h.get('napari_cuda_client_draw_interval_ms') or {})
                draw_mean_ms = d_hist.get('mean_ms')
                draw_last_ms = d_hist.get('last_ms')
                # Jitter metrics
                g = snap.get('gauges', {}) or {}
                c = snap.get('counters', {}) or {}
                jit_q = g.get('napari_cuda_jit_qdepth')
                # Per-second delivered/drop rates (delta counters)
                now = time.time()
                prev_t = float(self._hud_prev_jit_time or 0.0)
                if prev_t <= 0.0:
                    self._hud_prev_jit_time = now
                    self._hud_prev_jitter['delivered'] = int(c.get('napari_cuda_jit_delivered', 0) or 0)
                    self._hud_prev_jitter['dropped'] = int(c.get('napari_cuda_jit_dropped', 0) or 0)
                    self._hud_prev_jitter['reordered'] = int(c.get('napari_cuda_jit_reordered', 0) or 0)
                    self._hud_prev_jitter['duplicated'] = int(c.get('napari_cuda_jit_duplicated', 0) or 0)
                else:
                    dtj = max(1e-3, now - prev_t)
                    d_deliv = int(c.get('napari_cuda_jit_delivered', 0) or 0) - int(self._hud_prev_jitter['delivered'])
                    d_drop = int(c.get('napari_cuda_jit_dropped', 0) or 0) - int(self._hud_prev_jitter['dropped'])
                    jit_deliv_rate = max(0.0, float(d_deliv) / dtj)
                    jit_drop_rate = max(0.0, float(d_drop) / dtj)
                    # Update prev snapshot
                    self._hud_prev_jit_time = now
                    self._hud_prev_jitter['delivered'] = int(c.get('napari_cuda_jit_delivered', 0) or 0)
                    self._hud_prev_jitter['dropped'] = int(c.get('napari_cuda_jit_dropped', 0) or 0)
                    self._hud_prev_jitter['reordered'] = int(c.get('napari_cuda_jit_reordered', 0) or 0)
                    self._hud_prev_jitter['duplicated'] = int(c.get('napari_cuda_jit_duplicated', 0) or 0)
                jit_sched = (h.get('napari_cuda_jit_sched_delay_ms') or {})
                jit_sched_mean = jit_sched.get('mean_ms')
                # Presenter lateness (non-preview frames)
                late_hist = (h.get('napari_cuda_client_present_lateness_ms') or {})
                late_last_ms = late_hist.get('last_ms')
                late_mean_ms = late_hist.get('mean_ms')
                late_p90_ms = late_hist.get('p90_ms')
                # Derived
                d = snap.get('derived', {}) or {}
                present_fps = d.get('fps')
        except Exception:
            logger.debug('FPS HUD: metrics snapshot failed', exc_info=True)
        vtq_bit = f" vtq:{vt_q_len}" if isinstance(vt_q_len, int) else ""
        # Clear, orthogonal layout
        line1 = f"src:{active_str}  latency:{lat_ms} ms"
        line2 = f"ingress:{sub_fps:.1f}/s  consumed:{out_fps:.1f}/s  preview:{prev_fps:.1f}/s"
        line3 = f"queues: presenter[vt:{buf_vt} py:{buf_py}]  pipeline[vt:{q_vt} py:{q_py}]{vtq_bit}"
        txt = line1 + "\n" + line2 + "\n" + line3
        # Append decode/submit timings if available
        extra = []
        if active_str == 'pyav':
            if isinstance(dec_py_ms, (int, float)):
                extra.append(f"dec:{dec_py_ms:.2f}ms")
            if isinstance(render_pyav_ms, (int, float)):
                extra.append(f"render:{render_pyav_ms:.2f}ms")
        elif active_str == 'vt':
            if isinstance(vt_dec_ms, (int, float)):
                extra.append(f"dec:{vt_dec_ms:.2f}ms")
            if isinstance(render_vt_ms, (int, float)):
                extra.append(f"render:{render_vt_ms:.2f}ms")
            if isinstance(vt_submit_ms, (int, float)):
                extra.append(f"sub:{vt_submit_ms:.2f}ms")
        if extra:
            txt = txt + "\n" + "  ".join(extra)
        # Append draw-loop pacing and presented FPS if available
        loop_bits = []
        if isinstance(present_fps, (int, float)):
            loop_bits.append(f"present:{present_fps:.1f}fps")
        # Show last draw interval by default for responsiveness; optionally include mean
        if isinstance(draw_last_ms, (int, float)) and draw_last_ms > 0:
            loop_bits.append(f"loop:{draw_last_ms:.2f}ms")
        try:
            import os as _os
            if (_os.getenv('NAPARI_CUDA_HUD_LOOP_SHOW_MEAN') or '0') in ('1','true','yes'):
                if isinstance(draw_mean_ms, (int, float)) and draw_mean_ms > 0:
                    loop_bits.append(f"mean:{draw_mean_ms:.2f}ms")
        except Exception:
            pass
        # Presenter lateness summary (helps correlate judder)
        if isinstance(late_last_ms, (int, float)) and isinstance(late_mean_ms, (int, float)):
            loop_bits.append(f"late:{late_last_ms:.1f}/{late_mean_ms:.1f}ms")
        if isinstance(late_p90_ms, (int, float)) and late_p90_ms is not None:
            loop_bits.append(f"p90:{late_p90_ms:.1f}ms")
        # Optional memory usage (enable with NAPARI_CUDA_HUD_SHOW_MEM=1)
        try:
            import os as _os, sys as _sys
            if (_os.getenv('NAPARI_CUDA_HUD_SHOW_MEM') or '0') in ('1','true','yes'):
                cur_mb = None; max_mb = None
                # Prefer current RSS via psutil if available
                try:
                    import psutil as _ps
                    p = _ps.Process()
                    cur_mb = float(p.memory_info().rss) / (1024.0 * 1024.0)
                except Exception:
                    pass
                # High-water mark via resource
                try:
                    import resource as _res
                    rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
                    if _sys.platform == 'darwin':
                        max_mb = float(rss) / (1024.0 * 1024.0)
                    else:
                        max_mb = float(rss) / 1024.0
                except Exception:
                    pass
                if isinstance(cur_mb, float) and isinstance(max_mb, float):
                    loop_bits.append(f"mem:{cur_mb:.0f}/{max_mb:.0f}MB")
                elif isinstance(cur_mb, float):
                    loop_bits.append(f"mem:{cur_mb:.0f}MB")
                elif isinstance(max_mb, float):
                    loop_bits.append(f"mem_max:{max_mb:.0f}MB")
        except Exception:
            logger.debug('FPS HUD: mem calc failed', exc_info=True)
        if loop_bits:
            txt = txt + "\n" + "  ".join(loop_bits)
        # Optional 3D view HUD (volume tuning)
        try:
            import os as _os
            view_hud_enabled = (_os.getenv('NAPARI_CUDA_VIEW_HUD') or '0').lower() in ('1','true','yes','on')
        except Exception:
            view_hud_enabled = False
        if view_hud_enabled:
            view_txt = self._build_view_hud_text(mgr)
            if view_txt:
                txt = txt + "\n" + view_txt
        # Optional jitter line (shown when metrics are present)
        try:
            show_jit = (jit_q is not None) or (jit_deliv_rate is not None) or (jit_sched_mean is not None)
            if show_jit:
                parts = []
                if isinstance(jit_q, (int, float)):
                    parts.append(f"jitq:{int(jit_q)}")
                if isinstance(jit_deliv_rate, float):
                    parts.append(f"jit:{jit_deliv_rate:.1f}/s")
                if isinstance(jit_drop_rate, float) and jit_drop_rate > 0:
                    parts.append(f"drop:{jit_drop_rate:.1f}/s")
                if isinstance(jit_sched_mean, (int, float)):
                    parts.append(f"sched:{jit_sched_mean:.1f}ms")
                if parts:
                    txt = txt + "\n" + "  ".join(parts)
        except Exception:
            logger.debug('FPS HUD: jitter line failed', exc_info=True)
        self._fps_label.setText(txt)
        self._fps_label.adjustSize()

    def _build_view_hud_text(self, mgr) -> str | None:
        """Format a compact multi-line View HUD from manager snapshot.

        Returns a string with one or more lines, or None if unavailable.
        """
        try:
            if not hasattr(mgr, 'view_hud_snapshot'):
                return None
            vs = mgr.view_hud_snapshot()  # type: ignore[attr-defined]
        except Exception:
            logger.debug('FPS HUD: view snapshot failed', exc_info=True)
            return None
        if not isinstance(vs, dict) or not vs:
            return None
        try:
            # First line: mode + zoom/pan quick glance
            vol_flag = 'vol' if bool(vs.get('volume')) else 'img'
            ndisp = vs.get('ndisplay')
            vm = '3D' if bool(vs.get('vol_mode')) else '2D'
            zb = vs.get('zoom_base')
            zf = vs.get('last_zoom_factor')
            pdx = vs.get('last_pan_dx'); pdy = vs.get('last_pan_dy')
            line_v1 = (
                f"view:{vol_flag}/{vm} ndisp:{ndisp} zbase:{zb} zfac:{zf:.3f}"
                if isinstance(zf, (int, float))
                else f"view:{vol_flag}/{vm} ndisp:{ndisp} zbase:{zb}"
            )
            if isinstance(pdx, (int, float)) and isinstance(pdy, (int, float)):
                line_v1 += f" pan:({pdx:.1f},{pdy:.1f})"
            # Second line: render and multiscale summary
            rmode = vs.get('render_mode'); cmap = vs.get('colormap')
            clo = vs.get('clim_lo'); chi = vs.get('clim_hi')
            opa = vs.get('opacity'); sst = vs.get('sample_step')
            ms_pol = vs.get('ms_policy'); ms_lvl = vs.get('ms_level'); ms_n = vs.get('ms_levels')
            rbits: list[str] = []
            if rmode:
                rbits.append(f"mode:{rmode}")
            if isinstance(clo, (int, float)) and isinstance(chi, (int, float)):
                rbits.append(f"clim:[{clo:.2f},{chi:.2f}]")
            if cmap:
                rbits.append(f"map:{cmap}")
            if isinstance(opa, (int, float)):
                rbits.append(f"opac:{opa:.2f}")
            if isinstance(sst, (int, float)):
                rbits.append(f"step:{sst:.2f}")
            msbits: list[str] = []
            if ms_pol:
                msbits.append(f"pol:{ms_pol}")
            if isinstance(ms_lvl, int) and isinstance(ms_n, int):
                msbits.append(f"lvl:{ms_lvl}/{max(0, ms_n-1)}")
            line_v2 = "render: " + " ".join(rbits) if rbits else "render: -"
            if msbits:
                line_v2 += "  ms: " + " ".join(msbits)
            # Third line: last anchors if available
            aw = vs.get('last_zoom_widget_px'); av = vs.get('last_zoom_video_px'); asv = vs.get('last_zoom_anchor_px')
            line_v3 = None
            if isinstance(aw, (list, tuple)) and isinstance(av, (list, tuple)) and isinstance(asv, (list, tuple)):
                try:
                    line_v3 = (
                        f"anchor: w:({aw[0]:.1f},{aw[1]:.1f}) "
                        f"v:({av[0]:.1f},{av[1]:.1f}) s:({asv[0]:.1f},{asv[1]:.1f})"
                    )
                except Exception:
                    line_v3 = None
            # Compose
            out = line_v1 + "\n" + line_v2
            if line_v3:
                out = out + "\n" + line_v3
            return out
        except Exception:
            logger.debug('FPS HUD: view HUD format failed', exc_info=True)
            return None
