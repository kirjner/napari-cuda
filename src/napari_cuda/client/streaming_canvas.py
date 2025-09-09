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

import numpy as np
from qtpy import QtCore
from vispy import app as vispy_app

from napari._vispy.canvas import VispyCanvas

# Silence VisPy warnings about copying discontiguous data
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# No direct networking or shader setup here; StreamManager + GLRenderer own these


# Env parsing helpers
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    v = val.strip().lower()
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    return default


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
        env_smoke = _env_bool('NAPARI_CUDA_VT_SMOKE', False)
        self._vt_smoke = bool(vt_smoke or env_smoke)

        # Queue for decoded frames (latest-wins draining in draw)
        buf_n = _env_int('NAPARI_CUDA_CLIENT_BUFFER_FRAMES', 3)
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
            0.0, _env_float('NAPARI_CUDA_CLIENT_VT_LATENCY_MS', 0.0) / 1000.0
        )
        self._vt_buffer_limit = _env_int('NAPARI_CUDA_CLIENT_VT_BUFFER', 3)
        # Higher latency when falling back to PyAV (smoother on CPU decode)
        self._pyav_latency_s = max(
            0.0,
            _env_float(
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
        from napari_cuda.client.streaming.types import Source, TimestampMode

        ts_mode_env = (
            os.getenv('NAPARI_CUDA_CLIENT_VT_TS_MODE') or 'arrival'
        ).lower()
        ts_mode = (
            TimestampMode.ARRIVAL
            if ts_mode_env == 'arrival'
            else TimestampMode.SERVER
        )
        self._presenter = FixedLatencyPresenter(
            latency_s=self._vt_latency_s,
            buffer_limit=self._vt_buffer_limit,
            ts_mode=ts_mode,
        )
        self._source_mux = SourceMux(Source.PYAV)
        assert self._presenter is not None
        assert self._source_mux is not None
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
        self._vt_backlog_trigger = _env_int(
            'NAPARI_CUDA_CLIENT_VT_BACKLOG_TRIGGER', 16
        )
        # Workers are managed by StreamManager when enabled

        # Expected bitstream format from server ('avcc' or 'annexb'); default to AVCC
        self._stream_format = 'avcc'
        self._stream_format_set = False

        # PyAV decode decoupling
        self._pyav_in_q: queue.Queue[tuple[bytes, float|None]] = queue.Queue(
            maxsize=64
        )
        self._pyav_enqueued = 0
        self._pyav_backlog_trigger = _env_int(
            'NAPARI_CUDA_CLIENT_PYAV_BACKLOG_TRIGGER', 16
        )
        # PyAV worker is managed by StreamManager when enabled

        # Orchestrate with StreamManager unless in VT smoke mode
        self._use_manager = not self._vt_smoke
        if self._use_manager:
            from napari_cuda.client.streaming import StreamManager

            self._manager = StreamManager(
                scene_canvas=self._scene_canvas,
                server_host=self.server_host,
                server_port=self.server_port,
                state_port=self.state_port,
                vt_latency_s=self._vt_latency_s,
                pyav_latency_s=self._pyav_latency_s,
                vt_buffer_limit=self._vt_buffer_limit,
                vt_ts_mode=ts_mode.value,
                stream_format=self._stream_format,
                vt_backlog_trigger=self._vt_backlog_trigger,
                pyav_backlog_trigger=_env_int(
                    'NAPARI_CUDA_CLIENT_PYAV_BACKLOG_TRIGGER', 16
                ),
                vt_smoke=self._vt_smoke,
            )
            self._manager.start()
        else:
            # VT smoke thread (offline)
            self._streaming_thread = Thread(
                target=self._vt_smoke_worker, daemon=True
            )
            logger.info('StreamingCanvas in VT smoke test mode (offline)')
            self._streaming_thread.start()

        # Override draw to show video instead
        self._scene_canvas.events.draw.disconnect()
        self._scene_canvas.events.draw.connect(self._draw_video_frame)
        # Timer-driven display at target fps (use VisPy app timer to ensure GUI-thread delivery)
        fps = _env_float('NAPARI_CUDA_CLIENT_DISPLAY_FPS', 60.0)
        # Default to Qt timer for napari/Qt integration; enable vispy timer only if requested
        use_vispy_timer = (
            os.getenv('NAPARI_CUDA_CLIENT_VISPY_TIMER', '0') == '1'
        )
        interval = max(1.0 / max(1.0, fps), 1.0 / 120.0)
        if use_vispy_timer:
            self._display_timer = vispy_app.Timer(
                interval=interval,
                connect=lambda ev: self._scene_canvas.update(),
                start=True,
            )
            logger.info(
                'Video display initialized (vispy.Timer @ %.1f fps)',
                1.0 / interval,
            )
        else:
            # Fallback to Qt timer, ensure it belongs to the canvas' GUI thread
            self._display_timer = QtCore.QTimer(self._scene_canvas.native)
            self._display_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self._display_timer.setInterval(
                max(1, int(round(1000.0 / max(1.0, fps))))
            )
            self._display_timer.timeout.connect(
                self._scene_canvas.native.update
            )
            self._display_timer.start()
            logger.info(
                'Video display initialized (Qt QTimer @ %.1f fps)', fps
            )

        logger.info(
            f'StreamingCanvas initialized for {server_host}:{server_port}'
        )

        # Timestamp handling for VT scheduling
        self._vt_ts_mode = (
            os.getenv('NAPARI_CUDA_CLIENT_VT_TS_MODE') or 'server'
        ).lower()
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

    # No state-channel helpers here; StreamManager owns StateChannel

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

    def _vt_smoke_worker(self):
        from napari_cuda.client.streaming.smoke import start_vt_smoke_thread

        start_vt_smoke_thread(self._decoded_to_queue)

    def _stream_worker(self):
        """Background thread to receive and decode video stream via PixelReceiver."""
        # No direct stream worker here; StreamManager owns the receiver

    # No local decoder init; StreamManager owns decoders

    # No local AnnexB→AVCC conversion; handled in StreamManager/decoders

    # No local VT live decoding; handled in StreamManager

    # No local VT pixel buffer mapping here; handled in StreamManager/smoke

    def _decoded_to_queue(self, frame: np.ndarray) -> None:
        """Enqueue a decoded RGB frame with latest-wins behavior."""
        if self.frame_queue.full():
            with contextlib.suppress(queue.Empty):
                self.frame_queue.get_nowait()
        self.frame_queue.put(frame)

    def _draw_video_frame(self, event):
        """Draw via StreamManager when enabled; else legacy path for VT smoke."""
        if getattr(self, '_use_manager', False):
            try:
                self._manager.draw()
            except Exception:
                logger.exception('StreamManager draw failed')
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
