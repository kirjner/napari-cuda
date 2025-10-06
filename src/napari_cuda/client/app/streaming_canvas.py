"""
StreamingCanvas - Displays video stream from remote server.

This replaces the normal VispyCanvas with one that shows decoded video frames
instead of locally rendered content.
"""

import logging
import os
from qtpy import QtCore

from napari._vispy.canvas import VispyCanvas

# Silence VisPy warnings about copying discontiguous data
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# No direct networking or shader setup here; ClientStreamLoop + GLRenderer own these


from napari_cuda.utils.env import env_int, env_float
from napari_cuda.client.runtime.config import ClientConfig


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
        vs_env = (os.getenv('NAPARI_CUDA_VSYNC') or '').strip().lower()
        if vs_env:
            try:
                from qtpy import QtGui  # type: ignore
            except ImportError:
                logger.debug('VSync default format setup failed', exc_info=True)
            else:
                fmt = QtGui.QSurfaceFormat()
                if vs_env in ('0', 'off', 'false', 'no'):
                    fmt.setSwapInterval(0)
                    logger.info('GL vsync disabled via NAPARI_CUDA_VSYNC=0')
                else:
                    fmt.setSwapInterval(1)
                    logger.info('GL vsync enabled via NAPARI_CUDA_VSYNC=1')
                QtGui.QSurfaceFormat.setDefaultFormat(fmt)

        # Ensure we have a KeymapHandler; create a minimal one if not provided
        if key_map_handler is None:
            try:
                from napari.utils.key_bindings import KeymapHandler  # type: ignore
            except ImportError:
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
            else:
                handler = KeymapHandler()
                handler.keymap_providers = [viewer]
                key_map_handler = handler
        # Forward to base canvas
        super().__init__(viewer, key_map_handler, **kwargs)

        self._proxy_viewer = viewer
        self._deferred_window = None
        self._window_show_timer = None
        self._first_dims_ready = False

        self.server_host = server_host
        self.server_port = server_port
        self.state_port = int(
            state_port or int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081'))
        )
        # Presenter latency/buffer configuration derived from env
        self._vt_latency_s = max(
            0.0, env_float('NAPARI_CUDA_CLIENT_VT_LATENCY_MS', 0.0) / 1000.0
        )
        # If not explicitly set, derive a sane default from latency and a 60 Hz
        # output cadence: roughly ceil(latency * 60) + 2 frames. This avoids
        # trimming not-yet-due frames in SERVER mode with higher latency.
        buf_env = os.getenv('NAPARI_CUDA_CLIENT_VT_BUFFER')
        if buf_env is None or buf_env.strip() == '':
            import math

            derived = max(3, int(math.ceil(self._vt_latency_s * 60.0)) + 2)
            self._vt_buffer_limit = derived
        else:
            buf_val = int(buf_env)
            assert buf_val > 0, 'NAPARI_CUDA_CLIENT_VT_BUFFER must be positive'
            self._vt_buffer_limit = buf_val
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
        # Construct minimal client config (phase 0; no behavior change)
        # Estimate display FPS early from env to avoid NameError before DisplayLoop init
        fps_guess = env_float('NAPARI_CUDA_CLIENT_DISPLAY_FPS', 60.0)
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
        from napari_cuda.client.runtime.stream_runtime import ClientStreamLoop

        self._manager = ClientStreamLoop(
            scene_canvas=self._scene_canvas,
            server_host=self.server_host,
            server_port=self.server_port,
            state_port=self.state_port,
            vt_latency_s=self._vt_latency_s,
            pyav_latency_s=self._pyav_latency_s,
            vt_buffer_limit=self._vt_buffer_limit,
            stream_format='avcc',
            vt_backlog_trigger=env_int('NAPARI_CUDA_CLIENT_VT_BACKLOG_TRIGGER', 32),
            pyav_backlog_trigger=env_int('NAPARI_CUDA_CLIENT_PYAV_BACKLOG_TRIGGER', 16),
            client_cfg=self._client_cfg,
            on_first_dims_ready=self._on_first_dims_ready,
        )
        self._manager.start()
        self._manager.attach_viewer_proxy(viewer)
        if hasattr(viewer, 'attach_state_sender'):
            viewer.attach_state_sender(self._manager)  # type: ignore[attr-defined]
        self._manager.presenter_facade.bind_canvas(
            enable_dims_play=self.enable_dims_play,
        )

        logger.info('StreamingCanvas initialized for %s:%s', server_host, server_port)

    def defer_window_show(self, window) -> None:
        """Delay window visibility until first dims.update arrives (fallback timer)."""
        if window is None:
            return
        self._deferred_window = window
        if self._first_dims_ready:
            self._show_deferred_window()
            return
        window.hide()
        if self._window_show_timer is None:
            timer = QtCore.QTimer(window)
            timer.setSingleShot(True)
            timer.setTimerType(QtCore.Qt.PreciseTimer)  # type: ignore[attr-defined]
            # Allow ample time for dims.update to arrive before falling back.
            timer.setInterval(5000)

            def _fallback() -> None:
                logger.warning('StreamingCanvas: dims.update not received within fallback window; showing UI anyway')
                self._show_deferred_window()

            timer.timeout.connect(_fallback)
            self._window_show_timer = timer
        self._window_show_timer.start()

    def _cancel_window_timer(self) -> None:
        timer = self._window_show_timer
        if timer is not None and timer.isActive():
            timer.stop()

    def _show_deferred_window(self) -> None:
        window = self._deferred_window
        if window is None:
            return
        self._cancel_window_timer()
        window.show()

    def _on_first_dims_ready(self) -> None:
        self._first_dims_ready = True
        if self._deferred_window is None:
            self._cancel_window_timer()
            return
        self._show_deferred_window()
