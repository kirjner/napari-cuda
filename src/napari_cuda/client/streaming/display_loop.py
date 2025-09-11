from __future__ import annotations

"""
Display loop helper that centralizes timer selection and cadence for drawing.

Prefers VisPy's Timer for steady cadence, with a fallback to Qt's QTimer.
Provides start/stop and FPS changes without leaking timers.
"""

import logging
import os
from typing import Optional, Callable

from qtpy import QtCore
from vispy import app as vispy_app  # type: ignore

logger = logging.getLogger(__name__)


class DisplayLoop:
    def __init__(
        self,
        *,
        scene_canvas: object,
        callback: Optional[Callable[[], None]] = None,
        fps: Optional[float] = None,
        prefer_vispy: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        - scene_canvas: VisPy canvas wrapper with `.update()` and `.native` QWidget
        - callback: function to invoke each tick; defaults to `scene_canvas.update`
        - fps: target frames per second (default: env or 60)
        - prefer_vispy: whether to prefer VisPy timer (default: env or True when smoke)
        """
        self._canvas = scene_canvas
        self._callback = callback or getattr(scene_canvas, 'update')
        # Resolve FPS
        try:
            # Prefer explicit arg, else client display env, else smoke fps env
            env_fps = os.getenv('NAPARI_CUDA_CLIENT_DISPLAY_FPS')
            if env_fps is None:
                env_fps = os.getenv('NAPARI_CUDA_SMOKE_FPS')
            self._fps = float(fps if fps is not None else (env_fps if env_fps is not None else 60.0))
        except Exception:
            self._fps = 60.0
        # Choose timer backend
        env_vispy = os.getenv('NAPARI_CUDA_CLIENT_VISPY_TIMER')
        if prefer_vispy is not None:
            self._prefer_vispy = bool(prefer_vispy)
        elif env_vispy is not None:
            self._prefer_vispy = (env_vispy == '1')
        else:
            # Default to VisPy timer (steadier cadence)
            self._prefer_vispy = True
        self._timer = None
        self._using_vispy = False

    @property
    def fps(self) -> float:
        return float(self._fps)

    def set_fps(self, fps: float) -> None:
        self._fps = max(1.0, float(fps))
        if self._timer is not None:
            self.stop()
            self.start()

    def start(self) -> None:
        if self._timer is not None:
            return
        interval_s = 1.0 / max(1.0, float(self._fps))
        # Try VisPy first if requested
        if self._prefer_vispy:
            try:
                t = vispy_app.Timer(
                    interval=interval_s,
                    connect=lambda evt: self._safe_tick(),
                    start=True,
                )
                self._timer = t
                self._using_vispy = True
                logger.info('DisplayLoop: started vispy.Timer @ %.1f fps', 1.0 / interval_s)
                return
            except Exception:
                logger.debug('DisplayLoop: vispy.Timer failed; falling back to Qt', exc_info=True)
        # Fallback to Qt timer bound to the canvas' native widget
        try:
            qt_timer = QtCore.QTimer(self._canvas.native)
            qt_timer.setTimerType(QtCore.Qt.PreciseTimer)
            qt_timer.setInterval(max(1, int(round(1000.0 * interval_s))))
            qt_timer.timeout.connect(self._safe_tick)
            qt_timer.start()
            self._timer = qt_timer
            self._using_vispy = False
            logger.info('DisplayLoop: started Qt QTimer @ %.1f fps', 1.0 / interval_s)
        except Exception:
            logger.exception('DisplayLoop: failed to start any timer')

    def stop(self) -> None:
        t = self._timer
        self._timer = None
        if t is None:
            return
        try:
            if self._using_vispy:
                # VisPy Timer has a `stop` method
                t.stop()  # type: ignore[attr-defined]
            else:
                # Qt QTimer has a `stop` and will be GC'd with the widget
                t.stop()  # type: ignore[attr-defined]
        except Exception:
            logger.debug('DisplayLoop: stop failed', exc_info=True)

    def _safe_tick(self) -> None:
        try:
            cb = self._callback
            if cb is not None:
                cb()
            else:
                # Fallback to canvas.update via native
                self._canvas.update()
        except Exception:
            logger.debug('DisplayLoop: tick failed', exc_info=True)
