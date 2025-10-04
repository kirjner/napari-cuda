from __future__ import annotations

"""
Event loop stall monitor.

Runs a lightweight QTimer on the main thread, measures wall-clock gaps between
ticks, and logs/instruments when the gap exceeds a configured threshold. Optionally
issues a one-off GUI update to help recover from prolonged stalls.
"""

import logging
import time
from typing import Optional, Callable

from qtpy import QtCore

logger = logging.getLogger(__name__)


class EventLoopMonitor:
    def __init__(
        self,
        *,
        parent: object,
        metrics: Optional[object] = None,
        stall_threshold_ms: int = 250,
        sample_interval_ms: int = 100,
        on_stall_kick: Optional[Callable[[], None]] = None,
    ) -> None:
        self._metrics = metrics
        self._stall_ms = max(1, int(stall_threshold_ms))
        self._interval_ms = max(1, int(sample_interval_ms))
        self._kick = on_stall_kick
        self._timer: Optional[QtCore.QTimer] = None
        self._last_pc: float = 0.0
        # Bind to the same native widget so lifetime is coupled
        t = QtCore.QTimer(parent)  # type: ignore[arg-type]
        t.setTimerType(QtCore.Qt.PreciseTimer)
        t.setInterval(self._interval_ms)

        def _tick() -> None:
            try:
                now = time.perf_counter()
                last = float(self._last_pc or 0.0)
                self._last_pc = now
                if last <= 0.0:
                    return
                dt_ms = (now - last) * 1000.0
                if dt_ms >= float(self._stall_ms):
                    logger.warning(
                        "EventLoopMonitor: stall detected: %.1f ms (thr=%d ms)",
                        dt_ms,
                        self._stall_ms,
                    )
                    try:
                        if self._metrics is not None:
                            self._metrics.inc('napari_cuda_client_eventloop_stalls', 1.0)
                            self._metrics.set('napari_cuda_client_eventloop_last_stall_ms', float(dt_ms))
                    except Exception:
                        pass
                    # Optional gentle kick to help resume paints
                    try:
                        if self._kick is not None:
                            self._kick()
                    except Exception:
                        logger.debug("EventLoopMonitor: kick failed", exc_info=True)
            except Exception:
                logger.debug("EventLoopMonitor: tick failed", exc_info=True)

        t.timeout.connect(_tick)
        t.start()
        self._timer = t
        logger.info(
            "EventLoopMonitor started: interval=%d ms, stall_threshold=%d ms",
            self._interval_ms,
            self._stall_ms,
        )

    def stop(self) -> None:
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            logger.debug("EventLoopMonitor: stop failed", exc_info=True)

