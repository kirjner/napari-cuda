from __future__ import annotations

"""
PyAVPipeline - handles PyAV decode path decoupled from the coordinator.

- Owns its input queue and a worker thread.
- On decode, submits frames to the shared presenter and schedules a draw.
- Exposes simple enqueue() for the coordinator to feed bytes+timestamps.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Optional

from qtpy import QtCore

from napari_cuda.client.streaming.types import Source, SubmittedFrame

logger = logging.getLogger(__name__)

DecodeFn = Callable[[bytes | memoryview], Optional[object]]


@dataclass
class PyAVPipeline:
    presenter: object
    source_mux: object
    scene_canvas: object
    backlog_trigger: int = 16
    latency_s: float = 0.08
    metrics: object | None = None  # ClientMetrics-like (optional)
    schedule_next_wake: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        self._in_q: "queue.Queue[tuple[bytes | memoryview, float|None]]" = queue.Queue(maxsize=64)
        self._enqueued: int = 0
        self._decoder: Optional[DecodeFn] = None
        self._worker_started: bool = False
        self._stop = threading.Event()
        self._thread: Optional[Thread] = None
        # Gate per-decode GUI updates; now disabled by default since presenter wake drives draws
        import os as _os
        try:
            self._post_decode_update = (_os.getenv('NAPARI_CUDA_CLIENT_DECODE_UPDATE', '0') or '0') in ('1','true','yes')
        except Exception:
            self._post_decode_update = False

    def set_decoder(self, decode_fn: Optional[DecodeFn]) -> None:
        self._decoder = decode_fn

    def start(self) -> None:
        if self._worker_started:
            return
        self._stop.clear()

        def _worker() -> None:
            while not self._stop.is_set():
                try:
                    b, ts = self._in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                except Exception:
                    if self._stop.is_set():
                        break
                    logger.debug("PyAVPipeline: queue.get failed", exc_info=True)
                    continue
                if self._stop.is_set():
                    break
                arr = None
                try:
                    t0 = time.perf_counter()
                    dec = self._decoder
                    if dec is not None:
                        arr = dec(b)
                    t1 = time.perf_counter()
                    if self.metrics is not None:
                        self.metrics.observe_ms('napari_cuda_client_pyav_decode_ms', (t1 - t0) * 1000.0)
                except Exception:
                    logger.debug("PyAVPipeline: decode failed", exc_info=True)
                    arr = None
                if arr is None:
                    continue
                try:
                    self.presenter.submit(
                        SubmittedFrame(
                            source=Source.PYAV,
                            server_ts=float(ts) if ts is not None else None,
                            arrival_ts=time.perf_counter(),
                            payload=arr,
                            release_cb=None,
                        )
                    )
                    try:
                        t_prod = time.perf_counter()
                        logger.debug("PYAV_OUT out_mono=%.6f", float(t_prod))
                    except Exception:
                        pass
                except Exception:
                    logger.exception("PyAVPipeline: presenter submit failed")
                try:
                    if self._post_decode_update:
                        QtCore.QTimer.singleShot(0, self.scene_canvas.native.update)
                    if self.schedule_next_wake is not None:
                        self.schedule_next_wake()
                except Exception:
                    logger.debug("PyAVPipeline: schedule GUI update failed", exc_info=True)
            logger.debug("PyAVPipeline: worker exiting")

        t = Thread(target=_worker, daemon=True)
        t.start()
        self._thread = t
        self._worker_started = True

    def enqueue(self, b: bytes | memoryview, ts: Optional[float]) -> None:
        try:
            self._in_q.put_nowait((b, ts))
            self._enqueued += 1
            if self.metrics is not None:
                self.metrics.set('napari_cuda_client_pyav_qdepth', float(self._in_q.qsize()))
        except Exception:
            # Queue full; drop oldest by non-blocking drain, then retry once
            logger.debug("PyAVPipeline: enqueue failed; attempting drop-oldest and retry", exc_info=True)
            try:
                _ = self._in_q.get_nowait()
            except Exception:
                logger.debug("PyAVPipeline: drop-oldest failed (queue empty?)", exc_info=True)
            try:
                self._in_q.put_nowait((b, ts))
                self._enqueued += 1
                if self.metrics is not None:
                    self.metrics.inc('napari_cuda_client_pyav_dropped', 1.0)
                    self.metrics.set('napari_cuda_client_pyav_qdepth', float(self._in_q.qsize()))
            except Exception:
                logger.warning("PyAVPipeline: queue full; dropped frame")

    @property
    def enqueued(self) -> int:
        return self._enqueued

    # Introspection and control for coordinators
    def qsize(self) -> int:
        try:
            return int(self._in_q.qsize())
        except Exception:
            logger.debug("PyAVPipeline: qsize failed", exc_info=True)
            return 0

    def clear(self) -> None:
        try:
            while self._in_q.qsize() > 0:
                try:
                    _ = self._in_q.get_nowait()
                except Exception:
                    logger.debug("PyAVPipeline: get_nowait during clear failed", exc_info=True)
                    break
        except Exception:
            logger.debug("PyAVPipeline: clear failed", exc_info=True)

    def stop(self) -> None:
        if not self._worker_started:
            return
        self._stop.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=0.5)
            except Exception:
                logger.debug("PyAVPipeline: join failed", exc_info=True)
        self._thread = None
        try:
            while True:
                self._in_q.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            logger.debug("PyAVPipeline: queue drain during stop failed", exc_info=True)
        self._decoder = None
        self._worker_started = False
