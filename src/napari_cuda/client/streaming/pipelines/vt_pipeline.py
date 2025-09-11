from __future__ import annotations

"""
VTPipeline - encapsulates VT submit/drain and avcC/AnnexB normalization.

- Owns an input queue and two workers: submit (normalize+decode) and drain (present).
- Coordinates with the orchestrator using small callbacks for gating and caching.
"""

import logging
import queue
import time
from threading import Thread
from typing import Callable, Optional, Tuple

from qtpy import QtCore

from napari_cuda.client.streaming.types import Source, SubmittedFrame
from napari_cuda.codec.avcc import is_annexb, annexb_to_avcc, split_avcc_by_len

logger = logging.getLogger(__name__)


class VTPipeline:
    def __init__(
        self,
        *,
        presenter: object,
        source_mux: object,
        scene_canvas: object,
        backlog_trigger: int = 16,
        is_gated: Callable[[], bool] | None = None,
        on_backlog_gate: Callable[[], None] | None = None,
        request_keyframe: Callable[[], None] | None = None,
        on_cache_last: Callable[[object, bool], None] | None = None,
    ) -> None:
        self._presenter = presenter
        self._source_mux = source_mux
        self._scene_canvas = scene_canvas
        self._in_q: "queue.Queue[tuple[bytes, float | None]]" = queue.Queue(maxsize=64)
        self._backlog_trigger = int(backlog_trigger)
        self._decoder = None  # VTLiveDecoder-like
        self._started = False
        self._errors = 0
        self._enqueued = 0
        self._nal_length_size = 4
        # Coordination hooks
        self._is_gated = is_gated or (lambda: False)
        self._on_backlog_gate = on_backlog_gate or (lambda: None)
        self._request_keyframe = request_keyframe or (lambda: None)
        self._on_cache_last = on_cache_last or (lambda _pb, _persistent: None)
        # Last VT payload for redraw fallback
        self._last_payload = None  # type: ignore[var-annotated]
        self._last_persistent = False

    def start(self) -> None:
        if self._started:
            return

        def _submit_worker() -> None:
            while True:
                try:
                    data, ts = self._in_q.get()
                except Exception:
                    logger.debug("VTPipeline: queue.get failed", exc_info=True)
                    continue
                try:
                    self._decode_vt_live(data, ts)
                except Exception:
                    logger.exception("VTPipeline: decode submit failed")

        def _drain_worker() -> None:
            while True:
                try:
                    dec = self._decoder
                    if dec is None:
                        time.sleep(0.005)
                        continue
                    drained = False
                    while True:
                        item = dec.get_frame_nowait()
                        if not item:
                            break
                        drained = True
                        img_buf, pts = item
                        if self._is_gated():
                            # Drain but don't present while waiting for keyframe
                            from napari_cuda import _vt as vt  # type: ignore
                            vt.release_frame(img_buf)
                            continue
                        # Present and cache last
                        from napari_cuda import _vt as vt  # type: ignore
                        # Take an extra retain for last-frame fallback; update cache
                        vt.retain_frame(img_buf)
                        if self._last_payload is not None and not self._last_persistent:
                            vt.release_frame(self._last_payload)
                        self._last_payload = img_buf
                        self._last_persistent = True
                        self._on_cache_last(self._last_payload, True)
                        self._presenter.submit(
                            SubmittedFrame(
                                source=Source.VT,
                                server_ts=float(pts) if pts is not None else None,
                                arrival_ts=time.time(),
                                payload=img_buf,
                                release_cb=vt.release_frame,
                            )
                        )
                    if drained:
                        QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                    if not drained:
                        time.sleep(0.002)
                except Exception:
                    logger.debug("VTPipeline: drain worker error", exc_info=True)

        Thread(target=_submit_worker, daemon=True).start()
        Thread(target=_drain_worker, daemon=True).start()
        self._started = True

    def set_decoder(self, dec: Optional[object]) -> None:
        self._decoder = dec

    def update_nal_length_size(self, n: int) -> None:
        if n in (1, 2, 3, 4):
            self._nal_length_size = int(n)

    def enqueue(self, b: bytes, ts: Optional[float]) -> None:
        try:
            if self.qsize() >= max(2, self._backlog_trigger - 1):
                # Gate and request resync on next keyframe
                self._on_backlog_gate()
                self.clear()
                self._presenter.clear(Source.VT)
                self._request_keyframe()
            else:
                self._in_q.put_nowait((b, ts))
                self._enqueued += 1
        except queue.Full:
            # If enqueuing fails, gate and request keyframe
            self._on_backlog_gate()
            self._request_keyframe()

    def qsize(self) -> int:
        try:
            return int(self._in_q.qsize())
        except Exception:
            logger.debug("VTPipeline: qsize failed", exc_info=True)
            return 0

    def clear(self) -> None:
        try:
            while self._in_q.qsize() > 0:
                try:
                    _ = self._in_q.get_nowait()
                except Exception:
                    break
        except Exception:
            logger.debug("VTPipeline: clear failed", exc_info=True)

    def counts(self) -> Tuple[int, int, int] | None:
        try:
            if self._decoder is None:
                return None
            return self._decoder.counts()
        except Exception:
            return None

    def last_payload_info(self) -> Tuple[object | None, bool]:
        return self._last_payload, bool(self._last_persistent)

    # Internal decode submit with normalization
    def _decode_vt_live(self, data: bytes, ts: float | None) -> None:
        try:
            if not data or self._decoder is None:
                return
            target_len = int(self._nal_length_size or 4)
            if is_annexb(data):
                avcc_au = annexb_to_avcc(data, out_len=target_len)
            else:
                # Repackage AVCC to match current length size when possible
                nals = split_avcc_by_len(data, 4)
                if not nals:
                    nals = split_avcc_by_len(data, 2)
                if nals:
                    out = bytearray()
                    for n in nals:
                        out.extend(len(n).to_bytes(target_len, 'big'))
                        out.extend(n)
                    avcc_au = bytes(out)
                else:
                    avcc_au = data
            ok = self._decoder.decode(avcc_au, ts)
            if not ok:
                self._errors += 1
                if self._errors <= 3 or (self._errors % 50 == 0):
                    logger.warning("VTPipeline: VT decode submit failed (errors=%d)", self._errors)
                return
            self._errors = 0
            # Flush early on first few enqueues to reduce startup latency
            try:
                if self._enqueued <= 3:
                    self._decoder.flush()
            except Exception:
                logger.debug("VTPipeline: flush failed", exc_info=True)
        except Exception as e:
            self._errors += 1
            logger.exception("VTPipeline: decode/map failed (%d): %s", self._errors, e)
            if self._errors >= 3:
                logger.error("VTPipeline: disabling decoder after repeated errors")
                self._decoder = None
