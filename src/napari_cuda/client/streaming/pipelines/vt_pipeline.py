from __future__ import annotations

"""
VTPipeline - encapsulates VT submit/drain and avcC/AnnexB normalization.

- Owns an input queue and two workers: submit (normalize+decode) and drain (present).
- Coordinates with the coordinator using small callbacks for gating and caching.
"""

import logging
import queue
import time
import threading
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
        metrics: object | None = None,
        schedule_next_wake: Callable[[], None] | None = None,
    ) -> None:
        self._presenter = presenter
        self._source_mux = source_mux
        self._scene_canvas = scene_canvas
        # Input queue capacity (tunable via env to limit in-flight frames)
        import os as _os
        try:
            inq_cap = int(_os.getenv('NAPARI_CUDA_VT_INPUT_Q', '64') or '64')
        except Exception:
            inq_cap = 64
        self._in_q: "queue.Queue[tuple[bytes | memoryview, float | None]]" = queue.Queue(maxsize=max(2, int(inq_cap)))
        self._backlog_trigger = int(backlog_trigger)
        self._decoder = None  # VTLiveDecoder-like
        self._started = False
        self._errors = 0
        self._enqueued = 0
        self._nal_length_size = 4
        self._metrics = metrics
        # Optional periodic flush to nudge VT to retire internal surfaces.
        # Disabled by default; enable with NAPARI_CUDA_VT_PERIODIC_FLUSH_S.
        import os as _os
        try:
            self._periodic_flush_s = float(_os.getenv('NAPARI_CUDA_VT_PERIODIC_FLUSH_S', '0') or '0')
        except Exception:
            self._periodic_flush_s = 0.0
        self._last_flush_time = 0.0
        # Gate per-decode GUI updates; now disabled by default since presenter wake drives draws
        import os as _os
        try:
            self._post_decode_update = (_os.getenv('NAPARI_CUDA_CLIENT_DECODE_UPDATE', '0') or '0') in ('1','true','yes')
        except Exception:
            self._post_decode_update = False
        # Coordination hooks
        self._is_gated = is_gated or (lambda: False)
        self._on_backlog_gate = on_backlog_gate or (lambda: None)
        self._request_keyframe = request_keyframe or (lambda: None)
        self._on_cache_last = on_cache_last or (lambda _pb, _persistent: None)
        # Last VT payload for redraw fallback (optional cache)
        try:
            self._hold_cache = ((_os.getenv('NAPARI_CUDA_VT_HOLD_CACHE', '1') or '1').lower() in ('1','true','yes','on'))
        except Exception:
            self._hold_cache = True
        self._last_payload = None  # type: ignore[var-annotated]
        self._last_persistent = False
        # Scheduling hook to coordinator (optional)
        self._schedule_next_wake = schedule_next_wake or (lambda: None)
        # Drain coordination to reduce busy polling
        self._drain_event = threading.Event()
        # Repack scratch buffer to avoid per-frame allocs
        self._repack_buf = bytearray()

    def start(self) -> None:
        if self._started:
            return

        def _submit_worker() -> None:
            while True:
                try:
                    data, ts = self._in_q.get(timeout=0.1)
                except Exception:
                    # Timeout or queue error; nothing to submit
                    continue
                try:
                    self._decode_vt_live(data, ts)
                    # Decoded data may be ready for drain
                    self._drain_event.set()
                except Exception:
                    logger.exception("VTPipeline: decode submit failed")

        def _drain_worker() -> None:
            while True:
                try:
                    dec = self._decoder
                    if dec is None:
                        # Wait until decoder is set or new work arrives
                        self._drain_event.wait(0.05)
                        self._drain_event.clear()
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
                        # Optionally keep a last-frame cache; otherwise avoid extra retains
                        if self._hold_cache:
                            # Retain once for cache
                            vt.retain_frame(img_buf)
                            # Release previously cached payload
                            if self._last_payload is not None:
                                try:
                                    vt.release_frame(self._last_payload)
                                except Exception:
                                    logger.debug("VTPipeline: release previous cached payload failed", exc_info=True)
                            self._last_payload = img_buf
                            self._last_persistent = True
                            self._on_cache_last(self._last_payload, True)
                        else:
                            # No cache retained
                            self._last_payload = None
                            self._last_persistent = False
                            self._on_cache_last(None, False)
                        # Retain once for presenter-owned buffer entry
                        vt.retain_frame(img_buf)
                        self._presenter.submit(
                            SubmittedFrame(
                                source=Source.VT,
                                server_ts=float(pts) if pts is not None else None,
                                arrival_ts=time.perf_counter(),
                                payload=img_buf,
                                release_cb=vt.release_frame,
                            )
                        )
                        # Release the base reference obtained from vt_get_frame();
                        # retains above keep the buffer alive for cache/presenter as needed.
                        try:
                            vt.release_frame(img_buf)
                        except Exception:
                            logger.debug("VTPipeline: base release failed", exc_info=True)
                        # Nudge coordinator to schedule next-due wake (thread-safe via proxy)
                        try:
                            self._schedule_next_wake()
                        except Exception:
                            logger.debug("VTPipeline: schedule_next_wake failed", exc_info=True)
                    if drained and self._post_decode_update:
                        QtCore.QTimer.singleShot(0, self._scene_canvas.native.update)
                    # Optional periodic flush of VT async frames
                    if self._periodic_flush_s and self._periodic_flush_s > 0:
                        now = time.time()
                        if (now - self._last_flush_time) >= float(self._periodic_flush_s):
                            try:
                                dec.flush()  # type: ignore[attr-defined]
                            except Exception:
                                logger.debug("VTPipeline: periodic flush failed", exc_info=True)
                            self._last_flush_time = now
                    if not drained:
                        # Block briefly until new decoded frames are likely available
                        self._drain_event.wait(0.01)
                        self._drain_event.clear()
                except Exception:
                    logger.debug("VTPipeline: drain worker error", exc_info=True)

        Thread(target=_submit_worker, daemon=True).start()
        Thread(target=_drain_worker, daemon=True).start()
        # Optional periodic VT debug logging
        import os as _os
        _vt_debug = (_os.getenv('NAPARI_CUDA_VT_DEBUG', '0') or '0') in ('1','true','yes','on')
        if _vt_debug:
            def _vt_debugger() -> None:
                last = None
                while True:
                    try:
                        dec = self._decoder
                        if dec is not None and hasattr(dec, 'stats'):
                            s = dec.stats()  # type: ignore[attr-defined]
                            # s: (submits, outputs, qlen, drops, retains, releases)
                            if last is not None:
                                dt_sub = s[0] - last[0]
                                dt_out = s[1] - last[1]
                                dt_ret = s[4] - last[4]
                                dt_rel = s[5] - last[5]
                                outstanding = s[4] - s[5]
                                logger.info(
                                    "VT dbg: sub=%d out=%d q=%d drop=%d ret=%d rel=%d outstd=%d (+ret=%d +rel=%d)",
                                    s[0], s[1], s[2], s[3], s[4], s[5], outstanding, dt_ret, dt_rel,
                                )
                            else:
                                logger.info(
                                    "VT dbg: sub=%d out=%d q=%d drop=%d ret=%d rel=%d", s[0], s[1], s[2], s[3], s[4], s[5]
                                )
                            last = s
                        time.sleep(1.0)
                    except Exception:
                        time.sleep(1.0)
            Thread(target=_vt_debugger, daemon=True).start()
        self._started = True

    def set_decoder(self, dec: Optional[object]) -> None:
        self._decoder = dec

    def update_nal_length_size(self, n: int) -> None:
        if n in (1, 2, 3, 4):
            self._nal_length_size = int(n)

    def enqueue(self, b: bytes | memoryview, ts: Optional[float]) -> None:
        try:
            qd = self.qsize()
            if self._metrics is not None:
                self._metrics.set('napari_cuda_client_vt_qdepth', float(qd))
            if qd >= max(2, self._backlog_trigger - 1):
                # Gate and request resync on next keyframe
                self._on_backlog_gate()
                if self._metrics is not None:
                    self._metrics.inc('napari_cuda_client_vt_backlog_gates', 1.0)
                self.clear()
                self._presenter.clear(Source.VT)
                self._request_keyframe()
            else:
                self._in_q.put_nowait((b, ts))
                self._enqueued += 1
                self._drain_event.set()
        except queue.Full:
            # If enqueuing fails, gate and request keyframe
            self._on_backlog_gate()
            if self._metrics is not None:
                self._metrics.inc('napari_cuda_client_vt_backlog_gates', 1.0)
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
    def _decode_vt_live(self, data: bytes | memoryview, ts: float | None) -> None:
        try:
            if not data or self._decoder is None:
                return
            t0 = time.perf_counter()
            target_len = int(self._nal_length_size or 4)
            if is_annexb(data):
                avcc_au = annexb_to_avcc(data, out_len=target_len)
            else:
                # Repackage AVCC to match current length size when possible
                nals = split_avcc_by_len(data, 4)
                if not nals:
                    nals = split_avcc_by_len(data, 2)
                if nals:
                    out = self._repack_buf
                    out.clear()
                    for n in nals:
                        out.extend(len(n).to_bytes(target_len, 'big'))
                        out.extend(n)
                    avcc_au = bytes(out)
                else:
                    # Ensure bytes object for VT shim
                    avcc_au = data if isinstance(data, (bytes, bytearray)) else bytes(data)
            t_dec0 = time.perf_counter()
            ok = self._decoder.decode(avcc_au, ts)
            t_dec1 = time.perf_counter()
            if self._metrics is not None:
                self._metrics.observe_ms('napari_cuda_client_vt_decode_ms', (t_dec1 - t_dec0) * 1000.0)
            if not ok:
                self._errors += 1
                if self._errors <= 3 or (self._errors % 50 == 0):
                    logger.warning("VTPipeline: VT decode submit failed (errors=%d)", self._errors)
                if self._metrics is not None:
                    self._metrics.inc('napari_cuda_client_vt_decode_errors', 1.0)
                return
            self._errors = 0
            # Flush early on first few enqueues to reduce startup latency
            try:
                if self._enqueued <= 3:
                    self._decoder.flush()
            except Exception:
                logger.debug("VTPipeline: flush failed", exc_info=True)
            t1 = time.perf_counter()
            if self._metrics is not None:
                self._metrics.observe_ms('napari_cuda_client_vt_submit_ms', (t1 - t0) * 1000.0)
        except Exception as e:
            self._errors += 1
            logger.exception("VTPipeline: decode/map failed (%d): %s", self._errors, e)
            if self._errors >= 3:
                logger.error("VTPipeline: disabling decoder after repeated errors")
                self._decoder = None
