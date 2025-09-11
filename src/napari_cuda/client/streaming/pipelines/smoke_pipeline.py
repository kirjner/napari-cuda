from __future__ import annotations

"""
SmokePipeline - consolidated smoke runner for VT and PyAV paths.

Modes:
- encode-live: generate RGB → encode per tick → enqueue to target pipeline
- preencode: encode N frames once → replay AccessUnits at target FPS

Assumptions:
- Target pipelines implement enqueue(bytes, ts), clear(), qsize().
- Presenter and source_mux are shared components controlling mode/latency/source.
"""

import base64
import logging
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Iterable, List, Optional

from napari_cuda.client.streaming.types import Source, TimestampMode
from napari_cuda.client.streaming.smoke.generators import make_generator
from napari_cuda.codec.h264_encoder import H264Encoder, EncoderConfig
from napari_cuda.codec.avcc import AccessUnit

logger = logging.getLogger(__name__)


@dataclass
class SmokeConfig:
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    smoke_mode: str = 'checker'
    preencode: bool = False
    pre_frames: int = 180
    backlog_trigger: int = 16
    target: str = 'vt'  # 'vt' or 'pyav'
    vt_latency_s: Optional[float] = None
    pyav_latency_s: Optional[float] = None


class SmokePipeline:
    def __init__(
        self,
        *,
        config: SmokeConfig,
        presenter: object,
        source_mux: object,
        pipeline: object,
        init_vt_from_avcc: Optional[callable] = None,
    ) -> None:
        self.cfg = config
        self.presenter = presenter
        self.source_mux = source_mux
        self.pipeline = pipeline
        self._init_vt_from_avcc = init_vt_from_avcc
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._cache: List[AccessUnit] = []
        self._cache_ready = Event()
        self._submitted = 0
        self._dropped = 0

    def start(self) -> None:
        # Configure presenter and active source
        self.presenter.set_mode(TimestampMode.ARRIVAL)
        tgt = (self.cfg.target or 'vt').lower()
        if tgt == 'pyav':
            if self.cfg.pyav_latency_s is not None:
                self.presenter.set_latency(float(self.cfg.pyav_latency_s))
            self.source_mux.set_active(Source.PYAV)
        else:
            if self.cfg.vt_latency_s is not None:
                self.presenter.set_latency(float(self.cfg.vt_latency_s))
            self.source_mux.set_active(Source.VT)

        if self.cfg.preencode:
            self._thread = Thread(target=self._run_preencode_replay, daemon=True)
        else:
            self._thread = Thread(target=self._run_encode_live, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=0.1)

    # --- encode-live path ---
    def _run_encode_live(self) -> None:
        w, h, fps = int(self.cfg.width), int(self.cfg.height), float(self.cfg.fps)
        gen = make_generator(self.cfg.smoke_mode, w, h, fps)
        enc = self._open_encoder(w, h, fps)
        first_log = 0
        started = False
        while not self._stop.is_set():
            t0 = time.perf_counter()
            idx = self._submitted
            ts_this = float(idx) / max(1.0, fps)
            rgb = gen(idx)
            try:
                aus = enc.encode_rgb_frame(rgb, pixfmt='rgb24', pts=ts_this)
            except Exception:
                logger.debug('smoke: encode failed; reopening encoder', exc_info=True)
                try:
                    enc.close()
                except Exception:
                    logger.debug('smoke: encoder close failed', exc_info=True)
                enc = self._open_encoder(w, h, fps)
                continue
            # For VT target, initialize from avcC once
            if self.cfg.target == 'vt' and self._init_vt_from_avcc is not None:
                try:
                    avcc_b = enc.get_avcc_config()
                except Exception:
                    avcc_b = None
                if avcc_b:
                    try:
                        self._init_vt_from_avcc(base64.b64encode(avcc_b).decode('ascii'), w, h)
                    except Exception:
                        logger.debug('smoke: init_vt_from_avcc failed', exc_info=True)
                    # call once
                    self._init_vt_from_avcc = None
            # Enqueue
            for au in aus:
                if self.cfg.target == 'pyav' and not started:
                    if not au.is_keyframe:
                        continue
                    started = True
                self._enqueue_target(au.payload, au.pts)
                if first_log < 3:
                    logger.info('smoke: submitted AU len=%d ts=%.6f', len(au.payload), float(au.pts or ts_this))
                    first_log += 1
            # pacing
            target = 1.0 / max(1.0, fps)
            sleep = target - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # --- preencode → replay path ---
    def _run_preencode_replay(self) -> None:
        w, h, fps = int(self.cfg.width), int(self.cfg.height), float(self.cfg.fps)
        n = max(1, int(self.cfg.pre_frames or 180))
        # Build cache once
        self._build_cache(w, h, fps, n)
        # Replay loop
        i = 0
        t0 = time.perf_counter()
        while not self._stop.is_set():
            ts_this = float(i % n) / max(1.0, fps)
            au = self._cache[i % n]
            # Ensure we start at a keyframe for both targets
            if i == 0 and not au.is_keyframe:
                # find first keyframe
                k = next((j for j, a in enumerate(self._cache) if a.is_keyframe), 0)
                i = k
                au = self._cache[i % n]
            self._enqueue_target(au.payload, au.pts if au.pts is not None else ts_this)
            i += 1
            # pacing
            target = 1.0 / max(1.0, fps)
            sleep = target - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
            t0 = time.perf_counter()

    def _build_cache(self, w: int, h: int, fps: float, n: int) -> None:
        gen = make_generator(self.cfg.smoke_mode, w, h, fps)
        enc = self._open_encoder(w, h, fps)
        cache: List[AccessUnit] = []
        avcc_b: Optional[bytes] = None
        t_start = time.perf_counter()
        for i in range(n):
            ts_this = float(i) / max(1.0, fps)
            rgb = gen(i)
            aus = enc.encode_rgb_frame(rgb, pixfmt='rgb24', pts=ts_this)
            cache.extend(aus)
            if avcc_b is None:
                try:
                    avcc_b = enc.get_avcc_config()
                except Exception:
                    avcc_b = None
        try:
            enc.close()
        except Exception:
            logger.debug('smoke: encoder close failed', exc_info=True)
        if self.cfg.target == 'vt' and self._init_vt_from_avcc is not None and avcc_b:
            try:
                self._init_vt_from_avcc(base64.b64encode(avcc_b).decode('ascii'), w, h)
            except Exception:
                logger.debug('smoke: init_vt_from_avcc(pre) failed', exc_info=True)
            self._init_vt_from_avcc = None
        self._cache = cache
        self._cache_ready.set()
        dt = (time.perf_counter() - t_start) * 1000.0
        logger.info('smoke: preencoded %d frames in %.1f ms (avg %.2f ms/frame)', len(cache), dt, dt / max(1, len(cache)))

    def _open_encoder(self, w: int, h: int, fps: float) -> H264Encoder:
        last_err = None
        for name in ("h264_videotoolbox", "libx264", "h264"):
            try:
                e = H264Encoder(EncoderConfig(name=name, width=w, height=h, fps=fps))
                e.open()
                logger.info("smoke: encoder=%s opened (w=%d h=%d fps=%.1f)", name, w, h, fps)
                return e
            except Exception as e:
                logger.debug("smoke: failed to open encoder=%s", name, exc_info=True)
                last_err = e
                continue
        raise RuntimeError(f"No H.264 encoder available: {last_err}")

    def _enqueue_target(self, payload: bytes, ts: Optional[float]) -> None:
        tgt = (self.cfg.target or 'vt').lower()
        pl = self.pipeline
        qsz = int(pl.qsize())
        if qsz >= max(2, int(self.cfg.backlog_trigger) - 1):
            pl.clear()
            self.presenter.clear(Source.PYAV if tgt == 'pyav' else Source.VT)
            self._dropped += 1
        if payload:
            pl.enqueue(payload, ts)
            self._submitted += 1
        else:
            logger.debug('smoke: skipped empty AU payload')

    # Introspection
    def stats(self) -> dict:
        return {
            'submitted': int(self._submitted),
            'dropped': int(self._dropped),
            'target': self.cfg.target,
            'mode': 'preencode' if self.cfg.preencode else 'encode-live',
        }
