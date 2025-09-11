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
import os
import struct
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Iterable, List, Optional, Tuple

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
    # Phase 5 additions
    mem_cap_mb: int = 0  # 0 = unlimited in-memory
    pre_path: Optional[str] = None  # directory for disk-backed cache


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
        self._cache_bytes: int = 0
        # Disk-backed cache state
        self._use_disk: bool = False
        self._disk_index: List[Tuple[int, int, float, bool]] = []  # (offset,length,pts,is_key)
        self._disk_fh = None
        self._disk_path: Optional[str] = None
        self._cache_ready = Event()
        self._submitted = 0
        self._dropped = 0
        self._cache_count: int = 0

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
            idx = self._wrap_index(i)
            payload, pts, is_key = self._fetch_cached(idx)
            # Ensure we start at a keyframe for both targets
            if i == 0 and not is_key:
                i = self._find_first_key()
                idx = self._wrap_index(i)
                payload, pts, is_key = self._fetch_cached(idx)
            ts_this = float(idx) / max(1.0, fps)
            self._enqueue_target(payload, pts if pts is not None else ts_this)
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
        cap_bytes = max(0, int(self.cfg.mem_cap_mb or 0)) * 1024 * 1024
        pre_dir = (self.cfg.pre_path or '').strip() or None
        # Helper: switch to disk and move any existing cache
        def _switch_to_disk() -> None:
            if self._use_disk:
                return
            if pre_dir is None:
                return
            os.makedirs(pre_dir, exist_ok=True)
            path = os.path.join(pre_dir, 'smoke_preencode_cache.bin')
            try:
                fh = open(path, 'wb+')
            except Exception as e:
                logger.warning('smoke: failed to open disk cache %s: %s', path, e)
                return
            # Write any existing memory cache to disk
            offset = 0
            for a in cache:
                offset = self._write_record(fh, offset, a.payload, float(a.pts or 0.0), bool(a.is_keyframe))
            self._disk_fh = fh
            self._disk_path = path
            self._use_disk = True
            # Build index for written items
            self._disk_index = []
            off = 0
            for a in cache:
                length = len(a.payload)
                self._disk_index.append((off, length, float(a.pts or 0.0), bool(a.is_keyframe)))
                off += 4 + 1 + 8 + length
            # Clear memory cache tracking
            cache.clear()
            self._cache_bytes = 0
            logger.info('smoke: using disk-backed preencode cache at %s', path)

        for i in range(n):
            ts_this = float(i) / max(1.0, fps)
            rgb = gen(i)
            aus = enc.encode_rgb_frame(rgb, pixfmt='rgb24', pts=ts_this)
            for a in aus:
                if self._use_disk:
                    try:
                        assert self._disk_fh is not None
                        off = self._disk_index[-1][0] + 4 + 1 + 8 + self._disk_index[-1][1] if self._disk_index else 0
                        new_off = self._write_record(self._disk_fh, off, a.payload, float(a.pts or 0.0), bool(a.is_keyframe))
                        self._disk_index.append((off, len(a.payload), float(a.pts or 0.0), bool(a.is_keyframe)))
                    except Exception:
                        logger.debug('smoke: disk write failed', exc_info=True)
                else:
                    cache.append(a)
                    self._cache_bytes += len(a.payload)
                    # Enforce memory cap
                    if cap_bytes > 0 and self._cache_bytes > cap_bytes:
                        if pre_dir is not None:
                            _switch_to_disk()
                        else:
                            while cache and self._cache_bytes > cap_bytes:
                                old = cache.pop(0)
                                self._cache_bytes -= len(old.payload)
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
        self._cache_count = len(self._disk_index) if self._use_disk else len(self._cache)
        self._cache_ready.set()
        dt = (time.perf_counter() - t_start) * 1000.0
        logger.info('smoke: preencoded %d frames in %.1f ms (avg %.2f ms/frame) (cache=%s)', self._cache_count, dt, dt / max(1, self._cache_count), 'disk' if self._use_disk else 'mem')

    def _write_record(self, fh, offset: int, payload: bytes, pts: float, is_key: bool) -> int:
        try:
            fh.seek(offset)
            hdr = struct.pack('>I B d', int(len(payload)), 1 if is_key else 0, float(pts))
            fh.write(hdr)
            fh.write(payload)
            return offset + len(hdr) + len(payload)
        except Exception:
            logger.debug('smoke: write_record failed', exc_info=True)
            return offset

    def _wrap_index(self, i: int) -> int:
        n = max(1, int(self._cache_count or 1))
        return int(i % n)

    def _fetch_cached(self, idx: int) -> Tuple[bytes, Optional[float], bool]:
        if not self._use_disk:
            au = self._cache[idx]
            return au.payload, au.pts, bool(au.is_keyframe)
        try:
            off, length, pts, is_key = self._disk_index[idx]
            assert self._disk_fh is not None
            self._disk_fh.seek(off + 4 + 1 + 8)
            data = self._disk_fh.read(length)
            return data, pts, is_key
        except Exception:
            logger.debug('smoke: disk fetch failed', exc_info=True)
            return b'', None, False

    def _find_first_key(self) -> int:
        if not self._use_disk:
            for j, a in enumerate(self._cache):
                if a.is_keyframe:
                    return j
            return 0
        for j, ent in enumerate(self._disk_index):
            if ent[3]:
                return j
        return 0

    def _open_encoder(self, w: int, h: int, fps: float) -> H264Encoder:
        import os
        last_err = None
        # Allow forcing an encoder via env, else keep the default preference order
        forced = os.getenv('NAPARI_CUDA_SMOKE_ENCODER')
        names = [forced] if forced else ["h264_videotoolbox", "libx264", "h264"]
        for name in names:
            if not name:
                continue
            try:
                opts = None
                if name in ("libx264", "h264"):
                    # Favor fastest for smoke to keep FPS high
                    preset = os.getenv('NAPARI_CUDA_SMOKE_X264_PRESET', 'ultrafast')
                    tune = os.getenv('NAPARI_CUDA_SMOKE_X264_TUNE', 'zerolatency')
                    xtra = os.getenv('NAPARI_CUDA_SMOKE_X264_PARAMS', '')
                    params = 'keyint=1:min-keyint=1:scenecut=0:repeat-headers=1'
                    if xtra:
                        params = f"{params}:{xtra}"
                    opts = {
                        'preset': str(preset),
                        'tune': str(tune),
                        'bf': '0',
                        'x264-params': params,
                        'annexb': '0',
                    }
                    br = os.getenv('NAPARI_CUDA_SMOKE_BITRATE')
                    if br:
                        # Accept raw ffmpeg value (e.g., 4000k) or numeric kbps
                        if br.isdigit():
                            opts['b'] = f"{br}k"
                        else:
                            opts['b'] = br
                elif name == "h264_videotoolbox":
                    # Realtime mode to reduce latency/CPU on macOS
                    opts = {'realtime': '1'}
                e = H264Encoder(EncoderConfig(name=name, width=w, height=h, fps=fps, options=opts))
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
            'cache': 'disk' if self._use_disk else 'mem',
            'count': int(self._cache_count),
        }
