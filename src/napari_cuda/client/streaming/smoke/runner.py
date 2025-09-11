from __future__ import annotations

"""
Smoke runners: tight loops for encode→VT and encode→PyAV paths.

Avoids duplication in orchestrator while keeping zero hidden state.
"""

import base64
import time
from typing import Callable, Optional

from napari_cuda.client.streaming.smoke.generators import make_generator
from napari_cuda.client.streaming.smoke.submit import submit_pyav, submit_vt
from napari_cuda.client.streaming.types import Source, TimestampMode
from napari_cuda.codec.h264_encoder import H264Encoder, EncoderConfig


def run_encode_smoke(
    *,
    width: int,
    height: int,
    fps: float,
    smoke_mode: str,
    presenter: "object",
    source_mux: "object",
    vt_in_q: "object",
    vt_backlog_trigger: int,
    vt_latency_s: Optional[float] = None,
    init_vt_from_avcc: Optional[Callable[[str, int, int], None]] = None,
    after_vt_init: Optional[Callable[[], None]] = None,
    on_enqueued: Optional[Callable[[int], None]] = None,
    logger: Optional["object"] = None,
) -> None:
    # Mode and latency
    try:
        presenter.set_mode(TimestampMode.ARRIVAL)
        if vt_latency_s is not None:
            presenter.set_latency(float(vt_latency_s))
        source_mux.set_active(Source.VT)
    except Exception:
        pass
    # Encoder + generator
    enc = None
    def _make_encoder() -> H264Encoder:
        for name in ("h264_videotoolbox", "libx264", "h264"):
            try:
                e = H264Encoder(EncoderConfig(name=name, width=width, height=height, fps=fps))
                e.open()
                if logger:
                    try:
                        logger.info("encode-smoke: encoder=%s opened (w=%d h=%d fps=%.1f)", name, width, height, fps)
                    except Exception:
                        pass
                return e
            except Exception:
                if logger:
                    try:
                        logger.debug("encode-smoke: failed to open encoder=%s", name, exc_info=True)
                    except Exception:
                        pass
                continue
        raise RuntimeError("No H.264 encoder available")
    enc = _make_encoder()
    gen = make_generator(smoke_mode, width, height, fps)
    # Loop
    avcc_bytes: Optional[bytes] = None
    frame_idx = 0
    first_log = 0
    while True:
        t0 = time.perf_counter()
        ts_this = float(frame_idx) / float(max(1.0, fps))
        rgb = gen(frame_idx)
        try:
            au_list = enc.encode_rgb_frame(rgb, pixfmt='rgb24', pts=ts_this)
        except Exception:
            if logger:
                try:
                    logger.debug('encode-smoke: encode failed', exc_info=True)
                except Exception:
                    pass
            try:
                enc.close()
            except Exception:
                pass
            enc = _make_encoder()
            continue
        if avcc_bytes is None and init_vt_from_avcc is not None:
            try:
                avcc_b = enc.get_avcc_config()
            except Exception:
                avcc_b = None
            if avcc_b:
                avcc_bytes = avcc_b
                try:
                    init_vt_from_avcc(base64.b64encode(avcc_bytes).decode('ascii'), width, height)
                except Exception:
                    pass
                if after_vt_init is not None:
                    try:
                        after_vt_init()
                    except Exception:
                        pass
                if logger:
                    try:
                        logger.info('encode-smoke: initialized VT from encoder avcC')
                    except Exception:
                        pass
        try:
            submitted = submit_vt(au_list, vt_in_q, presenter, vt_backlog_trigger, ts_this)
            if on_enqueued:
                on_enqueued(int(submitted))
            if submitted and first_log < 3 and logger:
                try:
                    logger.info('encode-smoke: submitted AU len=%d ts=%.6f', len(au_list[0].payload), float(au_list[0].pts or ts_this))
                except Exception:
                    pass
                first_log += 1
        except Exception:
            if logger:
                try:
                    logger.debug('encode-smoke: submit failed', exc_info=True)
                except Exception:
                    pass
        # pacing
        target = 1.0 / max(1.0, fps)
        sleep = target - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)
        frame_idx += 1


def run_pyav_smoke(
    *,
    width: int,
    height: int,
    fps: float,
    smoke_mode: str,
    presenter: "object",
    source_mux: "object",
    pyav_in_q: "object",
    pyav_backlog_trigger: int,
    pyav_latency_s: Optional[float] = None,
    on_enqueued: Optional[Callable[[int], None]] = None,
    logger: Optional["object"] = None,
) -> None:
    # Mode and latency
    try:
        presenter.set_mode(TimestampMode.ARRIVAL)
        if pyav_latency_s is not None:
            presenter.set_latency(float(pyav_latency_s))
        source_mux.set_active(Source.PYAV)
    except Exception:
        pass
    # Encoder + generator
    enc = None
    def _make_encoder() -> H264Encoder:
        for name in ("h264_videotoolbox", "libx264", "h264"):
            try:
                e = H264Encoder(EncoderConfig(name=name, width=width, height=height, fps=fps))
                e.open()
                if logger:
                    try:
                        logger.info("pyav-smoke: encoder=%s opened", name)
                    except Exception:
                        pass
                return e
            except Exception:
                if logger:
                    try:
                        logger.debug("pyav-smoke: failed to open encoder=%s", name, exc_info=True)
                    except Exception:
                        pass
                continue
        raise RuntimeError("No H.264 encoder available")
    enc = _make_encoder()
    gen = make_generator(smoke_mode, width, height, fps)
    # Loop
    frame_idx = 0
    first_log = 0
    started = False
    while True:
        t0 = time.perf_counter()
        ts_this = float(frame_idx) / float(max(1.0, fps))
        rgb = gen(frame_idx)
        try:
            au_list = enc.encode_rgb_frame(rgb, pixfmt='rgb24', pts=ts_this)
        except Exception:
            if logger:
                try:
                    logger.debug('pyav-smoke: encode failed', exc_info=True)
                except Exception:
                    pass
            try:
                enc.close()
            except Exception:
                pass
            enc = _make_encoder()
            continue
        for au in au_list:
            if not started:
                if not au.is_keyframe:
                    continue
                started = True
            try:
                submitted = submit_pyav([au], pyav_in_q, presenter, pyav_backlog_trigger, ts_this)
                if on_enqueued:
                    on_enqueued(int(submitted))
                if submitted and first_log < 3 and logger:
                    try:
                        logger.info('pyav-smoke: submitted AU len=%d ts=%.6f', len(au.payload), float(au.pts or ts_this))
                    except Exception:
                        pass
                    first_log += 1
            except Exception:
                if logger:
                    try:
                        logger.debug('pyav-smoke: submit failed', exc_info=True)
                    except Exception:
                        pass
        # pacing
        target = 1.0 / max(1.0, fps)
        sleep = target - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)
        frame_idx += 1

