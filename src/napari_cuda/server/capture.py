"""Capture façade that composes GL capture, CUDA interop, and frame pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
import logging
import time
from threading import Lock

from napari_cuda.server.rendering.gl_capture import GLCapture
from napari_cuda.server.rendering.cuda_interop import CudaInterop
from napari_cuda.server.rendering.frame_pipeline import FramePipeline


logger = logging.getLogger(__name__)


@dataclass
class FrameTimings:
    render_ms: float
    blit_gpu_ns: Optional[int]
    blit_cpu_ms: float
    map_ms: float
    copy_ms: float
    convert_ms: float
    encode_ms: float
    pack_ms: float
    total_ms: float
    packet_bytes: Optional[int]
    capture_wall_ts: float = 0.0


class CaptureFacade:
    """Own the capture stack (GL capture, CUDA interop, frame pipeline)."""

    def __init__(self, *, width: int, height: int) -> None:
        self.gl_capture = GLCapture(width, height)
        self.cuda = CudaInterop(width, height)
        self.pipeline = FramePipeline(
            gl_capture=self.gl_capture,
            cuda=self.cuda,
            width=width,
            height=height,
            debug=None,
        )

    # --- Configuration --------------------------------------------------

    @property
    def enc_input_format(self) -> str:
        return self.pipeline.enc_input_format

    def set_enc_input_format(self, fmt: str) -> None:
        self.pipeline.set_enc_input_format(fmt)

    # --- Lifecycle ------------------------------------------------------

    def ensure(self) -> None:
        self.gl_capture.ensure()

    @property
    def texture_id(self) -> Optional[int]:
        return self.gl_capture.texture_id

    def initialize_cuda_interop(self) -> None:
        tex = self.gl_capture.texture_id
        if tex is None:
            raise RuntimeError("Capture texture is not available for CUDA interop")
        self.cuda.initialize(tex)

    def cleanup(self) -> None:
        try:
            self.cuda.cleanup()
        except Exception:  # pragma: no cover - defensive cleanup path
            logger.debug("CaptureFacade CUDA cleanup failed", exc_info=True)
        try:
            self.gl_capture.cleanup()
        except Exception:  # pragma: no cover - defensive cleanup path
            logger.debug("CaptureFacade GL cleanup failed", exc_info=True)

@dataclass
class FrameCapture:
    frame: Any
    blit_gpu_ns: Optional[int]
    blit_cpu_ms: float
    map_ms: float
    copy_ms: float
    convert_ms: float


@dataclass
class EncodedFrame:
    timings: FrameTimings
    packet: Optional[bytes]
    flags: int
    sequence: int


def capture_frame_for_encoder(
    facade: CaptureFacade,
    *,
    debug_cb: Optional[Callable[[int, int, int, object], None]] = None
) -> FrameCapture:
    t_b0 = time.perf_counter()
    blit_gpu_ns = facade.pipeline.capture_blit_gpu_ns()
    blit_cpu_ms = (time.perf_counter() - t_b0) * 1000.0
    map_ms, copy_ms = facade.pipeline.map_and_copy_to_torch(debug_cb)
    frame, convert_ms = facade.pipeline.convert_for_encoder()
    return FrameCapture(
        frame=frame,
        blit_gpu_ns=blit_gpu_ns,
        blit_cpu_ms=blit_cpu_ms,
        map_ms=map_ms,
        copy_ms=copy_ms,
        convert_ms=convert_ms,
    )


def encode_frame(
    *,
    capture: CaptureFacade,
    render_frame: Callable[[], float],
    obtain_encoder: Callable[[], Optional[Any]],
    encoder_lock: Optional[Lock],
    debug_dumper: Optional[Any],
    wall_time_fn: Callable[[], float] = time.time,
) -> EncodedFrame:
    """Render, capture, and encode a frame via the shared helpers."""

    wall_ts = wall_time_fn()
    render_ms = float(render_frame())

    debug_cb: Optional[Callable[[int, int, int, object], None]] = None
    if debug_dumper is not None:
        cfg = getattr(debug_dumper, "cfg", None)
        if cfg is not None and getattr(cfg, "enabled", False):
            remaining = getattr(cfg, "frames_remaining", 0)
            if remaining > 0:

                def _dump(tex_id: int, w: int, h: int, frame: object) -> None:
                    debug_dumper.dump_triplet(tex_id, w, h, frame)

                debug_cb = _dump

    capture_result = capture_frame_for_encoder(
        capture,
        debug_cb=debug_cb,
    )

    dst = capture_result.frame
    blit_gpu_ns = capture_result.blit_gpu_ns
    blit_cpu_ms = capture_result.blit_cpu_ms
    map_ms = capture_result.map_ms
    copy_ms = capture_result.copy_ms
    convert_ms = capture_result.convert_ms

    pkt: Optional[bytes]
    encode_ms: float
    pack_ms: float

    encoder = None
    if encoder_lock is not None:
        with encoder_lock:
            encoder = obtain_encoder()
            if encoder is None or not getattr(encoder, "is_ready", False):
                pkt = None
                encode_ms = 0.0
                pack_ms = 0.0
            else:
                pkt, timings = encoder.encode(dst)
                encode_ms = float(getattr(timings, "encode_ms", 0.0))
                pack_ms = float(getattr(timings, "pack_ms", 0.0))
    else:
        encoder = obtain_encoder()
        if encoder is None or not getattr(encoder, "is_ready", False):
            pkt = None
            encode_ms = 0.0
            pack_ms = 0.0
        else:
            pkt, timings = encoder.encode(dst)
            encode_ms = float(getattr(timings, "encode_ms", 0.0))
            pack_ms = float(getattr(timings, "pack_ms", 0.0))

    total_ms = render_ms + blit_cpu_ms + map_ms + copy_ms + convert_ms + encode_ms + pack_ms
    packet_bytes = len(pkt) if pkt is not None else None

    timings = FrameTimings(
        render_ms=render_ms,
        blit_gpu_ns=blit_gpu_ns,
        blit_cpu_ms=blit_cpu_ms,
        map_ms=map_ms,
        copy_ms=copy_ms,
        convert_ms=convert_ms,
        encode_ms=encode_ms,
        pack_ms=pack_ms,
        total_ms=total_ms,
        packet_bytes=packet_bytes,
        capture_wall_ts=wall_ts,
    )

    seq = int(getattr(encoder, "frame_index", 0)) if encoder is not None else 0

    return EncodedFrame(
        timings=timings,
        packet=pkt,
        flags=0,
        sequence=seq,
    )


__all__ = [
    "CaptureFacade",
    "FrameCapture",
    "FrameTimings",
    "EncodedFrame",
    "capture_frame_for_encoder",
    "encode_frame",
]
