from __future__ import annotations

import asyncio
import base64
import ctypes
import io
import logging
import os
from fractions import Fraction
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def start_vt_smoke_thread(put_frame: Callable[[np.ndarray], None]) -> None:
    """Start a VT smoke test in the current thread's event loop.

    Generates a synthetic H.264 stream with PyAV, decodes via VideoToolbox,
    and forwards RGB frames via `put_frame`.
    """

    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.get_event_loop().run_until_complete(_run_vt_smoke_test(put_frame))


async def _run_vt_smoke_test(put_frame: Callable[[np.ndarray], None]) -> None:
    width = int(os.getenv('NAPARI_CUDA_VT_SMOKE_W', '1280'))
    height = int(os.getenv('NAPARI_CUDA_VT_SMOKE_H', '720'))
    seconds = float(os.getenv('NAPARI_CUDA_VT_SMOKE_SECS', '0'))
    fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
    nframes = max(1, int(seconds * fps)) if seconds > 0 else None
    if nframes:
        logger.debug("VT smoke: generating %d frames at %dx%d @ %.1ffps", nframes, width, height, fps)
    else:
        logger.debug("VT smoke: generating infinite frames at %dx%d @ %.1ffps", width, height, fps)

    try:
        import av
    except Exception:
        logger.exception("VT smoke requires PyAV")
        return

    # Prefer AVCC end-to-end; allow Python packer fallback if Cython is missing
    os.environ.setdefault('NAPARI_CUDA_ALLOW_PY_FALLBACK', '1')

    enc = av.CodecContext.create('h264', 'w')
    enc.width = width
    enc.height = height
    enc.pix_fmt = 'yuv420p'
    enc.time_base = Fraction(1, int(round(fps)))
    enc.options = {
        'tune': 'zerolatency',
        'preset': 'veryfast',
        'bf': '0',
        'keyint': str(int(round(fps))),
        'sc_threshold': '0',
        # Let packer normalize; annexb may be ignored by some encoders
        'annexb': '0',
        'x264-params': f"keyint={int(round(fps))}:scenecut=0:repeat-headers=1",
    }

    from napari_cuda.server.bitstream import (
        ParamCache,
        pack_to_avcc,
        build_avcc_config,
    )

    # Initialize VT when SPS/PPS observed
    try:
        from napari_cuda.client.vt_decoder import VideoToolboxDecoder, is_vt_available
    except Exception as e:
        logger.error("VT frameworks missing: %s", e)
        return
    if not is_vt_available():
        logger.error("VideoToolbox not available on this system")
        return

    avcc_bytes: bytes | None = None
    vt = None
    decoded = 0
    cache = ParamCache()
    started = False

    i = 0
    while (nframes is None) or (i < nframes):
        # RGB gradient test pattern
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        r = (xv * 255).astype(np.uint8)
        g = (yv * 255).astype(np.uint8)
        b = ((xv * 0.5 + yv * 0.5) * 255).astype(np.uint8)
        frame = np.dstack([r, g, b])
        try:
            vframe = av.VideoFrame.from_ndarray(frame, format='rgb24')
            packets = enc.encode(vframe)
        except Exception:
            logger.debug("VT smoke: encode failed", exc_info=True)
            continue
        # Assemble packets for this frame into one AVCC AU using server packer
        payloads: list[bytes] = []
        for p in packets:
            try:
                payloads.append(p.to_bytes())
            except Exception:
                try:
                    payloads.append(bytes(p))
                except Exception:
                    payloads.append(memoryview(p).tobytes())
        au, is_key = pack_to_avcc(payloads, cache)
        if au is None:
            i += 1
            continue
        # Initialize VT from avcC built from cached SPS/PPS
        if avcc_bytes is None:
            try:
                avcc_b = build_avcc_config(cache)
                if avcc_b:
                    avcc_bytes = avcc_b
                    vt = VideoToolboxDecoder(avcc_bytes, width, height)
                    logger.info("VT smoke: initialized VT from packer avcC (SPS/PPS)")
            except Exception:
                logger.debug("VT smoke: build avcC failed", exc_info=True)
        if vt is None:
            i += 1
            continue
        if not started and not is_key:
            i += 1
            continue
        started = True
        img_buf = vt.decode(au)
        if img_buf is None:
            continue
            try:
                from Quartz import CoreVideo as CV  # type: ignore
                CV.CVPixelBufferLockBaseAddress(img_buf, 0)
                w = CV.CVPixelBufferGetWidth(img_buf)
                h = CV.CVPixelBufferGetHeight(img_buf)
                bpr = CV.CVPixelBufferGetBytesPerRow(img_buf)
                base = CV.CVPixelBufferGetBaseAddress(img_buf)
                size = int(bpr) * int(h)
                ctype_arr = ctypes.cast(int(base), ctypes.POINTER(ctypes.c_ubyte * size)).contents
                bgra = np.frombuffer(ctype_arr, dtype=np.uint8).reshape((int(h), int(bpr)//4, 4))
                rgb = bgra[:, :int(w), [2, 1, 0]].copy()
            finally:
                try:
                    CV.CVPixelBufferUnlockBaseAddress(img_buf, 0)
                except Exception:
                    logger.debug("VT smoke: unlock failed", exc_info=True)
            put_frame(rgb)
            decoded += 1
        i += 1
    logger.debug("VT smoke: decoded %d frames", decoded)
