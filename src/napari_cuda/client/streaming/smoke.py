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
    seconds = float(os.getenv('NAPARI_CUDA_VT_SMOKE_SECS', '3'))
    fps = float(os.getenv('NAPARI_CUDA_VT_SMOKE_FPS', '60'))
    nframes = max(1, int(seconds * fps))
    logger.debug("VT smoke: generating %d frames at %dx%d @ %.1ffps", nframes, width, height, fps)

    try:
        import av
    except Exception:
        logger.exception("VT smoke requires PyAV")
        return

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
        'annexb': '1',
    }

    def split_annexb(data: bytes) -> list[bytes]:
        out: list[bytes] = []
        i = 0
        n = len(data)
        idx: list[int] = []
        while i + 3 <= n:
            if data[i:i+3] == b'\x00\x00\x01':
                idx.append(i); i += 3
            elif i + 4 <= n and data[i:i+4] == b'\x00\x00\x00\x01':
                idx.append(i); i += 4
            else:
                i += 1
        idx.append(n)
        for a, b in zip(idx, idx[1:]):
            j = a
            while j < b and data[j] == 0:
                j += 1
            if j + 3 <= b and data[j:j+3] == b'\x00\x00\x01':
                j += 3
            elif j + 4 <= b and data[j:j+4] == b'\x00\x00\x00\x01':
                j += 4
            nal = data[j:b]
            if nal:
                out.append(nal)
        return out

    def annexb_to_avcc(data: bytes) -> bytes:
        nals = split_annexb(data)
        out = bytearray()
        for n in nals:
            out.extend(len(n).to_bytes(4, 'big'))
            out.extend(n)
        return bytes(out)

    def build_avcc(sps: bytes, pps: bytes) -> bytes:
        if len(sps) < 4:
            raise ValueError('SPS too short for avcC')
        profile = sps[1]
        compat = sps[2]
        level = sps[3]
        avcc = bytearray()
        avcc.append(1)
        avcc.append(profile)
        avcc.append(compat)
        avcc.append(level)
        avcc.append(0xFF)
        avcc.append(0xE1 | 1)
        avcc.extend(len(sps).to_bytes(2, 'big'))
        avcc.extend(sps)
        avcc.append(1)
        avcc.extend(len(pps).to_bytes(2, 'big'))
        avcc.extend(pps)
        return bytes(avcc)

    # Initialize VT when SPS/PPS observed
    try:
        from napari_cuda.client.vt_decoder import VideoToolboxDecoder, is_vt_available
    except Exception as e:
        logger.error("VT frameworks missing: %s", e)
        return
    if not is_vt_available():
        logger.error("VideoToolbox not available on this system")
        return

    avcc_bytes = None
    vt = None
    decoded = 0

    for i in range(nframes):
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
        for p in packets:
            try:
                data = p.to_bytes()
            except Exception:
                try:
                    data = bytes(p)
                except Exception:
                    data = memoryview(p).tobytes()
            # Extract SPS/PPS once
            if avcc_bytes is None:
                nals = split_annexb(data)
                sps = next((n for n in nals if (n[0] & 0x1F) == 7), None)
                pps = next((n for n in nals if (n[0] & 0x1F) == 8), None)
                if sps and pps:
                    avcc_bytes = build_avcc(sps, pps)
                    vt = VideoToolboxDecoder(avcc_bytes, width, height)
                    logger.info("VT smoke: initialized VT")
            if vt is None:
                continue
            avcc_au = annexb_to_avcc(data)
            img_buf = vt.decode(avcc_au)
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
    logger.debug("VT smoke: decoded %d frames", decoded)

