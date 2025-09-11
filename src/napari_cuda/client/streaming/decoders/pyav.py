from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PyAVDecoder:
    """Thin wrapper over PyAV H.264 decoder producing RGB arrays.

    Parameters
    - stream_format: 'avcc' or 'annexb'
    - pixfmt: 'rgb24' or 'bgr24' (converted to RGB)
    - swap_rb: force channel swap to diagnose RB issues
    """

    def __init__(self, stream_format: str = 'avcc', pixfmt: str = 'rgb24', swap_rb: bool = False) -> None:
        import av  # local import to avoid hard dep at import time

        self.codec = av.CodecContext.create('h264', 'r')
        self.stream_format = 'annexb' if (stream_format or '').lower().startswith('annex') else 'avcc'
        self.swap_rb = bool(swap_rb)
        self.pixfmt = pixfmt if pixfmt in {'rgb24', 'bgr24'} else 'rgb24'

    def decode(self, data: bytes) -> Optional[np.ndarray]:
        import av
        # Be resilient to state/format races: always normalize to Annex B
        try:
            from napari_cuda.codec.avcc import normalize_to_annexb  # type: ignore
            b, _ = normalize_to_annexb(data)
            packet = av.Packet(b)
        except Exception:
            packet = av.Packet(data)
        try:
            frames = self.codec.decode(packet)
        except av.AVError:
            # Common on boundary/flush-like packets; skip quietly
            logger.debug("PyAVDecoder: codec.decode AVError", exc_info=True)
            return None
        for frame in frames or []:
            try:
                arr = frame.to_ndarray(format=self.pixfmt)
            except Exception:
                logger.debug("PyAVDecoder: frame conversion failed", exc_info=True)
                continue
            if self.pixfmt == 'bgr24':
                arr = arr[..., ::-1].copy()
            if self.swap_rb:
                arr = arr[..., ::-1].copy()
            if not hasattr(self, '_logged_once'):
                try:
                    h, w, _c = arr.shape
                    logger.info("Client(PyAV) decode to %s -> RGB array (%dx%d)", self.pixfmt, w, h)
                except Exception:
                    logger.debug("PyAV first-frame logging failed", exc_info=True)
                setattr(self, '_logged_once', True)
            return arr
        return None
