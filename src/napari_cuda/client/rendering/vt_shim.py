from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _is_darwin() -> bool:
    return sys.platform == 'darwin'


class VTShimDecoder:
    """Thin Python wrapper around the native VT shim (macOS only).

    Phase 1: returns contiguous RGB frames (CPU copy). Phase 2 will switch to zero-copy GL.
    """

    def __init__(self, avcc: bytes, width: int, height: int, pixfmt: str | None = None) -> None:
        if not _is_darwin():
            raise RuntimeError("VideoToolbox shim is only available on macOS")
        try:
            from napari_cuda import _vt as vt  # type: ignore
        except Exception as e:
            raise RuntimeError(f"VT shim extension not available: {e}")
        self._vt = vt
        self._w = int(width)
        self._h = int(height)
        pf_env = (os.getenv('NAPARI_CUDA_CLIENT_VT_PIXFMT') or '').upper()
        # Default to BGRA for broader compatibility and faster CPU mapping
        pf = (pixfmt or pf_env or 'BGRA').upper()
        fourcc = 0x34323076 if pf in ('NV12', '420BIPLANAR') else 0x42475241  # 'NV12' or 'BGRA'
        self._sess = vt.create(avcc, self._w, self._h, fourcc)
        logger.info("VT shim session created: %dx%d pixfmt=%s", self._w, self._h, pf)

    def close(self) -> None:
        try:
            self._sess = None
        except Exception:
            logger.debug("VTShimDecoder.close: failed to clear session", exc_info=True)

    def flush(self) -> None:
        try:
            self._vt.flush(self._sess)
        except Exception:
            logger.debug("VTShimDecoder.flush: failed", exc_info=True)

    def counts(self) -> tuple[int, int, int]:
        try:
            a, b, c = self._vt.counts(self._sess)
            return int(a), int(b), int(c)
        except Exception:
            logger.debug("VTShimDecoder.counts: failed", exc_info=True)
            return (0, 0, 0)

    def stats(self) -> tuple[int, int, int, int, int, int]:
        """Return extended stats: (submits, outputs, qlen, drops, retains, releases)."""
        try:
            a, b, c, d, e, f = self._vt.stats(self._sess)
            return int(a), int(b), int(c), int(d), int(e), int(f)
        except Exception:
            logger.debug("VTShimDecoder.stats: failed", exc_info=True)
            return (0, 0, 0, 0, 0, 0)

    def decode(self, avcc_au: bytes, pts: Optional[float]) -> bool:
        try:
            rc = self._vt.decode(self._sess, avcc_au, float(pts) if pts is not None else 0.0)
            return rc == 0
        except Exception as e:
            logger.debug("VT shim decode failed: %s", e)
            return False

    def get_frame_nowait(self) -> Optional[tuple[object, Optional[float]]]:
        try:
            res = self._vt.get_frame(self._sess, 0.0)
            if res is None:
                return None
            cap, pts = res
            return cap, float(pts)
        except Exception:
            logger.debug("VTShimDecoder.get_frame_nowait: failed", exc_info=True)
            return None

    def get_frame(self, timeout: Optional[float] = None) -> Optional[tuple[object, Optional[float]]]:
        try:
            res = self._vt.get_frame(self._sess, float(timeout) if timeout is not None else 0.0)
            if res is None:
                return None
            cap, pts = res
            return cap, float(pts)
        except Exception:
            logger.debug("VTShimDecoder.get_frame: failed", exc_info=True)
            return None

    # Helpers to map/release frames from the shim capsule
    @staticmethod
    def map_to_rgb(_vt_module, capsule) -> np.ndarray:
        # Calls C helper to obtain contiguous RGB bytes and dimensions
        data, w, h = _vt_module.map_to_rgb(capsule)
        arr = np.frombuffer(data, dtype=np.uint8).reshape((int(h), int(w), 3))
        return arr

    def map_capsule_to_rgb(self, capsule) -> np.ndarray:
        return VTShimDecoder.map_to_rgb(self._vt, capsule)
