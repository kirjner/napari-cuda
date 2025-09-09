from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VTLiveDecoder:
    """Thin wrapper over the macOS VideoToolbox shim decoder used by the client.

    Exposes the minimal API used by StreamingCanvas:
    - decode(avcc_au: bytes, ts: Optional[float]) -> bool
    - get_frame_nowait() -> Optional[Tuple[object, Optional[float]]]
    - counts() -> tuple[int, int, int]
    - flush() -> None
    """

    def __init__(self, avcc: bytes, width: int, height: int) -> None:
        from napari_cuda.client.vt_shim import VTShimDecoder  # type: ignore

        self._shim = VTShimDecoder(avcc, width, height)

    def decode(self, avcc_au: bytes, ts: Optional[float]) -> bool:
        return bool(self._shim.decode(avcc_au, ts))

    def get_frame_nowait(self) -> Optional[Tuple[object, Optional[float]]]:
        item = self._shim.get_frame_nowait()
        if not item:
            return None
        return item

    def counts(self) -> tuple[int, int, int]:
        try:
            return self._shim.counts()
        except Exception:
            logger.debug("VT shim counts failed", exc_info=True)
            return (0, 0, -1)

    def flush(self) -> None:
        try:
            self._shim.flush()
        except Exception:
            logger.debug("VT shim flush failed", exc_info=True)

