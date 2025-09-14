from __future__ import annotations

import logging
from typing import Optional, Tuple

from napari_cuda.codec.avcc import is_annexb

logger = logging.getLogger(__name__)


class VTLiveDecoder:
    """Thin wrapper over the macOS VideoToolbox shim decoder used by the client.

    Exposes the minimal API used by StreamingCanvas:
    - decode(avcc_au: bytes, ts: Optional[float]) -> bool
    - get_frame_nowait() -> Optional[Tuple[object, Optional[float]]]
    - counts() -> tuple[int, int, int]
    - stats() -> tuple[int, int, int, int, int, int]
    - flush() -> None
    """

    def __init__(self, avcc: bytes, width: int, height: int) -> None:
        from napari_cuda.client.vt_shim import VTShimDecoder  # type: ignore

        self._shim = VTShimDecoder(avcc, width, height)

    def decode(self, avcc_au: bytes, ts: Optional[float]) -> bool:
        # Enforce invariants with assertions; let shim errors surface if any
        assert avcc_au, "empty AVCC access unit"
        assert not is_annexb(avcc_au), "VTLiveDecoder requires AVCC, got AnnexB input"
        return bool(self._shim.decode(avcc_au, ts))

    def get_frame_nowait(self) -> Optional[Tuple[object, Optional[float]]]:
        item = self._shim.get_frame_nowait()
        if not item:
            return None
        return item

    def counts(self) -> tuple[int, int, int]:
        return self._shim.counts()

    def flush(self) -> None:
        self._shim.flush()

    def stats(self) -> tuple[int, int, int, int, int, int]:
        return self._shim.stats()
