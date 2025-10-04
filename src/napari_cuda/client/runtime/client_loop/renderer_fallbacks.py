"""Helpers for renderer fallback behaviour (VT cache + PyAV frame reuse)."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Any

from napari_cuda.client.rendering.vt_frame import FrameLease


logger = logging.getLogger(__name__)


class RendererFallbacks:
    """Manages VT/PyAV fallback state for the client stream loop."""

    def __init__(self) -> None:
        self._last_vt_lease: Optional[FrameLease] = None
        self._last_pyav_frame: Optional[Any] = None

    # --- VT cache helpers -------------------------------------------------

    def update_vt_cache(self, payload: object, persistent: bool = False) -> None:
        """Store the latest VT lease for fallback draws."""

        self._last_vt_lease = payload if isinstance(payload, FrameLease) else None

    def try_enqueue_cached_vt(self, enqueue: Callable[[object], None]) -> bool:
        lease = self._last_vt_lease
        if lease is None:
            return False
        try:
            lease.acquire_renderer()
        except Exception:
            logger.debug("RendererFallbacks: VT cache acquire failed", exc_info=True)
            return False

        def _release(_: object, _lease=lease) -> None:
            try:
                _lease.release_renderer()
            except Exception:
                logger.debug("RendererFallbacks: VT cache release failed", exc_info=True)

        try:
            enqueue((lease, _release))
        except Exception:
            logger.debug("RendererFallbacks: enqueue cached VT frame failed", exc_info=True)
            try:
                _release(lease)
            except Exception:
                logger.debug("RendererFallbacks: release after failed enqueue failed", exc_info=True)
            return False
        return True

    def pop_vt_cache(self) -> Optional[FrameLease]:
        lease = self._last_vt_lease
        self._last_vt_lease = None
        return lease

    # --- PyAV fallback helpers --------------------------------------------

    def store_pyav_frame(self, frame: object) -> None:
        self._last_pyav_frame = frame

    def try_enqueue_pyav(self, enqueue: Callable[[object], None]) -> bool:
        if self._last_pyav_frame is None:
            return False
        try:
            enqueue(self._last_pyav_frame)
        except Exception:
            logger.debug("RendererFallbacks: enqueue PyAV fallback failed", exc_info=True)
            return False
        return True

    def clear_pyav(self) -> None:
        self._last_pyav_frame = None

    def reset(self) -> None:
        self._last_vt_lease = None
        self._last_pyav_frame = None
