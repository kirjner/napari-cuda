from __future__ import annotations

"""Utilities for managing VT frame ownership across threads."""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LeaseClosedError(RuntimeError):
    """Raised when operations are attempted on a released frame lease."""


class LeaseRole:
    """Simple role identifiers for VT frame leases."""

    DECODER = "decoder"
    CACHE = "cache"
    PRESENTER = "presenter"
    RENDERER = "renderer"


@dataclass(frozen=True)
class LeaseSnapshot:
    """Captures the outstanding refcounts for diagnostics."""

    counts: dict[str, int]


class FrameLease:
    """Reference-counted wrapper around a VT capsule.

    The lease tracks role-based retains and guarantees that the underlying
    VideoToolbox frame is released exactly once when the final role drops.
    All operations are thread-safe.
    """

    __slots__ = ("_capsule", "_closed", "_counts", "_lock", "_vt")

    def __init__(self, vt_module: object, capsule: object) -> None:
        self._vt = vt_module
        self._capsule = capsule
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {LeaseRole.DECODER: 1}
        self._closed = False

    # --- public API -----------------------------------------------------

    @property
    def capsule(self) -> object:
        return self._capsule

    def snapshot(self) -> LeaseSnapshot:
        with self._lock:
            return LeaseSnapshot(dict(self._counts))

    # Role helpers -------------------------------------------------------

    def acquire_cache(self) -> None:
        self._acquire(LeaseRole.CACHE)

    def release_cache(self) -> None:
        self._release(LeaseRole.CACHE)

    def acquire_presenter(self) -> None:
        self._acquire(LeaseRole.PRESENTER)

    def release_presenter(self) -> None:
        self._release(LeaseRole.PRESENTER)

    def acquire_renderer(self) -> None:
        self._acquire(LeaseRole.RENDERER)

    def release_renderer(self) -> None:
        self._release(LeaseRole.RENDERER)

    def release_decoder(self) -> None:
        self._release(LeaseRole.DECODER)

    def close(self) -> None:
        """Force release of all remaining roles (used on shutdown)."""

        to_release: dict[str, int]
        with self._lock:
            to_release = dict(self._counts)
        for role, count in to_release.items():
            for _ in range(count):
                try:
                    self._release(role)
                except Exception:
                    logger.debug("FrameLease.close: release failed", exc_info=True)

    # --- internals ------------------------------------------------------

    def _acquire(self, role: str) -> None:
        with self._lock:
            if self._closed:
                raise LeaseClosedError(f"lease already closed for {role}")
            try:
                self._vt.retain_frame(self._capsule)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - VT shim errors
                logger.debug("FrameLease: retain failed", exc_info=True)
                raise exc
            self._counts[role] = self._counts.get(role, 0) + 1

    def _release(self, role: str) -> None:
        with self._lock:
            count = self._counts.get(role, 0)
            if count <= 0:
                raise RuntimeError(f"release without acquire (role={role})")
            if count == 1:
                self._counts.pop(role, None)
            else:
                self._counts[role] = count - 1
            should_close = not self._counts and not self._closed
            if should_close:
                self._closed = True
        try:
            self._vt.release_frame(self._capsule)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - VT shim errors
            logger.debug("FrameLease: release failed", exc_info=True)
            raise exc
        if should_close:
            logger.debug("FrameLease closed for capsule %s", hex(id(self._capsule)))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        with self._lock:
            counts = dict(self._counts)
            closed = self._closed
        return f"FrameLease(capsule=0x{id(self._capsule):x}, counts={counts}, closed={closed})"
