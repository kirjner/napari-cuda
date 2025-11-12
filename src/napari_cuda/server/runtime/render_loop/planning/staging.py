"""Compatibility stubs for legacy staging helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


def drain_scene_updates(worker: "EGLRendererWorker") -> None:  # noqa: D401
    """Legacy hook retained for compatibility."""
    return None


__all__ = ["drain_scene_updates"]
