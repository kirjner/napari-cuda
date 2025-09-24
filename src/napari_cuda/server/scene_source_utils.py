"""Helpers for managing the worker's Zarr scene source."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from napari_cuda.server.zarr_source import ZarrSceneSource

logger = logging.getLogger(__name__)


def ensure_scene_source(worker) -> ZarrSceneSource:
    """Return a configured ``ZarrSceneSource`` and synchronise worker metadata."""

    if not getattr(worker, "_zarr_path", None):
        raise RuntimeError("No OME-Zarr path configured for scene source")

    source = getattr(worker, "_scene_source", None)
    if source is None:
        created = worker._create_scene_source()  # type: ignore[attr-defined]
        if created is None:
            raise RuntimeError("Failed to create ZarrSceneSource")
        source = created
        worker._scene_source = source  # type: ignore[attr-defined]

    target = source.current_level
    if getattr(worker, "_zarr_level", None):
        target = source.level_index_for_path(worker._zarr_level)  # type: ignore[attr-defined]

    if getattr(worker, "_log_layer_debug", False):
        key = (int(source.current_level), str(worker._zarr_level) if worker._zarr_level else None)  # type: ignore[attr-defined]
        last_key = getattr(worker, "_last_ensure_log", None)
        if last_key != key:
            logger.debug(
                "ensure_source: current=%d target=%d path=%s",
                int(source.current_level),
                int(target),
                worker._zarr_level,
            )
            worker._last_ensure_log = key  # type: ignore[attr-defined]
            worker._last_ensure_log_ts = time.perf_counter()  # type: ignore[attr-defined]

    with worker._state_lock:  # type: ignore[attr-defined]
        step = source.set_current_level(target, step=source.current_step)

    descriptor = source.level_descriptors[source.current_level]
    worker._active_ms_level = int(source.current_level)  # type: ignore[attr-defined]
    worker._zarr_level = descriptor.path or None  # type: ignore[attr-defined]
    worker._zarr_axes = ''.join(source.axes)  # type: ignore[attr-defined]
    worker._zarr_shape = descriptor.shape  # type: ignore[attr-defined]
    worker._zarr_dtype = str(source.dtype)  # type: ignore[attr-defined]

    axes_lower = [str(ax).lower() for ax in source.axes]
    if step:
        if 'z' in axes_lower:
            z_pos = axes_lower.index('z')
            worker._z_index = int(step[z_pos])  # type: ignore[attr-defined]
        else:
            worker._z_index = int(step[0])  # type: ignore[attr-defined]

    return source


def notify_scene_refresh(worker, step_hint: Optional[Tuple[int, ...]] = None) -> None:
    """Invoke the worker's scene refresh callback with the best step hint."""

    cb = getattr(worker, "_scene_refresh_cb", None)
    if cb is None:
        return

    hint = step_hint
    if hint is None:
        try:
            src = getattr(worker, "_scene_source", None)
            if src is not None:
                cur = getattr(src, 'current_step', None)
                if cur is not None:
                    hint = tuple(int(x) for x in cur)
        except Exception:
            hint = None
    if hint is None:
        try:
            viewer = getattr(worker, "_viewer", None)
            if viewer is not None:
                hint = tuple(int(x) for x in viewer.dims.current_step)  # type: ignore[attr-defined]
        except Exception:
            hint = None
    if hint is None and getattr(worker, "_z_index", None) is not None:
        hint = (int(worker._z_index),)

    try:
        cb(hint) if hint is not None else cb()  # type: ignore[misc]
    except TypeError:
        cb()
    except Exception:
        logger.debug("scene refresh callback failed", exc_info=True)


__all__ = ["ensure_scene_source", "notify_scene_refresh"]
