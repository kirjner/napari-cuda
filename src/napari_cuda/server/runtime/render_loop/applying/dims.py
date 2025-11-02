"""Dims snapshot application helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from napari_cuda.server.scene.viewport import RenderMode


def apply_dims_step(worker: Any, current_step: Sequence[int]) -> tuple[Optional[int], bool]:
    """Update viewer dims and source slice from the provided step."""

    if worker.viewport_state.mode is RenderMode.VOLUME:  # type: ignore[attr-defined]
        return None, False

    steps = tuple(int(x) for x in current_step)
    viewer = worker._viewer  # type: ignore[attr-defined]
    assert viewer is not None, "viewer must exist before applying dims"
    viewer.dims.current_step = steps  # type: ignore[attr-defined]

    source = worker._ensure_scene_source()  # type: ignore[attr-defined]
    axes = getattr(source, "axes", ())
    if isinstance(axes, str):
        axes = tuple(axes)

    z_index: Optional[int] = None
    if isinstance(axes, (list, tuple)) and "z" in axes:
        zi = axes.index("z")
        if zi < len(steps):
            z_index = int(steps[zi])

    if z_index is None:
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]
        return None, True

    current_z = getattr(worker, "_z_index", None)
    if current_z is not None and int(z_index) == int(current_z):
        worker._mark_render_tick_needed()  # type: ignore[attr-defined]
        return int(z_index), True

    zi = axes.index("z") if isinstance(axes, (list, tuple)) and "z" in axes else 0
    base = list(getattr(source, "current_step", steps) or steps)
    level_idx = int(worker._current_level_index())  # type: ignore[attr-defined]
    lvl_shape = source.level_shape(level_idx)
    if len(base) < len(lvl_shape):
        base.extend(0 for _ in range(len(lvl_shape) - len(base)))
    base[zi] = int(z_index)

    with worker._state_lock:  # type: ignore[attr-defined]
        source.set_current_slice(tuple(int(x) for x in base), level_idx)

    if worker._idr_on_z:  # type: ignore[attr-defined]
        worker._request_encoder_idr()  # type: ignore[attr-defined]

    worker._mark_render_tick_needed()  # type: ignore[attr-defined]
    return int(z_index), True


__all__ = ["apply_dims_step"]
