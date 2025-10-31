"""Worker-scoped helpers for constructing level contexts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from napari_cuda.server.data.lod import LevelContext, LevelDecision
from napari_cuda.server.data.roi import plane_scale_for_level
from napari_cuda.server.data.zarr_source import ZarrSceneSource


def build_level_context(
    decision: LevelDecision,
    *,
    source: ZarrSceneSource,
    prev_level: Optional[int],
    last_step: Optional[Sequence[int]],
) -> LevelContext:
    """Build an immutable context describing the chosen level."""

    level_idx = int(decision.selected_level)
    desc = source.level_descriptors[level_idx]
    axes = tuple(str(a) for a in source.axes)
    lower = [a.lower() for a in axes]

    prev_shape: Optional[Sequence[int]] = None
    if prev_level is not None and 0 <= int(prev_level) < len(source.level_descriptors):
        prev_shape = source.level_descriptors[int(prev_level)].shape

    step = _remap_step(prev_shape, desc.shape, axes, last_step)

    contrast = source.ensure_contrast(level=level_idx)
    sy, sx = plane_scale_for_level(source, level_idx)

    z_index = None
    if "z" in lower and len(step) > lower.index("z"):
        z_index = int(step[lower.index("z")])
    elif step:
        z_index = int(step[0])

    axes_str = "".join(axes)
    dtype_str = str(source.dtype)

    return LevelContext(
        level=level_idx,
        step=step,
        z_index=z_index,
        shape=tuple(int(s) for s in desc.shape),
        scale_yx=(float(sy), float(sx)),
        contrast=(float(contrast[0]), float(contrast[1])),
        axes=axes_str,
        dtype=dtype_str,
    )


def _remap_step(
    prev_shape: Optional[Sequence[int]],
    new_shape: Sequence[int],
    axes: Sequence[str],
    last_step: Optional[Sequence[int]],
) -> tuple[int, ...]:
    step = _clamp_step(new_shape, last_step)

    if prev_shape is None:
        return tuple(step)

    lower = [str(a).lower() for a in axes]
    if "z" not in lower:
        return tuple(step)

    zi = lower.index("z")
    if zi >= len(step) or zi >= len(prev_shape):
        return tuple(step)

    prev_len = int(prev_shape[zi])
    new_len = int(new_shape[zi])
    if prev_len <= 0 or new_len <= 0 or prev_len == new_len:
        return tuple(step)

    source_index = step[zi]
    if last_step is not None and len(last_step) > zi:
        try:
            source_index = int(last_step[zi])
        except Exception:
            source_index = step[zi]

    source_index = max(0, min(prev_len - 1, source_index)) if prev_len > 0 else source_index

    step[zi] = _proportional_z_remap(prev_len, new_len, source_index)
    return tuple(step)


def _clamp_step(
    shape: Sequence[int],
    values: Optional[Sequence[int]],
) -> list[int]:
    clamped: list[int] = []
    for idx, dim in enumerate(shape):
        bound = max(0, int(dim) - 1)
        value = 0
        if values is not None and idx < len(values):
            try:
                value = int(values[idx])
            except Exception:
                value = 0
        clamped.append(max(0, min(bound, value)))
    return clamped


def _proportional_z_remap(prev_len: int, new_len: int, index: int) -> int:
    if new_len <= 0:
        return 0

    if prev_len <= 1 or new_len <= 1:
        return max(0, min(new_len - 1, index))

    ratio = float(new_len - 1) / float(prev_len - 1)
    mapped = int(round(float(index) * ratio))
    return max(0, min(new_len - 1, mapped))


__all__ = ["build_level_context"]
