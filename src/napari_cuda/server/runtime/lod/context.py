"""Worker-scoped helpers for constructing level contexts."""

from __future__ import annotations

from collections.abc import Sequence

from napari_cuda.server.data.lod import LevelContext
from napari_cuda.server.data.roi import plane_scale_for_level
from napari_cuda.server.data.zarr_source import ZarrSceneSource


def build_level_context(
    *,
    source: ZarrSceneSource,
    level: int,
    step: Sequence[int],
) -> LevelContext:
    """Build an immutable context describing the chosen level."""

    level_idx = int(level)
    descriptors = source.level_descriptors
    if not descriptors or not (0 <= level_idx < len(descriptors)):
        raise AssertionError(f"level {level_idx} missing from source descriptors")

    desc = descriptors[level_idx]
    shape = tuple(int(dim) for dim in desc.shape)
    axes = tuple(str(axis) for axis in source.axes)
    step_tuple = tuple(int(v) for v in step)

    if len(step_tuple) != len(shape):
        raise AssertionError(
            f"step ndim {len(step_tuple)} != shape ndim {len(shape)} for level {level_idx}"
        )
    for idx, bound in enumerate(shape):
        if not (0 <= step_tuple[idx] < bound):
            raise AssertionError(
                f"step index {step_tuple[idx]} out of bounds [0, {bound}) for axis {idx}"
            )

    contrast = source.ensure_contrast(level=level_idx)
    sy, sx = plane_scale_for_level(source, level_idx)

    axes_lower = [axis.lower() for axis in axes]
    z_index = None
    if "z" in axes_lower:
        z_axis = axes_lower.index("z")
        z_index = int(step_tuple[z_axis])

    axes_str = "".join(axes)
    dtype_str = str(source.dtype)

    return LevelContext(
        level=level_idx,
        step=step_tuple,
        z_index=z_index,
        shape=shape,
        scale_yx=(float(sy), float(sx)),
        contrast=(float(contrast[0]), float(contrast[1])),
        axes=axes_str,
        dtype=dtype_str,
    )


__all__ = ["build_level_context"]
