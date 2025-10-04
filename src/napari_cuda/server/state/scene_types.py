"""Shared scene source types for multiscale adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable, TYPE_CHECKING

import numpy as np
import dask.array as da

if TYPE_CHECKING:
    from .zarr_source import LevelDescriptor


@dataclass(frozen=True)
class SliceROI:
    """Pixel-aligned rectangular region of interest in YX coordinates."""

    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    def clamp(self, max_y: int, max_x: int) -> "SliceROI":
        """Clamp the ROI to the provided bounds (half-open)."""

        y0 = max(0, min(self.y_start, max_y))
        y1 = max(y0, min(self.y_stop, max_y))
        x0 = max(0, min(self.x_start, max_x))
        x1 = max(x0, min(self.x_stop, max_x))
        return SliceROI(y0, y1, x0, x1)

    @property
    def height(self) -> int:
        return max(0, self.y_stop - self.y_start)

    @property
    def width(self) -> int:
        return max(0, self.x_stop - self.x_start)

    def is_empty(self) -> bool:
        return self.height <= 0 or self.width <= 0


@dataclass(frozen=True)
class SliceIOMetrics:
    """Estimated chunk counts and byte footprint for a slice request."""

    chunks: int
    bytes_est: int
    roi: Optional[SliceROI] = None


@runtime_checkable
class SceneSource(Protocol):
    """Protocol for multiscale sources that can service ROI-aware slices."""

    @property
    def axes(self) -> Tuple[str, ...]:
        ...

    @property
    def level_descriptors(self) -> Sequence["LevelDescriptor"]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...

    def slice(
        self,
        level: Optional[int],
        z_index: int,
        *,
        compute: bool = False,
        roi: Optional[SliceROI] = None,
    ) -> np.ndarray | da.Array:
        ...

    def estimate_slice_io(self, level: int, z_index: int, roi: Optional[SliceROI] = None) -> SliceIOMetrics:
        ...

    def level_shape(self, index: Optional[int] = None) -> Tuple[int, ...]:
        ...


__all__ = ["SliceROI", "SliceIOMetrics", "SceneSource"]
