"""Lightweight helper for reading OME-NGFF/OME-Zarr multiscale image stacks.

This module keeps the responsibilities tightly scoped so the EGL worker and
scene manager can share a single abstraction when bootstrapping napari layers
from OME-Zarr datasets.  The helper intentionally avoids additional
dependencies beyond ``dask.array``/``zarr`` and simple JSON parsing so it can
be extended later to support alternative formats with similar semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_AXES = ("z", "y", "x")


class ZarrSceneSourceError(RuntimeError):
    """Raised when a ZarrSceneSource cannot be constructed."""


@dataclass(frozen=True)
class LevelDescriptor:
    """Metadata for a single NGFF multiscale level."""

    index: int
    path: str
    shape: Tuple[int, ...]
    downsample: Tuple[float, ...]
    scale: Tuple[float, ...]


class ZarrSceneSource:
    """Wraps an OME-Zarr multiscale pyramid for napari consumption."""

    def __init__(
        self,
        root: str | Path,
        *,
        preferred_level: Optional[str] = None,
        axis_override: Optional[Sequence[str]] = None,
    ) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise ZarrSceneSourceError(f"Zarr root does not exist: {self._root}")

        self._preferred_level_name = preferred_level
        self._axes = tuple(axis_override) if axis_override else None
        self._levels: List[da.Array] = []
        self._level_descriptors: List[LevelDescriptor] = []
        self._contrast_limits: Optional[Tuple[float, float]] = None
        self._current_level: int = 0

        try:
            meta = self._read_zattrs(self._root / ".zattrs")
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ZarrSceneSourceError(f"Failed to read .zattrs: {exc}") from exc

        multiscales = self._first_multiscale_entry(meta)
        datasets_meta = multiscales.get("datasets", []) if multiscales else []

        if not datasets_meta:
            # Treat the root as a single-scale array
            arr = da.from_zarr(str(self._root))
            self._axes = self._axes or DEFAULT_AXES[: arr.ndim]
            scale = tuple(1.0 for _ in range(arr.ndim))
            self._levels = [arr]
            self._level_descriptors = [
                LevelDescriptor(
                    index=0,
                    path="",
                    shape=tuple(int(s) for s in arr.shape),
                    downsample=tuple(1.0 for _ in range(arr.ndim)),
                    scale=scale,
                )
            ]
            return

        axes = self._axes or self._parse_axes(multiscales)
        self._axes = axes

        level_arrays: List[da.Array] = []
        descriptors: List[LevelDescriptor] = []

        base_scale = None
        for idx, entry in enumerate(datasets_meta):
            rel_path = entry.get("path") or ""
            arr_path = self._root / rel_path if rel_path else self._root
            if not arr_path.exists():
                raise ZarrSceneSourceError(f"Missing dataset path: {arr_path}")

            darr = da.from_zarr(str(arr_path))
            if darr.ndim != len(axes):
                logger.warning(
                    "zarr_source: level %s ndim=%s does not match axes=%s", rel_path, darr.ndim, axes
                )

            level_scale = self._extract_scale(entry, axes)
            if base_scale is None:
                base_scale = level_scale
            downsample = self._relative_downsample(level_scale, base_scale)

            descriptors.append(
                LevelDescriptor(
                    index=idx,
                    path=str(rel_path),
                    shape=tuple(int(s) for s in darr.shape),
                    downsample=downsample,
                    scale=level_scale,
                )
            )
            level_arrays.append(darr)

        self._levels = level_arrays
        self._level_descriptors = descriptors

        if self._preferred_level_name:
            try:
                self._current_level = self.level_index_for_path(self._preferred_level_name)
            except ValueError:
                logger.debug(
                    "zarr_source: preferred level %s not found; defaulting to index 0",
                    self._preferred_level_name,
                )

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._axes or DEFAULT_AXES[: self.ndim]

    @property
    def ndim(self) -> int:
        if not self._levels:
            return 0
        return int(self._levels[0].ndim)

    @property
    def dtype(self) -> np.dtype:
        if not self._levels:
            raise ZarrSceneSourceError("No levels available")
        return np.dtype(self._levels[0].dtype)

    @property
    def levels(self) -> List[da.Array]:
        return list(self._levels)

    @property
    def level_descriptors(self) -> List[LevelDescriptor]:
        return list(self._level_descriptors)

    @property
    def current_level(self) -> int:
        return self._current_level

    def set_current_level(self, index: int) -> None:
        if not 0 <= index < len(self._levels):
            raise ValueError(f"Level index out of range: {index}")
        self._current_level = int(index)

    def level_index_for_path(self, path: str) -> int:
        for descriptor in self._level_descriptors:
            if descriptor.path == path:
                return descriptor.index
        raise ValueError(f"No level with path={path!r}")

    def get_level(self, index: int | None = None) -> da.Array:
        if index is None:
            index = self._current_level
        if not 0 <= index < len(self._levels):
            raise ValueError(f"Level index out of range: {index}")
        return self._levels[index]

    def level_shape(self, index: int | None = None) -> Tuple[int, ...]:
        descriptor = self._descriptor(index)
        return descriptor.shape

    def level_downsample(self, index: int | None = None) -> Tuple[float, ...]:
        descriptor = self._descriptor(index)
        return descriptor.downsample

    def level_scale(self, index: int | None = None) -> Tuple[float, ...]:
        descriptor = self._descriptor(index)
        return descriptor.scale

    def initial_step(self, z_index: Optional[int] = None) -> Tuple[int, ...]:
        shape = self.level_shape(0 if self._levels else None)
        axes = self.axes
        steps = [0] * len(shape)
        try:
            z_pos = axes.index("z")
        except ValueError:
            z_pos = 0
        if shape:
            if z_index is None:
                steps[z_pos] = int(shape[z_pos] // 2)
            else:
                steps[z_pos] = max(0, min(int(z_index), int(shape[z_pos]) - 1))
        return tuple(steps)

    def estimate_clims(self, *, level: Optional[int] = None, percentiles: Tuple[float, float] = (0.5, 99.5)) -> Tuple[float, float]:
        if self._contrast_limits is not None:
            return self._contrast_limits

        if level is None:
            level = self._current_level
        array = self.get_level(level)
        if array.ndim < 2:
            data = array.astype("float32").compute()
        else:
            axes = self.axes
            try:
                z_pos = axes.index("z")
            except ValueError:
                z_pos = 0
            plane_index = int(array.shape[z_pos] // 2)
            indexer: List[slice | int] = [slice(None)] * array.ndim
            indexer[z_pos] = plane_index
            slice_da = array[tuple(indexer)].astype("float32")
            data = slice_da.compute()
        data = np.asarray(data, dtype="float32")
        if data.size == 0:
            self._contrast_limits = (0.0, 1.0)
            return self._contrast_limits
        try:
            lo, hi = np.percentile(data, list(percentiles))
        except Exception as exc:
            logger.debug("zarr_source: percentile failed; falling back to min/max", exc_info=exc)
            lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(data))
            hi = float(np.nanmax(data))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        self._contrast_limits = (float(lo), float(hi))
        return self._contrast_limits

    # ------------------------------------------------------------------
    # Internal helpers

    def _descriptor(self, index: int | None) -> LevelDescriptor:
        if index is None:
            index = self._current_level
        if not 0 <= index < len(self._level_descriptors):
            raise ValueError(f"Level index out of range: {index}")
        return self._level_descriptors[index]

    @staticmethod
    def _read_zattrs(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    @staticmethod
    def _first_multiscale_entry(meta: Dict[str, object]) -> Dict[str, object]:
        entries = meta.get("multiscales") if isinstance(meta, dict) else None
        if isinstance(entries, list) and entries:
            entry = entries[0]
            if isinstance(entry, dict):
                return entry
        return {}

    @staticmethod
    def _parse_axes(multiscale: Dict[str, object]) -> Tuple[str, ...]:
        axes_meta = multiscale.get("axes") if isinstance(multiscale, dict) else None
        if isinstance(axes_meta, list) and axes_meta:
            axes: List[str] = []
            for entry in axes_meta:
                if isinstance(entry, dict):
                    name = entry.get("name")
                else:
                    name = entry
                axes.append(str(name or ""))
            return tuple(axes)
        return DEFAULT_AXES

    @staticmethod
    def _extract_scale(dataset_meta: Dict[str, object], axes: Sequence[str]) -> Tuple[float, ...]:
        transforms = dataset_meta.get("coordinateTransformations") if isinstance(dataset_meta, dict) else None
        if isinstance(transforms, list):
            for transform in transforms:
                if isinstance(transform, dict) and str(transform.get("type")).lower() == "scale":
                    scale_vals = transform.get("scale")
                    if isinstance(scale_vals, Iterable):
                        return tuple(float(v) for v in list(scale_vals))
        return tuple(1.0 for _ in axes)

    @staticmethod
    def _relative_downsample(level_scale: Tuple[float, ...], base_scale: Tuple[float, ...]) -> Tuple[float, ...]:
        downsample: List[float] = []
        for lvl, base in zip(level_scale, base_scale):
            if base == 0:
                downsample.append(1.0)
            else:
                downsample.append(float(lvl) / float(base))
        return tuple(downsample)


__all__ = [
    "LevelDescriptor",
    "ZarrSceneSource",
    "ZarrSceneSourceError",
]

