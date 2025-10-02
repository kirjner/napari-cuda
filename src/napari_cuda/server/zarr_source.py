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
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np

from .scene_types import SliceROI, SliceIOMetrics

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
        self._contrast_cache: Dict[int, Tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._current_level: int = 0
        self._current_step: Optional[Tuple[int, ...]] = None

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

        if self._level_descriptors:
            with self._lock:
                self._current_step = self.initial_step(level=self._current_level)

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
        with self._lock:
            return self._current_level

    @property
    def current_step(self) -> Optional[Tuple[int, ...]]:
        with self._lock:
            return tuple(self._current_step) if self._current_step is not None else None

    def set_current_slice(
        self,
        step: Sequence[int],
        level: int,
    ) -> Tuple[int, ...]:
        lvl = self._validate_level(level)
        clamped = self.initial_step(level=lvl, step=step)
        with self._lock:
            self._current_level = lvl
            self._current_step = tuple(clamped)
        return tuple(clamped)

    def level_index_for_path(self, path: str) -> int:
        for descriptor in self._level_descriptors:
            if descriptor.path == path:
                return descriptor.index
        raise ValueError(f"No level with path={path!r}")

    def get_level(self, index: int | None = None) -> da.Array:
        lvl = self._validate_level(index)
        return self._levels[lvl]

    def level_shape(self, index: int | None = None) -> Tuple[int, ...]:
        descriptor = self._descriptor(index)
        return descriptor.shape

    def level_downsample(self, index: int | None = None) -> Tuple[float, ...]:
        descriptor = self._descriptor(index)
        return descriptor.downsample

    def level_scale(self, index: int | None = None) -> Tuple[float, ...]:
        descriptor = self._descriptor(index)
        return descriptor.scale

    def initial_step(
        self,
        step_or_z: Optional[Sequence[int] | int] = None,
        *,
        level: Optional[int] = None,
        step: Optional[Sequence[int]] = None,
    ) -> Tuple[int, ...]:
        lvl = self._validate_level(level)
        descriptor = self._descriptor(lvl)
        shape = descriptor.shape
        axes = self.axes
        axes_lower = [str(ax).lower() for ax in axes]

        z_override: Optional[int] = None
        if step_or_z is not None and step is None:
            if isinstance(step_or_z, Sequence) and not isinstance(step_or_z, (str, bytes)):
                step = step_or_z  # type: ignore[assignment]
            else:
                z_override = int(step_or_z)

        values = [0] * len(shape)
        if step is not None:
            seq = list(step)
            for idx in range(min(len(shape), len(seq))):
                values[idx] = int(seq[idx])

        z_pos = axes_lower.index("z") if "z" in axes_lower else 0

        if z_override is not None and len(shape) > z_pos:
            values[z_pos] = int(z_override)
        elif step is None and len(shape) > 0:
            values[z_pos] = int(shape[z_pos] // 2)

        clamped: List[int] = []
        for idx, dim in enumerate(shape):
            hi = max(0, int(dim) - 1)
            if hi <= 0:
                clamped.append(0)
            else:
                clamped.append(max(0, min(int(values[idx]), hi)))
        return tuple(clamped)

    def ensure_contrast(
        self,
        level: Optional[int] = None,
        percentiles: Tuple[float, float] = (0.5, 99.5),
    ) -> Tuple[float, float]:
        lvl = self._validate_level(level)
        with self._lock:
            cached = self._contrast_cache.get(lvl)
        if cached is not None:
            return cached

        array = self.get_level(lvl)
        sample: da.Array
        if array.ndim <= 2:
            sample = array.astype("float32")
        else:
            axes = self.axes
            axes_lower = [str(ax).lower() for ax in axes]
            z_pos = axes_lower.index("z") if "z" in axes_lower else 0
            plane_index = int(array.shape[z_pos] // 2)
            indexer: List[slice | int] = [slice(None)] * array.ndim
            indexer[z_pos] = plane_index
            sample = array[tuple(indexer)].astype("float32")

        data = np.asarray(sample.compute(), dtype="float32")
        if data.size == 0:
            result = (0.0, 1.0)
        else:
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
            result = (float(lo), float(hi))

        with self._lock:
            self._contrast_cache[lvl] = result
        return result

    def estimate_clims(
        self,
        *,
        level: Optional[int] = None,
        percentiles: Tuple[float, float] = (0.5, 99.5),
    ) -> Tuple[float, float]:
        return self.ensure_contrast(level=level, percentiles=percentiles)

    def slice(
        self,
        level: Optional[int],
        z_index: int,
        *,
        compute: bool = False,
        roi: Optional[SliceROI] = None,
    ) -> da.Array | np.ndarray:
        lvl = self._validate_level(level)
        array = self.get_level(lvl)
        axes = self.axes
        axes_lower = [str(ax).lower() for ax in axes]
        descriptor = self._descriptor(lvl)

        def _axis_limit(axis_name: str, default_index: int) -> int:
            if axis_name in axes_lower:
                idx = axes_lower.index(axis_name)
                if 0 <= idx < len(descriptor.shape):
                    return int(descriptor.shape[idx])
            if descriptor.shape:
                idx = default_index if -len(descriptor.shape) <= default_index < len(descriptor.shape) else -1
                return int(descriptor.shape[idx])
            return 0

        y_limit = _axis_limit("y", max(-2, -len(descriptor.shape)))
        x_limit = _axis_limit("x", max(-1, -len(descriptor.shape)))
        roi_slice: Optional[SliceROI] = None
        if roi is not None:
            roi_slice = roi.clamp(y_limit, x_limit)

        if array.ndim <= 2 or "z" not in axes_lower:
            indexer: List[slice | int] = [slice(None)] * array.ndim
        else:
            z_pos = axes_lower.index("z") if "z" in axes_lower else 0
            dim = int(array.shape[z_pos])
            if dim <= 0:
                raise ValueError("Slice request on empty Z dimension")
            zi = max(0, min(int(z_index), dim - 1))
            indexer = [slice(None)] * array.ndim
            indexer[z_pos] = zi

            with self._lock:
                base = list(self._current_step) if self._current_step is not None else list(self.initial_step(level=lvl))
                if len(base) < len(array.shape):
                    base.extend([0] * (len(array.shape) - len(base)))
                base[z_pos] = zi
                self._current_step = tuple(base)

        if roi_slice is not None and not roi_slice.is_empty():
            try:
                if "y" in axes_lower:
                    y_pos = axes_lower.index("y")
                    indexer[y_pos] = slice(int(roi_slice.y_start), int(roi_slice.y_stop))
                if "x" in axes_lower:
                    x_pos = axes_lower.index("x")
                    indexer[x_pos] = slice(int(roi_slice.x_start), int(roi_slice.x_stop))
            except Exception:
                logger.debug("slice: ROI indexing failed; falling back to full slice", exc_info=True)

        view = array[tuple(indexer)].astype("float32")

        if not compute:
            return view

        lo, hi = self.ensure_contrast(level=lvl)
        slab = view.compute() if isinstance(view, da.Array) else np.asarray(view)
        slab = np.asarray(slab, dtype="float32")
        scale = max(1e-12, hi - lo)
        slab = (slab - lo) / scale
        np.clip(slab, 0.0, 1.0, out=slab)
        return slab

    def estimate_slice_io(
        self,
        level: int,
        z_index: int,
        roi: Optional[SliceROI] = None,
    ) -> SliceIOMetrics:
        lvl = self._validate_level(level)
        arr = self.get_level(lvl)
        axes_lower = [str(ax).lower() for ax in self.axes]
        descriptor = self._descriptor(lvl)
        dtype_size = int(np.dtype(getattr(arr, "dtype", self.dtype)).itemsize)

        def _axis_limit(axis_name: str, fallback_index: int) -> int:
            if axis_name in axes_lower:
                idx = axes_lower.index(axis_name)
                if 0 <= idx < len(descriptor.shape):
                    return int(descriptor.shape[idx])
            return int(descriptor.shape[fallback_index])

        y_limit = _axis_limit("y", max(0, len(descriptor.shape) - 2))
        x_limit = _axis_limit("x", max(0, len(descriptor.shape) - 1))
        roi_slice = roi.clamp(y_limit, x_limit) if roi is not None else SliceROI(0, y_limit, 0, x_limit)

        if roi_slice.is_empty():
            return SliceIOMetrics(chunks=0, bytes_est=0, roi=roi_slice)

        height = max(1, int(roi_slice.height))
        width = max(1, int(roi_slice.width))
        bytes_est = height * width * dtype_size

        chunks_attr = getattr(arr, "chunks", None)
        if chunks_attr is None:
            return SliceIOMetrics(chunks=0, bytes_est=bytes_est, roi=roi_slice)

        def _chunks_touched(chunk_sizes: Sequence[int], start: int, stop: int) -> int:
            if start >= stop:
                return 0
            offset = 0
            touched = 0
            for size in chunk_sizes:
                next_offset = offset + int(size)
                if stop <= offset:
                    break
                if start < next_offset:
                    touched += 1
                offset = next_offset
                if stop <= offset:
                    break
            return max(1, touched)

        chunk_count = 1
        for axis, chunk_sizes in enumerate(chunks_attr):
            sizes = tuple(int(s) for s in chunk_sizes)
            if not sizes:
                continue
            if axis < len(axes_lower):
                axis_name = axes_lower[axis]
            else:
                axis_name = ""
            if axis_name == "z":
                chunk_count *= _chunks_touched(sizes, int(z_index), int(z_index) + 1)
            elif axis_name == "y":
                chunk_count *= _chunks_touched(sizes, int(roi_slice.y_start), int(roi_slice.y_stop))
            elif axis_name == "x":
                chunk_count *= _chunks_touched(sizes, int(roi_slice.x_start), int(roi_slice.x_stop))
            else:
                chunk_count *= max(1, len(sizes))

        return SliceIOMetrics(chunks=int(chunk_count), bytes_est=int(bytes_est), roi=roi_slice)

    def level_volume(
        self,
        level: Optional[int] = None,
        *,
        compute: bool = False,
    ) -> da.Array | np.ndarray:
        lvl = self._validate_level(level)
        array = self.get_level(lvl).astype("float32")
        if not compute:
            return array
        lo, hi = self.ensure_contrast(level=lvl)
        volume = array.compute()
        volume = np.asarray(volume, dtype="float32")
        scale = max(1e-12, hi - lo)
        volume = (volume - lo) / scale
        np.clip(volume, 0.0, 1.0, out=volume)
        return volume

    # ------------------------------------------------------------------
    # Internal helpers

    def _descriptor(self, index: int | None) -> LevelDescriptor:
        lvl = self._validate_level(index)
        return self._level_descriptors[lvl]

    def _validate_level(self, index: int | None) -> int:
        if index is None:
            with self._lock:
                lvl = self._current_level
        else:
            lvl = int(index)
        if not 0 <= lvl < len(self._level_descriptors):
            raise ValueError(f"Level index out of range: {lvl}")
        return lvl

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
