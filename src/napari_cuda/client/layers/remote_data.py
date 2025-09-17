"""Array-like shims for remote layer metadata.

These helpers expose the minimal surface napari expects from layer data while
avoiding any large client-side allocations. They provide small cached previews
for UI affordances (thumbnails, auto-contrast) and return zero-filled slices for
any indexing operation. The real pixel data continues to arrive via the video
stream; the objects here simply satisfy `LayerDataProtocol`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple, Union, overload

import numpy as np

from napari.layers._data_protocols import LayerDataProtocol
from napari.layers._multiscale_data import MultiScaleData

from napari_cuda.protocol.messages import LayerSpec, MultiscaleLevelSpec, MultiscaleSpec

IndexKey = Union[int, slice, type(Ellipsis), None]

_PREVIEW_MAX = 8


def _safe_dtype(value: str | None) -> np.dtype:
    try:
        return np.dtype(value) if value is not None else np.dtype("float32")
    except Exception:
        return np.dtype("float32")


def _preview_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    if not shape:
        return (1,)
    dims: list[int] = []
    for length in shape:
        try:
            dim_len = int(length)
        except Exception:
            dim_len = 1
        dim_len = max(1, dim_len)
        dims.append(min(dim_len, _PREVIEW_MAX))
    return tuple(dims)


def _result_length(length: int, key: slice | int | None) -> int:
    if isinstance(key, int):
        return 1
    if key is None:
        return 1
    if not isinstance(key, slice):
        return length
    step = key.step if key.step not in (None, 0) else 1
    step = int(step) if isinstance(step, (int, float)) else 1
    start = key.start if key.start is not None else 0
    stop = key.stop if key.stop is not None else length
    try:
        start_i = int(start)
    except Exception:
        start_i = 0
    try:
        stop_i = int(stop)
    except Exception:
        stop_i = length
    if stop_i < start_i:
        start_i, stop_i = stop_i, start_i
    span = max(0, stop_i - start_i)
    if span == 0:
        return 0
    count = (span + abs(step) - 1) // abs(step)
    return max(1, min(count, length, _PREVIEW_MAX))


def _normalize_key(key: Union[IndexKey, Tuple[IndexKey, ...]], ndim: int) -> Tuple[IndexKey, ...]:
    if not isinstance(key, tuple):
        key_tuple: Tuple[IndexKey, ...] = (key,)
    else:
        key_tuple = key

    result: list[IndexKey] = []
    ellipsis_seen = False
    for item in key_tuple:
        if item is Ellipsis or isinstance(item, type(Ellipsis)):
            if ellipsis_seen:
                continue
            ellipsis_seen = True
            explicit = sum(
                1
                for v in key_tuple
                if v is not Ellipsis and not isinstance(v, type(Ellipsis)) and v is not None
            )
            remaining = max(0, ndim - explicit)
            result.extend([slice(None)] * remaining)
        else:
            result.append(item)

    while len(result) < ndim:
        result.append(slice(None))
    return tuple(result[:ndim])


class RemoteArray(LayerDataProtocol):
    """Minimal array stub that satisfies napari's data protocol."""

    def __init__(
        self,
        shape: Sequence[int],
        dtype: str | None,
        *,
        data_id: str | None = None,
        cache_version: int | None = None,
    ) -> None:
        self._shape = tuple(int(max(0, s)) for s in shape)
        if not self._shape:
            self._shape = (1,)
        self._dtype = _safe_dtype(dtype)
        self.data_id = data_id
        self.cache_version = cache_version
        self._preview = np.zeros(_preview_shape(self._shape), dtype=self._dtype)

    # --- LayerDataProtocol compliance -------------------------------------
    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        try:
            total = 1
            for dim in self._shape:
                total *= max(1, dim)
            return int(total)
        except Exception:
            return 0

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def __array__(self) -> np.ndarray:  # pragma: no cover - exercised implicitly
        return self._preview

    def __len__(self) -> int:
        return self._shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:  # pragma: no cover
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, key: IndexKey) -> np.ndarray: ...

    @overload
    def __getitem__(self, key: Tuple[IndexKey, ...]) -> np.ndarray: ...

    def __getitem__(self, key):
        normalized = _normalize_key(key, self.ndim)
        lengths: list[int] = []
        dim = 0
        for item in normalized:
            if item is None:
                lengths.append(1)
                continue
            if isinstance(item, (int, np.integer)):
                dim += 1
                continue
            if dim >= self.ndim:
                lengths.append(1)
                continue
            lengths.append(_result_length(self._shape[dim], item))
            dim += 1
        # Spill remaining dimensions
        for idx in range(dim, self.ndim):
            lengths.append(min(self._shape[idx], _PREVIEW_MAX))

        # Remove leading dimensions collapsed by integer indexing
        result_shape = [length for length in lengths if length not in (None,)]
        if not result_shape:
            return np.array(0, dtype=self._dtype)
        preview = np.zeros(tuple(result_shape), dtype=self._dtype)
        return preview

    # --- Mutation helpers -------------------------------------------------
    def update(self, *, shape: Sequence[int] | None = None, dtype: str | None = None, data_id: str | None = None, cache_version: int | None = None) -> None:
        if shape is not None:
            self._shape = tuple(int(max(0, s)) for s in shape) or (1,)
            self._preview = np.zeros(_preview_shape(self._shape), dtype=self._dtype)
        if dtype is not None:
            self._dtype = _safe_dtype(dtype)
            self._preview = self._preview.astype(self._dtype, copy=False)
        if data_id is not None:
            self.data_id = data_id
        if cache_version is not None:
            self.cache_version = cache_version

    @property
    def preview(self) -> np.ndarray:
        return self._preview


@dataclass
class RemotePreview:
    """Holds an optional preview image for a remote layer."""

    data: np.ndarray | None = None

    def update(self, data: np.ndarray | None) -> None:
        if data is None:
            self.data = None
            return
        try:
            arr = np.asarray(data, dtype=np.float32)
        except Exception:
            self.data = None
            return
        if arr.size == 0:
            self.data = None
            return
        np.clip(arr, 0.0, 1.0, out=arr)
        self.data = arr

    def as_thumbnail(self, rgb: bool, target_shape: Tuple[int, int]) -> np.ndarray:
        """Return a thumbnail-sized preview (RGB or scalar depending on ``rgb``)."""

        h, w = int(target_shape[0]), int(target_shape[1])
        if h <= 0 or w <= 0:
            return np.zeros((1, 1, 3 if rgb else 1), dtype=np.float32)

        src = self.data
        if src is None:
            shape = (h, w, 3) if rgb else (h, w)
            return np.zeros(shape, dtype=np.float32)

        arr = np.asarray(src)
        if arr.ndim == 0:
            arr = arr.reshape((1, 1))

        # Collapse leading dimensions until we have <= 3 dims (assuming final dims are spatial/color)
        while arr.ndim > 3:
            arr = arr.max(axis=0)

        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            arr = arr.max(axis=0)

        if arr.ndim == 2:
            arr = arr[..., np.newaxis]

        channels = arr.shape[-1]
        if rgb:
            if channels == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif channels >= 3:
                arr = arr[..., :3]
            else:  # channels == 2
                pad = np.zeros(arr.shape[:2] + (3 - channels,), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=-1)
        else:
            if channels > 1:
                arr = arr[..., 0]
            else:
                arr = arr[..., 0]
            arr = np.asarray(arr, dtype=arr.dtype)

        arr = arr.astype(np.float32, copy=False)
        arr = np.clip(arr, 0.0, 1.0, out=arr)

        if rgb:
            out = np.zeros((h, w, 3), dtype=np.float32)
        else:
            out = np.zeros((h, w), dtype=np.float32)
        copy_h = min(h, arr.shape[0])
        copy_w = min(w, arr.shape[1])
        if rgb:
            out[:copy_h, :copy_w, ...] = arr[:copy_h, :copy_w, : out.shape[-1]]
        else:
            out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return out

@dataclass(frozen=True)
class RemoteMultiscale:
    """Container for multiscale remote arrays."""

    arrays: Tuple[RemoteArray, ...]

    def as_multiscale(self) -> MultiScaleData:
        return MultiScaleData(self.arrays)


def build_remote_data(spec: LayerSpec) -> tuple[LayerDataProtocol | MultiScaleData, RemoteMultiscale | None]:
    data_id = None
    cache_version = None
    extras = spec.extras or {}
    if isinstance(extras, dict):
        data_id = extras.get("data_id") or extras.get("source_id")
        cache_version = extras.get("cache_version")
        try:
            if cache_version is not None:
                cache_version = int(cache_version)
        except Exception:
            cache_version = None

    if spec.multiscale and isinstance(spec.multiscale, MultiscaleSpec):
        arrays: list[RemoteArray] = []
        base_dtype = spec.dtype
        for level in spec.multiscale.levels:
            arrays.append(
                RemoteArray(
                    level.shape or spec.shape,
                    base_dtype,
                    data_id=data_id,
                    cache_version=cache_version,
                )
            )
        container = RemoteMultiscale(tuple(arrays))
        return container.as_multiscale(), container

    array = RemoteArray(spec.shape, spec.dtype, data_id=data_id, cache_version=cache_version)
    return array, None
