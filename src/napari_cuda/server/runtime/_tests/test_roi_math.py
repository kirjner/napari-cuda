from __future__ import annotations

import pytest

from napari_cuda.server.runtime.data import (
    SliceROI,
    align_roi_to_chunk_grid,
    chunk_shape_for_level,
    roi_chunk_signature,
)


class _StubLevelArray:
    def __init__(self, chunks: tuple[int, ...]) -> None:
        self.chunks = chunks


class _StubSource:
    def __init__(self, *, chunks: tuple[int, ...], axes: tuple[str, ...]) -> None:
        self._chunks = chunks
        self.axes = axes

    def get_level(self, level: int):
        return _StubLevelArray(self._chunks)


def test_chunk_shape_for_level_extracts_yx_axes() -> None:
    source = _StubSource(chunks=(1, 4, 8), axes=("t", "y", "x"))

    shape = chunk_shape_for_level(source, 0)

    assert shape == (4, 8)


def test_chunk_shape_for_level_returns_none_for_missing_chunks() -> None:
    class _NoChunksSource:
        axes = ("z", "y", "x")

        def get_level(self, level: int):
            return object()

    assert chunk_shape_for_level(_NoChunksSource(), 0) is None


@pytest.mark.parametrize(
    "roi,pad,expected",
    [
        (SliceROI(2, 18, 5, 27), 0, SliceROI(0, 20, 0, 28)),
        (SliceROI(2, 18, 5, 27), 1, SliceROI(0, 24, 0, 35)),
        (SliceROI(0, 8, 0, 8), 0, SliceROI(0, 8, 0, 14)),
    ],
)
def test_align_roi_to_chunk_grid(roi: SliceROI, pad: int, expected: SliceROI) -> None:
    chunk_shape = (4, 7)

    aligned = align_roi_to_chunk_grid(roi, chunk_shape, pad, height=24, width=40)

    assert aligned == expected


def test_roi_chunk_signature_matches_chunk_indices() -> None:
    roi = SliceROI(4, 16, 7, 27)
    chunk_shape = (4, 5)

    signature = roi_chunk_signature(roi, chunk_shape)

    assert signature == (1, 3, 1, 5)
