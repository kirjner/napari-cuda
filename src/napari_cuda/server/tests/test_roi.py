from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vispy import scene

from napari_cuda.server.data.roi import compute_viewport_roi
from napari_cuda.server.runtime.scene_types import SliceROI


class _DummyTransform:
    def __init__(self, scale: tuple[float, float] = (1.0, 1.0)) -> None:
        self._scale = scale
        self.matrix = np.eye(4)

    def imap(self, point):  # pragma: no cover - exercise via helper
        x, y, z = point
        sx, sy = self._scale
        return (x * sx, y * sy, z)


class _StubLevel:
    def __init__(self, index: int, shape: tuple[int, ...]) -> None:
        self.index = index
        self.shape = shape


class _StubLevelArray:
    def __init__(self, chunks: tuple[int, ...]) -> None:
        self.chunks = chunks


class _StubSource:
    def __init__(self) -> None:
        self.axes = ("z", "y", "x")
        self.level_descriptors = [_StubLevel(0, (3, 80, 100))]
        self._scale = [(1.0, 2.0, 4.0)]
        self._chunks = (1, 4, 5)

    dtype = np.dtype("float32")

    def level_scale(self, level: int):
        return self._scale[level]

    def get_level(self, level: int):
        return _StubLevelArray(self._chunks)


class _StubView:
    def __init__(self, width: int, height: int, *, scale: tuple[float, float] = (1.0, 1.0)) -> None:
        self._size = (width, height)
        self.scene = SimpleNamespace(transform=_DummyTransform(scale))
        self.camera = scene.cameras.PanZoomCamera(aspect=1.0)


def test_compute_viewport_roi_aligns_and_pads_chunks() -> None:
    source = _StubSource()
    view = _StubView(100, 80)
    result = compute_viewport_roi(
        view=view,
        canvas_size=(100, 80),
        source=source,
        level=0,
        align_chunks=True,
        chunk_pad=1,
        ensure_contains_viewport=True,
        edge_threshold=4,
        prev_roi=None,
        for_policy=False,
    )

    assert result.roi == SliceROI(0, 44, 0, 30)
    assert result.transform_signature is not None


def test_compute_viewport_roi_honours_prev_roi_hysteresis() -> None:
    source = _StubSource()
    view = _StubView(100, 80)
    base = compute_viewport_roi(
        view=view,
        canvas_size=(100, 80),
        source=source,
        level=0,
        align_chunks=True,
        chunk_pad=1,
        ensure_contains_viewport=True,
        edge_threshold=4,
        prev_roi=None,
        for_policy=False,
    )

    reused = compute_viewport_roi(
        view=view,
        canvas_size=(100, 80),
        source=source,
        level=0,
        align_chunks=True,
        chunk_pad=1,
        ensure_contains_viewport=True,
        edge_threshold=4,
        prev_roi=base.roi,
        for_policy=False,
    )

    assert reused.roi == base.roi


def test_compute_viewport_roi_for_policy_skips_alignment() -> None:
    source = _StubSource()
    view = _StubView(100, 80)
    result = compute_viewport_roi(
        view=view,
        canvas_size=(100, 80),
        source=source,
        level=0,
        align_chunks=False,
        chunk_pad=0,
        ensure_contains_viewport=False,
        edge_threshold=0,
        prev_roi=None,
        for_policy=True,
    )

    assert result.roi == SliceROI(0, 40, 0, 25)


def test_compute_viewport_roi_missing_transform_raises() -> None:
    source = _StubSource()
    view = SimpleNamespace(camera=scene.cameras.PanZoomCamera(aspect=1.0), scene=SimpleNamespace(transform=None))

    with pytest.raises(RuntimeError):
        compute_viewport_roi(
            view=view,
            canvas_size=(100, 80),
            source=source,
            level=0,
            align_chunks=True,
            chunk_pad=1,
            ensure_contains_viewport=True,
            edge_threshold=4,
            prev_roi=None,
            for_policy=False,
        )
