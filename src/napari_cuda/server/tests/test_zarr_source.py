from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from napari_cuda.server.data.zarr_source import ZarrSceneSource, ZarrSceneSourceError
from napari_cuda.server.data import SliceROI


def _write_multiscale(root: Path, datasets: list[dict]) -> None:
    data = {
        "multiscales": [
            {
                "name": "test",
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": datasets,
            }
        ]
    }
    (root / ".zattrs").write_text(json.dumps(data))


def test_zarr_scene_source_multiscale(tmp_path):
    root = tmp_path / "multi.zarr"
    store = zarr.open_group(str(root), mode="w")

    data0 = np.arange(2 * 4 * 6, dtype=np.uint16).reshape(2, 4, 6)
    data1 = data0[:, ::2, ::2]

    store.create_dataset("level_0", data=data0, chunks=(1, 2, 3))
    store.create_dataset("level_1", data=data1, chunks=(1, 2, 3))

    _write_multiscale(
        root,
        [
            {
                "path": "level_0",
                "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
            },
            {
                "path": "level_1",
                "coordinateTransformations": [{"type": "scale", "scale": [2.0, 3.0, 3.0]}],
            },
        ],
    )

    source = ZarrSceneSource(root)

    assert source.axes == ("z", "y", "x")
    assert source.ndim == 3
    assert len(source.levels) == 2
    assert source.level_descriptors[0].downsample == (1.0, 1.0, 1.0)
    assert source.level_descriptors[1].downsample == (2.0, 3.0, 3.0)
    assert source.level_descriptors[0].shape == (2, 4, 6)
    assert source.level_descriptors[1].shape == (2, 2, 3)

    clims = source.estimate_clims()
    assert isinstance(clims, tuple)
    assert len(clims) == 2
    # cached path
    assert source.estimate_clims() == clims

    initial = source.initial_step()
    assert len(initial) == 3
    assert 0 <= initial[0] < 2

    step = source.initial_step(level=1)
    source.set_current_slice(step, 1)
    assert source.current_level == 1
    assert source.get_level().shape == (2, 2, 3)


def test_zarr_scene_source_axes_override_and_preferred_level(tmp_path):
    root = tmp_path / "override.zarr"
    store = zarr.open_group(str(root), mode="w")

    data0 = np.arange(2 * 4 * 6, dtype=np.uint16).reshape(2, 4, 6)
    data1 = data0[:, ::2, ::2]

    store.create_dataset("coarse/level_0", data=data0, chunks=(1, 2, 3))
    store.create_dataset("coarse/level_1", data=data1, chunks=(1, 2, 3))

    _write_multiscale(
        root,
        [
            {
                "path": "coarse/level_0",
                "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
            },
            {
                "path": "coarse/level_1",
                "coordinateTransformations": [{"type": "scale", "scale": [2.0, 3.0, 3.0]}],
            },
        ],
    )

    source = ZarrSceneSource(root, preferred_level="coarse/level_1", axis_override=("t", "y", "x"))

    assert source.axes == ("t", "y", "x")
    assert source.current_level == 1
    assert tuple(source.level_scale()) == (2.0, 3.0, 3.0)
    assert tuple(source.level_downsample()) == (2.0, 3.0, 3.0)
    with pytest.raises(ValueError):
        source.level_index_for_path("missing")


def test_zarr_scene_source_single_scale(tmp_path):
    root = tmp_path / "single.zarr"
    data = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5)
    arr = zarr.open_array(str(root), mode="w", shape=data.shape, dtype=data.dtype, chunks=(2, 2))
    arr[:] = data
    (root / ".zattrs").write_text("{}")

    source = ZarrSceneSource(root)

    assert len(source.levels) == 1
    assert source.axes == ("z", "y")[: source.ndim]
    assert source.level_descriptors[0].downsample == tuple(1.0 for _ in range(source.ndim))


def test_zarr_scene_source_missing_path(tmp_path):
    root = tmp_path / "broken.zarr"
    root.mkdir()
    _write_multiscale(
        root,
        [
            {
                "path": "nope",
                "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}],
            }
        ],
    )

    with pytest.raises(ZarrSceneSourceError):
        ZarrSceneSource(root)


def test_slice_with_roi_and_metrics(tmp_path):
    root = tmp_path / "roi.zarr"
    store = zarr.open_group(str(root), mode="w")

    data = np.arange(3 * 8 * 10, dtype=np.uint16).reshape(3, 8, 10)
    store.create_dataset("level_0", data=data, chunks=(1, 4, 5))

    _write_multiscale(
        root,
        [
            {
                "path": "level_0",
                "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
            }
        ],
    )

    source = ZarrSceneSource(root)

    roi = SliceROI(2, 6, 3, 8)
    slab = source.slice(0, 1, compute=True, roi=roi)
    assert slab.shape == (roi.height, roi.width)

    metrics = source.estimate_slice_io(0, 1, roi=roi)
    assert metrics.bytes_est == roi.height * roi.width * data.dtype.itemsize
    assert metrics.chunks == 4  # two chunks along Y and two along X, single Z chunk
    assert metrics.roi == roi.clamp(8, 10)

    # ROI larger than plane clamps to bounds
    wide_roi = SliceROI(-5, 20, -3, 50)
    wide_metrics = source.estimate_slice_io(0, 0, roi=wide_roi)
    assert wide_metrics.roi == SliceROI(0, 8, 0, 10)
    wide_slab = source.slice(0, 0, compute=True, roi=wide_roi)
    assert wide_slab.shape == (8, 10)
