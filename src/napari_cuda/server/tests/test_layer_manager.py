from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import pytest
import numpy as np

from napari.components.viewer_model import ViewerModel

from napari_cuda.server.scene.layer_manager import ViewerSceneManager
from napari_cuda.server.rendering.viewer_builder import canonical_axes_from_source


class _StubWorker:
    def __init__(
        self,
        *,
        use_volume: bool,
        data_wh: tuple[int, int],
        data_d: Optional[int] = None,
        zarr_axes: Optional[str] = None,
        zarr_shape: Optional[tuple[int, ...]] = None,
        zarr_dtype: Optional[str] = None,
        volume_dtype: Optional[str] = None,
    ) -> None:
        self.use_volume = use_volume
        self._data_wh = data_wh
        self._is_ready = True
        if data_d is not None:
            self._data_d = data_d
        self._zarr_axes = zarr_axes
        self._zarr_shape = zarr_shape
        self._zarr_dtype = zarr_dtype
        self.volume_dtype = volume_dtype
        self._scene_source = None
        axes_tuple: tuple[str, ...]
        if zarr_axes:
            axes_tuple = tuple(zarr_axes)
        elif data_d is not None or (zarr_shape and len(zarr_shape) == 3):
            axes_tuple = ("z", "y", "x")
        else:
            axes_tuple = ("y", "x")

        if zarr_shape:
            shape_tuple = tuple(int(s) for s in zarr_shape)
        else:
            if len(axes_tuple) == 3:
                depth = int(data_d) if data_d is not None else 1
                shape_tuple = (depth, int(data_wh[1]), int(data_wh[0]))
            else:
                shape_tuple = (int(data_wh[1]), int(data_wh[0]))

        step = tuple(0 for _ in range(len(shape_tuple)))
        self._canonical_axes = canonical_axes_from_source(
            axes=axes_tuple,
            shape=shape_tuple,
            step=step,
            use_volume=use_volume,
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def viewer_model(self):
        return None


class _StubSceneState:
    def __init__(self, *, center=None, zoom=None, angles=None) -> None:
        self.center = center
        self.zoom = zoom
        self.angles = angles


@pytest.fixture()
def manager() -> ViewerSceneManager:
    return ViewerSceneManager((640, 480))


def _volume_state() -> Dict[str, Any]:
    return {
        "mode": "mip",
        "colormap": "gray",
        "clim": [0.0, 1.0],
        "opacity": 0.75,
        "sample_step": 1.0,
    }


def _multiscale_state() -> Dict[str, Any]:
    return {
        "levels": [
            {"path": "level_0", "downsample": [1.0, 1.0, 1.0], "shape": [20, 40, 60]},
            {"path": "level_1", "downsample": [1.0, 2.0, 2.0], "shape": [20, 20, 30]},
        ],
        "current_level": 0,
        "policy": "latency",
        "index_space": "base",
    }


def test_update_volume_scene(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(
        use_volume=True,
        data_wh=(60, 40),
        data_d=20,
        zarr_axes="zyx",
        zarr_shape=(20, 40, 60),
        zarr_dtype="float32",
    )
    scene_source = _StubSceneSource(current_level=0, scale=(1.0, 1.0, 1.0))
    manager.update_from_sources(
        worker=worker,
        scene_state=_StubSceneState(center=(10.0, 20.0, 30.0), zoom=2.5),
        multiscale_state=_multiscale_state(),
        volume_state=_volume_state(),
        current_step=(1, 2, 3),
        ndisplay=3,
        zarr_path="/data/sample.zarr",
        scene_source=scene_source,
    )

    meta = manager.dims_metadata()
    assert meta["ndim"] == 3
    assert meta["level_shapes"] == [[20, 40, 60], [20, 20, 30]]
    assert meta["current_level"] == 0
    assert meta["volume"] is True
    assert meta["controls"]["visible"] is True
    assert meta["multiscale"]["current_level"] == 0
    assert meta["multiscale"]["policy"] == "latency"

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    layer_block = snapshot.layers[0].block
    assert snapshot.metadata["source_path"] == "/data/sample.zarr"
    assert layer_block["source"]["kind"] == "ome-zarr"
    dims_block = snapshot.viewer.dims
    assert dims_block["ndisplay"] == 3
    camera_block = snapshot.viewer.camera
    assert camera_block["ndisplay"] == 3


def test_update_2d_scene(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(
        use_volume=False,
        data_wh=(128, 256),
        zarr_axes="yx",
        zarr_shape=(256, 128),
    )
    manager.update_from_sources(
        worker=worker,
        scene_state=_StubSceneState(center=(64.0, 32.0), zoom=1.5),
        multiscale_state=None,
        volume_state=None,
        current_step=(5, 6),
        ndisplay=2,
        zarr_path=None,
        scene_source=None,
    )

    meta = manager.dims_metadata()
    assert meta["ndim"] == 2
    assert meta["level_shapes"] == [[256, 128]]
    assert meta["current_level"] == 0
    assert meta.get("multiscale") is None
    assert meta.get("controls") is not None
    assert meta.get("volume") is False

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    dims_block = snapshot.viewer.dims
    assert dims_block["current_step"] == [5, 6]
    assert snapshot.metadata.get("source_path") is None


def test_empty_multiscale_levels(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(use_volume=False, data_wh=(32, 32))
    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state={"levels": []},
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        scene_source=None,
    )

    meta = manager.dims_metadata()
    assert meta.get("multiscale") is None

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    assert snapshot.layers[0].block.get("multiscale") is None


def test_update_with_worker_only(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(use_volume=False, data_wh=(32, 48), zarr_axes="yx", zarr_shape=(32, 48), zarr_dtype="float32")
    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        viewer_model=None,
        scene_source=None,
    )

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    layer_block = snapshot.layers[0].block
    assert layer_block["shape"] == [32, 48]
    assert layer_block["axis_labels"] == ["y", "x"]
    assert layer_block.get("source") is None


def test_worker_ndisplay_3d(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(
        use_volume=True,
        data_wh=(48, 32),
        data_d=4,
        zarr_axes="zyx",
        zarr_shape=(4, 32, 48),
        zarr_dtype="float32",
    )

    scene_source = _StubSceneSource(current_level=0, scale=(1.0, 1.0, 1.0))
    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=3,
        zarr_path=None,
        viewer_model=None,
        scene_source=scene_source,
    )

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    dims_block = snapshot.viewer.dims
    assert dims_block["ndisplay"] == 3
    assert dims_block["axis_labels"] == ["z", "y", "x"]
    assert snapshot.layers[0].block.get("volume") is True
    assert len(dims_block["displayed"]) == 3


def test_manager_uses_adapter_multiscale_state(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(
        use_volume=False,
        data_wh=(32, 32),
        zarr_axes="yx",
        zarr_shape=(16, 32),
        zarr_dtype="float32",
    )

    multiscale_state = {
        "levels": [
            {"path": "level_0", "downsample": [1.0, 1.0], "shape": [32, 32]},
            {"path": "level_1", "downsample": [2.0, 2.0], "shape": [16, 16]},
        ],
        "current_level": 1,
        "policy": "latency",
        "index_space": "base",
    }
    source = _StubSceneSource(
        current_level=1,
        scale=(0.5, 0.25),
        descriptors=[
            SimpleNamespace(index=0, path="level_0", shape=(32, 32), downsample=(1.0, 1.0), scale=(1.0, 1.0)),
            SimpleNamespace(index=1, path="level_1", shape=(16, 16), downsample=(2.0, 2.0), scale=(0.5, 0.5)),
        ],
    )

    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state=multiscale_state,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path="/tmp/test.zarr",
        scene_source=source,
    )

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    assert snapshot.metadata["source_path"] == "/tmp/test.zarr"
    layer_block = snapshot.layers[0].block
    ms_block = layer_block.get("multiscale")
    assert ms_block is not None
    assert ms_block["current_level"] == 1
    assert len(ms_block["levels"]) == 2
    assert ms_block["levels"][1]["downsample"] == [2.0, 2.0]
    assert layer_block["scale"] == [0.5, 0.25]


def test_manager_multiscale_levels_from_scene_source(manager: ViewerSceneManager) -> None:
    worker = _StubWorker(use_volume=False, data_wh=(64, 32))
    multiscale_state = {
        "levels": [
            {"path": "level_0", "downsample": [1.0, 1.0], "shape": [32, 64]},
            {"path": "level_1", "downsample": [2.0, 2.0], "shape": [16, 32]},
        ],
        "current_level": 1,
        "policy": "latency",
        "index_space": "base",
    }
    source = _StubSceneSource(
        current_level=1,
        scale=(0.25, 0.5),
        descriptors=[
            SimpleNamespace(index=0, path="level_0", shape=(32, 64), downsample=(1.0, 1.0), scale=(1.0, 1.0)),
            SimpleNamespace(index=1, path="level_1", shape=(16, 32), downsample=(2.0, 2.0), scale=(0.25, 0.5)),
        ],
    )

    manager.update_from_sources(
        worker=worker,
        scene_state=None,
        multiscale_state=multiscale_state,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path="/data/sample.zarr",
        scene_source=source,
    )

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    layer_block = snapshot.layers[0].block
    ms_block = layer_block["multiscale"]
    assert ms_block["current_level"] == 1
    assert ms_block["levels"][1]["path"] == "level_1"
    assert ms_block["levels"][1]["downsample"] == [2.0, 2.0]
    assert layer_block["scale"] == [0.25, 0.5]
    assert snapshot.metadata["source_path"] == "/data/sample.zarr"
class _StubSceneSource:
    def __init__(self, *, current_level: int = 0, scale: Sequence[float] = (1.0, 1.0, 1.0), descriptors: Optional[Sequence[SimpleNamespace]] = None) -> None:
        self.current_level = current_level
        self._scale = tuple(float(v) for v in scale)
        self.level_descriptors = list(descriptors or [])

    def level_scale(self, index: int) -> Sequence[float]:
        return self._scale
