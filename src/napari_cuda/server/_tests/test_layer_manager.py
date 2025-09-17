from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from napari_cuda.server.layer_manager import ViewerSceneManager


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
        if data_d is not None:
            self._data_d = data_d
        if zarr_axes is not None:
            self._zarr_axes = zarr_axes
        if zarr_shape is not None:
            self._zarr_shape = zarr_shape
        if zarr_dtype is not None:
            self._zarr_dtype = zarr_dtype
        if volume_dtype is not None:
            self.volume_dtype = volume_dtype


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
            {"path": "level_1", "downsample": [1.0, 2.0, 2.0]},
        ],
        "current_level": 0,
        "policy": "fixed",
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
    manager.update_from_sources(
        worker=worker,
        scene_state=_StubSceneState(center=(10.0, 20.0, 30.0), zoom=2.5),
        multiscale_state=_multiscale_state(),
        volume_state=_volume_state(),
        current_step=(1, 2, 3),
        ndisplay=3,
        zarr_path="/data/sample.zarr",
        extras={"note": "test"},
    )

    meta = manager.dims_metadata()
    assert meta["ndim"] == 3
    assert meta["sizes"] == [20, 40, 60]
    assert meta["volume"] is True
    assert meta["render"]["mode"] == "mip"
    assert meta["multiscale"]["current_level"] == 0
    assert meta["multiscale"]["policy"] == "fixed"

    scene = manager.scene_spec()
    assert scene is not None
    assert scene.layers[0].extras["zarr_path"] == "/data/sample.zarr"
    assert scene.dims.ndisplay == 3
    assert scene.camera is not None
    assert scene.camera.ndisplay == 3


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
        extras=None,
    )

    meta = manager.dims_metadata()
    assert meta["ndim"] == 2
    assert meta["sizes"] == [256, 128]
    assert meta.get("multiscale") is None
    assert meta.get("render") is None
    assert meta.get("volume") is False

    scene = manager.scene_spec()
    assert scene is not None
    assert scene.dims.current_step == [5, 6]
    assert scene.capabilities == ["layer.update", "layer.remove"]
