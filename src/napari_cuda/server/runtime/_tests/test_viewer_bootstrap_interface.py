from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_cuda.server.runtime.bootstrap.interface import (
    ViewerBootstrapInterface,
)


class DummyWorker:
    def __init__(self) -> None:
        self.width = 640
        self.height = 480
        self.volume_depth = 32
        self.viewport_state = SimpleNamespace()
        self._debug_policy = SimpleNamespace(worker=SimpleNamespace(debug_overlay=False, layer_interpolation="linear"))
        self._zarr_init_z = 7
        self._log_layer_debug = True
        self._current_level_index = None
        self._frame_calls: list[tuple[float, float, float]] = []
        self._plane_handle = None
        self._volume_handle = None
        self._ensure_plane_called = False
        self._ensure_volume_called = False

    # Worker helpers used by the facade -----------------------------------
    def _set_current_level_index(self, value: int) -> None:
        self._current_level_index = int(value)

    def _frame_volume_camera(self, w: float, h: float, d: float) -> None:
        self._frame_calls.append((w, h, d))

    def _register_plane_visual(self, node) -> None:
        self._plane_handle = node

    def _register_volume_visual(self, node) -> None:
        self._volume_handle = node

    def _ensure_plane_visual(self):
        self._ensure_plane_called = True
        return "plane-visual"

    def _ensure_volume_visual(self):
        self._ensure_volume_called = True
        return "volume-visual"


def test_viewer_bootstrap_interface_field_updates() -> None:
    worker = DummyWorker()
    facade = ViewerBootstrapInterface(worker)

    assert facade.width == 640
    assert facade.height == 480
    assert facade.volume_depth == 32
    assert facade.zarr_init_z == 7
    assert facade.log_layer_debug is True

    facade.set_canvas("canvas")
    facade.set_view("view")
    facade.set_viewer("viewer")
    facade.set_napari_layer("layer")
    facade.set_scene_source("scene-source")
    facade.set_current_level_index(3)
    facade.set_zarr_level("level_02")
    facade.set_zarr_axes("zyx")
    facade.set_zarr_shape([100, 200, 300])
    facade.set_zarr_dtype("float32")
    facade.set_zarr_clim((0.0, 1.0))
    facade.set_z_index(41)
    facade.set_volume_scale((1.0, 2.0, 3.0))
    facade.set_data_wh((512, 256))
    facade.set_data_depth(88)
    facade.set_bootstrap_full_roi(True)
    facade.set_debug_overlay("overlay")

    facade.frame_volume_camera(10.0, 20.0, 30.0)
    facade.register_plane_visual("plane-node")
    facade.register_volume_visual("volume-node")
    assert facade.ensure_plane_visual() == "plane-visual"
    assert facade.ensure_volume_visual() == "volume-visual"

    facade.set_volume_metadata(
        level_index=4,
        level_path="level_04",
        axes=("z", "y", "x"),
        shape=(16, 32, 64),
        dtype="uint16",
        clim=(5, 10),
    )

    # Verify worker fields were applied -----------------------------------
    assert worker.canvas == "canvas"
    assert worker.view == "view"
    assert worker._viewer == "viewer"
    assert worker._napari_layer == "layer"
    assert worker._scene_source == "scene-source"
    assert worker._current_level_index == 4  # overwritten by set_volume_metadata final call
    assert worker._zarr_level == "level_04"
    assert worker._zarr_axes == "zyx"
    assert worker._zarr_shape == (16, 32, 64)
    assert worker._zarr_dtype == "uint16"
    assert worker._zarr_clim == (5, 10)
    assert worker._z_index == 41
    assert worker._volume_scale == (1.0, 2.0, 3.0)
    assert worker._data_wh == (512, 256)
    assert worker._data_d == 88
    assert worker._bootstrap_full_roi is True
    assert worker._debug_overlay == "overlay"
    assert worker._frame_calls == [(10.0, 20.0, 30.0)]
    assert worker._plane_handle == "plane-node"
    assert worker._volume_handle == "volume-node"
    assert worker._ensure_plane_called is True
    assert worker._ensure_volume_called is True


def test_viewer_bootstrap_interface_lod_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = DummyWorker()
    facade = ViewerBootstrapInterface(worker)

    def fake_build(*, source, level, step):
        assert source == "scene"
        assert level == 3
        assert step == (1, 2, 3)
        return "built-context"

    def fake_resolve(worker_arg, source, level):
        assert worker_arg is worker
        assert source == "scene"
        assert level == 5
        return 7, True

    def fake_load(worker_arg, source, level):
        assert worker_arg is worker
        assert source == "scene"
        assert level == 9
        return "volume-bytes"

    monkeypatch.setattr(
        "napari_cuda.server.runtime.bootstrap.interface.lod_build_level_context",
        fake_build,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.bootstrap.interface.lod_resolve_volume_intent_level",
        fake_resolve,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.bootstrap.interface.lod_load_volume",
        fake_load,
    )

    context = facade.build_level_context(
        source="scene",
        level=3,
        step=(1, 2, 3),
    )
    resolved = facade.resolve_volume_intent_level("scene", 5)
    loaded = facade.load_volume("scene", 9)

    assert context == "built-context"
    assert resolved == (7, True)
    assert loaded == "volume-bytes"
