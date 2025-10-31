from __future__ import annotations

from types import SimpleNamespace

import pytest

import napari_cuda.server.data.lod as lod
from napari_cuda.server.viewstate import (
    RenderLedgerSnapshot,
)
from napari_cuda.server.runtime.render_loop.apply_interface import RenderApplyInterface
from napari_cuda.server.runtime.render_loop.apply.render_state.volume import (
    apply_volume_camera_pose,
    apply_volume_level,
)
from napari_cuda.server.runtime.viewport import ViewportState


class _FakeLayerLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log(
        self,
        *,
        enabled: bool,
        mode: str,
        level: int,
        z_index: int | None,
        shape: tuple[int, ...],
        contrast: tuple[float, float],
        downgraded: bool,
    ) -> None:
        self.calls.append(
            {
                "enabled": enabled,
                "mode": mode,
                "level": level,
                "z_index": z_index,
                "shape": shape,
                "contrast": contrast,
                "downgraded": downgraded,
            }
        )


def test_apply_volume_level_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    viewport_state = ViewportState()
    logger = _FakeLayerLogger()

    class _Worker:
        def __init__(self) -> None:
            self.viewport_state = viewport_state
            self._volume_scale = (1.0, 1.0, 1.0)
            self._data_wh = (0, 0)
            self._data_d = None
            self._layer_logger = logger
            self._log_layer_debug = True
            self.view = SimpleNamespace(camera=None)
            self._plane_visual_node = SimpleNamespace()
            self._volume_visual_node = SimpleNamespace()
            self._volume_max_bytes = None
            self._volume_max_voxels = None
            self._hw_limits = SimpleNamespace(volume_max_bytes=None, volume_max_voxels=None)
            self._napari_layer = None

        def _load_volume(self, _source: object, level: int) -> object:
            return ("volume", level)

        def _ensure_volume_visual(self):
            return SimpleNamespace()

    worker = _Worker()

    calls: dict[str, object] = {}

    def _fake_apply_volume_layer(
        *,
        layer: object,
        volume: object,
        contrast: tuple[float, float],
        scale: tuple[float, float, float],
        ensure_volume_visual: object,
    ) -> tuple[tuple[int, int], int]:
        calls["layer"] = layer
        calls["volume"] = volume
        calls["contrast"] = contrast
        calls["scale"] = scale
        calls["ensure_volume_visual"] = ensure_volume_visual
        return (128, 256), 64

    monkeypatch.setattr(
        "napari_cuda.server.runtime.render_loop.apply.render_state.volume.apply_volume_layer_data",
        _fake_apply_volume_layer,
    )

    class _Source:
        axes = ("z", "y", "x")
        level_descriptors = [SimpleNamespace(shape=(64, 256, 128)) for _ in range(3)]
        dtype = "float32"

        def level_scale(self, level: int) -> tuple[float, float, float]:
            return (0.5, 1.5, 3.0)

    context = lod.LevelContext(
        level=2,
        step=(8, 4, 2),
        z_index=None,
        shape=(64, 256, 128),
        scale_yx=(1.5, 3.0),
        contrast=(0.0, 1.0),
        axes="zyx",
        dtype="float32",
    )

    snapshot_iface = RenderApplyInterface(worker)
    result = apply_volume_level(
        snapshot_iface,
        source=_Source(),
        applied=context,
        downgraded=True,
    )

    assert result.level == 2
    assert result.downgraded is True
    assert result.data_wh == (128, 256)
    assert result.data_d == 64
    assert pytest.approx(result.scale) == (0.5, 1.5, 3.0)

    volume_state = viewport_state.volume
    assert volume_state.level == 2
    assert volume_state.downgraded is True
    assert volume_state.scale == (0.5, 1.5, 3.0)

    assert worker._volume_scale == (0.5, 1.5, 3.0)
    assert worker._data_wh == (128, 256)
    assert worker._data_d == 64

    assert logger.calls[-1]["mode"] == "volume"
    assert logger.calls[-1]["level"] == 2
    assert logger.calls[-1]["shape"] == (64, 256, 128)
    assert logger.calls[-1]["downgraded"] is True

    assert calls["scale"] == (0.5, 1.5, 3.0)
    assert calls["volume"] == ("volume", 2)
    assert calls["contrast"] == (0.0, 1.0)


class _FakeTurntableCamera:
    def __init__(self) -> None:
        self.center = (0.0, 0.0, 0.0)
        self.azimuth = 0.0
        self.elevation = 0.0
        self.roll = 0.0
        self.distance = 0.0
        self.fov = 0.0


def test_apply_volume_camera_pose_updates_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "napari_cuda.server.runtime.render_loop.apply.render_state.volume.TurntableCamera",
        _FakeTurntableCamera,
    )
    viewport_state = ViewportState()
    viewport_state.volume.update_pose(angles=(1.0, 2.0, 3.0))

    camera = _FakeTurntableCamera()
    worker = SimpleNamespace(
        viewport_state=viewport_state,
        view=SimpleNamespace(camera=camera),
    )

    snapshot = RenderLedgerSnapshot(
        volume_center=(10.0, 20.0, 30.0),
        volume_angles=(45.0, 30.0, 3.0),
        volume_distance=200.0,
        volume_fov=45.0,
    )

    snapshot_iface = RenderApplyInterface(worker)
    apply_volume_camera_pose(snapshot_iface, snapshot)

    vol_state = viewport_state.volume
    assert vol_state.pose.center == (10.0, 20.0, 30.0)
    assert vol_state.pose.angles == (45.0, 30.0, 3.0)
    assert vol_state.pose.distance == 200.0
    assert vol_state.pose.fov == 45.0

    assert camera.center == (10.0, 20.0, 30.0)
    assert camera.azimuth == 45.0
    assert camera.elevation == 30.0
    assert camera.roll == 3.0
    assert camera.distance == 200.0
    assert camera.fov == 45.0
