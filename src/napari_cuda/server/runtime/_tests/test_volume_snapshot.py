from __future__ import annotations

from types import SimpleNamespace

import pytest

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.scene_state_applier import SceneStateApplyContext, SceneStateApplier
from napari_cuda.server.runtime.state_structs import ViewportState
from napari_cuda.server.runtime.volume_snapshot import apply_volume_camera_pose, apply_volume_level


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

        def _get_level_volume(self, _source: object, level: int) -> object:
            return ("volume", level)

        def _build_scene_state_context(self, cam: object) -> SceneStateApplyContext:
            return SceneStateApplyContext(
                use_volume=True,
                viewer=None,
                camera=cam,
                visual=None,
                layer=None,
                scene_source=None,
                active_ms_level=0,
                z_index=None,
                last_roi=None,
                preserve_view_on_switch=False,
                sticky_contrast=False,
                idr_on_z=False,
                data_wh=(0, 0),
                volume_scale=None,
                state_lock=None,
                ensure_scene_source=lambda: None,
                plane_scale_for_level=lambda *_args: (1.0, 1.0),
                load_slice=lambda *_args: None,
                mark_render_tick_needed=lambda: None,
                request_encoder_idr=None,
            )

    worker = _Worker()

    calls: dict[str, object] = {}

    def _fake_apply_volume_layer(ctx: SceneStateApplyContext, *, volume: object, contrast: tuple[float, float]) -> tuple[tuple[int, int], int]:
        calls["ctx"] = ctx
        calls["volume"] = volume
        calls["contrast"] = contrast
        return (128, 256), 64

    monkeypatch.setattr(SceneStateApplier, "apply_volume_layer", staticmethod(_fake_apply_volume_layer))

    class _Source:
        axes = ("z", "y", "x")

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

    result = apply_volume_level(
        worker,
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

    ctx: SceneStateApplyContext = calls["ctx"]  # type: ignore[assignment]
    assert ctx.volume_scale == (0.5, 1.5, 3.0)
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
    monkeypatch.setattr("napari_cuda.server.runtime.volume_snapshot.TurntableCamera", _FakeTurntableCamera)
    viewport_state = ViewportState()
    viewport_state.volume.pose_angles = (1.0, 2.0, 3.0)

    camera = _FakeTurntableCamera()
    worker = SimpleNamespace(
        viewport_state=viewport_state,
        view=SimpleNamespace(camera=camera),
    )

    snapshot = RenderLedgerSnapshot(
        center=(10.0, 20.0, 30.0),
        angles=(45.0, 30.0),
        distance=200.0,
        fov=45.0,
    )

    apply_volume_camera_pose(worker, snapshot)

    vol_state = viewport_state.volume
    assert vol_state.pose_center == (10.0, 20.0, 30.0)
    assert vol_state.pose_angles == (45.0, 30.0, 3.0)
    assert vol_state.pose_distance == 200.0
    assert vol_state.pose_fov == 45.0

    assert camera.center == (10.0, 20.0, 30.0)
    assert camera.azimuth == 45.0
    assert camera.elevation == 30.0
    assert camera.roll == 3.0
    assert camera.distance == 200.0
    assert camera.fov == 45.0
