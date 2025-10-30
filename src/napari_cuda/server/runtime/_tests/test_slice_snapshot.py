from __future__ import annotations

from types import SimpleNamespace

import pytest

import napari_cuda.server.data.lod as lod
from napari_cuda.server.runtime.worker.snapshots.plane import apply_slice_level
from napari_cuda.server.runtime.data import SliceROI
from napari_cuda.server.runtime.viewport import ViewportState


class _FakeCamera:
    def __init__(self) -> None:
        self.range_calls: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.rect = None
        self.center = None
        self.zoom = None

    def set_range(self, *, x: tuple[float, float], y: tuple[float, float]) -> None:
        self.range_calls.append((x, y))


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
        shape: tuple[int, int],
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


def test_apply_slice_level_updates_plane_state(monkeypatch: pytest.MonkeyPatch) -> None:
    viewport_state = ViewportState()
    viewport_state.plane.applied_level = 0
    viewport_state.plane.applied_roi = None
    viewport_state.plane.applied_step = None

    camera = _FakeCamera()
    view = SimpleNamespace(camera=camera)
    logger = _FakeLayerLogger()
    emitted: list[str] = []
    result_cache: dict[str, object] = {}

    class _Worker:
        def __init__(self) -> None:
            self.viewport_state = viewport_state
            self.view = view
            self._emit_current_camera_pose = lambda reason: emitted.append(reason)
            self._sticky_contrast = False
            self._roi_align_chunks = False
            self._roi_pad_chunks = 0
            self._layer_logger = logger
            self._log_layer_debug = True
            self._z_index = 5
            self._viewport_runner = None
            self._data_wh = (0, 0)
            self._data_d = None
            self._napari_layer = None

    worker = _Worker()
    viewport_state.plane.update_pose(
        rect=(0.0, 0.0, 80.0, 40.0),
        center=(20.0, 10.0),
        zoom=1.5,
    )

    roi = SliceROI(2, 6, 4, 12)

    def _fake_plane_wh(_source: object, _level: int) -> tuple[int, int]:
        return (40, 80)

    def _fake_roi(_worker: object, _source: object, _level: int) -> SliceROI:
        return roi

    def _fake_apply(
        _worker: object,
        _source: object,
        level: int,
        roi_in: SliceROI,
        *,
        update_contrast: bool,
        step: object = None,
    ) -> tuple[int, int]:
        result_cache["level"] = level
        result_cache["roi"] = roi_in
        result_cache["update_contrast"] = update_contrast
        worker._data_wh = (roi_in.width, roi_in.height)
        worker._data_d = None
        return (roi_in.height, roi_in.width)

    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.snapshots.plane.plane_wh_for_level",
        _fake_plane_wh,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.snapshots.plane.viewport_roi_for_level",
        _fake_roi,
    )
    monkeypatch.setattr(
        "napari_cuda.server.runtime.worker.snapshots.plane.apply_slice_roi",
        _fake_apply,
    )

    context = lod.LevelContext(
        level=3,
        step=(3, 0, 0),
        z_index=3,
        shape=(1, 40, 80),
        scale_yx=(1.0, 1.0),
        contrast=(0.0, 1.0),
        axes="zyx",
        dtype="float32",
    )

    apply_slice_level(worker, source=object(), applied=context)

    assert emitted == ["slice-apply"]
    assert result_cache["level"] == 3
    assert result_cache["roi"] == roi
    assert result_cache["update_contrast"] is True

    plane_state = viewport_state.plane
    assert plane_state.applied_level == 3
    assert plane_state.applied_step == (3, 0, 0)
    assert plane_state.applied_roi == roi
    assert plane_state.applied_roi_signature is None

    assert logger.calls[-1]["mode"] == "slice"
    assert logger.calls[-1]["level"] == 3
    assert logger.calls[-1]["shape"] == (roi.height, roi.width)

    assert camera.range_calls == []
    assert camera.rect.pos == (0.0, 0.0)
    assert camera.rect.size == (80.0, 40.0)
    assert camera.center == (20.0, 10.0)
    assert camera.zoom == 1.5
