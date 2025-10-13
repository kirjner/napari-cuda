from __future__ import annotations

from dataclasses import dataclass

import pytest

from napari_cuda.server.runtime.camera_controller import (
    CameraDeltaOutcome,
    CameraDebugFlags,
    apply_camera_deltas,
)
from napari_cuda.server.scene import CameraDeltaCommand


@dataclass
class _StubView:
    pass


@dataclass
class _StubCamera:
    zoom_calls: list[float]
    pan_calls: list[tuple[float, float]]

    def __post_init__(self) -> None:
        self.center = (0.0, 0.0)

    def zoom(self, factor: float, *, center) -> None:  # type: ignore[override]
        self.zoom_calls.append(float(factor))
        self.center = tuple(center)


@pytest.fixture(autouse=True)
def _patch_camops(monkeypatch):
    def _zoom_2d(cam, factor, anchor, canvas, view):  # type: ignore[unused-ignore]
        cam.zoom(float(factor), center=anchor)

    def _pan_2d(cam, dx, dy, canvas, view):  # type: ignore[unused-ignore]
        cam.pan_calls.append((float(dx), float(dy)))

    monkeypatch.setattr("napari_cuda.server.runtime.camera_controller.camops.apply_zoom_2d", _zoom_2d)
    monkeypatch.setattr("napari_cuda.server.runtime.camera_controller.camops.apply_pan_2d", _pan_2d)
    yield


def test_apply_camera_deltas_zoom_marks_render_and_policy() -> None:
    cam = _StubCamera([], [])
    cmd = CameraDeltaCommand(kind="zoom", factor=1.5, command_seq=7)
    render_marks: list[bool] = []
    policy_marks: list[bool] = []

    outcome = apply_camera_deltas(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(100, 100),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
        mark_render_tick_needed=lambda: render_marks.append(True),
        trigger_policy_refresh=lambda: policy_marks.append(True),
    )

    assert render_marks == [True]
    assert policy_marks == [True]
    assert isinstance(outcome, CameraDeltaOutcome)
    assert outcome.camera_changed is True
    assert outcome.policy_triggered is True
    assert outcome.last_command_seq == 7
    assert outcome.last_target == "main"
    assert cam.zoom_calls == [pytest.approx(1.5)]


def test_apply_camera_deltas_pan_without_zoom_passthrough() -> None:
    cam = _StubCamera([], [])
    cmd = CameraDeltaCommand(kind="pan", dx_px=3.0, dy_px=-2.0, command_seq=11)
    outcome = apply_camera_deltas(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(200, 200),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
    )
    assert outcome.camera_changed is True
    assert outcome.policy_triggered is True
    assert outcome.last_command_seq == 11
    assert outcome.last_target == "main"
    assert cam.pan_calls == [(3.0, -2.0)]


def test_apply_camera_deltas_requires_positive_zoom() -> None:
    cam = _StubCamera([], [])
    bad = CameraDeltaCommand(kind="zoom", factor=0.0)
    with pytest.raises(ValueError):
        apply_camera_deltas(
            [bad],
            camera=cam,
            view=_StubView(),
            canvas_size=(100, 100),
            reset_camera=lambda c: None,
            debug_flags=CameraDebugFlags(),
        )
