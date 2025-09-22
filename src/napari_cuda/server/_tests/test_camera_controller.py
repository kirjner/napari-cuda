from __future__ import annotations

from dataclasses import dataclass

import pytest

from napari_cuda.server.camera_controller import (
    CameraCommandOutcome,
    CameraDebugFlags,
    apply_camera_commands,
)
from napari_cuda.server.state_machine import CameraCommand


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

    monkeypatch.setattr("napari_cuda.server.camera_controller.camops.apply_zoom_2d", _zoom_2d)
    monkeypatch.setattr("napari_cuda.server.camera_controller.camops.apply_pan_2d", _pan_2d)
    yield


def test_apply_camera_commands_zoom_records_intent() -> None:
    cam = _StubCamera([], [])
    cmd = CameraCommand(kind="zoom", factor=1.5)
    outcome = apply_camera_commands(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(100, 100),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
    )
    assert isinstance(outcome, CameraCommandOutcome)
    assert outcome.camera_changed is True
    assert outcome.policy_triggered is True
    assert outcome.zoom_intent == pytest.approx(1.5)
    assert cam.zoom_calls == [pytest.approx(1.5)]


def test_apply_camera_commands_pan_without_zoom_passthrough() -> None:
    cam = _StubCamera([], [])
    cmd = CameraCommand(kind="pan", dx_px=3.0, dy_px=-2.0)
    outcome = apply_camera_commands(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(200, 200),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
    )
    assert outcome.camera_changed is True
    assert outcome.policy_triggered is True
    assert outcome.zoom_intent is None
    assert cam.pan_calls == [(3.0, -2.0)]


def test_apply_camera_commands_requires_positive_zoom() -> None:
    cam = _StubCamera([], [])
    bad = CameraCommand(kind="zoom", factor=0.0)
    with pytest.raises(ValueError):
        apply_camera_commands(
            [bad],
            camera=cam,
            view=_StubView(),
            canvas_size=(100, 100),
            reset_camera=lambda c: None,
            debug_flags=CameraDebugFlags(),
        )
