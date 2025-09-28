from __future__ import annotations

from dataclasses import dataclass

import pytest

from napari_cuda.server.camera_controller import (
    CameraCommandOutcome,
    CameraDebugFlags,
    apply_camera_commands,
)
from napari_cuda.server.server_scene import ServerSceneCommand


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


def test_apply_camera_commands_zoom_records_intent_and_callbacks() -> None:
    cam = _StubCamera([], [])
    cmd = ServerSceneCommand(kind="zoom", factor=1.5)
    zooms: list[float] = []
    render_marks: list[bool] = []
    policy_marks: list[bool] = []

    outcome = apply_camera_commands(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(100, 100),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
        mark_render_tick_needed=lambda: render_marks.append(True),
        trigger_policy_refresh=lambda: policy_marks.append(True),
        record_zoom_intent=lambda ratio: zooms.append(ratio),
        last_zoom_hint_ts=None,
        zoom_hint_hold_s=0.0,
        now_fn=lambda: 50.0,
    )

    assert render_marks == [True]
    assert policy_marks == [True]
    assert isinstance(outcome, CameraCommandOutcome)
    assert outcome.camera_changed is True
    assert outcome.policy_triggered is True
    assert zooms == [pytest.approx(1 / 1.5)]
    assert outcome.zoom_intent == pytest.approx(1 / 1.5)
    assert outcome.last_zoom_hint_ts == pytest.approx(50.0)
    assert outcome.interaction_ts == pytest.approx(50.0)
    assert cam.zoom_calls == [pytest.approx(1.5)]


def test_apply_camera_commands_zoom_honours_hold_window() -> None:
    cam = _StubCamera([], [])
    cmd = ServerSceneCommand(kind="zoom", factor=2.0)
    recorded: list[float] = []

    outcome = apply_camera_commands(
        [cmd],
        camera=cam,
        view=_StubView(),
        canvas_size=(120, 90),
        reset_camera=lambda c: None,
        debug_flags=CameraDebugFlags(),
        record_zoom_intent=lambda ratio: recorded.append(ratio),
        last_zoom_hint_ts=10.0,
        zoom_hint_hold_s=5.0,
        now_fn=lambda: 12.0,
    )

    assert recorded == []
    assert outcome.zoom_intent is None
    assert outcome.last_zoom_hint_ts == 10.0


def test_apply_camera_commands_pan_without_zoom_passthrough() -> None:
    cam = _StubCamera([], [])
    cmd = ServerSceneCommand(kind="pan", dx_px=3.0, dy_px=-2.0)
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
    assert outcome.last_zoom_hint_ts is None
    assert outcome.interaction_ts is not None
    assert cam.pan_calls == [(3.0, -2.0)]


def test_apply_camera_commands_requires_positive_zoom() -> None:
    cam = _StubCamera([], [])
    bad = ServerSceneCommand(kind="zoom", factor=0.0)
    with pytest.raises(ValueError):
        apply_camera_commands(
            [bad],
            camera=cam,
            view=_StubView(),
            canvas_size=(100, 100),
            reset_camera=lambda c: None,
            debug_flags=CameraDebugFlags(),
        )
