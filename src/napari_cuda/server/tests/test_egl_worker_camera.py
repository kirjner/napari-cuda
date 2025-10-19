from __future__ import annotations

from dataclasses import dataclass

import pytest

from vispy.scene.cameras import PanZoomCamera

from napari_cuda.server.runtime.camera_controller import CameraDeltaOutcome
from napari_cuda.server.runtime.render_update_mailbox import RenderUpdateMailbox
from napari_cuda.server.runtime.egl_worker import EGLRendererWorker
from napari_cuda.server.scene import CameraDeltaCommand


@dataclass
class _StubView:
    camera: PanZoomCamera


class _StubWorker:
    def __init__(self) -> None:
        self.view = _StubView(PanZoomCamera())
        self.canvas = None
        self.width = 640
        self.height = 480
        self._debug_zoom_drift = False
        self._debug_pan = False
        self._debug_orbit = False
        self._debug_reset = False
        self._level_policy_refresh_needed = False
        self._render_mailbox = RenderUpdateMailbox()
        self._user_interaction_seen = False
        self._last_interaction_ts = 0.0
        self._capture = None
        self._camera_pose_callback = None
        self._zoom_hints: list[float] = []
        self.use_volume = False
        self._pose_seq = 1
        self._max_camera_command_seq = 0

    def _mark_render_tick_needed(self) -> None:
        pass

    def _apply_camera_reset(self, _camera) -> None:
        pass

    def _record_zoom_hint(self, commands) -> None:  # pragma: no cover - replaced during test
        pass


def test_process_camera_deltas_invokes_pose_callback(monkeypatch) -> None:
    worker = _StubWorker()
    camera = worker.view.camera
    camera.center = (12.0, 34.0)
    camera.zoom = 2.0

    captured = []

    def _record_zoom_hint(commands) -> None:
        for command in commands:
            if command.kind == "zoom" and command.factor is not None and command.factor > 0.0:
                factor = float(command.factor)
                ratio = factor if factor <= 1.0 else 1.0 / factor
                worker._zoom_hints.append(ratio)
                worker._render_mailbox.record_zoom_hint(ratio)

    worker._record_zoom_hint = _record_zoom_hint  # type: ignore[assignment]
    worker._camera_pose_callback = captured.append
    worker._snapshot_camera_pose = EGLRendererWorker._snapshot_camera_pose.__get__(worker, _StubWorker)  # type: ignore[attr-defined]
    worker._emit_current_camera_pose = EGLRendererWorker._emit_current_camera_pose.__get__(worker, _StubWorker)  # type: ignore[attr-defined]

    def _fake_process(self, commands):
        self._mark_render_tick_needed()
        self._level_policy_refresh_needed = True
        self._user_interaction_seen = True
        return CameraDeltaOutcome(
            camera_changed=True,
            policy_triggered=True,
            last_command_seq=int(commands[-1].command_seq),
            last_target=str(commands[-1].target),
        )

    monkeypatch.setattr(
        "napari_cuda.server.runtime.egl_worker._process_camera_deltas",
        _fake_process,
    )

    command = CameraDeltaCommand(kind="zoom", factor=1.25, command_seq=9)
    EGLRendererWorker.process_camera_deltas(worker, [command])  # type: ignore[arg-type]

    assert worker._level_policy_refresh_needed is True
    assert worker._user_interaction_seen is True
    assert worker._last_interaction_ts > 0.0
    hint = worker._render_mailbox.consume_zoom_hint(max_age=1.0)
    assert hint is not None and hint.ratio == pytest.approx(0.8)
    assert len(captured) == 1
    pose = captured[0]
    assert pose.command_seq == 10
    assert pose.target == "main"
    assert pose.zoom == pytest.approx(2.0)
    assert pose.center is not None
    assert tuple(pose.center[:2]) == pytest.approx((12.0, 34.0))
