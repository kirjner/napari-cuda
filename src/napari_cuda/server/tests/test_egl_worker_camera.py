from __future__ import annotations

from dataclasses import dataclass

import pytest

from types import SimpleNamespace
from vispy.scene.cameras import PanZoomCamera

from napari_cuda.server.runtime.camera.controller import CameraDeltaOutcome
from napari_cuda.server.runtime.ipc.mailboxes import RenderUpdateMailbox
from napari_cuda.server.runtime.worker import EGLRendererWorker
from napari_cuda.server.runtime.render_loop.ticks import camera as camera_tick
from napari_cuda.server.scene import CameraDeltaCommand
from napari_cuda.server.runtime.viewport import RenderMode, ViewportState


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
        self._render_mailbox = RenderUpdateMailbox()
        self._user_interaction_seen = False
        self._last_interaction_ts = 0.0
        self._capture = None
        self._camera_pose_callback = None
        self._zoom_hints: list[float] = []
        self.use_volume = False
        self._pose_seq = 1
        self._max_camera_command_seq = 0
        self._evaluate_level_policy = lambda: None
        self._viewport_runner = SimpleNamespace(
            ingest_camera_deltas=lambda _commands: None,
            update_camera_rect=lambda _rect: None,
        )
        self._last_plane_pose = None
        self._last_volume_pose = None
        self._viewport_state = ViewportState(mode=RenderMode.PLANE)
        self._viewport_state.plane.applied_level = 0
        self._viewport_state.plane.target_level = 0
        self._level_policy_suppressed = False
        self.viewport_state = self._viewport_state

    def _current_panzoom_rect(self):
        return None

    def _mark_render_tick_needed(self) -> None:
        pass

    def _apply_camera_reset(self, _camera) -> None:
        pass

def test_process_camera_deltas_invokes_pose_callback(monkeypatch) -> None:
    worker = _StubWorker()
    camera = worker.view.camera
    camera.center = (12.0, 34.0)
    camera.zoom = 2.0

    captured = []

    def _record_zoom_hint(_worker, commands) -> None:
        for command in commands:
            if command.kind == "zoom" and command.factor is not None and command.factor > 0.0:
                factor = float(command.factor)
                ratio = 1.0 / factor
                worker._zoom_hints.append(ratio)
                worker._render_mailbox.record_zoom_hint(ratio)

    monkeypatch.setattr(camera_tick, "record_zoom_hint", _record_zoom_hint)
    monkeypatch.setattr(
        camera_tick.viewport,
        "run",
        lambda w: w._emit_current_camera_pose("camera-delta"),
    )
    worker._camera_pose_callback = captured.append
    worker._snapshot_camera_pose = EGLRendererWorker._snapshot_camera_pose.__get__(worker, _StubWorker)  # type: ignore[attr-defined]
    worker._emit_current_camera_pose = EGLRendererWorker._emit_current_camera_pose.__get__(worker, _StubWorker)  # type: ignore[attr-defined]
    worker._emit_pose_from_camera = EGLRendererWorker._emit_pose_from_camera.__get__(worker, _StubWorker)  # type: ignore[attr-defined]
    worker._pose_from_camera = EGLRendererWorker._pose_from_camera.__get__(worker, _StubWorker)  # type: ignore[attr-defined]

    def _fake_process(tick_iface, commands):
        worker = tick_iface.worker
        tick_iface.mark_render_tick_needed()
        worker._user_interaction_seen = True
        for command in commands:
            if getattr(command, "kind", None) == "zoom" and command.factor not in (None, 0):
                ratio = 1.0 / float(command.factor)
                tick_iface.render_mailbox.record_zoom_hint(ratio)
                worker._zoom_hints.append(ratio)
        return CameraDeltaOutcome(
            camera_changed=True,
            policy_triggered=True,
            last_command_seq=int(commands[-1].command_seq),
            last_target=str(commands[-1].target),
        )

    monkeypatch.setattr(
        "napari_cuda.server.runtime.render_loop.ticks.camera._process_camera_deltas",
        _fake_process,
    )

    expected_zoom = worker.view.camera.zoom_factor

    command = CameraDeltaCommand(kind="zoom", factor=1.25, command_seq=9)
    camera_tick.process_commands(worker, [command])

    assert worker._user_interaction_seen is True
    assert worker._last_interaction_ts > 0.0
    hint = worker._render_mailbox.consume_zoom_hint(max_age=1.0)
    assert hint is not None and hint.ratio == pytest.approx(0.8)
    assert len(captured) == 1
    pose = captured[0]
    assert pose.command_seq == 10
    assert pose.target == "main"
    assert pose.zoom == pytest.approx(expected_zoom)
    assert pose.center is not None
    assert tuple(pose.center[:2]) == pytest.approx((12.0, 34.0))
