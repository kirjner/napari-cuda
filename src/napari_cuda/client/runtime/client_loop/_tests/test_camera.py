from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger, IntentRecord
from napari_cuda.client.control.emitters import NapariCameraIntentEmitter
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop import camera
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState


class DummyCamera:
    def __init__(self) -> None:
        self.center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.zoom: float = 1.0
        self.angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class DummyViewer:
    def __init__(self) -> None:
        self.camera = DummyCamera()


def _make_state(qtbot: Any) -> tuple[
    ControlStateContext,
    camera.CameraState,
    NapariCameraIntentEmitter,
    List[Tuple[IntentRecord, str]],
]:
    cam_env = SimpleNamespace(
        zoom_base=1.2,
        camera_rate_hz=120.0,
    )
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    cam_state = camera.CameraState.from_env(cam_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    state_ledger = ClientStateLedger()
    cam_state.cam_min_dt = 0.0
    dispatched: List[Tuple[IntentRecord, str]] = []

    def dispatch(pending_update: IntentRecord, origin: str) -> bool:
        dispatched.append((pending_update, origin))
        return True

    emitter = NapariCameraIntentEmitter(
        ledger=state_ledger,
        state=control_state,
        loop_state=loop_state,
        dispatch_state_update=dispatch,
        ui_call=None,
        log_camera_info=False,
    )
    return control_state, cam_state, emitter, dispatched


def test_emit_camera_set_from_viewer_posts_absolute(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()
    viewer.camera.center = (1.0, 2.0, 3.0)
    viewer.camera.zoom = 2.5
    viewer.camera.angles = (0.1, 0.2, 0.3)

    sent = camera.emit_camera_set_from_viewer(
        emitter,
        cam_state,
        viewer,
        origin='test',
        force=True,
    )

    assert sent is True
    assert dispatched, "expected state.update dispatch"
    pending, origin = dispatched[-1]
    assert origin == 'test'
    assert pending.scope == 'camera'
    assert pending.key == 'set'
    assert pending.value == {
        'center': [1.0, 2.0, 3.0],
        'zoom': 2.5,
        'angles': [0.1, 0.2, 0.3],
    }


def test_emit_camera_set_from_viewer_throttles_identical_payload(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()
    viewer.camera.center = (1.0, 2.0, 3.0)
    viewer.camera.zoom = 2.0

    assert camera.emit_camera_set_from_viewer(
        emitter,
        cam_state,
        viewer,
        origin='first',
        force=False,
    )
    assert len(dispatched) == 1

    # Second call with identical payload should be suppressed when not forced
    assert camera.emit_camera_set_from_viewer(
        emitter,
        cam_state,
        viewer,
        origin='second',
        force=False,
    ) is False
    assert len(dispatched) == 1


def test_handle_wheel_zoom_emits_absolute_update(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()
    viewer.camera.zoom = 1.5

    camera.handle_wheel_zoom(
        emitter,
        cam_state,
        {'angle_y': 120, 'x_px': 10, 'y_px': 20},
        widget_to_video=lambda x, y: (x, y),
        server_anchor_from_video=lambda x, y: (x, y),
        log_dims_info=False,
        viewer=viewer,
    )

    assert dispatched, "expected state.update dispatch"
    pending, origin = dispatched[-1]
    assert origin == 'wheel'
    assert pending.key == 'set'
    assert 'zoom' in pending.value


def test_handle_pointer_drag_sends_absolute(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()

    camera.handle_pointer(
        emitter,
        cam_state,
        {'phase': 'down', 'x_px': 0, 'y_px': 0, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
        viewer=viewer,
    )

    viewer.camera.center = (5.0, 6.0, 7.0)
    camera.handle_pointer(
        emitter,
        cam_state,
        {'phase': 'move', 'x_px': 5, 'y_px': 3, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
        viewer=viewer,
    )

    assert any(entry[0].key == 'set' for entry in dispatched), "expected camera.set intent"
    pending, _origin = dispatched[-1]
    assert pending.value.get('center') == [5.0, 6.0, 7.0]


def test_reset_camera_uses_viewer_snapshot(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()
    viewer.camera.center = (9.0, 8.0, 7.0)
    viewer.camera.zoom = 3.0

    assert camera.reset_camera(
        emitter,
        origin='ui',
        viewer=viewer,
        state=cam_state,
    )
    pending, origin = dispatched[-1]
    assert origin == 'ui'
    assert pending.key == 'set'
    assert pending.value['zoom'] == 3.0


def test_zoom_steps_at_center_updates_viewer(qtbot) -> None:
    _control_state, cam_state, emitter, dispatched = _make_state(qtbot)
    viewer = DummyViewer()
    viewer.camera.zoom = 2.0
    # keep last cursor location reasonable
    cam_state.cursor_wx = 100.0
    cam_state.cursor_wy = 50.0

    camera.zoom_steps_at_center(
        emitter,
        cam_state,
        steps=1,
        widget_to_video=lambda x, y: (x, y),
        server_anchor_from_video=lambda x, y: (x, y),
        log_dims_info=False,
        vid_size=(200, 100),
        viewer=viewer,
    )

    assert dispatched, "expected camera.set intent"
    pending, origin = dispatched[-1]
    assert origin == 'keys'
    assert pending.key == 'set'
    assert 'zoom' in pending.value
    assert viewer.camera.zoom != 2.0
