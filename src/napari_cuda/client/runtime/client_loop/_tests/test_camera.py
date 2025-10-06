from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List, Tuple

import pytest

from napari_cuda.client.control.client_state_ledger import IntentRecord, ClientStateLedger
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.runtime.client_loop import camera
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState


def _make_state(
) -> tuple[
    ControlStateContext,
    camera.CameraState,
    ClientLoopState,
    ClientStateLedger,
    List[Tuple[IntentRecord, str]],
    Callable[[IntentRecord, str], bool],
]:
    cam_env = SimpleNamespace(
        zoom_base=1.2,
        camera_rate_hz=120.0,
        orbit_deg_per_px_x=0.5,
        orbit_deg_per_px_y=0.25,
    )
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    cam_state = camera.CameraState.from_env(cam_env)
    loop_state = ClientLoopState()
    state_store = ClientStateLedger()
    cam_state.cam_min_dt = 0.0
    dispatched: List[Tuple[IntentRecord, str]] = []

    def dispatch(pending_update: IntentRecord, origin: str) -> bool:
        dispatched.append((pending_update, origin))
        return True

    return control_state, cam_state, loop_state, state_store, dispatched, dispatch


def test_handle_wheel_zoom_posts_camera_zoom() -> None:
    control_state, cam_state, loop_state, state_store, dispatched, dispatch = _make_state()

    camera.handle_wheel_zoom(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'angle_y': 120, 'x_px': 10, 'y_px': 20},
        widget_to_video=lambda x, y: (x, y),
        server_anchor_from_video=lambda x, y: (x, y),
        log_dims_info=False,
    )

    assert dispatched, "expected state.update dispatch"
    pending, origin = dispatched[-1]
    assert origin == 'wheel'
    assert pending.scope == 'camera'
    assert pending.key == 'zoom'
    assert pending.value == {'factor': 1.2, 'anchor_px': [10.0, 20.0]}
    assert cam_state.last_zoom_factor is not None
    assert cam_state.last_zoom_widget_px == (10.0, 20.0)


def test_handle_pointer_pan_posts_delta() -> None:
    control_state, cam_state, loop_state, state_store, dispatched, dispatch = _make_state()

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'down', 'x_px': 0, 'y_px': 0, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'move', 'x_px': 5, 'y_px': 3, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'up', 'x_px': 5, 'y_px': 3, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    assert dispatched, "expected pan dispatch"
    pan_updates = [entry for entry in dispatched if entry[0].key == 'pan']
    assert pan_updates, "expected at least one pan entry"
    assert any(
        update.scope == 'camera' and update.value == {'dx_px': 5.0, 'dy_px': 3.0}
        for update, _origin in pan_updates
    ), "expected pan delta to carry accumulated values"


def test_handle_pointer_orbit_posts_delta() -> None:
    control_state, cam_state, loop_state, state_store, dispatched, dispatch = _make_state()
    control_state.dims_ready = True
    control_state.dims_meta['mode'] = 'volume'
    control_state.dims_meta['ndisplay'] = 3

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'down', 'x_px': 0, 'y_px': 0, 'mods': 0x2000},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=True,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'move', 'x_px': 4, 'y_px': 2, 'mods': 0x2000},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=True,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        control_state,
        cam_state,
        loop_state,
        state_store,
        dispatch,
        {'phase': 'up', 'x_px': 4, 'y_px': 2, 'mods': 0x2000},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=True,
        alt_mask=0x2000,
    )

    orbit_updates = [(entry[0], entry[1]) for entry in dispatched if entry[0].key == 'orbit']
    assert orbit_updates, "expected orbit dispatch"
    non_zero = [(update, src) for update, src in orbit_updates if any(abs(v) > 0 for v in update.value.values())]
    assert non_zero, "expected orbit delta to carry rotation"
    pending, origin = non_zero[-1]
    assert pending.scope == 'camera'
    assert pending.metadata['mode'] == 'orbit'


def test_reset_and_set_camera_send_payloads() -> None:
    control_state, cam_state, loop_state, state_store, dispatched, dispatch = _make_state()

    assert camera.reset_camera(
        control_state,
        loop_state,
        state_store,
        dispatch,
        origin='ui',
    ) is True
    assert dispatched, "expected reset dispatch"
    reset_pending, reset_origin = dispatched[-1]
    assert reset_origin == 'ui'
    assert reset_pending.scope == 'camera'
    assert reset_pending.key == 'reset'
    assert reset_pending.value == {'reason': 'ui'}

    assert camera.set_camera(
        control_state,
        loop_state,
        state_store,
        dispatch,
        center=(1, 2, 3),
        zoom=2.5,
        angles=(0.1, 0.2, 0.3),
        origin='ui',
    ) is True
    set_pending, set_origin = dispatched[-1]
    assert set_origin == 'ui'
    assert set_pending.scope == 'camera'
    assert set_pending.key == 'set'
    assert set_pending.value == {
        'center': [1.0, 2.0, 3.0],
        'zoom': 2.5,
        'angles': [0.1, 0.2, 0.3],
    }
