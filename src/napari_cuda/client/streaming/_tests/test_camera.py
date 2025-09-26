from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from napari_cuda.client.streaming.client_loop import camera
from napari_cuda.client.streaming.client_loop.loop_state import ClientLoopState


class FakeChannel:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> bool:
        self.payloads.append(dict(payload))
        return True


def _make_state() -> tuple[camera.CameraState, ClientLoopState, FakeChannel]:
    env = SimpleNamespace(
        zoom_base=1.2,
        camera_rate_hz=120.0,
        orbit_deg_per_px_x=0.5,
        orbit_deg_per_px_y=0.25,
    )
    state = camera.CameraState.from_env(env)
    loop_state = ClientLoopState()
    channel = FakeChannel()
    loop_state.state_channel = channel
    state.cam_min_dt = 0.0
    return state, loop_state, channel


def test_handle_wheel_zoom_posts_camera_zoom() -> None:
    state, loop_state, channel = _make_state()

    camera.handle_wheel_zoom(
        state,
        loop_state,
        {'angle_y': 120, 'x_px': 10, 'y_px': 20},
        widget_to_video=lambda x, y: (x, y),
        server_anchor_from_video=lambda x, y: (x, y),
        log_dims_info=False,
    )

    assert channel.payloads[-1]['type'] == 'camera.zoom_at'
    assert state.last_zoom_factor is not None
    assert state.last_zoom_widget_px == (10.0, 20.0)


def test_handle_pointer_pan_posts_delta() -> None:
    state, loop_state, channel = _make_state()

    camera.handle_pointer(
        state,
        loop_state,
        {'phase': 'down', 'x_px': 0, 'y_px': 0, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        state,
        loop_state,
        {'phase': 'move', 'x_px': 5, 'y_px': 3, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    camera.handle_pointer(
        state,
        loop_state,
        {'phase': 'up', 'x_px': 5, 'y_px': 3, 'mods': 0},
        widget_to_video=lambda x, y: (x, y),
        video_delta_to_canvas=lambda dx, dy: (dx, dy),
        log_dims_info=False,
        in_vol3d=False,
        alt_mask=0x2000,
    )

    assert any(payload['type'] == 'camera.pan_px' for payload in channel.payloads)


def test_reset_and_set_camera_send_payloads() -> None:
    state, loop_state, channel = _make_state()

    assert camera.reset_camera(loop_state, origin='ui') is True
    assert channel.payloads[-1]['type'] == 'camera.reset'

    assert camera.set_camera(loop_state, center=(1, 2, 3), zoom=2.5, angles=(0.1, 0.2, 0.3), origin='ui') is True
    assert channel.payloads[-1]['type'] == 'set_camera'
