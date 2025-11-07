from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.mirrors import NapariCameraMirror
from napari_cuda.client.control.control_state import ControlStateContext
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.protocol.envelopes import build_notify_camera


class _StubEmitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def record_confirmed(self, key: str, value: dict) -> None:
        self.calls.append((key, dict(value)))


class _StubPresenter:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict]] = []

    def apply_camera_update(self, *, mode: str, payload: dict) -> None:
        self.updates.append((mode, dict(payload)))


@pytest.mark.usefixtures("qtbot")
def test_ingest_notify_camera_updates_state_and_presenter() -> None:
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    ledger = ClientStateLedger()
    presenter = _StubPresenter()
    mirror = NapariCameraMirror(
        ledger=ledger,
        state=control_state,
        loop_state=loop_state,
        presenter=presenter,  # type: ignore[arg-type]
        ui_call=None,
        log_camera_info=False,
    )
    emitter = _StubEmitter()
    mirror.attach_emitter(emitter)  # prime with empty snapshot

    frame = build_notify_camera(
        session_id='sess-1',
        payload={'mode': 'zoom', 'origin': 'server', 'delta': {'factor': 1.25, 'anchor_px': [4.0, 8.0]}},
    )

    mirror.ingest_notify_camera(frame)

    assert control_state.camera_state['zoom'] == {'factor': 1.25, 'anchor_px': [4.0, 8.0]}
    confirmed = ledger.confirmed_value('camera', 'main', 'zoom')
    assert confirmed == {'factor': 1.25, 'anchor_px': [4.0, 8.0]}
    assert presenter.updates[-1] == ('zoom', {'factor': 1.25, 'anchor_px': [4.0, 8.0]})
    assert emitter.calls[-1] == ('zoom', {'factor': 1.25, 'anchor_px': [4.0, 8.0]})


@pytest.mark.usefixtures("qtbot")
def test_replay_last_payload_reinvokes_presenter() -> None:
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    ledger = ClientStateLedger()
    presenter = _StubPresenter()
    mirror = NapariCameraMirror(
        ledger=ledger,
        state=control_state,
        loop_state=loop_state,
        presenter=presenter,  # type: ignore[arg-type]
        ui_call=None,
        log_camera_info=False,
    )

    frame = build_notify_camera(
        session_id='sess-2',
        payload={'mode': 'pan', 'origin': 'server', 'delta': {'dx_px': 3.0, 'dy_px': -2.0}},
    )

    mirror.ingest_notify_camera(frame)
    presenter.updates.clear()

    mirror.replay_last_payload()

    assert presenter.updates == [('pan', {'dx_px': 3.0, 'dy_px': -2.0})]


@pytest.mark.usefixtures("qtbot")
def test_ingest_notify_camera_pose_updates_state() -> None:
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    ledger = ClientStateLedger()
    presenter = _StubPresenter()
    mirror = NapariCameraMirror(
        ledger=ledger,
        state=control_state,
        loop_state=loop_state,
        presenter=presenter,  # type: ignore[arg-type]
        ui_call=None,
        log_camera_info=False,
    )

    pose_payload = {'center': [10.0, 20.0], 'zoom': 0.5}
    frame = build_notify_camera(
        session_id='sess-3',
        payload={'mode': 'pose', 'origin': 'worker', 'state': pose_payload},
    )

    mirror.ingest_notify_camera(frame)

    assert control_state.camera_state['pose'] == pose_payload
    confirmed = ledger.confirmed_value('camera', 'main', 'pose')
    assert confirmed == pose_payload
    assert presenter.updates[-1] == ('pose', pose_payload)
