from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.emitters import NapariLayerIntentEmitter
from napari_cuda.client.control.mirrors.napari_layer_mirror import NapariLayerMirror
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.data.registry import RemoteLayerRegistry
from napari_cuda.client.rendering.presenter_facade import PresenterFacade
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.client.runtime.client_loop.scheduler import CallProxy
from napari_cuda.protocol.messages import (
    FrameEnvelope,
    NotifyLayersFrame,
    NotifyLayersPayload,
    NotifySceneFrame,
    NotifyScenePayload,
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    PROTO_VERSION,
)


def _nullcontext():
    class _Null:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    return _Null()


class FakeLayers(list):
    def __init__(self) -> None:
        super().__init__()
        self.events = SimpleNamespace(blocker=_nullcontext)

    def move(self, src: int, dst: int) -> None:
        item = self.pop(src)
        self.insert(dst, item)


class ViewerStub:
    def __init__(self) -> None:
        self.layers = FakeLayers()
        self._suppress_forward_flag = False


class PresenterStub(PresenterFacade):
    def __init__(self) -> None:
        super().__init__()
        self.layer_updates: list[Any] = []

    def apply_layer_delta(self, message: Any) -> None:  # type: ignore[override]
        self.layer_updates.append(message)


def _make_control_state() -> ControlStateContext:
    env = type(
        "Env",
        (),
        {
            "dims_rate_hz": 60.0,
            "wheel_step": 1,
            "settings_rate_hz": 30.0,
            "dims_z": None,
            "dims_z_min": None,
            "dims_z_max": None,
        },
    )
    state = ControlStateContext.from_env(env)
    state.session_id = "session-test"
    return state


def _layer_block(remote_id: str = "layer-1") -> dict[str, Any]:
    return {
        "layer_id": remote_id,
        "layer_type": "image",
        "name": "demo",
        "ndim": 2,
        "shape": [1, 1],
        "dtype": "float32",
        "axis_labels": ["y", "x"],
        "contrast_limits": [0.0, 1.0],
        "metadata": {},
        "render": {"mode": "mip"},
        "controls": {
            "visible": True,
            "opacity": 0.5,
            "rendering": "mip",
            "colormap": "gray",
            "gamma": 1.0,
            "contrast_limits": [0.0, 1.0],
        },
    }


def _make_scene_frame(block: dict[str, Any]) -> NotifySceneFrame:
    payload = NotifyScenePayload(viewer={}, layers=(block,), metadata=None, policies=None)
    envelope = FrameEnvelope(
        type=NOTIFY_SCENE_TYPE,
        version=PROTO_VERSION,
        session="session-test",
        frame_id="scene-1",
        timestamp=0.0,
        seq=0,
        delta_token="token-0",
    )
    return NotifySceneFrame(envelope=envelope, payload=payload)


def _make_layers_frame(layer_id: str, changes: dict[str, Any]) -> NotifyLayersFrame:
    payload = NotifyLayersPayload(layer_id=layer_id, changes=changes)
    envelope = FrameEnvelope(
        type=NOTIFY_LAYERS_TYPE,
        version=PROTO_VERSION,
        session="session-test",
        frame_id="layers-1",
        timestamp=1.0,
        seq=1,
        delta_token="token-1",
    )
    return NotifyLayersFrame(envelope=envelope, payload=payload)


@pytest.fixture
def gui_call(qtbot):
    proxy = CallProxy()
    yield proxy
    proxy.deleteLater()


@pytest.fixture
def mirror_setup(gui_call):
    state = _make_control_state()
    ledger = ClientStateLedger(clock=lambda: 0.0)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    registry = RemoteLayerRegistry()
    presenter = PresenterStub()
    viewer = ViewerStub()
    dispatch_calls: list[tuple[Any, str]] = []

    def _dispatch(pending: Any, origin: str) -> bool:
        dispatch_calls.append((pending, origin))
        return True

    emitter = NapariLayerIntentEmitter(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        dispatch_state_update=_dispatch,
        ui_call=gui_call,
        log_layers_info=False,
        tx_interval_ms=0,
    )
    mirror = NapariLayerMirror(
        ledger=ledger,
        state=state,
        loop_state=loop_state,
        registry=registry,
        presenter=presenter,
        viewer_ref=lambda: viewer,
        ui_call=gui_call,
        log_layers_info=False,
    )
    mirror.attach_emitter(emitter)
    return mirror, emitter, registry, presenter, viewer, ledger, dispatch_calls


def test_ingest_scene_snapshot_primes_layers(mirror_setup) -> None:
    mirror, emitter, registry, presenter, viewer, ledger, dispatch_calls = mirror_setup
    block = _layer_block()
    mirror.ingest_scene_snapshot(_make_scene_frame(block))

    snapshot = registry.snapshot()
    assert snapshot.layers[0].layer.remote_id == "layer-1"
    assert viewer.layers[0].remote_id == "layer-1"
    value = ledger.confirmed_value("layer", "layer-1", "opacity")
    assert value == pytest.approx(0.5)
    assert dispatch_calls == []

    mirror.replay_last_payload()
    assert viewer.layers[0].remote_id == "layer-1"
    assert presenter.layer_updates == []


def test_ingest_layer_delta_updates_layer(mirror_setup) -> None:
    mirror, emitter, registry, presenter, viewer, ledger, dispatch_calls = mirror_setup
    block = _layer_block()
    mirror.ingest_scene_snapshot(_make_scene_frame(block))

    mirror.ingest_layer_delta(_make_layers_frame("layer-1", {"opacity": 0.8}))

    layer = registry.snapshot().layers[0].layer
    assert layer.opacity == pytest.approx(0.8)
    confirmed = ledger.confirmed_value("layer", "layer-1", "opacity")
    assert confirmed == pytest.approx(0.8)
    assert presenter.layer_updates[-1].layer_id == "layer-1"


def test_mirror_asserts_on_non_gui_thread(mirror_setup) -> None:
    mirror, *_ = mirror_setup
    frame = _make_scene_frame(_layer_block())
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            mirror.ingest_scene_snapshot(frame)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors
    assert isinstance(errors[0], AssertionError)
