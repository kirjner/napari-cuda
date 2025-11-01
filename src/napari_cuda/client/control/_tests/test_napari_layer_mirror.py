from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy import QtCore

from napari_cuda.client.control.client_state_ledger import ClientStateLedger
from napari_cuda.client.control.mirrors import NapariLayerMirror
from napari_cuda.client.control.state_update_actions import ControlStateContext
from napari_cuda.client.data.registry import RemoteLayerRegistry
from napari_cuda.client.data.remote_image_layer import RemoteImageLayer
from napari_cuda.client.runtime.client_loop.loop_state import ClientLoopState
from napari_cuda.protocol.snapshots import (
    LayerDelta,
    LayerSnapshot,
    SceneSnapshot,
    ViewerSnapshot,
)


class _StubPresenter:
    def __init__(self) -> None:
        self.deltas: list[LayerDelta] = []

    def apply_layer_delta(self, message: LayerDelta) -> None:
        self.deltas.append(message)


def test_layer_mirror_applies_registry_updates(qtbot) -> None:
    from napari_cuda.client.app.proxy_viewer import ProxyViewer

    viewer = ProxyViewer(offline=True)
    ctrl_env = SimpleNamespace(dims_rate_hz=60.0, wheel_step=1.0, settings_rate_hz=30.0)
    control_state = ControlStateContext.from_env(ctrl_env)
    loop_state = ClientLoopState()
    loop_state.gui_thread = QtCore.QThread.currentThread()
    ledger = ClientStateLedger()
    registry = RemoteLayerRegistry()
    presenter = _StubPresenter()

    mirror = NapariLayerMirror(
        ledger=ledger,
        state=control_state,
        loop_state=loop_state,
        registry=registry,
        presenter=presenter,  # type: ignore[arg-type]
        viewer_ref=lambda: viewer,
        ui_call=object(),
        log_layers_info=False,
    )

    block = {
        "layer_id": "layer-1",
        "layer_type": "image",
        "name": "remote",
        "ndim": 2,
        "shape": [16, 32],
        "dtype": "float32",
        "axis_labels": ["y", "x"],
        "controls": {"visible": True, "opacity": 0.5},
    }

    snapshot = SceneSnapshot(
        viewer=ViewerSnapshot(settings={}, dims={}, camera={}),
        layers=(LayerSnapshot(layer_id=block["layer_id"], block=dict(block)),),
        policies={},
        metadata={},
    )

    registry.apply_snapshot(snapshot)

    assert len(viewer.layers) == 1
    remote_layer = viewer.layers[0]
    assert isinstance(remote_layer, RemoteImageLayer)
    assert remote_layer.visible is True
    assert pytest.approx(remote_layer.opacity, rel=1e-6) == 0.5

    registry.apply_delta(LayerDelta(layer_id=block["layer_id"], changes={"opacity": 0.25}))
    assert pytest.approx(remote_layer.opacity, rel=1e-6) == 0.25

    registry.apply_delta(LayerDelta.removal(block["layer_id"]))
    assert len(viewer.layers) == 0
    viewer.close()
