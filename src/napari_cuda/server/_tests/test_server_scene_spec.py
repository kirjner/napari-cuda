from __future__ import annotations

import threading
from typing import Any

import pytest

from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import ServerSceneData
from napari_cuda.server.control.state_update_engine import (
    apply_dims_state_update,
    apply_layer_state_update,
)
from napari_cuda.server.control.scene_snapshot_builder import (
    build_layer_controls_payload,
    build_notify_dims_from_result,
    build_notify_layers_delta_payload,
    build_notify_scene_payload,
)
from napari_cuda.protocol import (
    EnvelopeParser,
    NOTIFY_SCENE_TYPE,
    ResumableTopicSequencer,
    build_notify_scene_snapshot,
)


@pytest.fixture
def scene() -> ServerSceneData:
    return ServerSceneData()


@pytest.fixture
def manager() -> ViewerSceneManager:
    mgr = ViewerSceneManager((640, 480))
    mgr.update_from_sources(
        worker=None,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
    )
    return mgr


def test_build_notify_scene_payload_round_trip(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    viewer_settings = {"fps_target": 60.0, "canvas_size": [640, 480]}
    payload = build_notify_scene_payload(
        scene,
        manager,
        viewer_settings=viewer_settings,
    )
    cached = scene.last_scene_snapshot
    assert cached is not None
    assert cached == payload.to_dict()

    sequencer = ResumableTopicSequencer(topic=NOTIFY_SCENE_TYPE)
    frame = build_notify_scene_snapshot(
        session_id="sess-1234",
        viewer=payload.viewer,
        layers=payload.layers,
        policies=payload.policies,
        ancillary=payload.ancillary,
        timestamp=0.0,
        sequencer=sequencer,
    )

    parser = EnvelopeParser()
    parsed = parser.parse_notify_scene(frame.to_dict())

    assert parsed.envelope.seq == 0
    assert parsed.envelope.delta_token is not None
    viewer_block = parsed.payload.viewer
    assert viewer_block["settings"]["fps_target"] == 60.0
    dims_block = viewer_block["dims"]
    assert int(dims_block.get("ndisplay", 0)) == 2
    assert len(parsed.payload.layers) == 1


def test_build_notify_layers_delta_payload(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    lock = threading.RLock()
    result = apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.5,
    )
    assert result is not None

    manager.update_from_sources(
        worker=None,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        layer_controls=scene.layer_controls,
    )

    payload = build_notify_layers_delta_payload(result)
    assert payload.layer_id == "layer-0"
    assert payload.changes == {"opacity": 0.5}

    baseline = build_layer_controls_payload("layer-0", scene)
    assert baseline is not None
    assert baseline.layer_id == "layer-0"
    assert baseline.changes.get("opacity") == 0.5


def test_build_notify_dims_from_result(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    lock = threading.RLock()
    meta: dict[str, Any] = {
        "ndim": 3,
        "order": ["z", "y", "x"],
        "range": [[0, 9], [0, 9], [0, 9]],
    }
    result = apply_dims_state_update(
        scene,
        lock,
        meta,
        axis="z",
        prop="step",
        value=5,
    )
    assert result is not None

    payload = build_notify_dims_from_result(
        result,
        ndisplay=2,
        mode="volume",
        source="server",
    )
    assert payload.current_step == (5, 0, 0)
    assert payload.ndisplay == 2
    assert payload.mode == "volume"
    assert payload.source == "server"


def test_viewer_scene_manager_prefers_control_state(scene: ServerSceneData) -> None:
    from napari_cuda.server.server_scene import LayerControlState

    manager = ViewerSceneManager((640, 480))
    scene.layer_controls["layer-0"] = LayerControlState(opacity=0.42)

    manager.update_from_sources(
        worker=None,
        scene_state=None,
        multiscale_state=None,
        volume_state=None,
        current_step=None,
        ndisplay=2,
        zarr_path=None,
        extras={"opacity": 1.0},
        layer_controls=scene.layer_controls,
    )

    snapshot = manager.scene_snapshot()
    assert snapshot is not None
    layer_block = snapshot.layers[0].block
    assert "opacity" not in layer_block["extras"]
    assert layer_block["controls"]["opacity"] == 0.42
