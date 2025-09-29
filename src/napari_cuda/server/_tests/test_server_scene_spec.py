from __future__ import annotations

import threading
from typing import Any

import pytest

from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import ServerSceneData
from napari_cuda.server.server_state_updates import (
    apply_dims_state_update,
    apply_layer_state_update,
)
from napari_cuda.server.server_scene_spec import (
    build_scene_spec_json,
    build_scene_spec_message,
    build_state_update_payload,
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


def test_build_scene_spec_message_caches(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    message = build_scene_spec_message(scene, manager)
    assert scene.last_scene_spec is not None
    assert scene.last_scene_spec_json is not None
    assert message.to_json() == scene.last_scene_spec_json


def test_build_scene_spec_json_round_trip(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    json_payload = build_scene_spec_json(scene, manager)
    assert json_payload == scene.last_scene_spec_json


def test_build_state_update_payload_layer(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
    lock = threading.RLock()
    result = apply_layer_state_update(
        scene,
        lock,
        layer_id="layer-0",
        prop="opacity",
        value=0.5,
        client_id="client-a",
        client_seq=4,
        interaction_id="drag-1",
        phase="update",
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

    payload = build_state_update_payload(scene, manager, result=result)
    assert payload["type"] == "state.update"
    assert payload["scope"] == "layer"
    assert payload["target"] == "layer-0"
    assert payload["value"] == 0.5
    assert payload["server_seq"] == result.server_seq
    assert payload["client_id"] == "client-a"
    versions = payload.get("control_versions") or {}
    assert versions.get("opacity", {}).get("server_seq") == result.server_seq
    assert payload["controls"]["opacity"] == 0.5
    assert payload.get("interaction_id") == "drag-1"
    assert payload.get("phase") == "update"


def test_build_state_update_payload_dims(scene: ServerSceneData, manager: ViewerSceneManager) -> None:
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
        client_id="client-z",
        client_seq=3,
        interaction_id="dims-1",
        phase="commit",
    )
    assert result is not None

    payload = build_state_update_payload(scene, manager, result=result)
    assert payload["type"] == "state.update"
    assert payload["scope"] == "dims"
    assert payload["target"] == "z"
    assert payload["value"] == 5
    assert payload["server_seq"] == result.server_seq
    assert payload["last_client_id"] == "client-z"
    assert payload.get("interaction_id") == "dims-1"
    assert payload.get("phase") == "commit"
    versions = payload.get("control_versions") or {}
    assert versions.get("step", {}).get("server_seq") == result.server_seq
    assert isinstance(payload.get("meta"), dict)
    assert payload["meta"].get("order") == meta["order"]


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

    spec = manager.scene_spec()
    assert spec is not None
    layer = spec.layers[0]
    assert layer.extras is not None
    assert "opacity" not in layer.extras
    assert layer.controls is not None
    assert layer.controls.get("opacity") == 0.42
