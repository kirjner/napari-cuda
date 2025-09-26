from __future__ import annotations

from typing import Any

import pytest

from napari_cuda.server.layer_manager import ViewerSceneManager
from napari_cuda.server.server_scene import ServerSceneData
from napari_cuda.server.server_scene_spec import (
    build_scene_spec_json,
    build_scene_spec_message,
    build_dims_payload,
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


class _StubSceneSource:
    def __init__(self) -> None:
        self.current_level = 0
        self.level_descriptors = [
            type("Desc", (), {"shape": (32, 32), "downsample": (1, 1), "path": "level_0"})
        ]


@pytest.mark.parametrize("step", ([0, 0], [3], [7, 1, 2]))
def test_build_dims_payload(scene: ServerSceneData, step: list[int]) -> None:
    meta = {"ndim": max(1, len(step)), "order": ["z", "y", "x"], "volume": False}
    payload = build_dims_payload(
        scene,
        step_list=step,
        last_client_id="client-1",
        meta=meta,
        worker_scene_source=_StubSceneSource(),
        use_volume=False,
    )
    assert payload["type"] == "dims.update"
    assert payload["last_client_id"] == "client-1"
    assert "meta" in payload
    # incr dims_seq wrap handled inside helper; ensure cached payload matches
    assert scene.last_dims_payload is payload


@pytest.mark.parametrize("bad_step", ([], [1, 2, 3, 4]))
def test_build_dims_payload_invalid(scene: ServerSceneData, bad_step: list[int]) -> None:
    meta = {"ndim": 2, "order": ["y", "x"], "volume": False}
    with pytest.raises(AssertionError):
        build_dims_payload(
            scene,
            step_list=bad_step,
            last_client_id=None,
            meta=meta,
            worker_scene_source=None,
            use_volume=False,
        )
