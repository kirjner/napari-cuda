from __future__ import annotations

from typing import Optional

import numpy as np

from napari_cuda.server.control import state_reducers as reducers
from napari_cuda.server.scene import snapshot_render_state, snapshot_scene
from napari_cuda.server.scene import blocks as scene_blocks
import napari_cuda.server.scene.builders as scene_builders
from napari_cuda.server.scene.blocks import camera_block_from_payload
from napari_cuda.server.ledger import ServerStateLedger


def _seed_plane_ledger(ndisplay: int = 2) -> ServerStateLedger:
    ledger = ServerStateLedger()
    reducers.reduce_bootstrap_state(
        ledger,
        step=(0, 0, 0),
        axis_labels=("z", "y", "x"),
        order=(0, 1, 2),
        level_shapes=((10, 20, 30),),
        levels=({"index": 0, "shape": [10, 20, 30]},),
        current_level=0,
        ndisplay=ndisplay,
        origin="test.bootstrap",
    )
    reducers.reduce_camera_update(
        ledger,
        center=(5.0, 10.0, 0.0) if ndisplay >= 3 else (5.0, 10.0),
        zoom=1.0,
        rect=(0.0, 0.0, 20.0, 30.0),
        origin="test.camera",
        bump_op_seq=False,
    )
    ledger.record_confirmed(
        "viewport",
        "active",
        "state",
        {"mode": "volume" if ndisplay >= 3 else "plane", "level": 0},
        origin="test.bootstrap",
    )
    ledger.record_confirmed("layer", "layer-0", "visible", True, origin="test.plane")
    return ledger


reducers.ENABLE_VIEW_AXES_INDEX_BLOCKS = True
scene_blocks.ENABLE_VIEW_AXES_INDEX_BLOCKS = True
scene_builders.ENABLE_VIEW_AXES_INDEX_BLOCKS = True


def test_snapshot_scene_plane() -> None:
    ledger = _seed_plane_ledger()
    render_state = snapshot_render_state(ledger)

    scene = snapshot_scene(
        render_state=render_state,
        ledger_snapshot=ledger.snapshot(),
        canvas_size=(640, 480),
        fps_target=60.0,
    )

    dims_block = scene.viewer.dims
    assert dims_block["ndisplay"] == 2
    assert dims_block["current_step"] == [0, 0, 0]
    assert dims_block["axis_labels"] == ["z", "y", "x"]

    camera_block = scene.viewer.camera
    assert camera_block["ndisplay"] == 2
    assert camera_block["center"] == [5.0, 10.0]
    assert camera_block["rect"] == [0.0, 0.0, 20.0, 30.0]

    layer_block = scene.layers[0].block
    assert layer_block["volume"] is False
    assert layer_block["controls"]["visible"] is True


def test_snapshot_scene_volume_camera() -> None:
    ledger = _seed_plane_ledger(ndisplay=3)
    reducers.reduce_camera_update(
        ledger,
        center=(15.0, 25.0, 35.0),
        angles=(45.0, 30.0, 0.0),
        distance=100.0,
        fov=60.0,
        origin="test.volume",
        bump_op_seq=False,
    )
    ledger.record_confirmed("volume", "main", "rendering", "mip", origin="test.volume")

    render_state = snapshot_render_state(ledger)

    scene = snapshot_scene(
        render_state=render_state,
        ledger_snapshot=ledger.snapshot(),
        canvas_size=(800, 600),
        fps_target=30.0,
    )

    camera_block = scene.viewer.camera
    assert camera_block["ndisplay"] == 3
    assert camera_block["center"] == [15.0, 25.0, 35.0]
    assert camera_block["angles"] == [45.0, 30.0, 0.0]
    assert camera_block["distance"] == 100.0
    assert camera_block["fov"] == 60.0

    settings = scene.viewer.settings
    assert settings["volume_enabled"] is True
    assert settings["fps_target"] == 30.0

    layer_block = scene.layers[0].block
    assert layer_block["controls"]["visible"] is True


def test_snapshot_scene_thumbnail_provider() -> None:
    ledger = _seed_plane_ledger()
    render_state = snapshot_render_state(ledger)
    thumbnail = np.ones((4, 4), dtype=np.uint8)

    def provider(_layer_id: str) -> Optional[np.ndarray]:
        return thumbnail

    scene = snapshot_scene(
        render_state=render_state,
        ledger_snapshot=ledger.snapshot(),
        canvas_size=(320, 240),
        fps_target=30.0,
        thumbnail_provider=provider,
    )

    layer_metadata = scene.layers[0].block.get("metadata")
    assert layer_metadata is not None
    expected_thumbnail = np.stack([thumbnail, thumbnail, thumbnail], axis=-1).tolist()
    assert layer_metadata.get("thumbnail") == expected_thumbnail


def test_snapshot_scene_without_default_layer() -> None:
    ledger = _seed_plane_ledger()
    ledger.clear_scope("layer")
    render_state = snapshot_render_state(ledger)

    scene = snapshot_scene(
        render_state=render_state,
        ledger_snapshot=ledger.snapshot(),
        canvas_size=(320, 240),
        fps_target=30.0,
        default_layer_id=None,
    )

    assert scene.layers == ()
    assert scene.metadata["status"] == "idle"


def test_render_snapshot_includes_block_snapshot_with_restore_caches() -> None:
    ledger = _seed_plane_ledger()
    plane_cache = reducers.load_plane_restore_cache(ledger)
    volume_cache = reducers.load_volume_restore_cache(ledger)
    render_state = snapshot_render_state(ledger)

    blocks = render_state.block_snapshot
    assert blocks is not None
    assert blocks.view.mode == "plane"
    assert blocks.index is not None
    assert blocks.plane_restore == plane_cache
    assert blocks.volume_restore == volume_cache
