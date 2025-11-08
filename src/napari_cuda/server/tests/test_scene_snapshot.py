from __future__ import annotations

from typing import Optional

import numpy as np

from napari_cuda.server.scene import snapshot_render_state, snapshot_scene
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import (
    AxisExtent,
    DimsSpec,
    DimsSpecAxis,
    dims_spec_to_payload,
)


def _seed_plane_ledger(ndisplay: int = 2) -> ServerStateLedger:
    ledger = ServerStateLedger()
    ledger.record_confirmed("dims", "main", "current_step", (0, 0, 0), origin="test.plane")
    axes = (
        DimsSpecAxis(
            index=0,
            label="z",
            role="z",
            displayed=ndisplay >= 3,
            order_position=0,
            current_step=0,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(10,),
            per_level_world=(AxisExtent(start=0.0, stop=9.0, step=1.0),),
        ),
        DimsSpecAxis(
            index=1,
            label="y",
            role="y",
            displayed=True,
            order_position=1,
            current_step=0,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(20,),
            per_level_world=(AxisExtent(start=0.0, stop=19.0, step=1.0),),
        ),
        DimsSpecAxis(
            index=2,
            label="x",
            role="x",
            displayed=True,
            order_position=2,
            current_step=0,
            margin_left_steps=0.0,
            margin_right_steps=0.0,
            margin_left_world=0.0,
            margin_right_world=0.0,
            per_level_steps=(30,),
            per_level_world=(AxisExtent(start=0.0, stop=29.0, step=1.0),),
        ),
    )
    displayed_axes: tuple[int, ...] = (1, 2) if ndisplay == 2 else (0, 1, 2)
    spec = DimsSpec(
        version=1,
        ndim=3,
        ndisplay=ndisplay,
        order=(0, 1, 2),
        displayed=displayed_axes,
        current_level=0,
        current_step=(0, 0, 0),
        level_shapes=((10, 20, 30),),
        plane_mode=ndisplay < 3,
        axes=axes,
        levels=(
            {"index": 0, "shape": [10, 20, 30], "downsample": [1.0, 1.0, 1.0]},
        ),
        labels=("z", "y", "x"),
    )
    ledger.record_confirmed("dims", "main", "dims_spec", dims_spec_to_payload(spec), origin="test.plane")
    ledger.record_confirmed(
        "camera_plane",
        "main",
        "center",
        (5.0, 10.0, 0.0),
        origin="test.plane",
    )
    ledger.record_confirmed("camera_plane", "main", "zoom", 1.0, origin="test.plane")
    ledger.record_confirmed(
        "camera_plane",
        "main",
        "rect",
        (0.0, 0.0, 20.0, 30.0),
        origin="test.plane",
    )
    ledger.record_confirmed("layer", "layer-0", "visible", True, origin="test.plane")
    return ledger


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
    assert camera_block["center"] == [5.0, 10.0, 0.0]
    assert camera_block["rect"] == [0.0, 0.0, 20.0, 30.0]

    layer_block = scene.layers[0].block
    assert layer_block["volume"] is False
    assert layer_block["multiscale"]["levels"][0]["shape"] == [10, 20, 30]


def test_snapshot_scene_volume_camera() -> None:
    ledger = _seed_plane_ledger(ndisplay=3)
    ledger.record_confirmed(
        "camera_volume",
        "main",
        "center",
        (15.0, 25.0, 35.0),
        origin="test.volume",
    )
    ledger.record_confirmed(
        "camera_volume",
        "main",
        "angles",
        (45.0, 30.0, 0.0),
        origin="test.volume",
    )
    ledger.record_confirmed("camera_volume", "main", "distance", 100.0, origin="test.volume")
    ledger.record_confirmed("camera_volume", "main", "fov", 60.0, origin="test.volume")
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
