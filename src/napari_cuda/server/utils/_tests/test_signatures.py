from __future__ import annotations

from dataclasses import replace

from napari_cuda.server.scene import LayerVisualState, RenderLedgerSnapshot
from napari_cuda.server.scene.blocks import (
    AxisBlock,
    AxisExtentBlock,
    AxesBlock,
    CameraBlock,
    IndexBlock,
    LodBlock,
    PlaneCameraBlock,
    ViewBlock,
    VolumeCameraBlock,
)
from napari_cuda.shared.dims_spec import AxisExtent, DimsSpec, DimsSpecAxis
from napari_cuda.server.utils.signatures import (
    SignatureToken,
    axes_block_signature,
    camera_block_signature,
    dims_content_signature,
    index_block_signature,
    layer_content_signature,
    layer_inputs_signature,
    layers_block_signature,
    lod_block_signature,
    scene_content_signature,
    snapshot_versions,
    view_block_signature,
)


def _sample_axes_spec() -> DimsSpec:
    level_shapes = ((100, 100), (50, 50))
    axes = []
    for idx, label in enumerate(("z", "y")):
        per_steps = tuple(shape[idx] if idx < len(shape) else 1 for shape in level_shapes)
        per_world = tuple(AxisExtent(start=0.0, stop=float(max(0, step - 1)), step=1.0) for step in per_steps)
        axes.append(
            DimsSpecAxis(
                index=idx,
                label=label,
                role=label,
                displayed=idx in (0, 1),
                order_position=idx,
                current_step=1 if idx == 0 else 0,
                margin_left_steps=0.0,
                margin_right_steps=0.0,
                margin_left_world=0.0,
                margin_right_world=0.0,
                per_level_steps=per_steps,
                per_level_world=per_world,
            )
        )
    return DimsSpec(
        version=1,
        ndim=2,
        ndisplay=2,
        order=(0, 1),
        displayed=(0, 1),
        current_level=0,
        current_step=(1, 0),
        level_shapes=level_shapes,
        plane_mode=True,
        axes=tuple(axes),
        levels=tuple({"index": idx, "shape": list(shape)} for idx, shape in enumerate(level_shapes)),
        labels=None,
    )


def _sample_snapshot() -> RenderLedgerSnapshot:
    spec = _sample_axes_spec()
    return RenderLedgerSnapshot(
        ndisplay=spec.ndisplay,
        current_step=spec.current_step,
        current_level=spec.current_level,
        order=spec.order,
        displayed=spec.displayed,
        axis_labels=tuple(axis.label for axis in spec.axes),
        level_shapes=spec.level_shapes,
        dims_mode="plane",
        dims_spec=spec,
        layer_values={
            "layer-0": LayerVisualState(
                layer_id="layer-0",
                visible=True,
                opacity=0.5,
                extra={"volume.rendering": "mip"},
            )
        },
        plane_center=(1.0, 2.0),
        plane_zoom=1.25,
        plane_rect=(0.0, 0.0, 50.0, 60.0),
        volume_center=(10.0, 11.0, 12.0),
        volume_angles=(30.0, 15.0, 5.0),
        volume_distance=100.0,
        volume_fov=60.0,
        volume_mode="mip",
        volume_colormap="gray",
        volume_clim=(0.0, 1.0),
        volume_opacity=0.3,
        volume_sample_step=2.0,
    )


def test_scene_signature_reflects_layer_changes() -> None:
    snapshot = _sample_snapshot()
    base = scene_content_signature(snapshot)
    assert isinstance(base, SignatureToken)

    # Mutate layer opacity; expect signature change.
    layer_state = snapshot.layer_values["layer-0"]
    snapshot.layer_values["layer-0"] = layer_state.with_updates(updates={"opacity": 0.9})
    updated = scene_content_signature(snapshot)
    assert updated.changed(base)


def test_layer_inputs_signature_filters_volume_in_plane_mode() -> None:
    snapshot = _sample_snapshot()
    token = layer_inputs_signature(snapshot, "layer-0")
    visuals = dict(token.value)["visuals"]
    assert all(not key.startswith("volume.") for key, _ in visuals)

    # Switching to 3D should keep volume extras.
    snapshot_3d = replace(snapshot, ndisplay=3)
    token_3d = layer_inputs_signature(snapshot_3d, "layer-0")
    visuals_3d = dict(token_3d.value)["visuals"]
    assert any(key.startswith("volume.") for key, _ in visuals_3d)


def test_layer_content_signature_includes_metadata_values() -> None:
    state = LayerVisualState(
        layer_id="layer-1",
        visible=True,
        metadata={"foo": 1},
        thumbnail={"shape": (16, 16)},
    )
    token = layer_content_signature(state)
    keys = {key for key, _ in token.value}
    assert "metadata" in keys
    assert "thumbnail" in keys


def test_dims_content_signature_sorts_level_items() -> None:
    token_a = dims_content_signature(
        current_step=(0, 0),
        current_level=1,
        ndisplay=2,
        mode="plane",
        displayed=None,
        axis_labels=("z", "y"),
        order=(0, 1),
        labels=None,
        levels=(
            {"shape": (100, 100), "downsample": (1, 1)},
            {"shape": (50, 50), "downsample": (2, 2)},
        ),
        level_shapes=((100, 100), (50, 50)),
    )
    token_b = dims_content_signature(
        current_step=(0, 0),
        current_level=1,
        ndisplay=2,
        mode="plane",
        displayed=None,
        axis_labels=("z", "y"),
        order=(0, 1),
        labels=None,
        levels=({"downsample": (1, 1), "shape": (100, 100)}, {"downsample": (2, 2), "shape": (50, 50)}),
        level_shapes=((100, 100), (50, 50)),
    )
    assert token_a.value == token_b.value


def test_snapshot_versions_apply_updates_mapping() -> None:
    snapshot = RenderLedgerSnapshot(
        dims_version=3,
        view_version=4,
        multiscale_level_version=5,
        camera_versions={"plane.zoom": 6, "volume.fov": 7},
    )
    gate = snapshot_versions(snapshot)
    cache: dict[tuple[str, str, str], int] = {}
    gate.apply(cache)
    assert cache[("dims", "main", "current_step")] == 3
    assert cache[("dims", "main", "dims_spec")] == 4
    assert cache[("multiscale", "main", "level")] == 5
    assert cache[("camera_plane", "main", "zoom")] == 6
    assert cache[("camera_volume", "main", "fov")] == 7


def test_layers_block_signature_sorts_layer_ids() -> None:
    layers_a = {
        "layer-b": LayerVisualState(layer_id="layer-b", visible=True),
        "layer-a": LayerVisualState(layer_id="layer-a", opacity=0.5),
    }
    layers_b = {
        "layer-a": LayerVisualState(layer_id="layer-a", opacity=0.5),
        "layer-b": LayerVisualState(layer_id="layer-b", visible=True),
    }
    token_a = layers_block_signature(layers_a)
    token_b = layers_block_signature(layers_b)
    assert token_a.value == token_b.value


def test_view_block_signature_changes_when_axes_change() -> None:
    block = ViewBlock(mode="plane", displayed_axes=(0, 1), ndim=2)
    token = view_block_signature(block)
    updated = view_block_signature(ViewBlock(mode="volume", displayed_axes=(0, 1, 2), ndim=3))
    assert updated.changed(token)


def test_axes_block_signature_includes_extent_metadata() -> None:
    axes = AxesBlock(
        axes=(
            AxisBlock(
                axis_id=0,
                label="z",
                role="depth",
                displayed=True,
                world_extent=AxisExtentBlock(start=0.0, stop=10.0, step=1.0),
                margin_left_world=0.0,
                margin_right_world=0.0,
            ),
        )
    )
    token = axes_block_signature(axes)
    replay = axes_block_signature(axes)
    assert token.value == replay.value


def test_index_and_lod_block_signatures_cover_core_fields() -> None:
    index_token = index_block_signature(IndexBlock(value=(1, 2, 3)))
    new_index_token = index_block_signature(IndexBlock(value=(1, 2, 4)))
    assert new_index_token.changed(index_token)

    lod_token = lod_block_signature(LodBlock(level=1, roi=(1, 2), policy="roi"))
    no_roi_token = lod_block_signature(LodBlock(level=1, roi=None, policy="roi"))
    assert no_roi_token.changed(lod_token)


def test_camera_block_signature_changes_for_plane_pose() -> None:
    block = CameraBlock(
        plane=PlaneCameraBlock(rect=(0.0, 0.0, 5.0, 6.0), center=(1.0, 2.0), zoom=1.5),
        volume=VolumeCameraBlock(center=(0.0, 0.0, 0.0), angles=(0.0, 0.0, 0.0), distance=10.0, fov=45.0),
    )
    token = camera_block_signature(block)
    updated = camera_block_signature(
        CameraBlock(
            plane=PlaneCameraBlock(rect=(0.0, 0.0, 5.0, 6.0), center=(2.0, 3.0), zoom=1.5),
            volume=block.volume,
        )
    )
    assert updated.changed(token)
