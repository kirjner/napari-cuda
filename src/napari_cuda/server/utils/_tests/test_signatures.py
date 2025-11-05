from __future__ import annotations

from dataclasses import replace

from napari_cuda.server.scene import LayerVisualState, RenderLedgerSnapshot
from napari_cuda.server.utils.signatures import (
    SignatureToken,
    dims_content_signature,
    layer_content_signature,
    layer_inputs_signature,
    scene_content_signature,
    snapshot_versions,
)
from napari_cuda.shared.axis_spec import (
    derive_axis_labels,
    derive_margins,
    fabricate_axis_spec,
)


def _sample_snapshot() -> RenderLedgerSnapshot:
    spec = fabricate_axis_spec(
        ndim=3,
        ndisplay=2,
        current_level=0,
        level_shapes=[(64, 32, 16)],
        order=(0, 1, 2),
        displayed=(1, 2),
        labels=("z", "y", "x"),
        current_step=(1, 0, 0),
    )
    margins_left, margins_right = derive_margins(spec, prefer_world=True)
    return RenderLedgerSnapshot(
        ndisplay=spec.ndisplay,
        current_step=(1, 0, 0),
        current_level=spec.current_level,
        dims_mode="plane",
        axis_labels=tuple(derive_axis_labels(spec)),
        order=spec.order,
        displayed=spec.displayed,
        level_shapes=spec.level_shapes,
        margin_left=margins_left,
        margin_right=margins_right,
        axes=spec,
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
    axis_spec = fabricate_axis_spec(
        ndim=2,
        ndisplay=2,
        current_level=1,
        level_shapes=[(100, 100), (50, 50)],
        order=(0, 1),
        displayed=None,
        labels=("z", "y"),
    )
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
        downgraded=False,
        axis_spec=axis_spec,
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
        downgraded=False,
        axis_spec=axis_spec,
    )
    assert token_a.value == token_b.value


def test_snapshot_versions_apply_updates_mapping() -> None:
    snapshot = replace(
        _sample_snapshot(),
        dims_version=3,
        view_version=4,
        multiscale_level_version=5,
        camera_versions={"plane.zoom": 6, "volume.fov": 7},
    )
    gate = snapshot_versions(snapshot)
    cache: dict[tuple[str, str, str], int] = {}
    gate.apply(cache)
    assert cache[("dims", "main", "current_step")] == 3
    assert cache[("view", "main", "ndisplay")] == 4
    assert cache[("multiscale", "main", "level")] == 5
    assert cache[("camera_plane", "main", "zoom")] == 6
    assert cache[("camera_volume", "main", "fov")] == 7
