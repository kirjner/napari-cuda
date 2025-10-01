from dataclasses import replace
import os

import numpy as np

import pytest

try:
    from napari_cuda.client.layers import LayerRecord, RegistrySnapshot, RemotePreview
    from napari_cuda.client.layers.remote_data import RemoteArray
    from napari_cuda.client.layers.registry import RemoteLayerRegistry
    from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
    from napari_cuda.protocol.messages import (
        LayerRenderHints,
        LayerRemoveMessage,
        LayerSpec,
        LayerUpdateMessage,
        NotifyScenePayload,
    )
    NAPARI_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent import guard
    NAPARI_AVAILABLE = False
    NAPARI_IMPORT_ERROR = str(exc)
    pytestmark = pytest.mark.skip(reason=f"napari unavailable: {NAPARI_IMPORT_ERROR}")
    LayerRecord = RegistrySnapshot = RemoteArray = RemoteLayerRegistry = RemoteImageLayer = RemotePreview = object  # type: ignore[assignment]
    LayerRenderHints = LayerRemoveMessage = LayerSpec = LayerUpdateMessage = NotifyScenePayload = object  # type: ignore[assignment]


def make_layer_spec(**overrides) -> LayerSpec:
    base = dict(
        layer_id="layer-1",
        layer_type="image",
        name="remote",
        ndim=3,
        shape=[16, 32, 48],
        dtype="float32",
        axis_labels=["z", "y", "x"],
        scale=[1.0, 2.0, 2.0],
        translate=[0.0, 0.0, 0.0],
        contrast_limits=[0.0, 1.0],
        metadata={"source": "test"},
        render=LayerRenderHints(mode="mip", opacity=0.75, visibility=True, colormap="gray"),
        extras={"data_id": "abc", "cache_version": 1},
    )
    base.update(overrides)
    return LayerSpec(**base)


def test_remote_array_protocol():
    arr = RemoteArray(shape=[100, 200, 3], dtype="uint16", data_id="abc")
    assert arr.ndim == 3
    assert arr.shape == (100, 200, 3)
    assert arr.size == 100 * 200 * 3
    preview = arr.__array__()
    assert preview.shape == (8, 8, 3)
    assert preview.dtype == arr.dtype
    slice_preview = arr[10, slice(20, 40), slice(None, None, 2)]
    assert slice_preview.shape == (8, 2)
    arr.update(shape=[50, 60], dtype="float32", cache_version=2)
    assert arr.shape == (50, 60)
    assert arr.dtype == arr.__array__().dtype
    assert arr.cache_version == 2


def test_remote_image_layer_applies_spec():
    spec = make_layer_spec()
    layer = RemoteImageLayer(spec)
    assert layer.remote_id == spec.layer_id
    assert layer.name == spec.name
    assert layer.axis_labels == tuple(spec.axis_labels)
    assert tuple(map(float, layer.contrast_limits)) == tuple(spec.contrast_limits)
    assert layer.opacity == pytest.approx(0.75)
    assert layer.visible is True
    updated = make_layer_spec(
        name="remote-updated",
        contrast_limits=[-1.0, 2.0],
        render=LayerRenderHints(opacity=0.5, visibility=False, gamma=1.2),
    )
    layer.update_from_spec(updated)
    assert layer.name == "remote-updated"
    assert tuple(map(float, layer.contrast_limits)) == (-1.0, 2.0)
    assert layer.opacity == pytest.approx(0.5)
    assert layer.visible is False
    assert layer.metadata.get("source") == "test"
    assert "thumbnail" not in layer.metadata


def test_remote_preview_handles_none_and_arrays():
    preview = RemotePreview()
    thumb_rgb = preview.as_thumbnail(True, (16, 16))
    assert thumb_rgb.shape == (16, 16, 3)
    assert np.allclose(thumb_rgb, 0.0)
    data = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    preview.update(data)
    thumb_scalar = preview.as_thumbnail(False, (8, 8))
    assert thumb_scalar.shape == (8, 8)
    assert np.isclose(thumb_scalar[0, 0], 0.0)
    assert np.isclose(thumb_scalar.max(), 1.0)

    line = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    preview.update(line)
    thumb_line = preview.as_thumbnail(False, (4, 16))
    assert thumb_line.shape == (4, 16)
    expected = np.zeros((4, 16), dtype=np.float32)
    expected[0, :16] = line[:16]
    np.testing.assert_allclose(thumb_line, expected, atol=1e-6)


def test_remote_image_layer_uses_metadata_thumbnail():
    preview = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(4, 4, 3)
    spec = make_layer_spec(shape=list(preview.shape), ndim=preview.ndim, metadata={"source": "test", "thumbnail": preview.tolist()})
    layer = RemoteImageLayer(spec)
    assert layer._remote_preview.data is not None
    np.testing.assert_allclose(layer._remote_preview.data, preview.astype(np.float32))
    assert "thumbnail" not in layer.metadata
    updated_preview = np.zeros_like(preview)
    updated = make_layer_spec(shape=list(preview.shape), ndim=preview.ndim, metadata={"source": "test", "thumbnail": updated_preview.tolist()})
    layer.update_from_spec(updated)
    assert layer._remote_preview.data is not None
    np.testing.assert_allclose(layer._remote_preview.data, updated_preview.astype(np.float32))
    assert "thumbnail" not in layer.metadata


def test_remote_layer_registry_lifecycle():
    registry = RemoteLayerRegistry()
    snapshots = []
    registry.add_listener(snapshots.append)
    spec = make_layer_spec()
    payload = NotifyScenePayload(viewer={"dims": {}, "camera": {}, "settings": {}}, layers=(spec.to_dict(),))
    registry.apply_scene(payload)
    assert snapshots
    first = snapshots[-1]
    assert first.ids() == (spec.layer_id,)
    record = first.layers[0]
    assert record.layer.remote_id == spec.layer_id
    update_spec = make_layer_spec(render=LayerRenderHints(opacity=0.25))
    registry.apply_update(LayerUpdateMessage(layer=update_spec, partial=True))
    updated_record = registry.snapshot().layers[0]
    assert updated_record.layer.opacity == pytest.approx(0.25)
    preview = np.ones((4, 4, 1), dtype=np.float32)
    new_spec = make_layer_spec(metadata={"source": "test", "added": True, "thumbnail": preview.tolist()})
    registry.apply_update(LayerUpdateMessage(layer=new_spec, partial=True))
    latest = registry.snapshot().layers[0]
    assert latest.layer.metadata["added"] is True
    assert latest.layer.metadata["source"] == "test"
    assert "thumbnail" not in latest.layer.metadata
    assert latest.layer._remote_preview.data is not None
    np.testing.assert_allclose(latest.layer._remote_preview.data, preview.astype(np.float32))
    registry.remove_layer(LayerRemoveMessage(layer_id=spec.layer_id))
    assert not registry.snapshot().layers


def test_proxy_viewer_sync_layers():
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from napari_cuda.client.proxy_viewer import ProxyViewer

    viewer = ProxyViewer(offline=True)
    spec = make_layer_spec()
    layer = RemoteImageLayer(spec)
    snapshot = RegistrySnapshot(layers=(LayerRecord(layer_id=spec.layer_id, spec=spec, layer=layer),))
    viewer._sync_remote_layers(snapshot)
    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    assert layer.visible is True
    empty_snapshot = RegistrySnapshot(layers=tuple())
    viewer._sync_remote_layers(empty_snapshot)
    assert len(viewer.layers) == 0


def test_proxy_viewer_applies_displayed_axes():
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from napari_cuda.client.proxy_viewer import ProxyViewer

    viewer = ProxyViewer(offline=True)
    viewer._apply_remote_dims_update(
        current_step=[0, 0, 0],
        ndisplay=3,
        ndim=3,
        dims_range=[(0.0, 3.0, 1.0), (0.0, 31.0, 1.0), (0.0, 47.0, 1.0)],
        order=[0, 1, 2],
        axis_labels=["z", "y", "x"],
        sizes=[4, 32, 48],
        displayed=[0, 1, 2],
    )
    assert viewer.dims.ndisplay == 3
    assert viewer.dims.displayed == (0, 1, 2)
    ranges = tuple(tuple(r) for r in viewer.dims.range)
    assert ranges[0] == (0.0, 3.0, 1.0)
    assert viewer.dims.ndim == 3
    assert tuple(float(x) for x in viewer.dims.point) == (0.0, 0.0, 0.0)
    assert viewer._suppress_forward is False


def test_remote_image_layer_updates_thumbnail_from_preview():
    spec = make_layer_spec(shape=[4, 4, 3], ndim=3, render=LayerRenderHints(opacity=1.0, visibility=True))
    layer = RemoteImageLayer(spec)
    preview = np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3)
    layer.update_preview(preview)
    layer._update_thumbnail()
    assert layer.loaded is True
    base = layer._remote_preview.as_thumbnail(layer.rgb, layer._thumbnail_shape[:2])
    if layer.rgb:
        expected_rgba = np.concatenate([base, np.ones(base.shape[:2] + (1,), dtype=base.dtype)], axis=2)
    else:
        low, high = layer.contrast_limits
        downsampled = np.clip(base, low, high)
        color_range = high - low
        if color_range != 0:
            downsampled = (downsampled - low) / color_range
        downsampled = downsampled ** layer.gamma
        color_array = layer.colormap.map(downsampled.ravel())
        expected_rgba = color_array.reshape((*downsampled.shape, 4))
        expected_rgba[..., 3] *= layer.opacity
    thumb = layer.thumbnail
    region = thumb[: expected_rgba.shape[0], : expected_rgba.shape[1], :]
    if np.issubdtype(region.dtype, np.integer):
        region_float = region.astype(np.float32) / 255.0
        np.testing.assert_allclose(region_float, expected_rgba, atol=1.0 / 255 + 1e-6)
    else:
        np.testing.assert_array_almost_equal(region, expected_rgba)


def test_remote_image_layer_gamma_handles_1d_preview():
    spec = make_layer_spec()
    layer = RemoteImageLayer(spec)
    preview = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    layer.update_preview(preview)

    layer._slice_input = replace(layer._slice_input, ndisplay=3)
    layer.gamma = 0.5

    assert layer.thumbnail.shape == layer._thumbnail_shape
