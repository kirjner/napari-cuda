import os

import pytest

try:
    from napari_cuda.client.layers import LayerRecord, RegistrySnapshot
    from napari_cuda.client.layers.remote_data import RemoteArray
    from napari_cuda.client.layers.registry import RemoteLayerRegistry
    from napari_cuda.client.layers.remote_image_layer import RemoteImageLayer
    from napari_cuda.protocol.messages import (
        LayerRenderHints,
        LayerRemoveMessage,
        LayerSpec,
        LayerUpdateMessage,
        SceneSpec,
        SceneSpecMessage,
    )
    NAPARI_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent import guard
    NAPARI_AVAILABLE = False
    NAPARI_IMPORT_ERROR = str(exc)
    pytestmark = pytest.mark.skip(reason=f"napari unavailable: {NAPARI_IMPORT_ERROR}")
    LayerRecord = RegistrySnapshot = RemoteArray = RemoteLayerRegistry = RemoteImageLayer = object  # type: ignore[assignment]
    LayerRenderHints = LayerRemoveMessage = LayerSpec = LayerUpdateMessage = SceneSpec = SceneSpecMessage = object  # type: ignore[assignment]


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


def test_remote_layer_registry_lifecycle():
    registry = RemoteLayerRegistry()
    snapshots = []
    registry.add_listener(snapshots.append)
    spec = make_layer_spec()
    registry.apply_scene(SceneSpecMessage(scene=SceneSpec(layers=[spec])))
    assert snapshots
    first = snapshots[-1]
    assert first.ids() == (spec.layer_id,)
    record = first.layers[0]
    assert record.layer.remote_id == spec.layer_id
    update_spec = make_layer_spec(render=LayerRenderHints(opacity=0.25))
    registry.apply_update(LayerUpdateMessage(layer=update_spec, partial=True))
    updated_record = registry.snapshot().layers[0]
    assert updated_record.layer.opacity == pytest.approx(0.25)
    new_spec = make_layer_spec(metadata={"added": True})
    registry.apply_update(LayerUpdateMessage(layer=new_spec, partial=True))
    latest = registry.snapshot().layers[0]
    assert latest.layer.metadata["added"] is True
    assert latest.layer.metadata["source"] == "test"
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
    assert layer.visible is False
    empty_snapshot = RegistrySnapshot(layers=tuple())
    viewer._sync_remote_layers(empty_snapshot)
    assert len(viewer.layers) == 0
