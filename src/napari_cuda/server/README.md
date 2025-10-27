# napari-cuda Server Notes

## Scene / Layer Synchronisation Handshake (Phase 0)

The server now advertises its authoritative scene through `notify.scene`
snapshots. When a state websocket client connects:

1. `EGLHeadlessServer` builds a `SceneSnapshot` via `snapshot_scene`. The
   snapshot contains JSON-safe layer blocks plus viewer metadata (dims/camera).
2. The server sends `notify.scene(seq=0)` with that payload. Each layer block
   carries shape, dtype, axis labels, multiscale levels, a `source` section, and
   a `controls` map with the canonical intent state (opacity, visible, etc.).
3. Subsequent changes reuse the same helpers:
   - bulk refresh → `notify.scene`
   - per-layer updates → `notify.layers`
   - dims metadata → `notify.dims`

`snapshot_dims_metadata()` still underpins the HUD, and the latest dims
snapshot is pulled directly from the server ledger via `ServerDimsMirror`.

A typical `notify.scene` payload looks like:

```json
{
  "viewer": {
    "settings": {"fps_target": 60.0, "canvas_size": [640, 480]},
    "dims": {
      "ndim": 3,
      "axis_labels": ["z", "y", "x"],
      "order": ["z", "y", "x"],
      "sizes": [20, 40, 60],
      "range": [[0, 19], [0, 39], [0, 59]],
      "displayed": [1, 2],
      "ndisplay": 3
    },
    "camera": {"center": [10.0, 20.0, 30.0], "zoom": 2.5, "ndisplay": 3}
  },
  "layers": [
    {
      "layer_id": "layer-0",
      "layer_type": "image",
      "name": "napari-cuda",
      "ndim": 3,
      "shape": [20, 40, 60],
      "dtype": "float32",
      "axis_labels": ["z", "y", "x"],
      "render": {"mode": "mip", "opacity": 0.8},
      "multiscale": {
        "levels": [
          {"shape": [20, 40, 60], "downsample": [1.0, 1.0, 1.0], "path": "level_0"},
          {"shape": [10, 20, 30], "downsample": [1.0, 2.0, 2.0], "path": "level_1"}
        ],
        "current_level": 0,
        "metadata": {"policy": "latency", "index_space": "base"}
      },
      "volume": true,
      "scale": [1.0, 1.0, 1.0],
      "source": {"kind": "ome-zarr"},
      "controls": {"visible": true, "opacity": 0.8, "contrast_limits": [0.0, 1.0]}
    }
  ],
  "policies": {"multiscale": {"policy": "latency", "active_level": 0}},
  "metadata": {"source_path": "/data/sample.zarr"}
}
```

Clients subscribe via `StateChannel`, mirror the layer list into the
`ProxyViewer`, and rely on follow-up `notify.layers`/`notify.dims` deltas to stay
in sync.
