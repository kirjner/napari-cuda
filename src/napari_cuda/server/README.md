# napari-cuda Server Notes

## Scene / Layer Synchronisation Handshake (Phase 0)

The server now advertises its authoritative scene using the `scene.spec` payload
defined in `napari_cuda.protocol.messages`. When a state websocket client
connects:

1. `EGLHeadlessServer` refreshes the `ViewerSceneManager`, composing a
   `SceneSpec` from the active renderer (`LayerSpec`, `DimsSpec`, `CameraSpec`).
2. The server sends a JSON message with `type: "scene.spec"`, `version: 1`, and
   `scene` containing:
   - `layers`: list of layer descriptors (id, type, shape, dtype, render hints,
     multiscale description, extras such as `zarr_path`)
   - `dims`: ndim/order/sizes/ranges/current step/ndisplay
   - `camera`: optional snapshot of the viewer camera
   - `capabilities`: currently `state.update`, `layer.remove`
3. Subsequent changes (e.g. dataset swap, level switch) resend
   `scene.spec`, while fine‑grained updates flow through `state.update` or
   `layer.remove` as those envelopes are emitted.

`ViewerSceneManager.dims_metadata()` now underpins `dims.update` payloads, so the
legacy HUD continues to work alongside the richer scene spec.

```json
{
  "type": "scene.spec",
  "version": 1,
  "scene": {
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
        "extras": {"is_volume": true, "zarr_path": "/data/sample.zarr"}
      }
    ],
    "dims": {
      "ndim": 3,
      "axis_labels": ["z", "y", "x"],
      "order": ["z", "y", "x"],
      "sizes": [20, 40, 60],
      "range": [[0, 19], [0, 39], [0, 59]],
      "current_step": [0, 0, 0],
      "displayed": [1, 2],
      "ndisplay": 3
    },
    "camera": {
      "center": [10.0, 20.0, 30.0],
      "zoom": 2.5,
      "ndisplay": 3
    },
    "capabilities": ["state.update", "layer.remove"]
  }
}
```

Client code should subscribe to these messages via the `StateChannel` and keep
its `ProxyViewer` layer list in sync.
