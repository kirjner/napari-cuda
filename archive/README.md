This folder contains legacy or experimental modules that are not part of the active server path.

Archived modules:

- legacy_server/
  - cuda_streaming_layer.py
  - headless_server.py (Qt-based)
  - render_thread.py (Qt/CUDA thread)
- legacy_cuda/
  - pynv_encoder.py (standalone NVENC wrapper)
- legacy_client/
  - minimal_client.py (legacy control channel client used during early protocol bring-up)

Reason: The active server uses the asyncio EGL path in `src/napari_cuda/server/egl_headless_server.py`
with `EGLRendererWorker` and direct NVENC encode. Keeping a single encode path reduces confusion.

If you need to reference these for historical context, please do so from here without importing
them into production paths.
