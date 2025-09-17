napari-cuda Streaming (EGL + NVENC)

- Default encoder input: YUV444 (planar), BT.709 limited, for correct colors and smooth playback.
  - Override via `NAPARI_CUDA_ENCODER_INPUT_FMT=ARGB|ABGR|YUV444`.
  - YUV444 path converts RGBA->YUV444 on-GPU and feeds planar (3H x W) to NVENC.

- Client decoding:
  - Minimal client decodes to `rgb24` using PyAV; no client-side color conversion.
  - Header reserved field remains for compatibility but carries no color hints.

- Debugging:
  - Frame dumps: `NAPARI_CUDA_DEBUG_FRAMES=3 NAPARI_CUDA_DUMP_DIR=logs/napari_cuda_frames`.
  - Bitstream dump (Annex B): `NAPARI_CUDA_DUMP_BITSTREAM=120` to `logs/bitstreams/` (ensure dir exists).

- Performance:
  - Server drains the pixel queue on overflow to avoid latency buildup.
  - Encoder resets on client connect to guarantee an immediate keyframe.

Quick start

Server
  uv run python -m napari_cuda.server.egl_headless_server --width 1920 --height 1080 --fps 60 --animate --volume

Client
  uv run python -m napari_cuda.client.minimal_client --host 127.0.0.1 --state-port 8081 --pixel-port 8082

Scene + Layer Handshake (client focus)

- The state channel now emits structured messages in addition to the legacy `video_config` + `dims.update` flow.
  - `scene.spec`: full snapshot of the server scene (layer specs, dims, camera, capability hints).
  - `layer.update`: incremental metadata change for a single layer (name, shape, multiscale state, etc.).
  - `layer.remove`: notification that a layer was removed on the server.
- Client responsibilities (current phase):
  - `StateChannel` dispatches these payloads to optional callbacks; `StreamCoordinator` caches the latest scene/layer specs under a mutex for consumers.
  - Nothing is rendered locally yet—the streamed texture remains authoritative—but UI code can inspect `_latest_scene_spec`/`_scene_layers` to reason about extents before the remote layer registry lands.
  - Reconnect handling is automatic: the cached scene is refreshed when a new `scene.spec` arrives after websocket recovery.
- Near-term follow-up (tracked separately):
  1. Introduce a `RemoteLayerRegistry` that builds napari `Layer` proxies from the cached specs so the viewer UI has accurate sliders/icons without waiting for server interactions.
  2. Mirror layer intents (visibility, rename, etc.) via the same state channel once capability flags advertise support.
  3. Expand docs with API examples once the registry is in place; for now, the cache is internal but can be inspected for debugging via `StreamCoordinator._scene_layers`.

2D Content And Artifacts

- Why random noise looks “patchy/filmy”
  - H.264 is block-based and optimized for natural images. A fully random 2D texture has high spatial entropy and triggers macroblock artifacts during motion (pan/zoom). You may see boxes popping in/out as prediction updates.
  - Volumetric MIP tends to be low frequency and compresses smoothly, so it looks “clean” at the same bitrate.

- Recommendations for 2D demos
  - Use low-frequency or structured 2D patterns (gradient, color bars, checkerboard) or real image data rather than random noise.
  - Select pattern via `NAPARI_CUDA_2D_PATTERN=gradient|bars|checker|noise` (default: `gradient`).
  - To show a custom image (e.g., a cat), set `NAPARI_CUDA_2D_PATTERN=image` and provide `NAPARI_CUDA_2D_IMAGE=/path/to/your.png`.
  - If random/noisy 2D inputs are important, consider slightly higher bitrate or a higher-quality preset.
  - Keep YUV444 when color fidelity is important; NV12 (4:2:0) is fine for most content and typically compresses more efficiently, but with subsampled chroma.

- Practical tips
  - Bitrate: start around 8–10 MB/s at 1080p60 and adjust by eye.
  - Preset/tuning: keep `tuning_info=low_latency` for responsiveness; bump preset quality if needed.
  - Dimensions: prefer even width/height for best encoder compatibility.

Notes On Formats

- YUV444 vs NV12
  - YUV444 keeps full chroma detail. Slightly more bandwidth and decode cost, but simplest to guarantee color correctness.
  - NV12 (4:2:0) halves chroma samples (interleaved UV plane). Generally a bit faster and more compressible at similar perceptual quality; minimal color difference on most content.
  - Both can produce correct colors as long as the server’s RGB->YUV uses the intended matrix/range (we use BT.709 limited for 1080p).
