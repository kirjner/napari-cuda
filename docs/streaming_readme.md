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
