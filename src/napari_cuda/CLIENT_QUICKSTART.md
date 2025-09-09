napari-cuda Client Quickstart (macOS VT)

Recommended VT settings (smooth, low-latency)
- NAPARI_CUDA_VT_STATS=info
- NAPARI_CUDA_CLIENT_VT_LATENCY_MS=16
- NAPARI_CUDA_CLIENT_VT_TS_MODE=server
- NAPARI_CUDA_VT_BACKEND=1
- Run: uv run napari-cuda-client

Notes
- VT backend expects AVCC from the server; this server sends AVCC and announces it via the state channel.
- Fixed-latency presenter is used for both VT and PyAV; “server” mode schedules by server timestamp + learned offset.
- Fallback to CPU decoding: set NAPARI_CUDA_VT_BACKEND=off (PyAV). Arrival mode can be used if server timestamps are unavailable.

Troubleshooting
- If VT initializes but no frames appear, a keyframe may be needed; the client auto-requests one after VT (re)init.
- For clearer debugging, websocket client logs are kept at WARNING; set NAPARI_CUDA_VT_STATS=debug for detailed presenter stats.
