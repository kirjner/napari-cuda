napari-cuda Client Quickstart (macOS VT)

Recommended settings (smooth, low-latency)
- Use ARRIVAL timing by default (now the client default)
- Start with 16–24 ms latency for interactive work
- Example: `NAPARI_CUDA_VT_STATS=info NAPARI_CUDA_CLIENT_VT_LATENCY_MS=16 uv run napari-cuda-client`

Defaults and behavior
- Timestamp mode: default `arrival` (set `NAPARI_CUDA_CLIENT_VT_TS_MODE=server` to use server timestamps)
- VT pixel format: default `BGRA` (override with `NAPARI_CUDA_CLIENT_VT_PIXFMT`)
- Presenter stats: 1 Hz dedicated timer when `NAPARI_CUDA_VT_STATS=1|info|debug`
- Keyframe logs: client logs “Keyframe detected (seq=…)”; server logs “Server: IDR sent (seq=…)”
- Join warmup (arrival mode): auto adds a small extra latency at start then ramps down to target to avoid phase‑lock ticks
- PyAV fallback: presenter automatically switches to a higher latency for smoothness (configurable)

Environment knobs (client)
- `NAPARI_CUDA_CLIENT_VT_LATENCY_MS` — base latency target (default 0)
- `NAPARI_CUDA_CLIENT_VT_TS_MODE` — `arrival` (default) or `server`
- `NAPARI_CUDA_VT_BACKEND` — `1|shim` to enable VT (default), `0|off|false|no` to force PyAV
- `NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS` — latency to use when falling back to PyAV (default 50)
- Warmup (arrival mode only):
  - `NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MS` — fixed extra (overrides auto)
  - `NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MARGIN_MS` — auto margin above one frame (default 2 ms)
  - `NAPARI_CUDA_CLIENT_STARTUP_WARMUP_MAX_MS` — cap on extra (default 24 ms)
  - `NAPARI_CUDA_CLIENT_STARTUP_WARMUP_WINDOW_S` — ramp duration (default 0.75 s)
- Display timer:
  - `NAPARI_CUDA_CLIENT_DISPLAY_FPS` — paint cadence (default 60)
  - `NAPARI_CUDA_CLIENT_VISPY_TIMER=1` — use vispy.Timer instead of Qt QTimer

Notes
- VT backend expects AVCC from the server; the server sends AVCC and announces it via the state channel.
- Fixed‑latency presenter is used for both VT and PyAV; arrival mode ignores server clocks and is most robust at very low latency.

Troubleshooting
- No picture after connect: wait for first keyframe; client auto‑requests one and logs on receipt.
- Occasional “discontinuity” warnings: server dropped a frame (latest‑wins queue). Client will resync on next keyframe without stalling in arrival mode.
- Brief ticks on join with very small latencies: arrival warmup auto‑sizes to ~one frame; increase base latency by a few ms if needed.
