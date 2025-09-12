napari-cuda Client Quickstart (macOS VT)

Recommended settings (smooth, low-latency)
- Start with 16–24 ms latency for interactive work
- Example: `NAPARI_CUDA_VT_STATS=info NAPARI_CUDA_CLIENT_VT_LATENCY_MS=16 uv run napari-cuda-client`

Defaults and behavior
- Timestamping: client prefers server timestamps with a learned offset; falls back to arrival time only when server PTS is missing.
- VT pixel format: default `BGRA` (override with `NAPARI_CUDA_CLIENT_VT_PIXFMT`)
- Presenter stats: 1 Hz dedicated timer when `NAPARI_CUDA_VT_STATS=1|info|debug`
- Keyframe logs: client logs “Keyframe detected (seq=…)”; server logs “Server: IDR sent (seq=…)”
- Join warmup: auto adds a small extra latency at start then ramps down to target to avoid phase‑lock ticks
- PyAV fallback: presenter automatically switches to a higher latency for smoothness (configurable)

Environment knobs (client)
- `NAPARI_CUDA_CLIENT_VT_LATENCY_MS` — base latency target (default 0)
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
  - `NAPARI_CUDA_CLIENT_VT_BUFFER` — presenter buffer size in frames. If unset, the client derives a default of `ceil(latency_s*60)+2` to avoid trimming not‑yet‑due frames in SERVER mode.

Notes
- VT backend expects AVCC from the server; the server sends AVCC and announces it via the state channel.
- Fixed‑latency presenter is used for both VT and PyAV.

Offline smoke (no network)
- `NAPARI_CUDA_SMOKE=1` enables offline testing without a server.
- `NAPARI_CUDA_SMOKE_SOURCE={vt|pyav}` selects the path:
  - `vt`   (default): PyAV H.264 encode → AVCC → VT decode → zero‑copy render.
  - `pyav`: PyAV H.264 encode → PyAV decode (CPU) → RGB render.
- Dimensions / cadence / pattern:
  - `NAPARI_CUDA_SMOKE_W` (default 1280)
  - `NAPARI_CUDA_SMOKE_H` (default 720)
  - `NAPARI_CUDA_SMOKE_FPS` (default 60)
  - `NAPARI_CUDA_SMOKE_MODE={checker|gradient|mip_turntable}` (default `checker`)
  - Aliases: `mip_turntable` can also be selected via `turntable` or `volume` for convenience.
  - Turntable controls:
    - `NAPARI_CUDA_SMOKE_TT_DPS` — degrees per second (default 30; honors `NAPARI_CUDA_TURNTABLE_DPS` too)
    - `NAPARI_CUDA_SMOKE_TT_ELEV` — elevation angle in degrees (default 30)
- Latency knobs used during smoke runs:
  - `vt`: `NAPARI_CUDA_CLIENT_VT_LATENCY_MS`
  - `pyav`: `NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS`
- Pack/encode notes (vt/pyav):
  - Uses the server AVCC packer; if the Cython packer isn’t built, set `NAPARI_CUDA_ALLOW_PY_FALLBACK=1`.
  - VT expects AVCC; smoke paths normalize AUs to AVCC and gate on keyframes.
- Diagnostics:
  - One‑time info logs for smoke start, VT init, and first zero‑copy draw.
  - Presenter stats with `NAPARI_CUDA_VT_STATS=info|debug` (1 Hz).
  - First few AU submissions log AU length and timestamp for visibility.

Jittered smoke (network simulation)
- Enable: `NAPARI_CUDA_JIT_ENABLE=1`
- Timing model:
  - `NAPARI_CUDA_JIT_BASE_MS` — base latency added to each AU (default 0)
  - `NAPARI_CUDA_JIT_MODE={uniform|normal|pareto}` (default `uniform`)
  - `NAPARI_CUDA_JIT_JITTER_MS` — ±half‑range (uniform) or sigma (normal) (default 20)
  - Pareto heavy tail: `NAPARI_CUDA_JIT_PARETO_ALPHA` (default 2.5), `NAPARI_CUDA_JIT_PARETO_SCALE` (ms, default 5)
  - Bursts: `NAPARI_CUDA_JIT_BURST_P` (default 0.03) and `NAPARI_CUDA_JIT_BURST_MS` or `BURST_MIN_MS`/`BURST_MAX_MS`
- Loss/reorder/duplication:
  - `NAPARI_CUDA_JIT_LOSS_P` (default 0.0; TCP/WebSocket are reliable)
  - `NAPARI_CUDA_JIT_REORDER_P` (default 0.0; TCP preserves order), `NAPARI_CUDA_JIT_REORDER_ADV_MS` (default 5)
  - `NAPARI_CUDA_JIT_DUP_P` (default 0.0), `NAPARI_CUDA_JIT_DUP_MS` (default 10)
- Bandwidth cap:
  - `NAPARI_CUDA_JIT_BW_KBPS` (default 0=disabled), `NAPARI_CUDA_JIT_BURST_BYTES` (default 32768)
- PTS policy:
  - `NAPARI_CUDA_JIT_AFFECT_PTS` — also delays PTS (default 0)
  - `NAPARI_CUDA_JIT_TS_SOURCE={encode|send}` — when `send`, stamp PTS at submit time (server-like). Optional `NAPARI_CUDA_JIT_TS_BIAS_MS` shifts PTS by a constant.
- Other:
  - `NAPARI_CUDA_JIT_QUEUE_CAP` (default 512), `NAPARI_CUDA_JIT_SEED` (default 1234)
- Example scenarios:
  - Mild Wi‑Fi: `NAPARI_CUDA_JIT_ENABLE=1 NAPARI_CUDA_JIT_BASE_MS=10 NAPARI_CUDA_JIT_JITTER_MS=15 NAPARI_CUDA_JIT_LOSS_P=0.005 NAPARI_CUDA_JIT_REORDER_P=0.01`
  - Heavy tail: `NAPARI_CUDA_JIT_ENABLE=1 NAPARI_CUDA_JIT_MODE=pareto NAPARI_CUDA_JIT_PARETO_ALPHA=2 NAPARI_CUDA_JIT_PARETO_SCALE=8 NAPARI_CUDA_JIT_BURST_P=0.03 NAPARI_CUDA_JIT_BURST_MS=100`
  - Bandwidth cap: `NAPARI_CUDA_JIT_ENABLE=1 NAPARI_CUDA_JIT_BW_KBPS=4000 NAPARI_CUDA_JIT_BURST_BYTES=65536`

Jitter presets (quick setup)
- CLI flags (launcher):
  - `--jitter` → applies `mild` preset (envs you set explicitly still override)
  - `--jitter-preset {off,mild,heavy,cap4mbps,wifi30}` → selects a preset
- Env fallback (without launcher flags):
  - `NAPARI_CUDA_JIT_PRESET={off|mild|heavy|cap4mbps|wifi30}` — applied on smoke start
- Override behavior:
  - Any explicit env you set wins over the preset (presets use setdefault)
- Examples:
  - `uv run napari-cuda-client --smoke --jitter` (mild jitter)
  - `NAPARI_CUDA_JIT_JITTER_MS=25 uv run napari-cuda-client --smoke --jitter` (override jitter amplitude)
  - `NAPARI_CUDA_JIT_PRESET=cap4mbps uv run napari-cuda-client --smoke`

- Server-like PTS stamping in jitter:
  - `NAPARI_CUDA_JIT_TS_SOURCE=send` (use submit time as PTS)
  - Optionally set `NAPARI_CUDA_SERVER_TS_BIAS_MS=0` to remove client bias

All‑intra for robustness
- When jitter is enabled, the smoke encoder prefers all‑intra GOPs to avoid reference loss artifacts:
  - For `libx264`, we already set `keyint=1:min-keyint=1:bf=0`.
  - For `h264_videotoolbox`, we set `g=1`, `bf=0`, `max_key_interval=1` when jitter is on.
- Override:
  - Set `NAPARI_CUDA_SMOKE_ALLINTRA=0` to allow P‑frames (expect artifacts if frames are dropped before the next keyframe).

Troubleshooting
- No picture after connect: wait for first keyframe; client auto‑requests one and logs on receipt.
- Occasional “discontinuity” warnings: server dropped a frame (latest‑wins queue). Client will resync on next keyframe.
- Brief ticks on join with very small latencies: warmup auto‑sizes to ~one frame; increase base latency by a few ms if needed.
