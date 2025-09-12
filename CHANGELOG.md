Changelog (client-side)

Unreleased

- Arrival mode is now the default timestamp mode (server mode optional).
- Default VT pixfmt switched to BGRA (more reliable mapping path).
- Presenter stats moved to a dedicated 1 Hz timer (`NAPARI_CUDA_VT_STATS`).
- Client logs keyframes (“Keyframe detected (seq=…)”) to verify IDR flow.
- Startup warmup (arrival mode): auto-sized extra latency to ~one frame with ramp-down window; reduces initial phase-lock jitter.
- PyAV fallback uses a higher latency for smooth playback (`NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS`).
- Discontinuity handling hardened: on seq gaps, request keyframe; arrival mode keeps decoding; server mode gates until IDR.
- VT drain worker: decouple VT output from draw loop; prevents decoder queue build-up.
- Robust keyframe detection from bitstream NAL types in addition to header flags.
- Defaults updated:
  - `NAPARI_CUDA_CLIENT_VT_LATENCY_MS=0` (was 80)
  - Removed `NAPARI_CUDA_CLIENT_VT_TS_MODE`; client always uses server timestamps.
- New envs and tuning:
  - `NAPARI_CUDA_CLIENT_PYAV_LATENCY_MS` (default 50)
  - `NAPARI_CUDA_CLIENT_STARTUP_WARMUP_*` (MS, MARGIN_MS, MAX_MS, WINDOW_S)
  - `NAPARI_CUDA_SERVER_TS_BIAS_MS` (tiny negative bias in server mode)

Notes

- Server also prefers keyframes during coalescing and logs “Server: IDR sent (seq=…)”.
- See `src/napari_cuda/CLIENT_QUICKSTART.md` for recommended settings and troubleshooting.
