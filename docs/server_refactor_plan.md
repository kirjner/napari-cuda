# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot
- `egl_worker.py`: 2.4k LOC, still hosts camera/state/ROI/policy/encode logic. Longest blocks remain `__init__` (252 lines), `_apply_pending_state` (239), `_viewport_roi_for_level` (186), `_apply_level_internal` (182), `_maybe_select_level` (154).
- `try:` count: 129 in worker. Many wrap logging/sample instrumentation. `getattr` (85) and `isinstance` (17) still pepper hot paths.
- Env/env_bool sprinkled across init (encoder, ROI, debug). No centralized logging/flag policy yet.
- Server (`egl_headless_server.py`) untouched in this pass; still >2k LOC.

## Guiding Principles
Refer back to `server_refactor_tenets.md` for the non-negotiable tenets (Degodify, Hostility to "Just In Case").
1. **No "just in case" guards in hot paths**: Logging, attributes we control, and invariants should assert, not silently fall back.
2. **Single responsibilities**: Worker should delegate to thin helpers for ROI, policy, encode, state application, etc.
3. **Explicit boundaries**: Only I/O (GL/CUDA/NVENC/websockets) earns try/except; all else should fail loudly.
4. **Testable helpers**: Move math/policy/ROI code into pure modules with docstring specs.

## Phase Plan

### Phase A — NVENC Boundary ✅
- **Status**: Completed in `rendering/encoder.py`; `EGLRendererWorker` now instantiates `Encoder` and delegates `setup()`, `encode()`, `reset()`, `request_idr()`, and `force_idr()` through the helper guarded by `_enc_lock`.
- **Behavior changes**: NVENC creation + logging paths moved out of `_init_encoder`; worker no longer imports `PyNvVideoCodec` or reads NVENC env vars directly, and encode timings flow through `EncoderTimings`.
- **Follow-ups**:
  - Add targeted unit tests for `Encoder` (mock `PyNvVideoCodec`) covering preset kwargs, fallback path, IDR toggles, and logging guards.
  - Move residual encode-path env reads (e.g. `NAPARI_CUDA_DUMP_RAW`, debug swizzle toggles) into config/logging policy during Phase D so they share the same initialization flow.

### Phase B — State/Camera Core
- **State machine extraction**: Pull `apply_state`, `_apply_pending_state`, `process_camera_commands`, and `_render_frame_and_maybe_eval`’s state transition pieces into a new `state_machine.py` that exposes:
  - `PendingStateQueue`: dataclass capturing queued scene/camera state with validation helpers + `apply_to_view(viewer)` for render-thread commits.
  - `CameraCommandProcessor`: pure helper that consumes `CameraCommand` sequences, folds zoom intent for the policy, and raises on unexpected command kinds.
  - `StateSignature`: utility for change detection (mirrors current `_last_state_sig`).
- **Integration plan**:
  - Worker becomes a thin coordinator: enqueue state via `PendingStateQueue.enqueue(state)` and call `pending.apply(view, layer_set)` on the render thread.
  - Replace ad-hoc `getattr` guards with asserts once `ViewerModel`/camera invariants are satisfied post-initialization.
  - Hoist zoom intent + `_pending_zoom_ratio` mutations into the helper so `_maybe_select_level` only consumes stable intents.
- **Testing scope**:
  - Unit tests for `PendingStateQueue` (2D/3D camera states, volume params) and `CameraCommandProcessor` (zoom/pan/orbit/reset ordering, zoom intent cooldown) under `src/napari_cuda/server/_tests/test_state_machine.py`.
  - Regression test ensuring state signature changes trigger policy eval while identical sequences short-circuit.

### Phase C — ROI & LOD Minimization
- **ROI helper**: Move `_viewport_roi_for_level`, related chunk alignment, oversampling to `lod.py` or new `roi.py`. Worker should request `roi = compute_roi(view_context)` and let helper raise on failure.
- **LOD selection**: Replace `_maybe_select_level` with composition of (1) input intent (zoom queue), (2) pure `lod.select_level(current, overs_map, thresholds)`. No env reads inside worker; thresholds come from config.

### Phase D — Logging & Debug Policy
- Establish `logging_policy.py` or config entries controlling debug flags. Remove per-call env parsing (`NAPARI_CUDA_DEBUG_*`). Convert to bools resolved at init via `ServerCtx`.
- Strip logging try/except; ensure format strings are safe or pre-format values.

### Phase E — Server Decomposition
- Mirror worker work: extract `PixelBroadcaster`, `StateServer`, `SceneSpecBuilder`. Apply same hostility to try/except & getattr.

### Targets & Metrics
- Module size goals (track in weekly snapshots):
  - `egl_worker.py` → target 700–900 LOC (never above 900); reductions come from camera/state, ROI/LOD, and capture helpers.
  - `state_machine.py` (new) → 200–300 LOC covering pending state application and camera command processing.
  - `roi.py` / `lod.py` (new) → 250–300 LOC handling viewport ROI math and oversampling thresholds.
  - `capture.py` (new) → ≤200 LOC for render→blit glue; `cuda_interop.py` (existing) → ≤150 LOC once trimmed to map/unmap and cleanup.
  - `rendering/encoder.py` → ~300 LOC after config/env plumbing moves into ServerCtx.
  - `egl_headless_server.py` → 800–1,000 LOC after websocket, watchdog, and metrics helpers extract.
- Reduce worker `try:` count <40 (boundary-only) & `getattr` <30 in final state.
- Limit longest worker methods <120 lines (hard cap 80 ideal).
- Shift env read paths entirely into `ServerCtx`/new config modules.

### Supporting Work
- **Docs**: Capture module responsibilities & invariants in `docs/server_refactor_plan.md` and inline docstrings.
- **Tests**: Add unit tests around new helpers (encoder, ROI/LOD, state machine) for regression control. Prefer pure functions for easy testing.
- **Lint rules**: Introduce ruff rules to flag `except Exception` & `logger.*` inside try/except.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
