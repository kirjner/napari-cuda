# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot (2025-09-22)
- `egl_worker.py`: 1,801 LOC (down ~470 from the pre-extraction snapshot and -119 since the last checkpoint). Longest blocks are `__init__` (254 lines), `_viewport_roi_for_level` (75), and `_evaluate_level_policy` (109); `_apply_level_internal` still sits <80 lines after the helper extraction.
- `scene_state_applier.py`: 216 LOC capturing the dims/Z and volume update logic previously embedded in the worker.
- `lod.py`: 413 LOC after shedding the duplicate ROI helper and unused `LevelDecision` scaffolding. Policy helpers (`LevelPolicy*`, `select_level`) now live beside `apply_level` while we stage the final Phase C trims.
- `try:` count trimmed to ≈60 with camera/state hot-path guards removed; remaining blocks sit at subsystem boundaries (EGL/CUDA/NVENC) and legacy ROI helpers queued for Phase C/D.
- `getattr` usage in the worker now ≈33 after asserting invariants in the state queue + camera paths. Further reductions arrive with Phase C decomposition.
- Env/env_bool still sprinkled across init (encoder, ROI, debug). Centralised logging/config policy remains scheduled for Phase D.
- Server (`egl_headless_server.py`) untouched in this pass; still 2,249 LOC.

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

### Phase B — State/Camera Core (in flight)
- **Implemented**:
  - `state_machine.py` now owns scene snapshots, zoom intent tracking, and signature change detection; `_pending_*` fields and state signatures were removed from the worker.
  - `camera_controller.py` applies zoom/pan/orbit/reset commands and returns intent metadata so the worker only schedules renders/policy once.
  - `roi_applier.py` handles ROI drift detection + slab placement, with `render_tick` delegating ROI refreshes to `SceneStateApplier` rather than reimplementing translate logic.
  - `SceneStateApplier` extracted from `drain_scene_updates`; dims/Z updates and 3D volume params are now applied through a procedural context. Hot-path `try/except` and `hasattr` guards were removed in favour of assertions, aligning with the “hostility to just-in-case” tenet.
  - Method renames landed: `_apply_pending_state` → `drain_scene_updates`, `_render_frame_and_maybe_eval` → `render_tick`, clarifying responsibilities for future refactors.
  - Zoom intent handling normalized (factor >1 ⇒ invert) and rate-limited; LOD thresholds soften to 1.20 during active zoom hints.
  - Unit coverage exists for the state machine, camera controller, and `SceneStateApplier` helpers.
- `roi.py` now owns viewport ROI math and plane helpers; the worker simply wraps caching/log bookkeeping, eliminating the bespoke ROI code path and associated guard soup.
- `lod.py` now contains both the selector policy and the apply helpers. Near-term goal is to trim it back under 400 LOC by pushing ROI math that doesn't need selection context into `roi.py` and collapsing historic LevelDecision scaffolding once the compute-only napari path lands.
- `SceneStateApplier` honours `preserve_view_on_switch`; unit coverage now guards against camera resets when panning before a Z change.
- **Remaining milestones to close Phase B**:
  1. **Capture/CUDA extraction** — hoist render capture + CUDA interop into dedicated helpers so the worker only orchestrates timings/logging (goal: trim ~200 LOC). ✅ `FramePipeline` now owns capture/encode plumbing.
  2. **Guard + naming audit** — ✅ renamed to `SceneStateQueue`/`PendingSceneUpdate`, replaced `_policy_eval_pending` with `_level_policy_refresh_needed`, and removed hot-path `try/except`/`getattr` usage in camera + ROI application. Current counts: 60 `try:` / 33 `getattr` in `egl_worker.py` (boundary-only holds).
  3. **ServerCtx policy surface** — ✅ `ServerCtx.policy` is now authoritative and callers must provide an explicit ctx (`EGLRendererWorker` no longer falls back to `load_server_ctx()`; experiments/headless server wire it through).
  4. **Integration tests** — ✅ added zoom-intent coverage and preserve-view smoke harness in `src/napari_cuda/server/_tests/test_worker_integration.py` to exercise the worker pipeline end-to-end.

### Phase C — ROI & LOD Minimization
- Phase transition: Phase B prerequisites (guard audit, integration coverage, explicit `ServerCtx`) are satisfied; Phase C work can now begin while capture/logging follow-ups proceed in parallel.
- **ROI helper**: `compute_viewport_roi` now lives in `roi.py` and the worker routes through it with cached signatures. Remaining work is to delete the legacy duplicate path in `lod.py`, converge the tests on the shared helper, and collapse the worker-side log scaffolding once the new telemetry lands.
- **LOD selection**: `_evaluate_level_policy` already delegates to `lod.select_level`; next trim is to remove the unused `LevelDecision` scaffolding and cap the module <400 LOC by pushing the lingering ROI math into `roi.py`.
- **View preservation**: Render harness landed (`test_render_tick_preserve_view_smoke`) so we can safely delete the old fallback logging once the shared ROI helper is universal.
- **Milestones (status + next action)**:
  1. **ROI consolidation** — Worker uses `roi.compute_viewport_roi`; drop `lod.compute_viewport_roi`, update `lod` tests to import the shared helper, and remove the remaining ROI fallback logging from the worker.
  2. **Slim `lod.py`** — Remove `LevelDecision` + dormant napari scaffolding, cap the file <400 LOC, and extend `test_lod_selector.py` with oversampling + cooldown coverage for the slimmed API.
  3. **Capture façade** — Reuse the existing `rendering.gl_capture.GLCapture`, `rendering.cuda_interop.CudaInterop`, and `rendering.frame_pipeline.FramePipeline` by fronting them with a thin `server.capture` module so `_set_level_with_budget`/render tick shed the current init/reset boilerplate without duplicating capture logic; add smoke coverage that exercises the façade in the worker fixture.
  4. **Regression sweep** — After each extraction, run ROI + worker integration tests (`uv run pytest src/napari_cuda/server/_tests/test_roi.py src/napari_cuda/server/_tests/test_worker_integration.py`) and log the new LOC metrics (`wc -l src/napari_cuda/server/egl_worker.py`).

### Phase D — Logging & Debug Policy
- Establish `logging_policy.py` or config entries controlling debug flags. Remove per-call env parsing (`NAPARI_CUDA_DEBUG_*`). Convert to bools resolved at init via `ServerCtx`.
- Strip logging try/except; ensure format strings are safe or pre-format values.

### Phase E — Server Decomposition
- Mirror worker work: extract `PixelBroadcaster`, `StateServer`, `SceneSpecBuilder`. Apply same hostility to try/except & getattr.

### Targets & Metrics
- Module size goals (track in weekly snapshots):
  - `egl_worker.py` → target 700–900 LOC (never above 900); reductions come from camera/state, ROI/LOD, and capture helpers.
  - `state_machine.py` (new) → 200–300 LOC covering pending state application and camera command processing.
  - `roi.py` → ≤250 LOC focused purely on ROI math once the remaining helpers move across.
  - `lod.py` → <400 LOC after we delete the legacy `LevelDecision` scaffolding, shift ROI math to `roi.py`, and collapse unused code paths when the napari-gated selector matures.
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

## Immediate Next Steps
1. Finish ROI consolidation: delete `lod.compute_viewport_roi`, repoint its callers/tests to `roi.compute_viewport_roi`, and strip the worker’s fallback logging now that integration coverage exists.
2. Trim `lod.py`: remove the dead `LevelDecision` scaffolding, keep only selector/policy helpers, and extend unit coverage before recording the updated LOC snapshot.
3. Draft the capture façade module skeleton (`capture.py` composing `rendering.gl_capture.GLCapture`, `rendering.cuda_interop.CudaInterop`, and `rendering.frame_pipeline.FramePipeline`) and outline the migration strategy so we can schedule the LOC drop once the ROI/LOD cleanup lands.
4. Draft the `ServerCtx` logging/debug config surface so env probes migrate in Phase D without another worker churn.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
