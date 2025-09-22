# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot (2025-09-22)
- `egl_worker.py`: 1,640 LOC (down ~627 from the pre-extraction snapshot and -41 since the last checkpoint). Longest blocks are `__init__` (235 lines), `_apply_level_internal` (72), and `_evaluate_level_policy` (70) after the helper splits.
- `scene_state_applier.py`: 216 LOC capturing the dims/Z and volume update logic previously embedded in the worker.
- `lod.py`: 526 LOC after adding policy-evaluation helpers. Further trims will move selector plumbing and cooldown logging into dedicated modules.
- `roi.py`: 403 LOC after absorbing the viewport debug + ROI caching helpers. Further consolidation will follow once the worker delegates slice refresh entirely to `roi_applier`.
- `capture.py`: 94 LOC encapsulating GL capture, CUDA interop, and frame pipeline orchestration for the worker.
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
  1. **Capture/CUDA extraction** — hoist render capture + CUDA interop into dedicated helpers so the worker only orchestrates timings/logging (goal: trim ~200 LOC). ✅ `CaptureFacade` now fronts GL capture, CUDA interop, and the frame pipeline.
  2. **Guard + naming audit** — ✅ renamed to `SceneStateQueue`/`PendingSceneUpdate`, replaced `_policy_eval_pending` with `_level_policy_refresh_needed`, and removed hot-path `try/except`/`getattr` usage in camera + ROI application. Current counts: 60 `try:` / 33 `getattr` in `egl_worker.py` (boundary-only holds).
  3. **ServerCtx policy surface** — ✅ `ServerCtx.policy` is now authoritative and callers must provide an explicit ctx (`EGLRendererWorker` no longer falls back to `load_server_ctx()`; experiments/headless server wire it through).
  4. **Integration tests** — ✅ added zoom-intent coverage and preserve-view smoke harness in `src/napari_cuda/server/_tests/test_worker_integration.py` to exercise the worker pipeline end-to-end.

### Phase C — ROI & LOD Minimization
- Phase transition: Phase B prerequisites (guard audit, integration coverage, explicit `ServerCtx`) are satisfied; Phase C work can now begin while capture/logging follow-ups proceed in parallel.
- **ROI helper**: `roi.resolve_viewport_roi` now handles debug snapshots, caching, and fallbacks; the worker simply forwards canvas metadata. Integration tests cover the shared helper and the ROI unit suite.
- **LOD selection**: `_evaluate_level_policy` already delegates to `lod.select_level`; budget application now flows through `level_budget.apply_level_with_budget` so the worker only wires callbacks.
- **View preservation**: Render harness landed (`test_render_tick_preserve_view_smoke`) and the ROI helper is universal, so camera assertions now fail fast instead of logging fallbacks.
- **Milestones (status + next action)**:
  1. **ROI consolidation** — ✅ Done. Worker calls the shared helper, `lod.compute_viewport_roi` is removed, and fallback logging/log counters were excised.
  2. **Slim `lod.py`** — ✅ Done for Phase C scope. Budget orchestration moved into `level_budget.apply_level_with_budget`; residual policy trimming is deferred to the next pass.
  3. **Capture façade** — ✅ Done. `CaptureFacade` exposes `capture_frame_for_encoder`, shrinking the worker’s encode path. Follow-up: fold resize handling once dynamic canvas sizing lands.
  4. **Regression sweep** — Ongoing. Worker + LOD suites green (`uv run pytest src/napari_cuda/server/_tests/test_lod_selector.py src/napari_cuda/server/_tests/test_worker_integration.py`); queue ROI unit suite next and keep recording LOC snapshots (`wc -l src/napari_cuda/server/{egl_worker,lod,roi,capture}.py`).

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
