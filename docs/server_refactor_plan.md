# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot (2025-09-22)
- `egl_worker.py`: 1,920 LOC (down ~350 from the pre-extraction snapshot). Longest blocks are `__init__` (254 lines), `_viewport_roi_for_level` (87), and `_evaluate_level_policy` (109); `_apply_level_internal` now delegates to helpers and is <80 lines.
- `scene_state_applier.py`: 216 LOC capturing the dims/Z and volume update logic previously embedded in the worker.
- `lod.py`: 553 LOC after folding the selector helpers back into the core LOD module. The policy helpers (`LevelPolicy*`, `select_level`) now live beside `apply_level`/ROI math, removing the redundant `lod_selector.py` layer.
- `try:` count trending down (≈110) with hot-path guards removed from camera/state application. Remaining broad `try:` live at subsystem boundaries (EGL/CUDA/NVENC) and ROI math; targets to trim continue in Phase C/D.
- `getattr` usage in the worker dropped (≈60 → ≈35) by asserting invariants in the new helper. Further reductions arrive with Phase C decomposition.
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
- **In progress**:
  - Worker naming cleanup (`SceneStateCoordinator`, `SceneUpdateBundle`, `policy_eval_requested`) to better reflect ownership.
- **Next extraction targets**:
  - **Capture/interop split**: hoist capture + CUDA glue into `capture.py`/`cuda_interop.py` so the worker only orchestrates timing and logging.
  - **Guard audit**: prune the remaining ROI fallback logs and broad `except Exception` blocks now that `compute_viewport_roi` is authoritative.
  - **ServerCtx wiring**: lift level-policy thresholds and logging toggles into a shared config object to keep `__init__` moving toward the 200 LOC goal.
- **Tests still to add**:
  - Integration-style test exercising level selection under a sequence of zoom hints to lock the relaxed thresholds in place.

### Phase C — ROI & LOD Minimization
- **ROI helper**: Move `_viewport_roi_for_level`, related chunk alignment, oversampling to `lod.py` or new `roi.py`. Worker should request `roi = compute_roi(view_context)` and let helper raise on failure.
- **LOD selection**: `_evaluate_level_policy` delegates to `lod.select_level`, so zoom hints and thresholds are handled in a pure helper with single env reads during init.
- **View preservation**: Promote the new unit guard into an integration-level check once the ROI extraction lands, so render-thread camera state stays stable without manual smoke tests.

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
- Audit ROI and policy guards now that `compute_viewport_roi`/`lod.select_level` own the math; replace fallback logging with assertions and keep only boundary logging.
- Backfill a scripted render smoke test (or integration test) that exercises Z updates with `preserve_view_on_switch=True`, using the new unit coverage as a baseline.
- Carve the capture/CUDA glue into `capture.py`/`cuda_interop.py`, shrinking `egl_worker.py` toward the 1.5k LOC mark.
- Land the zoom-hint integration test around the new selector before relaxing thresholds further.
- Draft the `ServerCtx` logging/debug config surface so env probes migrate in Phase D without another worker churn.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
