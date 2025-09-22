# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot
- `egl_worker.py`: 2,203 LOC (down ~200 from the pre-extraction snapshot) after moving scene application into `SceneStateApplier`. Longest blocks now `__init__` (~250 lines), `_viewport_roi_for_level` (186), `_apply_level_internal` (182), `_maybe_select_level` (154); `_apply_pending_state` is ~80 lines and purely orchestrates helpers.
- `scene_state_applier.py`: 203 LOC capturing the dims/Z and volume update logic previously embedded in the worker.
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
- **In progress**:
  - Worker naming cleanup (`SceneStateCoordinator`, `SceneUpdateBundle`, `policy_eval_requested`) to better reflect ownership.
- **Next extraction targets**:
- **ROI refresh integration**: follow through on translating remaining ROI fetches (policy, capture) to use the new helper consistently so no worker path pokes at napari internals directly.
  - **Guard audit**: continue removing remaining defensive branches in ROI math once helper coverage lands.
- **Tests still to add**:
  - Integration-style test exercising level selection under a sequence of zoom hints to lock the relaxed thresholds in place.

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

## Immediate Next Steps
- Begin Phase C by extracting `_viewport_roi_for_level` and related oversampling math into `lod.py`/`roi.py`, further shrinking worker LOC and eliminating remaining defensive branches.
- Land the integration-style test for zoom-hint-driven level selection to guard policy behaviour.
- Prepare for Phase D by sketching a `ServerCtx` logging/debug policy so environment probes can be centralised when logging refactor begins.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
