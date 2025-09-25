# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot (2025-09-24)
- `egl_worker.py`: 980 LOC (finally under four digits after hoisting scene, ROI, and camera reset helpers). Longest blocks are `__init__` (91 lines), `_evaluate_level_policy` (77), and `_build_scene_state_context` (24).
- `worker_runtime.py`: 492 LOC covering scene-source bootstrap, ROI refresh, camera reset, and multiscale orchestration shared by the worker.
- `scene_state_applier.py`: 357 LOC handling dims/Z and volume state application.
- `lod.py`: 651 LOC; policy runner + budget helpers remain, with further slimming deferred.
- `roi.py`: 502 LOC after absorbing viewport ROI and plane helpers; still slated for a later trim once phase-d policy work moves logging.
- `capture.py`: 260 LOC encapsulating GL capture, CUDA interop, and the frame pipeline façade.
- `try:` count in the worker sits at ≈36 with hot-path guards removed; remaining blocks are subsystem boundaries (EGL/CUDA/NVENC).
- `getattr` usage in the worker now ≈28 after asserting invariants in state/camera helpers.
- Server (`egl_headless_server.py`) still untouched in this pass at 2,249 LOC.

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
  - Method renames landed: `_apply_pending_state` → `drain_scene_updates`, `_render_frame_and_maybe_eval` → `render_tick`, and `_apply_level_internal` → `_apply_level`, clarifying responsibilities for future refactors.
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
- **Worker LOC reduction roadmap (pre-Phase E)**: target ≈−490 LOC to bring `egl_worker.py` under 1,200 before we decompose the server layer. Execute the extractions below while Phase C remains in flight:
  - Add procedural helpers in a `display_mode.py` module for `_apply_ndisplay_switch` and `_reset_volume_step`; worker just logs and forwards (≈−120 LOC).
  - Extend `camera_controller` with a `process_commands` wrapper that subsumes `process_camera_commands` and `_log_zoom_drift` (≈−80 LOC).
  - Collapse `_set_level_with_budget`, `_perform_level_switch`, `_configure_camera_for_mode`, and `_viewport_roi_for_level` into procedural functions inside `worker_runtime.py` (≈−150 LOC).
  - Hoist `_ensure_scene_source` and `_notify_scene_refresh` into plain helpers in `scene_source.py` alongside the existing Zarr source logic (≈−80 LOC).
  - Push `cleanup` into capture/encoder lifecycle helpers and fold `_refresh_slice_if_needed` into `roi_applier`’s procedural API (≈−60 LOC).

- Phase transition: Phase B prerequisites (guard audit, integration coverage, explicit `ServerCtx`) are satisfied; Phase C work can now begin while capture/logging follow-ups proceed in parallel.
- **ROI helper**: `roi.resolve_viewport_roi` now handles debug snapshots, caching, and fallbacks; the worker simply forwards canvas metadata. Integration tests cover the shared helper and the ROI unit suite.
- **LOD selection**: `_evaluate_level_policy` already delegates to `lod.select_level`; budget application now flows through `level_budget.apply_level_with_budget` so the worker only wires callbacks.
- **View preservation**: Render harness landed (`test_render_tick_preserve_view_smoke`) and the ROI helper is universal, so camera assertions now fail fast instead of logging fallbacks.
- **Milestones (status + next action)**:
  1. **ROI consolidation** — ✅ Done. Worker calls the shared helper, `lod.compute_viewport_roi` is removed, and fallback logging/log counters were excised.
  2. **Slim `lod.py`** — ✅ Done for Phase C scope. Budget orchestration moved into `level_budget.apply_level_with_budget`; residual policy trimming is deferred to the next pass.
  3. **Capture façade** — ✅ Done. `CaptureFacade` exposes `capture_frame_for_encoder`, shrinking the worker’s encode path. Follow-up: fold resize handling once dynamic canvas sizing lands.
  4. **Regression sweep** — Ongoing. Worker + LOD suites green (`uv run pytest src/napari_cuda/server/_tests/test_lod_selector.py src/napari_cuda/server/_tests/test_worker_integration.py`); queue ROI unit suite next and keep recording LOC snapshots (`wc -l src/napari_cuda/server/{egl_worker,lod,roi,capture}.py`).
  5. **Residual god-object trimming** — In flight. The worker still orchestrates capture/encode timing, camera command side-effects, scene draining, and budget logging. Each will be delegated to the new helpers below to push the worker under 1,600 LOC.

### Phase D — Logging & Debug Policy (next)
- **D0: Inventory freeze** — capture the current env + debug landscape (`rg "NAPARI_CUDA_"` counts, `try:` totals) and stash the figures in this doc. Acceptance: one table covering env keys by domain (config, logging, dumps, metrics) plus the worker/package try counts so we can prove the cleanup landed.
- Baseline snapshot (2025-09-24):
  | Domain | Unique env keys | Modules touching env directly |
  | --- | --- | --- |
  | Core server/encoder config | 38 | `config.py`, `egl_headless_server.py`, `rendering/encoder.py`, `bitstream.py`
  | Logging + debug toggles | 23 (`NAPARI_CUDA_LOG_*`, `NAPARI_CUDA_DEBUG*`) | `config.py`, `egl_worker.py`, `debug_tools.py`, `rendering/adapter_scene.py`, `frame_pipeline.py`, `egl_headless_server.py`
  | Dump / capture controls | 4 (`NAPARI_CUDA_DUMP_*`) | `config.py`, `debug_tools.py`, `rendering/frame_pipeline.py`
  | Policy + metrics | 14 (`NAPARI_CUDA_LEVEL_*`, `NAPARI_CUDA_METRICS_*`, `NAPARI_CUDA_POLICY_EVENT_PATH`, `NAPARI_CUDA_PRESERVE_VIEW`, `NAPARI_CUDA_STICKY_CONTRAST`, `NAPARI_CUDA_OVERSAMPLING_HYSTERESIS`) | `config.py`, `policy.py`, `metrics.py`, `egl_worker.py`

  | Metric | Current value | Collection command |
  | --- | --- | --- |
  | `try:` blocks in `egl_worker.py` | 23 | `rg -c "try:" src/napari_cuda/server/egl_worker.py` |
  | `try:` blocks in `src/napari_cuda/server` | 345 | `rg -c "try:" src/napari_cuda/server | awk -F: '{sum+=$2} END {print sum}'` |
- **D1: Debug policy surface** — introduce `logging_policy.py` (or `config.debug`) that materialises a `DebugPolicy` dataclass from `ServerCtx`. Enumerate every logging/dump toggle we found (`NAPARI_CUDA_LOG_*`, `NAPARI_CUDA_DEBUG_*`, dump + policy flags) with type hints and defaults. Acceptance: the dataclass is the only place that knows about these env keys and ships descriptive docstrings.
- **D1: Debug policy surface** ✅ — `logging_policy.py` now materialises `DebugPolicy` (with nested dataclasses) from `ServerCtx`. Every logging/dump flag lives there with docstrings + type hints, and unit coverage exists in `_tests/test_logging_policy.py`.
- **D2: Context + CLI wiring** ✅ — `load_server_ctx` builds and carries the policy; `EGLHeadlessServer` logs the snapshot and passes it through to worker/capture. CLI still controls env overrides pre-load; future work can add explicit flags if needed.
- **D3: Worker + pipeline adoption** ✅ — worker/rendering/capture stack consume the policy directly. `DebugDumper` and `FramePipeline` respect policy budgets, and raw dumps no longer mutate `os.environ`. `git grep os.getenv src/napari_cuda/server/egl_worker.py` is clean.
- **D4: Rendering + bitstream alignment** ✅ — `ServerCtx` now exposes `encoder_runtime`/`bitstream` dataclasses feeding `rendering/encoder.py` and `bitstream.py`; `AdapterScene` pulls interpolation from the debug policy; bitstream packer wiring is configured via `configure_bitstream(self._ctx.bitstream)`. Direct env reads in these modules are gone—`rg "NAPARI_CUDA_" src/napari_cuda/server/rendering` and `rg "NAPARI_CUDA_" src/napari_cuda/server/bitstream.py` are clean.
- **D5: Logging guard cleanup** — audit `try/except` blocks that only defend logging/formatting (`debug_tools.DebugDumper.log_env_once`, `_apply_encoder_profile`, ROI logging). Replace internal guards with assertions or helper methods that catch once at the boundary (`logger.exception`). Acceptance: worker `try:` count <25, server package <50, with before/after numbers recorded in this doc.
- **D6: Docs, tests, lint policy** — document the policy flow in `docs/server_architecture.md`, add unit coverage for env resolution + policy wiring, and codify a Ruff-backed lint policy (documented in this phase) that enforces the new logging/debug rules. Extend `ruff.toml` (or equivalent) with rules that block fresh `os.getenv`/`env_bool` calls in the server package and flag logging-only `try/except` guards. Acceptance: new tests live under `src/napari_cuda/server/_tests/test_logging_policy.py`, the lint policy is documented in this doc + `docs/server_architecture.md`, and CI Ruff runs fail on regressions.

Implementation slices:
1. **Policy scaffolding** ✅ — `logging_policy.py` landed; `ServerCtx` now owns `debug_policy` and startup logs the resolved snapshot.
2. **Worker + capture refactor** ✅ — worker + capture stack consume the policy, debug budgets are tracked in-process, and CUDA interop flags derive from the policy instead of env reads.
3. **Render + encoder sweep** ✅ — rendering, adapter, and bitstream modules now consume `ServerCtx.encoder_runtime`, `ServerCtx.bitstream`, and the policy. NVENC env reads are gone, bitstream toggles flow through `configure_bitstream`, and napari dims transitions rely purely on model state (no env fallbacks).
4. **Cleanup + guard audit** — remove remaining direct env reads (policy/patterns/CLI), tighten logging guards, wire Ruff lint/test enforcement, and refresh the metrics table above.

### Phase E — Server Decomposition
- Mirror worker work: extract `PixelBroadcaster`, `StateServer`, `SceneSpecBuilder`. Apply same hostility to try/except & getattr.

### Phase F — Worker State Extraction (later)
- Introduce an explicit `WorkerState` data bag capturing mutable fields consumed by helpers.
- Gradually migrate helpers (`ensure_scene_source`, `refresh_worker_slice_if_needed`, `reset_worker_camera`, etc.) to accept/return state snapshots instead of mutating `self`.
- Refactor the worker into a thin event loop that wires inputs/outputs between state, helpers, and rendering primitives, aligning with the data-oriented plan.

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

## Progress (2025-09-22)
- Capture and encoding now flow through `capture.encode_frame(...)`, and the worker records the helper’s `timings`/`packet` outputs with orientation flags preserved.
- `camera_controller.apply_camera_commands` regained the zoom/pan callback wiring (indent fix) and the SceneState tests exercise the render/policy hooks.
- `scene_state_applier.drain_updates(...)` applies dims/camera fields, issues render marks even without a camera, and returns `SceneDrainResult` consumed by the worker.
- `lod.apply_level_with_context(...)` coordinates budget checks, ROI cache eviction, and switch timing; new unit coverage asserts downgrade and cache behaviour.
- The render loop now delegates to `server.render_loop.run_render_tick(...)`, shrinking `render_tick` to a thin wrapper that just wires callbacks.
- LOD policy evaluation funnels through `lod.run_policy_switch(...)`, keeping decision logging + budget handling out of the worker.
- ROI refreshes rely on `roi_applier.refresh_slice_for_worker(...)` plus `roi.resolve_worker_viewport_roi(...)`, so the worker no longer carries PanZoom bootstrap logic.
- Animation gating lives in `server.camera_animator.animate_if_enabled(...)`, keeping VisPy camera checks bundled for reuse.

## Immediate Next Steps
1. Finish Phase C trims by extracting policy metrics + level apply bookkeeping so `egl_worker.py` drops to ~1.3 k LOC, then reassess hotspots toward the 1.2 k target.
2. Re-run end-to-end GPU smoke (headless server loop) and the full server pytest suite after the next helper drop to confirm selection + capture still cooperate.
3. Backfill docstrings on the new helpers (`render_loop.run_render_tick`, `lod.run_policy_switch`, `roi.resolve_worker_viewport_roi`) and link them from dev notes for the server refactor.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
