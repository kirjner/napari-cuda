# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot (2025-09-24)
- `egl_worker.py`: 994 LOC (finally under four digits after hoisting scene, ROI, and camera reset helpers). Longest blocks are `__init__` (91 lines), `_evaluate_level_policy` (77), and `_build_scene_state_context` (24).
- `worker_runtime.py`: 492 LOC covering scene-source bootstrap, ROI refresh, camera reset, and multiscale orchestration shared by the worker.
- `scene_state_applier.py`: 357 LOC handling dims/Z and volume state application.
- `lod.py`: 651 LOC; policy runner + budget helpers remain, with further slimming deferred.
- `roi.py`: 502 LOC after absorbing viewport ROI and plane helpers; still slated for a later trim once phase-d policy work moves logging.
- `capture.py`: 260 LOC encapsulating GL capture, CUDA interop, and the frame pipeline façade.
- `try:` count in the worker sits at ≈36 with hot-path guards removed; remaining blocks are subsystem boundaries (EGL/CUDA/NVENC).
- `getattr` usage in the worker now ≈28 after asserting invariants in state/camera helpers.
- Server (`egl_headless_server.py`) now sits at 2,181 LOC after the broadcaster split.

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
  - `server_scene_queue.py` now owns scene snapshots, zoom intent tracking, and signature change detection; `_pending_*` fields and state signatures were removed from the worker.
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
  2. **Guard + naming audit** — ✅ renamed to `ServerSceneQueue`/`PendingServerSceneUpdate`, replaced `_policy_eval_pending` with `_level_policy_refresh_needed`, and removed hot-path `try/except`/`getattr` usage in camera + ROI application. Current counts: 60 `try:` / 33 `getattr` in `egl_worker.py` (boundary-only holds).
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
- **D0: Inventory freeze** — capture the current env + debug landscape (`rg "NAPARI_CUDA_"` counts, `try:` totals) and stash the figures in this doc. Acceptance: up-to-date tables covering env domains and guard counts so we can prove the cleanup landed.
- Snapshot (2025-09-25):
  | Domain | Unique env keys | Notes |
  | --- | --- | --- |
  | Encoder/runtime | 21 | NVENC presets + byte budgets live in `config.py`; `egl_headless_server.py` still reads `NAPARI_CUDA_BROADCAST_FPS` directly. |
  | Logging/debug | 21 | All parsed via `logging_policy.py`; worker helpers flip behaviour off `DebugPolicy`. |
  | Runtime ops | 17 | Host/port sizing, queue depth, animation toggles, HW limits. |
  | Metrics | 4 | Metrics server port/refresh and the policy event log path. |
  | Zarr selectors | 4 | Dataset axes/levels/z hints for NGFF sources. |
  | Patterns | 3 | Synthetic image helpers retained for demos/tests. |
  | Debug | 1 | `NAPARI_CUDA_DEBUG` JSON (flags, dumps, worker tweaks). |
  | Encoder/runtime | 1 | `NAPARI_CUDA_ENCODER_CONFIG` JSON (encode/runtime/bitstream overrides). |
  | Policy | 1 | `NAPARI_CUDA_POLICY_CONFIG` JSON (LOD + oversampling settings). |

  | Metric | Current value | Collection command |
  | --- | --- | --- |
  | `try:` blocks in `egl_worker.py` | 18 | `rg -c "try:" src/napari_cuda/server/egl_worker.py` |
  | `try:` blocks in `src/napari_cuda/server` | 342 | `uv run python - <<'PY' ...` *(inline reducer summing `rg -c` output)* |
- **D1: Debug policy surface** — introduce `logging_policy.py` (or `config.debug`) that materialises a `DebugPolicy` dataclass from `ServerCtx`. Enumerate every logging/dump toggle we found (`NAPARI_CUDA_LOG_*`, `NAPARI_CUDA_DEBUG_*`, dump + policy flags) with type hints and defaults. Acceptance: the dataclass is the only place that knows about these env keys and ships descriptive docstrings.
- **D1: Debug policy surface** ✅ — `logging_policy.py` now materialises `DebugPolicy` (with nested dataclasses) from `ServerCtx`. Every logging/dump flag lives there with docstrings + type hints, and unit coverage exists in `_tests/test_logging_policy.py`.
- **D2: Context + CLI wiring** ✅ — `load_server_ctx` builds and carries the policy; `EGLHeadlessServer` logs the snapshot and passes it through to worker/capture. CLI still controls env overrides pre-load; future work can add explicit flags if needed.
- **D3: Worker + pipeline adoption** ✅ — worker/rendering/capture stack consume the policy directly. `DebugDumper` and `FramePipeline` respect policy budgets, and raw dumps no longer mutate `os.environ`. `git grep os.getenv src/napari_cuda/server/egl_worker.py` is clean.
- **D4: Rendering + bitstream alignment** ✅ — `ServerCtx` now exposes `encoder_runtime`/`bitstream` dataclasses feeding `rendering/encoder.py` and `bitstream.py`; `AdapterScene` pulls interpolation from the debug policy; bitstream packer wiring is configured via `configure_bitstream(self._ctx.bitstream)`. Direct env reads in these modules are gone—`rg "NAPARI_CUDA_" src/napari_cuda/server/rendering` and `rg "NAPARI_CUDA_" src/napari_cuda/server/bitstream.py` are clean.
- **D5: Debug flag consolidation** ✅ — `NAPARI_CUDA_DEBUG` now carries JSON (`{"enabled": true, "flags": [...], "dumps": {...}, "worker": {...}}`) parsed by `logging_policy`. All legacy `NAPARI_CUDA_DEBUG_*` / `NAPARI_CUDA_LOG_*` vars are gone, tests updated, and docs refreshed.
- **D6: Encoder override bundle** ✅ — `NAPARI_CUDA_ENCODER_CONFIG` replaces the NVENC/bitstream env sprawl. `load_server_ctx` hydrates encode/runtime/bitstream structures from JSON, CLI profiles merge through `_build_encoder_override_env`, and `egl_headless_server` no longer peeks at `NAPARI_CUDA_BROADCAST_FPS`.
- **D7: Logging guard cleanup** — audit `try/except` blocks that only defend logging/formatting (`debug_tools.DebugDumper.log_env_once`, budget loggers, adapter scene stats). Replace internal guards with assertions or boundary helpers so only I/O surfaces retain broad catches. Acceptance: worker `try:` count stays ≤18 while the whole server package drops below 250 on the path to the <50 target post-refactor, with before/after numbers recorded here. 2025-09-26 snapshot: `egl_worker.py` remains at 18 `try:` blocks; package total is 333 (down from 339 after trimming `debug_tools`, `hw_limits`, and `bitstream`). The remaining 134 guards live in `egl_headless_server.py` and are deferred to Phase E per the decomposition boundary.
- **D8: Docs, tests, lint policy** — document the consolidated policy flow in `docs/server_architecture.md`, add regression coverage for the new parsers, and codify a Ruff-backed lint policy that blocks fresh `os.getenv`/`env_bool` usage and logging-only try/except blocks inside `src/napari_cuda/server`. Acceptance: lint rule enabled, docs updated, and CI fails on regressions. 2025-09-26: `docs/server_architecture.md` already reflects the policy flow; the config-context regression test now lives in `_tests/test_config_ctx.py` and the global pytest `testpaths` includes `napari_cuda` so it runs by default. Lint gating remains to be wired into Ruff after the Phase E server trims unblock broader enforcement.

Implementation slices:
1. **Policy scaffolding** ✅ — `logging_policy.py` landed; `ServerCtx` now owns `debug_policy` and startup logs the resolved snapshot.
2. **Worker + capture refactor** ✅ — worker + capture stack consume the policy, debug budgets are tracked in-process, and CUDA interop flags derive from the policy instead of env reads.
3. **Render + encoder sweep** ✅ — rendering, adapter, and bitstream modules now consume `ServerCtx.encoder_runtime`, `ServerCtx.bitstream`, and the policy. NVENC env reads are gone, bitstream toggles flow through `configure_bitstream`, and napari dims transitions rely purely on model state (no env fallbacks).
4. **Cleanup + guard audit** — remove remaining direct env reads (policy/patterns/CLI), tighten logging guards, wire Ruff lint/test enforcement, and refresh the metrics table above.

### Phase E — Server Decomposition
- Guiding style: follow the Casey Muratori “bag of data + free functions” playbook. Prefer simple dataclasses or TypedDict snapshots that describe state explicitly, and keep helpers procedural rather than adding new classes or inheritance.
- Sequence of work (**work-in-progress**):
  1. **Baseline snapshot** — record current LOC, `try:` counts, and websocket handler responsibilities for `egl_headless_server.py` (targets: trim from 2.2 k LOC toward 1.0 k, keep existing guard counts steady during extraction). *Status 2025-09-27:* 1,394 LOC (`wc -l`), 75 `try:` blocks (`rg -c "try:"`), with the remaining hot spots in `_handle_pixel`, `_broadcast_loop`, and the metrics writers.
  2. **PixelBroadcaster split** — move websocket broadcast/writeback logic into `pixel_broadcaster.py`, exposing pure functions that accept a broadcaster state bag (maps of clients, queue metrics, watchdog timestamps) and emit packets or scheduling decisions. Ensure encode pacing stays untouched; add focused unit tests around queue coalescing and watchdog cooldown. *Status 2025-09-27:* Implemented via `PixelBroadcastState`/`PixelBroadcastConfig`; `_broadcast_loop` delegates to `pixel_broadcaster.broadcast_loop`, LOC trimmed to 2,181 and broadcaster unit tests cover safe-send pruning + bypass keyframe delivery. Server `try:` count dropped to 120 (package total 330).
  3. **ServerScene split** — relocate the state-channel queue helpers into `server_scene_queue.py` (renaming types to `ServerSceneQueue`, `ServerSceneCommand`, `PendingServerSceneUpdate`) and introduce `server_scene.py` exposing the mutable `ServerSceneData` bag. The bag must carry the latest `ServerSceneState`, camera command deque, dims sequencing, volume/multiscale metadata, policy metrics snapshot, SceneSpec caches, and policy log bookkeeping so intent handlers operate on a single data source.
  4. **SceneSpecBuilder extraction** — centralise SceneSpec construction in `server_scene_spec.py`, consuming the viewer snapshot + config databag and returning the serialisable payload. Cover with unit tests for axis/dims permutations.
     - Implement `build_scene_spec(scene: ServerSceneData, manager: ViewerSceneManager, ctx: ServerCtx) -> dict` as a pure function; no new classes.
     - Move existing `_pending_scene_spec` / `_last_scene_spec` bookkeeping in `egl_headless_server` into `ServerSceneData` so broadcasts read/write through the bag.
     - Include a helper for dims payloads (`build_dims_payload(...)`) that clamps ranges using `ServerSceneData` before rebroadcasting.
  5. **Re-wire headless server** — refactor `egl_headless_server.py` to orchestrate these helpers, keeping only lifecycle wiring, thread startup, and boundary I/O. Any transient state should live in explicit structs handed to the helpers.
     - Replace direct attribute mutations with helper calls that operate on `ServerSceneData` (e.g. `server_scene.update_volume_state(...)`).
     - Emit immutable `ServerSceneState` snapshots by copying from the bag before handing off to the worker.
     - Ensure the dims/spec broadcast helpers only read from the bag; no hidden state on the server object.
     - Remaining LOC now sits in `_handle_pixel`, metrics/event plumbing, worker lifecycle glue, and the embedded Dash server; future slices target those clusters with focused helpers.
     - State-channel orchestration lives in `server_scene_control.py` (`handle_state`, `process_state_message`, `broadcast_dims_update`, `rebroadcast_meta`, metrics writers); `EGLHeadlessServer` now only wires the websocket handlers to these helpers.
     - Helpers operate as free functions over the scene bag and server interface, matching the data-oriented style we committed to for Phase E.
  6. **Worker→control queue** — route worker refresh notifications through a dedicated channel so metadata stays ahead of each broadcast:
     - Introduce a lock-safe worker→control queue (parallel to `ServerSceneQueue`) owned by the asyncio loop; `_on_scene_refresh` enqueues the authoritative step + hints instead of scheduling a broadcast directly.
     - Teach the control loop to drain that queue before emitting `dims.update` / `scene.spec`, updating `ServerSceneData` and `ViewerSceneManager` inside the same critical section so `build_dims_payload` always sees aligned axes.
     - Delete the `pending_worker_step` scaffolding once the queue is wired, keeping the scene bag free of cross-thread scratch state and eliminating the ad-hoc flush helper.

  7. **Pixel channel extraction ✅** — `_handle_pixel`, `_broadcast_loop`, encoder resets, and watchdog orchestration now live in `pixel_channel.py`:
     - `PixelChannelState` wraps the existing broadcaster bag and tracks avcC cache, pacing bypass, drops, and watchdog handles while `PixelChannelConfig` captures the static codec geometry.
     - `EGLHeadlessServer` delegates client attach/detach, keyframe forcing, queue draining, and video-config broadcasts to the new helpers; only lifecycle wiring remains inline.
     - Added focused tests (`test_pixel_channel.py`) covering queue overflow drop counting, video-config caching, and watchdog cooldown behaviour without spinning up websockets.
  8. **Metrics/Dash helper ✅** — metrics startup/shutdown now lives in `metrics_server.py`:
     - `metrics_core.Metrics` remains the shared aggregator; `metrics_server.start_metrics_dashboard` boots Dash while `update_policy_metrics` encapsulates gauge updates + JSON dumps.
     - `EGLHeadlessServer` lost `_start/_stop_metrics_server` and simply retains the runner handle; docs now describe the split under “Metrics Helpers.”
  9. **Worker lifecycle module (in progress)** — `worker_lifecycle.py` now owns worker start/stop + scene refresh wiring:
     - `WorkerLifecycleState` tracks the thread, worker instance, and stop event; `start_worker(server, loop, state)`/`stop_worker(state)` encapsulate bootstrap and teardown.
     - Follow-ups: tighten remaining defensive guards in the helpers and remove any lingering direct lifecycle code from `egl_headless_server`; coverage now includes focussed tests (`test_worker_lifecycle.py`) for notification draining and keyframe scheduling.
  10. **Intent helper extraction ✅** — state intents now funnel through `server_scene_intents.py`:
     - Dims and volume mutations operate on `ServerSceneData` via pure helpers that take a lock + metadata, keeping the websocket dispatcher thin.
     - Legacy `_apply_dims_intent` and volume clamp utilities were dropped from `EGLHeadlessServer`; new unit tests (`test_server_scene_intents.py`) cover range clamping, axis resolution, and volume updates.
     - Upcoming MCP/bridge work can reuse these helpers directly, ensuring a single authoritative path for all intent surfaces.
  11. **Preset registry ✅** — profile/env toggles now flow through a dedicated registry:
     - `server/presets.py` maps preset tokens to structured overrides across `ServerConfig`, `EncodeCfg`, `EncoderRuntime`, and `BitstreamRuntime`.
     - Streaming profiles (`latency`, `quality`) now override encode bitrate, NVENC runtime tuning, and the `ServerConfig.profile` flag alongside the NVENC preset tiers (`P1`–`P7`).
     - `load_server_ctx()` resolves `NAPARI_CUDA_PRESET` once, merges overrides via `_apply_dataclass_overrides`, and no longer relies on ad-hoc env reads inside NVENC plumbing.
     - Unit coverage (`test_presets.py`) verifies case-insensitive resolution, streaming-profile overrides, and that registry entries dominate JSON runtime settings.
  12. **Smoke + regression** — after each extraction, run `uv run napari-cuda-server …` (with `--debug --log-sends`) and the server pytest subset (`uv run pytest src/napari_cuda/server/_tests/test_*.py`) to catch behavioural drift.
     - Add a focused pytest (`test_server_scene_data.py`) to cover the new helper functions and dim-sequence wraparound once they land.
  13. **Docs + metrics refresh** — update this plan with new LOC/guard totals, and augment `docs/server_architecture.md` with module responsibilities once the decomposition lands.
     - Document the `server_scene` helpers inline (docstrings) and add a `ServerScene` section to the architecture doc once the spec builder is in place.
  14. **Layer intent bridge alignment** — prep the server for client layer control intents by keeping logic data-oriented:
     - Specify the `image.intent.*` payload schema (opacity, blending, contrast, gamma, colormap, projection, interpolation, depiction) alongside validation helpers.
     - Extend `ServerSceneData` with render-property fields so handlers mutate data snapshots instead of `self`.
     - Add procedural helpers to apply each mutation, enqueue worker commands, and trigger authoritative rebroadcasts.
     - Cover with focused tests asserting state updates and outgoing `layer.update` mirrors.
     - ✅ Canonical layer extras now live in `ServerSceneData.layer_state`; `server_scene_intents.apply_layer_intent` normalizes payloads, the worker applies updates through `SceneStateApplier`, and `server_scene_spec.build_layer_update_payload` drives `layer.update` acks (tests: `test_server_scene_intents.py`, `test_server_scene_spec.py`). Remaining work: wire the client bridge + expand coverage for projection/depiction once the schema is finalised.
     - ✅ Layer controls now live in `LayerControlState`; the worker consumes updates via `SceneStateApplier`, and `layer.update` includes a `controls` payload while keeping `extras` mirrored for compatibility.
     - ⬜ Remaining cleanup: rename the planar payload from `extras` to `controls` once clients have migrated, and extend coverage to mode switches (2D↔3D) to ensure render hints stay aligned.
  15. **ServerScene documentation** — update `docs/server_architecture.md` to describe `ServerSceneData`, `ServerSceneQueue`, and related helpers so downstream consumers understand the mutable vs. immutable scene boundaries.
- Apply the same hostility to `try/except` & `getattr` counts as on the worker: helpers should assert on invariants and reserve broad guards strictly for websocket/NVENC boundary failures.

### Phase F — Worker State Extraction (later)
- Introduce an explicit `WorkerState` data bag capturing mutable fields consumed by helpers.
- Gradually migrate helpers (`ensure_scene_source`, `refresh_worker_slice_if_needed`, `reset_worker_camera`, etc.) to accept/return state snapshots instead of mutating `self`.
- Refactor the worker into a thin event loop that wires inputs/outputs between state, helpers, and rendering primitives, aligning with the data-oriented plan.

### Targets & Metrics
- Module size goals (track in weekly snapshots):
  - `egl_worker.py` → target 700–900 LOC (never above 900); reductions come from camera/state, ROI/LOD, and capture helpers.
  - `server_scene_queue.py` (new) → 200–300 LOC covering pending state application and camera command processing.
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
