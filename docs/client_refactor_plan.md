# Client Streamlining & Degodification Plan

## Current Snapshot (2025-09-25)
- `streaming/client_stream_loop.py`: 1,903 LOC after the Phase B bootstrap. Still owns dims intents, decode orchestration, presenter policy, registry mirroring, Qt wakeups, and logging in one class. Only the VT shim fallback keeps a `try` block in the hot path; `getattr` calls remain purged.
- `streaming_canvas.py`: 829 LOC mixing Qt bootstrap, decoder gatekeeping, presenter config, and the smoke harness. Dummy keymap proxies and repeated env lookups remain.
- `streaming/state.py`: 401 LOC combining websocket lifecycle, payload normalisation, throttled requests, and debug logger wiring. Most helpers still live as nested try/except with minimal test coverage.
- `client/proxy_viewer.py`: 517 LOC; mirrors the napari viewer, rate limits dims, owns timers, and forwards events. Naming still reflects legacy socket days.
- Tests remain limited to the websocket shim and the new helper modules; no direct coverage yet for intent ack bookkeeping, presenter scheduling, or dims replay.
- Known runtime issue: VT zero-copy draw segfaults shortly after `"VT gate lifted on keyframe"` once the presenter flips to VT. See `docs/debugging/vt_zero_copy_crash.md` for spike notes and reproduction details.

## Guiding Principles
1. **Plain data + direct functions**: prefer module-level helpers over class stacks. If a helper is only called once, inline it. Keep call stacks shallow so debug traces read naturally.
2. **Boundary-only guards**: assertions inside client logic; reserve `try/except` for websocket I/O, PyAV/VT calls, and Qt signal emission. Log failures at the boundary.
3. **One env read, one place**: resolve flags/config at startup into a simple struct; pass values explicitly rather than lazily reading env vars mid-frame.
4. **Explicit state struct**: represent coordinator state with dataclasses or plain dicts. No indirection layers for “maybe someday” abstraction.
5. **Meaningful names**: rename surfaces to describe behaviour (`ClientStreamLoop`, `DimsIntentTracker`, `send_step_request`). Avoid placeholder terms like “coordinator” or “pending”.
6. **Tests before extraction**: every new pure helper gets a colocated test; leverage recorded payloads (e.g. `dims_update.json`) for smoke coverage.

## Branch Strategy
- Commit current work on `layer-sync-phase2`, then create `client-refactor` from that point (mirroring `server-refactor`).
- Use short-lived topic branches (`client-refactor/phase-a-state-channel`, etc.) rebasing onto `client-refactor` only.
- Periodically rebase `client-refactor` on `layer-sync-phase2` and validate against `server-refactor` in a dedicated integration branch before targeting `main`.

## Phase Plan

### Phase A — State Channel & Dims Intake
- [done] Extracted payload normalisation into `streaming/dims_payload.py` (2025-09-22) with `_tests/test_dims_payload.py`.
- [done] Replaced `_normalize_meta` and `_normalize_current_step` with straight helpers in `streaming/state.py`.
- [done] Introduced a lightweight `StateChannelLoop` wrapper and renamed callbacks (`handle_scene_spec`, `handle_layer_update`).
- Add tests covering dims ack handling and scene caching.

### Phase B — Client Stream Loop Degodification
- Rename `StreamCoordinator` → `ClientStreamLoop` and collapse constructor; move thread bookkeeping into a `LoopState` dataclass.
- Extract VT/PyAV pipeline toggles into module-level helpers returning simple structs (no classes unless absolutely required by Qt).
- Delete `_CallProxy`/`_WakeProxy`; replace with direct `QtCore.QMetaObject.invokeMethod` usage.
- Assert invariants (e.g. decoder readiness) instead of logging-only fallbacks.

#### Phase B Work Plan (target: ≤1,000 LOC in `ClientStreamLoop` post-split)
- **Bootstrap & rename** *(done)*
  1. `LoopState` dataclass introduced to gather threads, channel refs, and cached payloads.
  2. Class/public rename to `ClientStreamLoop`; docs/tests/imports updated.
  3. `send_json` renamed to `post` across channel/controller layers.
- **Scheduler & wake helper**
  1. ✓ `_WakeProxy`/`_CallProxy` now live in `client_loop/scheduler.py`, still canvas-owned; behaviour untouched pending tests.
  2. ✓ Unit coverage (`client_loop/_tests/test_scheduler.py`) locks in GUI-thread delivery from worker threads.
  3. Optionally evolve toward a `ClientLoopBus(QtCore.QObject)` that exposes typed signals (`wake_requested`, `apply_snapshot`, `invoke`); revisit only after latency instrumentation is in place (see `#scheduler-spike`).
- **Pipeline + telemetry helpers**
  1. ✓ VT/PyAV pipeline factories now live in `client_loop/pipelines.py`; behaviour unchanged and still schedule wakes via the canvas.
  2. ✓ Telemetry helpers (`client_loop/telemetry.py`) centralise stats/log timers and metrics enablement with `_tests/test_telemetry.py` coverage.
  3. Write regression tests that simulate keyframe gating + dims replay to ensure `_last_dims_payload` survives each extraction.
- **Config & env plumbing**
  1. ✓ `client_loop_config.py` now loads warmup, metrics, dims, and input env knobs once; loop ctor consumes the struct and `_tests/test_config.py` locks parsing behaviour.
  2. After env centralisation, sweep remaining `try`/`getattr` usage in the loop to boundary-only assertions.
     - ✓ Draw/watchdog path now assertion-based; remaining blanket guards live in smoke harness + renderer fallbacks.
- **Exit criteria**
  - `ClientStreamLoop` ≤1,000 LOC with `<10` `try` blocks and ≤10 `getattr` calls (stretch goal: 0 in hot paths).
  - New helper modules each ≤200 LOC with ≥85 % coverage.
  - Qt objects (proxies/bus) remain canvas-owned; lifetimes explicit.

### Phase C — Presentation & Canvas Simplification
- Formalise a `ClientConfig` struct resolved once at canvas init; drop dummy keymap layers and env re-reads.
- Trim `StreamingCanvas` to focus on Qt wiring and rendering entry points. Move presenter warmup, VT gate state, and fallback buffer policy into `presentation.py` module-level helpers.
- Add smoke test feeding recorded frames to ensure presenter & renderer interplay.

### Phase D — Viewer Intent Surface
- Rename key methods to describe actions (`queue_dims_update`, `flush_dims_update`).
- Extract throttle state into a simple helper (`dims_intent.py`) with unit tests ensuring rate limiting.
- Remove unused “offline” pathways, keep ProxyViewer as a direct forwarder with explicit public methods.

### Phase E — Logging & Instrumentation
- Centralise debug toggles via `client_logging.py`; convert env flags into booleans at startup.
- Ensure every boundary catch logs via `logger.exception`; hot path logs use `logger.debug(..., stacklevel=2)`.
- Align metrics with the new surfaces (`metrics.ClientMetrics.record_presenter_latency`) and add tests for metric increments.

### Scheduler Spike Notes (#scheduler-spike)
- Replacing `_WakeProxy`/`_CallProxy` with `QTimer.singleShot` significantly increased wake latency and broke dims slider replay. See `docs/archive_local/scheduler_spike.md` for details.
- Keep the existing Qt proxies during Phase B extractions; any redesign must preserve GUI-thread delivery and be backed by latency instrumentation.
- Potential follow-up: a `ClientLoopBus(QtCore.QObject)` exposing typed signals (`wake_requested`, `apply_snapshot`, `invoke`) while still leveraging Qt's queued connections.

## Naming Targets
- `StreamCoordinator` → `ClientStreamLoop`
- `_dims_meta` → `dims_state`
- `_pending_intents` → `pending_intents`
- `_vt_gate_lift_time` → `vt_gate_open_at`
- `send_json` → `post`
- Tests follow the same vocabulary (`test_dims_payload_normalises_axes`).

## Metrics & Checkpoints
- Phase A: drive `client/streaming/state.py` to ≤350 LOC (today ~490) and keep `controllers.py` under 200 LOC while adding loop/ack tests.
- Phase B: after rename, cap `streaming/client_stream_loop.py` at 900–1,000 LOC with <25 `try` and <20 `getattr` calls; break out helper modules ≤200 LOC each.
- Phase C: trim `streaming_canvas.py` to ≤500 LOC by outsourcing presenter/VT glue into helpers capped at 200 LOC.
- Maintain >85% coverage on every new helper module (dims payload, presenter policy, intent throttle) and snapshot LOC/try metrics per phase.

## Immediate Next Steps
1. Extract the smoke harness and renderer fallback paths into `client_loop/smoke_helpers.py` + `renderer_fallbacks.py`, replace the remaining blanket `try/except` blocks with explicit assertions/logging at those helper boundaries, and re-run the focused client loop tests.
2. After the helper extraction, recount `ClientStreamLoop` safeguards (current: 1 `try`, 0 `getattr`) and keep the doc snapshot in sync as we chip away at the remaining VT fallback guard.
3. Author the Phase C presenter/canvas split outline (draft `docs/client_presenter_facade.md`) specifying which warmup, VT gate, and render responsibilities move out of `streaming_canvas.py`, along with the regression tests required before refactoring.
4. Run the VT zero-copy spike checklist (see below) before touching renderer code paths so we do not widen the segfault window mid-refactor.

## VT Zero-Copy Spike (2025-09-25)
- Failure signature: client logs `"VT gate lifted on keyframe (seq=15713); presenter=VT"` and crashes inside `GLRenderer._draw_vt_texture` on the immediate paint tick; the crash propagates through Vispy's `paintGL` into Qt's event loop.
- Scope: reproducible on the current `client-refactor` tip with the VT pipeline enabled; the PyAV pipeline alone does not crash.
- Impact: streaming client hard exits (Fatal Python error: Segmentation fault) with multiple worker threads still alive; this blocks further QA of the refactor until mitigated.
- Spike tasks:
  - Capture native backtraces via `lldb -- uv run napari-cuda-client` with the VT safe flag flipped both on/off.
  - Ensure `renderer.py` around `_draw_vt_texture` continues to emit guard logs for cache rebuilds, GL state, and texture IDs without requiring debug env toggles.
  - Audit VT retain/release counts during the gate transition (`RemoteLayerRegistry` swap) to confirm we do not double-release keyframe payloads.
  - Update `docs/debugging/vt_zero_copy_crash.md` with findings and ensure mitigations are tracked as blockers before Phase C kicks off.

### Phase C — VT Pipeline Re-architecture (2025-09-26 spike)

**Motivation**

- Repeated VT segfaults point to unsafe capsule ownership: decoder, cache, presenter, and renderer all `retain`/`release` the same buffer independently.
- Cache state is shared via raw capsules (`_last_payload`) with no synchronisation, so presenter clears or gate transitions can release a buffer while Qt still draws it.
- The existing refcount “anomaly” logging adds noise but cannot prevent double-release; we need structural lifetime ownership instead of heuristics.

**Guiding Shape**

1. (done) Introduce a `FrameLease` abstraction that wraps a VT capsule plus a role-based refcount (`decoder`, `cache`, `presenter`, `renderer`). Only the lease calls into VideoToolbox.
2. (done) Pipelines create leases as soon as `dec.get_frame_nowait()` returns and drop the decoder role immediately after wiring cache + presenter consumers.
3. (done) Presenter buffers store leases (not raw capsules) and release the presenter role whenever frames are trimmed or cleared.
4. (done) Renderer acquires the renderer role before drawing, hands the capsule to the GL shim, then releases on completion via the presenter-supplied callback. Fallback uploads do the same.
5. (done) Cache/gate logic manipulates lease roles only—no direct capsule swapping—so gate entry/exit merely toggles submissions without invalidating the cache lease. The stream loop now reuses the cached lease automatically when no fresh VT frame is dequeued.

**Implementation Steps**

1. (done) `client/streaming/vt_frame.py`: define `FrameLease`, role enum, and context helpers (`lease.acquire_renderer()`, etc.), plus assertions for mismatched release.
2. (done) `pipelines/vt_pipeline.py`:
   - Replace `_last_payload` with `_cache_lease` guarded by a lock.
   - Emit `SubmittedFrame` objects carrying leases; make `release_cb` call `lease.release_presenter`.
   - Simplify `clear()` to release cache/presenter roles without touching capsules; gating clears now preserve the cached lease.
3. (done) `presenter.py`: store leases inside `_BufItem`, releasing presenter role when items fall out of the deque.
4. (done) `client_stream_loop.py`: update draw queue to push leases; when no frame arrives, reuse `_cache_lease` by acquiring the renderer role via `_try_enqueue_cached_vt_frame()`.
5. (done) `renderer.py`: accept leases with explicit release callbacks and hand capsules to the GL shim; renderer never invents its own release logic.
6. (done) Delete the VT retain/release audit scaffolding added during the spike—lease invariants make it redundant.

**Outstanding Lease Cleanup (as of 2025-09-25)**

- (done) Add a deterministic shutdown path that quiesces the VT pipeline before the decoder is torn down; Qt `aboutToQuit` now drives `ClientStreamLoop.stop()` and the launcher calls it explicitly during cleanup.
- (done) Remove the legacy VT retain/release tracing now that shutdown is hardened.
- Once shutdown is stable, re-exercise the spike harness (ndisplay flips, camera resets, back-to-back gates) with the default logging pipeline to confirm no regressions.

**Validation Plan**

- Run manual scenarios (`ndisplay` flips, camera resets, backlog gating) with `NAPARI_CUDA_VT_GL_SAFE=1` and the new lease instrumentation logs.
- Track outstanding renderer roles via debug counters to confirm zero at idle.
- Capture docs updates in `docs/debugging/vt_zero_copy_crash.md` once behaviour stabilises.

> Branch note: work continues on `client-refactor` (see branch strategy above). We stashed the spike tweaks and will reimplement VT on a feature branch stemming from the clean refactor base.
