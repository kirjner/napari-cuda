# Client Streamlining & Degodification Plan

## Current Snapshot (2025-09-22)
- `streaming/coordinator.py`: 2,441 LOC (still a god object). Handles dims intents, decode orchestration, presenter policy, registry mirroring, Qt wakeups, and logging in one class; >40 `try` blocks and pervasive env reads in hot paths.
- `streaming_canvas.py`: 829 LOC mixing Qt bootstrap, decoder gatekeeping, presenter config, and smoke harness. Retains dummy keymap proxies and repeated env lookups.
- `streaming/state.py`: 401 LOC combining websocket lifecycle, payload normalisation, throttled requests, and debug logger wiring. Most helpers live as nested try/except with minimal test coverage.
- `proxy_viewer.py`: 517 LOC; mirrors napari viewer, rate limits dims, owns timers, and forwards events. Naming still reflects legacy socket days.
- Tests exist only for the websocket shim. No coverage for intent ack bookkeeping, presenter scheduling, or dims replay.

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

#### Phase B Work Plan (target: three PRs, ≤1,000 LOC in `ClientStreamLoop` post-split)
- **Bootstrap & rename**
  1. Cut `ClientStreamLoop` scaffolding: introduce a `LoopState` dataclass bundling threads, channel references, and cached payloads.
  2. Rename the class + public surface (`attach_viewer_mirror`, `post`, etc.) without behaviour changes; snapshot metrics (`LOC`, `try`, `getattr`).
  3. Move the lingering env reads for rate limits and watchdogs into a `client_loop_config.py` helper with colocated unit tests.
- **Scheduler & wake extraction**
  1. Lift `_schedule_next_wake`, `_WakeProxy`, `_CallProxy`, and the timer bookkeeping into `client_loop/scheduler.py`; expose a pure `schedule_next_wake(loop_state, when)` API with unit tests covering coalescing and error handling.
  2. Replace inline Qt signal wiring with `QtCore.QMetaObject.invokeMethod`, asserting GUI-thread dispatch in tests via a minimal Qt harness.
  3. Gate pipelines through the new scheduler module and document cross-thread guarantees.
- **Pipeline + telemetry helpers**
  1. Split VT/PyAV gate logic into `client_loop/pipeline_vt.py` and `client_loop/pipeline_pyav.py` helpers (≤200 LOC each) returning simple dataclasses of callbacks.
  2. Funnel metrics/logging toggles through a `ClientMetricsFacade` wrapper so Phase E can reuse the surface.
  3. Write regression tests that simulate keyframe gating + dims replay to ensure `_last_dims_payload` survives the refactor.
- **Exit criteria**
  - `ClientStreamLoop` ≤1,000 LOC with `<25` `try` blocks and `<20` `getattr` calls.
  - New helper modules each ≤200 LOC with ≥85 % coverage.
  - No direct Qt objects stored on the loop; lifetime owned by the scene canvas.

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

## Naming Targets
- `StreamCoordinator` → `ClientStreamLoop`
- `_dims_meta` → `dims_state`
- `_pending_intents` → `pending_intents`
- `_vt_gate_lift_time` → `vt_gate_open_at`
- `send_json` → `post`
- Tests follow the same vocabulary (`test_dims_payload_normalises_axes`).

## Metrics & Checkpoints
- Phase A: drive `client/streaming/state.py` to ≤350 LOC (today ~490) and keep `controllers.py` under 200 LOC while adding loop/ack tests.
- Phase B: after rename, cap `streaming/coordinator.py` (future `ClientStreamLoop`) at 900–1,000 LOC with <25 `try` and <20 `getattr` calls; break out helper modules ≤200 LOC each.
- Phase C: trim `streaming_canvas.py` to ≤500 LOC by outsourcing presenter/VT glue into helpers capped at 200 LOC.
- Maintain >85% coverage on every new helper module (dims payload, presenter policy, intent throttle) and snapshot LOC/try metrics per phase.

## Immediate Next Steps
1. Land the `ClientStreamLoop` rename + `LoopState` skeleton (no behavioural change).
2. Extract the scheduler/wake helpers into `client_loop/scheduler.py` with unit tests covering coalescing + Qt dispatch.
3. Stage the VT/PyAV pipeline helper modules and move env parsing into `client_loop_config.py` ahead of logic edits.
