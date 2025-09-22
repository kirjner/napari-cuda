# Client Streamlining & Degodification Plan

## Current Snapshot (2025-09-22)
- `streaming/client_stream_loop.py`: 2,271 LOC (down 124 with config + telemetry + guard pruning; still a god object). Handles dims intents, decode orchestration, presenter policy, registry mirroring, Qt wakeups, and logging in one class; >40 `try` blocks and pervasive env reads in hot paths.
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
- **Exit criteria**
  - `ClientStreamLoop` ≤1,000 LOC with `<25` `try` blocks and `<20` `getattr` calls.
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
1. Sweep remaining `try`/`getattr` hotspots (renderer fallback + smoke harness) now that config centralises inputs; replace with explicit assertions or helper methods.
2. Snapshot remaining try-count metrics and earmark targets for the next extraction slice; align docs/tests with the 2,271 LOC baseline.
3. Draft the Phase C presenter facade proposal while the telemetry hooks are fresh (document expected surfaces before editing `streaming_canvas.py`).
