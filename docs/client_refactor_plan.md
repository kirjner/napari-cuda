# Client Streamlining & Degodification Plan

## Current Snapshot (2025-09-22)
- `streaming/coordinator.py`: 2,444 LOC (still a god object). Handles dims intents, decode orchestration, presenter policy, registry mirroring, Qt wakeups, and logging in one class; >40 `try` blocks and pervasive env reads in hot paths.
- `streaming_canvas.py`: 829 LOC mixing Qt bootstrap, decoder gatekeeping, presenter config, and smoke harness. Retains dummy keymap proxies and repeated env lookups.
- `streaming/state.py`: 489 LOC combining websocket lifecycle, payload normalisation, throttled requests, and debug logger wiring. Most helpers live as nested try/except with minimal test coverage.
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
- Extract payload normalisation into `streaming/dims_payload.py` with pure functions, covered by `_tests/test_dims_payload.py`.
- Replace `_normalize_meta` and `_normalize_current_step` with straight helpers; remove broad exception swallowing.
- Introduce a lightweight `StateClient` struct to own websocket handles; rename callbacks (`handle_scene_spec`, `handle_layer_update`).
- Add tests covering dims ack handling and scene caching.

### Phase B — Client Stream Loop Degodification
- Rename `StreamCoordinator` → `ClientStreamLoop` and collapse constructor; move thread bookkeeping into a `LoopState` dataclass.
- Extract VT/PyAV pipeline toggles into module-level helpers returning simple structs (no classes unless absolutely required by Qt).
- Delete `_CallProxy`/`_WakeProxy`; replace with direct `QtCore.QMetaObject.invokeMethod` usage.
- Assert invariants (e.g. decoder readiness) instead of logging-only fallbacks.

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
- Reduce `client/streaming/coordinator.py` under 1,000 LOC by Phase C; track `try` count <25 and `getattr` <20.
- Limit `streaming_canvas.py` to ≤500 LOC after presenter extraction.
- Achieve >85% coverage on new helpers (`dims_payload`, presenter policy, intent throttle).
- Record metrics snapshots per phase inside this doc.

## Immediate Next Steps
1. Commit/stash current local changes on `layer-sync-phase2` and branch `client-refactor`.
2. Promote this document to source control alongside the new branch.
3. Begin Phase A by extracting dims payload helpers and backfilling tests using `dims_update.json` as a fixture.
