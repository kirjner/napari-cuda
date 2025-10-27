# Code Cleanup Plan
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Purpose
Offensive coding tenet: assertions over guards unless interfacing external systems.

Outline concrete hygiene tasks for both client and server before deeper
architecture refactors. Tasks are grouped so we can parallelize without
colliding with forthcoming store/projection work.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Client Cleanups
Offensive coding tenet: make invariants explicit and crash on violation.

1. ~~**Retire legacy streaming shells**~~
   - ~~Delete `client/streaming/client_stream_loop.py`, `state.py`, `state_ledger.py` once imports are redirected to new modules.~~
2. ~~**Slim down PresenterFacade**~~
   - ~~Remove `legacy_draw` wiring and reconnect logic.~~
   - ~~Strip unused HUD/metrics hooks guarded by env flags.~~
3. ~~**ProxyViewer minimalism**~~
   - ~~Move slider/camera forwarding into intent bridge; keep ProxyViewer focused on Qt integration only.~~
4. **Consolidate config reads**
   - Replace scattered `os.getenv` calls with helper functions in a single
     config module.
Offensive coding tenet: no defensive fallbacks, no silent failures.

5. **Logging hygiene**
   - Replace ad-hoc `logger.debug(..., exc_info=True)` in hot paths with
     assertions or structured errors.
6. **Remove dead tests**
   - Delete or rewrite legacy streaming tests that target removed modules once
     new architecture tests exist.
Offensive coding tenet: assertions over guards unless interfacing external systems.

## Server Cleanups
Offensive coding tenet: remove try/except scaffolding that hides bugs.

1. **Archive empty stubs**
  - Remove `server/runtime/runtime_loop.py`, `server/control/pixel_channel.py` after ensuring
     callers reference new modules.
2. **Split control channel**
   - Move WebSocket setup to a new `transport.py`.
   - Extract resume/history logic into `history_store.py`.
3. **Scene apply consolidation**
   - Merge duplicated logic between `scene_state_applier.py` and
     `scene/snapshot.py`; create projection classes ahead of architecture change.
4. **Config normalization**
   - Ensure all modules consume a single `ServerConfig` object rather than
     reading env vars directly.
Offensive coding tenet: make invariants explicit and crash on violation.

5. **Logging consistency**
   - Standardize on structured logs for key events; remove string concatenation
     debug statements.
6. ~~**Deprecation cleanup**~~
   - ~~Delete compatibility shims in `protocol/envelopes.py` after consumers migrate to greenfield imports.~~
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Shared Tasks
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Replace `try/except` blocks guarding internal logic with assertions.
- Introduce pre-commit hooks for import sorting and dead-code detection.
- Document module ownership in `docs/README.md` so future changes know where to
  live.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Execution Order
Offensive coding tenet: make invariants explicit and crash on violation.

1. Finish server/client architecture design docs (this step).
2. Tackle low-risk deletions (legacy shells, logging cleanups).
3. Proceed with architecture refactors using the new store/bridge projections.
4. Revisit cleanup list after each phase to remove completed items and add new
   findings.
Offensive coding tenet: no defensive fallbacks, no silent failures.
