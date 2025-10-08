# Server State Architecture

Offensive coding tenets: assert invariants, no fallback branches, surface failures the moment they occur.

## Objectives

- Align the server with the client’s “ledger + emitters + mirrors” architecture.
- Replace the ad-hoc shared scene bag with a single authoritative `ServerStateLedger`.
- Keep command staging separate (queue) but rename and document it using the same vocabulary.
- Ensure render worker feedback (dims/multiscale) ingests into the ledger before any broadcasts.
- Document the concurrency boundary between asyncio control loop and the render worker thread.

## High-Level Flow

```
Control Handlers (emit intents) ──▶ ServerCommandQueue (pending intents)
                                   │
                                   ▼
Render Worker Drain               Render Worker Emits Applied State
                                   │
                                   ▼
                            ServerStateLedger
                                   │
                            ┌──────┴──────┐
                            ▼             ▼
                    ServerDimsMirror   Other mirrors/broadcasters
```

### Terminology

- **ServerCommandQueue** (rename of `RenderMailbox`): coalesces control intents before the render worker applies them.
- **ServerStateLedger**: authoritative property store, mirroring the client’s `ClientStateLedger`.
- **ServerDimsMirror**: subscriber that broadcasts dims/multiscale updates to clients (and any other consumers) once the ledger confirms them.

## Detailed Changes

### New Modules

- `src/napari_cuda/server/state/server_state_ledger.py`
  - Public API mirrors the client: `record_confirmed`, `batch_record_confirmed`, `subscribe`, `subscribe_all`, `snapshot`.
  - Stores per-property tuples `(scope, target, key)` with confirmed value, origin, timestamp, optional metadata/version.
  - Emits deterministic callbacks (offensive coding: raises if subscriber misbehaves).

- `src/napari_cuda/server/state/server_mirrors.py`
  - Defines `ServerDimsMirror`.
  - Subscribes to ledger keys:
    - `('view', 'main', 'ndisplay')`
    - `('view', 'main', 'displayed')`
    - `('dims', 'main', 'current_step')`
    - `('multiscale', 'main', 'level')`
    - plus level metadata (`levels`, `downgraded`), axis labels, etc.
  - Builds `NotifyDimsPayload` from the ledger snapshot.
  - Invokes async broadcaster (`_broadcast_dims_state`) on the control loop via `server._schedule_coro`.

### Updated Modules

- `src/napari_cuda/server/runtime/runtime_mailbox.py` → rename to `server_command_queue.py`.
  - Class renamed to `ServerCommandQueue`.
  - Method names updated (`enqueue_delta`, `drain_pending` remain but under the new class).
  - All imports adjusted in `worker_lifecycle.py`, `control_channel_server.py`, tests.

- `src/napari_cuda/server/runtime/worker_lifecycle.py`
  - `_handle_scene_refresh` renamed to `_ingest_scene_refresh`.
  - The routine now records worker-generated dims payloads in the ledger (ndisplay, displayed axes, current step, multiscale level metadata, axis labels) while continuing to update `server._scene.latest_state` for consumers that still read from the legacy data bag.
  - Duplicate payload guard remains so unchanged snapshots short-circuit before touching the ledger.
  - Post-frame sync (after `capture_and_encode_packet`) records the applied step via `ledger.record_confirmed('dims', 'main', 'current_step', worker._last_step, origin='worker_frame')`.
  - `WorkerSceneNotificationQueue` and `_process_worker_notifications` are removed entirely; the worker no longer enqueues `WorkerSceneNotification` objects once the ledger + mirror path is live.
  - Ledger entry points enforce thread safety with an internal re-entrant lock; render-worker writes happen on the worker thread, control-loop writes stay on the asyncio event loop, and both sides assert the expected thread affinity.

- `src/napari_cuda/server/control/control_channel_server.py`
  - `EGLHeadlessServer` now wires up a `ServerDimsMirror` that subscribes to the ledger and reuses `_broadcast_dims_state` to push confirmed dims payloads to clients.
  - `process_worker_notifications` and the worker notification queue have been deleted; dims flow exclusively through the ledger/mirror path.
  - Control reducers still update `server._scene` for camera/volume/layer state. Future work will migrate those scopes to the ledger once readers are refactored.

- `src/napari_cuda/server/state/server_scene.py`
  - Documented as legacy transitional bag; dims metadata is populated from the ledger/mirror path, while camera/volume/layer state still flows through `ServerSceneState`.
  - TODOs track removing `last_dims_payload`, `last_scene_seq`, and eventually deleting `scene_state.py` after downstream consumers migrate.

- Tests updated:
  - `src/napari_cuda/server/tests/test_worker_lifecycle.py`: ledger-aware fixtures assert `_ingest_scene_refresh` emits `NotifyDimsPayload` via the mirror.
  - `src/napari_cuda/server/tests/test_state_channel_updates.py`: instantiate `ServerStateLedger`/`ServerDimsMirror` and verify dims frames are driven by ledger snapshots.
  - `src/napari_cuda/server/tests/test_worker_integration.py`: fixtures seed canonical axes after the refactor.
  - Adjust fixtures to reference `ServerCommandQueue` and delete the worker notification queue scaffolding.

### Control-State Integration

- `state_update_engine.py` still mutates `server._scene` for camera/volume/layer state. Ledger support currently covers dims metadata; migrating other scopes remains on the future roadmap.
- Ledger records include `timestamp`/`origin`; the history store can consume these once additional scopes move over.

## Implementation Plan

1. **Rename Command Queue**
   - Rename module/class, fix imports, adjust tests.
   - Touch every reference: server bootstrap, `test_worker_lifecycle`, `test_state_channel_updates`, `test_egl_headless_server`, and helper fixtures that construct the queue.
2. **Land `ServerStateLedger`**
   - Implement API with unit tests mirroring `ClientStateLedger` semantics (dedupe, subscription, timestamps).
3. **Integrate Worker Ingestion**
   - Replace `_handle_scene_refresh` with `_ingest_scene_refresh`.
   - Update post-frame sync to use ledger rather than `_scene.latest_state`.
4. **Introduce `ServerDimsMirror`**
   - Wire into server init.
   - Remove `process_worker_notifications`.
   - Mirror subscribes to ledger, builds `NotifyDimsPayload`, uses `_broadcast_dims_state`.
5. **Land `control/state_reducers`**
   - Move all reducer logic out of `state_update_engine.py` into a new `control/state_reducers.py` that writes to the ledger and returns `StateUpdateResult`.
   - Rename `server/state/server_state_ledger.py` to `control/state_ledger.py` and update imports/tests.
6. **Drive render ingest from the ledger**
   - Add `runtime/scene_ingest.py` to build the render snapshot from the ledger.
   - Update worker lifecycle / `_enqueue_latest_state_for_worker` to call the new helper rather than reading `_scene.latest_state`.
7. **Delete legacy scene bag**
   - Remove `ServerSceneState`/`ServerSceneData.latest_state` access, drop `state_update_engine.py`, `server_state_updates.py`, and the `state/` package once all callers use the ledger.
   - Relocate any remaining helpers (plane restore, camera ops) into `runtime/` or focused modules.
8. **Update Tests & Docs**
   - Adjust server tests to use ledger.
   - Ensure doc references match new module names.

## Legacy Scene Data Migration

- **Current step / multiscale metadata**: ingested by `_ingest_scene_refresh` and broadcast via the dims mirror; `_scene.last_dims_payload` remains as a compatibility cache.
- **Camera / Volume / Layer state**: still stored on `ServerSceneState`; migrating these scopes to the ledger is tracked as follow-up work.
- **Plane restore state**: stays render-worker owned; we retain `server._scene.plane_restore_state` until the ledger exposes an equivalent worker-only stash or we move it into a dedicated worker-side structure.
- **Command queue inputs**: rename every occurrence of `RenderMailbox` to `ServerCommandQueue`, including fixtures (`test_worker_lifecycle`, `test_state_channel_updates`, `test_egl_headless_server`) and helper factories.
- **Thread safety**: ledger methods assert they are invoked under the ledger lock; add a short inline docstring describing the render-thread vs asyncio-thread contract so future contributors do not regress into lock-free mutations.

## Naming & Symmetry with Client

- `ClientStateLedger` ↔ `ServerStateLedger`.
- `napari_dims_mirror` ↔ `ServerDimsMirror`.
- Client intent emitters ↔ server control handlers feeding `ServerCommandQueue`.
- Only dims currently needs a server mirror because the render worker is authoritative for multiscale switches. Camera/layer mirrors can be introduced later if the worker ever emits those scopes.

## Concurrency Notes

- Ledger mutations happen on their respective threads:
  - Render worker thread ingests dims state; ledger methods must be thread-safe (explicit locks make crashes obvious rather than silently failing).
  - Control loop thread records confirmed values for camera/layer updates.
- Mirrors run on the control loop (async) and must schedule broadcasts via `server._schedule_coro`.

## Testing Strategy

- **Ledger unit tests**: seed, dedupe, subscription, timestamp origins.
- **Worker lifecycle tests**: ingest dims, confirm ledger updated, no direct `_scene` mutation.
- **Control channel integration**: ensure ledger subscriptions trigger `notify.dims` with correct payloads.
- **Concurrent mutation tests**: simulate worker-thread vs control-loop writes to verify the ledger locking asserts hold.
- **Regression smoke**: run `uv run pytest src/napari_cuda/server/tests -q` before landing.

## Future Work

- Extend ledger to camera/layer scopes if/when worker emits those updates.
- Remove legacy `_scene` bag entirely after consumers switch to ledger snapshots.
