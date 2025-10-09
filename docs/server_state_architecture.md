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
Intent Emitters ──▶ Reducers ──▶ ServerLedgerUpdate ──▶ Confirmation Queue
      │                         │                              │
      │                         ▼                              │
      └─────────────▶ ServerCommandQueue ◀─── Render Worker ───┘
                                     │
                                     ▼
                             ServerStateLedger
                                     │
                             ┌───────┴───────┐
                             ▼               ▼
                     ServerDimsMirror   (future mirrors)
```

### Terminology

- **ServerCommandQueue** (rename of `RenderMailbox`): coalesces control intents before the render worker applies them.
- **ServerStateLedger**: authoritative property store, mirroring the client’s `ClientStateLedger`.
- **ServerDimsMirror**: subscriber that broadcasts dims/multiscale updates to clients (and any other consumers) once the ledger confirms them.
- **RenderSceneSnapshot**: immutable render-thread input built from the ledger; replaces the legacy `ServerSceneState` bag.

## Detailed Changes

### New (or Relocated) Modules

- `src/napari_cuda/server/runtime/scene_ingest.py`
  - Builds immutable `RenderSceneSnapshot` objects from the ledger plus transitional caches on `ServerSceneData`.
  - Normalises layer deltas and drains them so the worker sees each update once.

- `src/napari_cuda/server/runtime/scene_state_applier.py`
  - Migrated from the deleted `server/state` package.
  - Applies a `RenderSceneSnapshot` to the VisPy view/camera and reports the resulting metadata back to the worker loop.

- `src/napari_cuda/server/control/state_ledger.py`
  - Houses `ServerStateLedger`, mirroring the client module and providing thread-safe mutation/observer APIs.

- `src/napari_cuda/server/control/mirrors/dims_mirror.py`
  - Contains `ServerDimsMirror`, the ledger subscriber responsible for emitting `notify.dims` frames.
  - Schedules broadcasts via the control loop rather than relying on worker notifications.

- `src/napari_cuda/server/scene/`
  - `data.py` houses `ServerSceneData`, camera command queues, and helper utilities (`layer_controls_to_dict`, `prune_control_metadata`, etc.).
  - `plane_restore_state.py` keeps the render-thread-only plane restore helper.
  - `layer_manager.py` now reflects the new namespace used by control payload builders and tests.

- `src/napari_cuda/server/runtime/camera_{controller,animator,ops}.py`
  - Camera helpers formerly under `server/state/` live alongside the worker implementation that consumes them.

### Updated Modules

- `src/napari_cuda/server/runtime/server_command_queue.py`
  - Renamed from the legacy mailbox and now stores `RenderSceneSnapshot` values.
  - Signature tracking ensures the worker only triggers expensive policy refreshes when the snapshot actually changes.

- `src/napari_cuda/server/runtime/worker_lifecycle.py`
  - The render loop acquires the state lock, calls `build_render_snapshot`, and hands the immutable snapshot to the worker.
  - Post-frame acknowledgements keep the ledger in sync without mutating ad-hoc bags.

- `src/napari_cuda/server/control/control_channel_server.py`
  - Baseline responses compose payloads from the ledger snapshot plus `ServerSceneData` control metadata.
  - State reducers schedule broadcasts via `ServerDimsMirror` or the existing notify helpers instead of mutating a legacy scene bag.

- `src/napari_cuda/server/control/state_reducers.py`
  - Reducers normalise intents into `ServerLedgerUpdate` objects that the confirmation queue drains into the ledger.
  - Continues to manage per-layer control metadata but records every confirmed value in the ledger.
  - Future work will inline the remaining `_scene` fallbacks once consumers switch to ledger-only reads.

- Test suite
  - Worker, control, and integration tests now import helpers from `server/scene`, `server/runtime`, and `control/mirrors`.
  - All fixtures operate on `RenderSceneSnapshot` objects, matching the runtime behavior.
  - Confirmation queue drains, mirror broadcasts, and stale acknowledgement guards have dedicated regression coverage.

### Control-State Integration

- `state_reducers.py` still mutates `server._scene` for camera/volume/layer state. Ledger support currently covers dims metadata; migrating other scopes remains on the future roadmap.
- Ledger records include `timestamp`/`origin`; the history store can consume these once additional scopes move over.

## Reducer → Ledger → Mirror Lifecycle

1. **Reducers produce ledger updates**
   - Control handlers route intents through reducers that emit `ServerLedgerUpdate` instances. Only dims metadata uses this path today; camera, volume, and layer scopes still rely on transitional `_scene` mutations.
2. **Updates queue for confirmation**
   - Each reducer result enqueues into the worker confirmation queue. The render worker drains the queue, applies the state, and replays authoritative metadata back into the queue for the ledger to commit.
   - Stale confirmations (e.g., submitting step 0 after step 1) are dropped; the queue maintains the highest monotonic step per axis.
3. **Ledger commits and mirrors broadcast**
   - Confirmed entries land in `ServerStateLedger`. Mirrors observe ledger commits and broadcast payloads once per mutation. `ServerDimsMirror` is the first producer; additional mirrors will follow as more scopes migrate.

### Worker confirmation queue

- The confirmation queue captures render-thread feedback and ensures ledger commits reflect the worker’s last applied state.
- Queue guards prevent regressions:
  - Step regression guard retains the latest confirmed step and ignores lower values.
  - `relative_step_size` remains clamped until we refactor the volume state machine.
  - Plane restore metadata still flows outside the confirmation payload; ledger entries currently omit restore metadata.

### Current limitations

- 2D↔3D transitions briefly reset the step index to 0 before stabilising on the stored plane step. A smoke test asserts that we recover to the mirror’s final state without crashing.
- Volume transitions still rely on an attribute guard that blocks stale `relative_step_size` until the worker-side volume pipeline is rewritten.
- Plane restore metadata is not yet transported through `WorkerStateUpdateConfirmation`; future work will extend the confirmation payload.
- All reducers eventually target the ledger, but today only dims flows complete the reducer → confirmation → ledger pipeline.

## Event-Driven Ingest Roadmap

- **Render loop emits explicit state events**: decouple the render tick from state ingestion so the worker only schedules `ingest_scene_refresh` when it mutates slice/camera/volume state. Pixel streaming and policy evaluation keep their own cadence.
- **Ledger becomes the worker’s source of truth**: on every confirmed control-side update (dims step, camera restore), push the confirmed values back into the worker before the next render tick so `_last_step` (and related fields) always match the ledger.
- **Skip payload construction for no-ops**: once the worker mirrors the ledger, `ingest_scene_refresh` can return immediately if the worker hasn’t changed state—no duplicate payloads hit the ledger or mirror.
- **Mirror broadcasts only ledger mutations**: with the above in place, `ServerDimsMirror` receives exactly one event per real mutation, eliminating dim “snap back” behaviour entirely.
- **Remove transitional guards and caches**: delete the temporary worker-side dedupe cache and leftover compatibility scaffolding once the event-driven path owns the flow. Revisit `relative_step_size` guards when the volume transition path ships.

## Implementation Plan

1. **Extend ledger coverage**
   - Push camera, volume, and layer reducers to write authoritative values into the ledger and drop redundant `_scene` mutations.
   - Update acknowledgements and mirrors once the ledger is the only source of truth.
2. **Ledger-driven baselines**
   - Rework control-channel baseline helpers to compose `notify.scene` / `notify.layers` payloads directly from the ledger snapshot.
   - Remove `pending_layer_updates` after all paths drain through the ledger.
3. **Prune compatibility caches**
   - ✅ `last_dims_payload` removed; dims baselines now come from the ledger via `ServerDimsMirror`.
   - Cull `last_scene_snapshot` and other transitional fields once remaining consumers migrate.
   - Decide whether plane-restore metadata should stay render-thread only or move into the ledger.
4. **Protocol alignment**
   - Document that `notify.scene` and `notify.layers` originate from ledger reads.
   - Plan any protocol adjustments (if needed) alongside client updates so both sides remain in lockstep.
5. **Observability & docs**
   - Ensure dashboards, diagrams, and onboarding docs reflect the new module layout (`server/scene`, `server/runtime`, `control/mirrors`).
   - Keep test guidance focused on `RenderSceneSnapshot` to discourage regressions back to mutable bags.

## Legacy Data Migration Status

- **Render snapshot**: `ServerSceneData.latest_state` now stores a `RenderSceneSnapshot`; the old `ServerSceneState` dataclass is gone.
- **Layer deltas**: reducers still stage pending layer updates in `ServerSceneData.pending_layer_updates`. These remain until baselines broadcast purely from the ledger.
- **Dims cache**: Removed. `ServerDimsMirror` now owns the dedupe signature and the ledger acts as the single source of truth.
- **Plane restore**: `PlaneRestoreState` stays render-thread owned for now. Revisit once we confirm whether the ledger should encode the metadata.
- **Thread safety**: the ledger continues to assert thread affinity. New call sites must keep that contract explicit (render worker vs asyncio loop).

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
