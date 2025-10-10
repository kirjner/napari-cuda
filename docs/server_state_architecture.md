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
Client (state.update)
   │
   ▼
Reducers (control handlers)
   │                 ┌───────────────────────────────────────────────┐
   │                 │                                               │
   │                 ▼                                               │
   │        LatestIntent (continuous, latest‑wins)           ControlCommandQueue
   │        (dims/camera/view/layers per‑key latest)         (discrete, FIFO)
   │                 │                                               │
   │                 ▼                                               │
   │        Render Worker (applied‑first): consumes LatestIntent,    │
   │        renders from applied snapshot only, then confirms        │
   │        applied state (dims/level/view/camera/layers).           │
   │                 │                                               │
   ▼                 ▼                                               ▼
ServerStateLedger ◀── Confirmations (applied)                 Command acks/results
   │
   ├─────────▶ notify.dims / notify.scene / notify.layers / notify.camera
   ▼
Mirrors (e.g., ServerDimsMirror)
```

### Terminology

- **RenderUpdateQueue** (rename of `RenderMailbox`): queues discrete commands before the render worker applies them.
- **ServerStateLedger**: authoritative property store, mirroring the client’s `ClientStateLedger`.
- **ServerDimsMirror**: subscriber that broadcasts dims/multiscale updates to clients (and any other consumers) once the ledger confirms them.
- **RenderLedgerSnapshot**: immutable render-thread input built from the ledger; replaces the legacy `ServerSceneState` bag.
- **AppliedSeqs**: small per-scope counters on the server that record the last applied confirmation sequence (e.g., `dims`, `view`, later `camera`, `layers`). Used only for skip decisions.

## Detailed Changes

### New (or Relocated) Modules

- `src/napari_cuda/server/runtime/render_ledger_snapshot.py`
  - Builds immutable `RenderLedgerSnapshot` objects from the ledger plus transitional caches on `ServerSceneData`.
  - Normalises layer deltas and drains them so the worker sees each update once.

- `src/napari_cuda/server/runtime/scene_state_applier.py`
  - Migrated from the deleted `server/state` package.
  - Applies a `RenderLedgerSnapshot` to the VisPy view/camera and reports the resulting metadata back to the worker loop.

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

- `src/napari_cuda/server/runtime/render_update_queue.py`
  - Renamed from the legacy mailbox and now stores `RenderLedgerSnapshot` values.
  - Signature tracking ensures the worker only triggers expensive policy refreshes when the snapshot actually changes.

- `src/napari_cuda/server/runtime/worker_lifecycle.py`
  - The render loop calls a single frame input builder to merge `LatestIntent` onto the baseline snapshot per tick.
  - After applying, the worker emits applied-first confirmations; the server commits these to the ledger without requiring a matching id for continuous scopes.
  - The loop can skip an entire tick when no new desires exist (latest desired seq per scope ≤ last applied seq per scope), there are no discrete commands, and no animation/diagnostics are active.

- `src/napari_cuda/server/control/control_channel_server.py`
  - Baseline responses compose payloads from the ledger snapshot plus `ServerSceneData` control metadata.
  - State reducers write continuous updates to `LatestIntent` (dims/camera/view/layers) and schedule command acks; they do not mutate legacy scene bags nor push continuous updates into FIFO queues.

- `src/napari_cuda/server/control/state_reducers.py`
  - For continuous domains (dims/camera/view/layers), reducers normalise and store desired targets in `LatestIntent` with per-key monotonic `seq` tracking.
  - Discrete commands continue to queue via `ControlCommandQueue`.
  - `ServerStateLedger` remains authoritative; reducers no longer optimistically project into worker snapshots.

- `src/napari_cuda/server/runtime/worker_runtime.py`
  - Level switches no longer synthesise policy dims intents; the worker emits confirmations after applied state.
  - Helper utilities now lean on `LatestIntent` and snapshots instead of pending-step caches.

- `src/napari_cuda/server/runtime/render_ledger_snapshot.py`
  - Provides helpers to build ledger-backed render snapshots and overlay in-flight `LatestIntent` values for dims/view before the worker consumes them.

- `src/napari_cuda/server/control/pixel_channel.py`
  - Tracks avcC broadcasts separately from frame enqueueing; publishes stream configs immediately when available and caches the last snapshot for reconnects.

### Removed / Simplified

- No more `pending_dims_step`, optimistic ledger mutations, or synthetic policy intents.
- Legacy `RenderMailbox` renamed/repurposed; doubled caching removed in favour of the new snapshot builder.
- The worker no longer reaches into the scene bag directly; everything flows through the ledger snapshot consumed each tick.

## Concurrency Boundary

- **Async control loop (main thread)**:
  - Receives state updates and commands from clients.
  - Writes desired targets into `LatestIntent` for continuous scopes.
  - Enqueues discrete commands into `ControlCommandQueue`.
  - Receives worker confirmations, writes them into the ledger, and drives mirrors (`notify.dims`, `notify.scene`, etc.).

- **Render worker thread**:
  - Pulls a `RenderLedgerSnapshot` from the ledger each tick via `render_ledger_snapshot.pull_render_snapshot`.
  - Reads `LatestIntent` to determine desired dims/view/camera/layer targets.
  - Applies state, renders, and emits applied-first confirmations (dims step, level metadata, view, camera).
  - Pushes encoded frames to the pixel channel and avcC snapshots when available.

The ledger is the only authoritative store for applied state. Mirrors subscribe to ledger updates and broadcast from there. There are no “fast paths” that bypass ledger writes.

## Naming & Symmetry with Client

- `ClientStateLedger` ↔ `ServerStateLedger`.
- `napari_dims_mirror` ↔ `ServerDimsMirror`.
- Client intent emitters ↔ server control handlers writing to `LatestIntent` (continuous) or `ControlCommandQueue` (discrete).
- Mirrors broadcast from the ledger after applied confirmations; render worker remains authoritative for multiscale switches and applied camera/layer state.

## LatestIntent: Continuous Control State

`LatestIntent` is a thread-safe store keyed by (scope, key) with monotonic `seq` tracking per key. Continuous domains use it instead of FIFO queues:

- `dims`: key = axis identifier, value = absolute step tuple, `seq` per axis.
- `view`: key = `"ndisplay"`, value ∈ {2, 3}, `seq`.
- `camera`: keys for pose components (center/zoom/angles), values are absolute targets, `seq`.
- `layers`: key = `(layer_id, prop)`, value = typed update (opacity, visibility, colormap, etc.), `seq`.

Reducers write desired targets using `LatestIntent.set(...)`. They no longer stash pending steps or mutate the worker snapshot directly. Discrete operations (e.g., reset, export) still enqueue commands.

On the render thread, each frame:

1. `render_ledger_snapshot.pull_render_snapshot` constructs a baseline snapshot from the ledger and merges clamped desired dims/view targets from `LatestIntent`.
2. The worker applies the snapshot, renders, and emits confirmations reflecting the applied state.
3. The server writes confirmations to the ledger, updates `AppliedSeqs`, and drives mirrors.

This removes races between optimistic projections and applied state. Skip logic now compares desired/apply `seq` rather than tuples.

## ControlCommandQueue: Discrete, Order-Sensitive Actions

Discrete operations retain a FIFO queue:

- Reducers enqueue commands (with ids) and return acknowledgements.
- The worker executes commands serially and produces command result confirmations when necessary.
- IDs remain required for discrete intents so the server can correlate acks/results.

Continuous updates never enter this queue.

## Invariants & Eliminations

- No optimistic writes to the render snapshot.
- Applied state (worker confirmations) is the single source of truth for dims/view/camera/layers.
- `LatestIntent` holds desired state only; applied truth is tracked in the ledger and `AppliedSeqs`.
- Discrete commands remain FIFO and must produce responses/acks tied to their ids.

## Migration Plan (Dims First)

1. Route dims reducers through `LatestIntent` with per-axis `seq` tracking.
2. Remove optimistic scene bag writes and the `pending_dims_step` cache.
3. Have the worker merge desired dims from `LatestIntent`, apply, and confirm applied step/level.
4. Accept applied-first dims confirmations; update the ledger and mirrors.
5. Extend the same pattern to view, camera, layers.

## Concurrency Notes

- Ledger writes happen under explicit locks; worker thread writes must assert thread affinity where necessary.
- Mirrors run on the asyncio loop; broadcasts are scheduled via `_schedule_coro`.
- The render loop is responsible for pushing avcC snapshots via `pixel_channel.publish_avcc` so reconnects have immediate stream metadata.

## Testing Strategy

- **Ledger tests**: subscription, duplication, timestamp origins.
- **Worker lifecycle tests**: merge latest intents, confirm ledger writes, ensure no direct `_scene` mutation.
- **Control integration**: confirm `notify.dims` broadcasts follow ledger updates.
- **Concurrency regression**: simulate worker vs control loop writes to exercise locking.
- **End-to-end smoke**: `uv run pytest -q` across server tests before landing changes.

## Future Work

- Apply the applied-first + LatestIntent model to camera and layer scopes.
- Consider renaming protocol `intent_id` to `command_id` for clarity (dims no longer rely on it).
- Remove remaining legacy scene bag usage once all consumers switch to ledger snapshots.
