# Idle Bootstrap Refactor Plan

## Context
- When the CUDA server starts without an active dataset, `EGLHeadlessServer._enter_idle_state()` seeds dims metadata but still fabricates a default `layer-0` scene snapshot via `snapshot_scene`.
- That placeholder layer propagates through the ledger, `ServerLayerMirror`, notify payload builders, and the client registry, so the napari client shows an empty layer panel instead of the welcome screen.
- The ledger never clears those bootstrap layer entries, so reloading/unloading datasets leaves behind stale layer state that resumes across reconnects.

## Goals
- Emit a truly empty scene while idle (zero layers, welcome overlay visible) without breaking downstream consumers.
- Keep layer state, thumbnails, and keyframe scheduling dormant until a validated dataset is active.
- Ensure unload back to idle removes layer state from both the ledger and connected clients.
- Preserve dims/camera metadata so HUD widgets stay functional even in idle mode.
- Backstop the change with coverage across server/control/client layers.

## Non‑Goals
- Reworking dataset discovery or validation pipelines.
- Changing the control protocol schema (notify payload shapes stay the same).
- Introducing new hot-path exception handling (retain “no blanket try/except” rule).

## Current Behavior Summary
- `snapshot_scene` always injects a synthetic `layer-0` using `DEFAULT_LAYER_ID` / `DEFAULT_LAYER_NAME`.
- `_default_layer_id()` returns `layer-0` when the cached snapshot is empty, causing helpers such as `_send_layer_thumbnail`, `_schedule_keyframe_and_thumbnail`, and view update handlers to assume a layer exists.
- `ServerLayerMirror.start()` seeds `_latest_states` from the ledger and subscribes to volume/multiscale updates assuming a default layer resolver always returns a concrete ID.
- Ledger bootstrap writes dims/view entries only, but notify baseline builders replay the synthetic layer controls into the ledger on connect, keeping the placeholder alive.
- The client registry accepts the fabricated layer and instantiates a `RemoteImageLayer`, which keeps napari’s Layers dock visible despite the welcome overlay.

## Proposed Changes

### Server Bootstrap & Scene Snapshot
- Teach `snapshot_scene` to support an “empty scene” mode:
  - Skip layer construction when no layer metadata is available (e.g., idle bootstrap).
  - Factor viewer/dims metadata construction out so dims/camera blocks are still emitted without needing a layer.
  - Gate thumbnail acquisition when no default layer is present.
- Replace `DEFAULT_LAYER_*` usage during idle entry:
  - `_enter_idle_state()` should invoke the scene builder in empty-mode, store that snapshot, and cache history without touching the layer scope.
  - `_default_layer_id()` should return `None` when the cached snapshot has zero layers.
- Update all call sites that consume `_default_layer_id()` to short-circuit gracefully (`notify.baseline`, view state update handlers, worker lifecycle thumbnail scheduling, keyframe helpers).
- Harden `_start_mirrors_if_needed()` so `ServerLayerMirror.start()` tolerates a ledger with no layer scopes and simply waits for the first real layer.

### Ledger Lifecycle & Unload
- Add an explicit API (e.g., `ServerStateLedger.clear_scope(scope, target_prefix=None)` or `drop_prefix("layer", layer_id)`) to remove confirmed entries and associated version counters.
- On dataset unload (and idle entry), broadcast `notify.layers` removals before clearing ledger state:
  - Emit a `LayerVisualState(layer_id, extra={"removed": True})` through `broadcast_layers_delta` so clients drop the layer immediately.
  - Use the new ledger API to purge residual layer, multiscale, and volume entries tied to the dataset.
- Ensure the resumable history store resets layer epochs so reconnecting clients do not replay stale layer deltas once we return to idle.

### Mirrors, Thumbnails, and Scheduling
- Guard `ServerLayerMirror._map_event_locked()` and `_record_snapshot()` so volume/multiscale updates without a default layer simply no-op.
- Skip thumbnail/keyframe scheduling when `_default_layer_id()` yields `None`.
- Teach pixel/worker lifecycle bootstrap to check for a populated scene snapshot before scheduling the initial thumbnail task.

### Client Ingestion & Presentation
- Verify `RemoteLayerRegistry.apply_snapshot` already emits an empty snapshot when `scene.layers` is empty; add assertions or unit tests to lock in that behavior.
- Update `NapariLayerMirror.ingest_scene_snapshot` and registry listeners to tolerate zero-layer snapshots without creating placeholder layers or emitting bogus attach/detach cycles.
- Ensure the welcome overlay toggle continues to rely on `metadata.status` and `layer_count`, which will now be zero at idle.

### Notify Baseline & Resume Flow
- Adjust `notify.baseline` helpers:
  - When `default_visuals` is empty, skip thumbnail scheduling and layer delta emission.
  - Avoid seeding `bootstrap.layer_defaults` into the ledger while idle; only perform that work once a dataset is active.
- Confirm resumable replay (`ResumePlan`) handles an empty scene envelope (no layer frames) without errors.

## Implementation Order
1. **Branch & Ledger API**
   - Create a working branch (`feature/idle-ledger-reset` or similar).
   - Add scoped removal helpers to `ServerStateLedger` and cover them with unit tests (state + version eviction).
2. **Idle Bootstrap Rework**
   - Teach `snapshot_scene` and `_enter_idle_state()` to emit empty-layer snapshots.
   - Update `_default_layer_id()` plus all call sites to tolerate `None`.
3. **Unload & Purge Flow**
   - Broadcast `removed` deltas before dataset teardown, then clear layer/multiscale/volume scopes with the new ledger API.
   - Reset resumable history epochs after purge so reconnects don’t replay stale layer deltas.
4. **Baseline & Scheduling Guards**
   - Short-circuit notify baseline builders, keyframe/thumbnail scheduling, and worker bootstrap when no layer ID is available.
5. **Mirror Adjustments**
   - Harden `ServerLayerMirror` for zero-layer ledgers and ensure volume/multiscale events are ignored without a default layer.
6. **Client Updates**
   - Verify the registry + layer mirror handle empty snapshots; adjust tests/mocks that assumed `layer-0` always exists.
7. **Test Suite**
   - Add the server/client/integration coverage detailed below; re-run permission-denied discovery regression.

## Testing Strategy
- **Server unit tests**
  - `test_state_channel_updates`: add expectations that idle baselines emit zero layers and no thumbnail scheduling occurs.
  - Ledger purge tests verifying `clear_scope` drops entries and versions.
  - Permission-denied discovery regression (already added) to ensure the crawl still bypasses protected directories.
- **Client unit tests**
  - Registry snapshot for empty scene + delta removal path.
  - Layer mirror welcome overlay toggles when `layer_count == 0`.
- **Integration / end-to-end**
  - Idle handshake smoke: client connects before dataset load and sees welcome overlay with no layer UI.
  - Load → unload cycle: dataset load seeds real layers, unload emits removal, idle baseline returns to zero.
  - Resume store replay: connect, disconnect, reconnect at idle should not resurrect stale layers.

## Open Questions / Follow-Ups
- Decide on the ledger purge API shape (`clear_scope` vs. targeted delete) and whether it should broadcast an event for observers.
- Ensure volume/multiscale state resets alongside layer removal to avoid lingering state in future loads.
- Confirm there are no consumers (dash dashboard, metrics, etc.) that assume `DEFAULT_LAYER_ID` always exists; update them if discovered during implementation.
- Consider documenting the new idle invariants (zero layers, metadata.status == "idle") in developer docs so future features respect the contract.
