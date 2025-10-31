# Server Runtime State Migration Plan

Single-source our render state so the worker, viewport runner, controller, and tests consume the same information. This doc captures (a) every plane/volume datum that exists today, (b) the desired end state, and (c) the checkpoints we expect to pass through. It replaces the stale architecture plans that referenced removed mailboxes and “render txn” scaffolding.

## 1. Current State Inventory (truth table)

### 1.1 Snapshot ingest → apply
- `render_loop.apply.snapshots.apply.apply_render_snapshot` (`src/napari_cuda/server/runtime/render_loop/apply/snapshots/apply.py:50`) resolves snapshot ops before touching napari state, short-circuits when the signature matches `_last_snapshot_signature`, and only then suppresses fit callbacks while applying dims.
- `_resolve_snapshot_ops` computes the per-mode metadata (level intent, ROI alignment, slice signature) and `_apply_snapshot_ops` performs the actual mutations, invoking `apply_volume_level` or `apply_slice_level` as needed.
- Volume pose handling is still routed through `apply_volume_camera_pose` and plane pose via `apply_slice_camera_pose`.

### 1.2 Worker fields mutated during apply
- Mode and level: `worker.use_volume`, `worker._active_ms_level`, `worker._level_downgraded`, `worker._last_dims_signature` (`render_loop/apply/snapshots/apply.py:209`, `render_loop/apply/snapshots/apply.py:298`).
- Plane metadata: `_data_wh`, `_last_slice_signature`, `_current_panzoom_rect`, `_viewport_runner.state.*` (`render_loop/apply/snapshots/apply.py:120`–`render_loop/apply/snapshots/apply.py:150`, `render_loop/apply/snapshots/apply.py:418`).
- Volume metadata: `_volume_scale`, `_data_d` (`render_loop/apply/snapshots/apply.py:404`), plus logging via `_layer_logger`.
- Camera pose emission: `_emit_current_camera_pose` invoked with reasons `slice-apply` or `level-reload` (`render_loop/apply/snapshots/apply.py:430`, `worker/egl.py:1623`).

### 1.3 Viewport runner internals
- The runner maintains plane targets (`target_level`, `target_step`, `pose`, `pending_roi`, reload flags, pose reason) and short-circuits volume handling by clearing reload flags when `target_ndisplay >= 3` (`src/napari_cuda/server/runtime/viewport/runner.py`).
- ROI intent resolution depends on `roi_resolver` to call `viewport_roi_for_level` and stores signatures in `pending_roi_signature` (`viewport/runner.py:214`).

### 1.4 Worker loop interactions
- `render_loop.ticks.viewport.run` asks the `ViewportRunner` for a `ViewportPlan`. If the plan contains a slice task and we are still in plane mode, it routes through `render_loop.apply.snapshots.plane.apply_slice_roi` (`src/napari_cuda/server/runtime/render_loop/ticks/viewport.py`).
- Snapshot drain (`egl_worker.py:1488`) rebuilds `SceneStateApplyContext`, updates `_z_index`, `_data_wh`, and mirrors runner state by calling `mark_level_applied` when the staged level matches the target.
- `ensure_scene_source` and `reset_worker_camera` (server/runtime/worker_runtime.py) mutate `worker._active_ms_level`, `_zarr_*`, `_data_wh`, and depend on `worker.use_volume` to branch camera resets (`worker_runtime.py:16`–`worker_runtime.py:105`).

### 1.5 Controller + resume surface
- The snapshot helpers in `napari_cuda.server.scene.snapshot` read `worker.use_volume`, `_active_ms_level`, `_ledger_*` helpers, and napari viewer metadata when constructing `SceneSnapshot` payloads.
- `egl_headless_server._refresh_scene_snapshot` builds scene snapshots directly from the ledger, deriving volume state from `snapshot.volume_*` plus any worker metadata (`src/napari_cuda/server/app/egl_headless_server.py:580`–`egl_headless_server.py:648`).
- State-channel tests seed ledger caches (`view_cache.plane`, `camera_plane.*`) and expect mode switches to drive a single ROI apply and pose emit (`src/napari_cuda/server/tests/test_state_channel_ingest.py:79` and `_helpers/state_channel.py:251`).
- The shared `camera.*` namespace is now deprecated; reducers and restore transactions emit only scoped `camera_plane.*` and `camera_volume.*` entries so plane ACKs return `[x, y]` centers while volume ACKs preserve full 3‑D pose.

### 1.6 Tests that directly hit these internals
- `test_render_snapshot.py` stubs `worker.use_volume`/`_active_ms_level` to validate multi-scale transitions (`src/napari_cuda/server/tests/test_render_snapshot.py:45`–`test_apply_snapshot_multiscale_exit_volume`).
- `test_scene_state_applier.py` assumes `SceneStateApplyContext.last_roi` matches runner state (`scene_state_applier.py:207`).
- `test_egl_worker_camera.py` patches `_viewport_runner` for pose emission (`src/napari_cuda/server/tests/test_egl_worker_camera.py:24`–`test_process_camera_deltas_invokes_pose_callback`).

## 2. Desired End State (steady-state architecture)

### 2.1 Unified runtime state
- Replace ad-hoc `ViewportRunnerState` + worker fields with a shared `ViewportState` dataclass containing `PlaneState` and `VolumeState`. This struct should sit in `src/napari_cuda/server/runtime/viewport/state.py`.
- `PlaneState` owns: controller target (`target_level`, `target_step`, `pose`), applied (`applied_level`, `applied_roi`, signatures), pending reload flags, pose reason, and zoom hints.
- `VolumeState` owns: current level, cached pose (`center`, `angles`, `distance`, `fov`), downgrade flag, and any cached extents/scale.
- `ViewportState.mode` (enum) replaces `worker.use_volume`.

-### 2.2 Snapshot helpers
- Plane path resolved by `_resolve_snapshot_ops` (plane branch) and applied via `_apply_snapshot_ops` plus `slice_snapshot.apply_slice_level`:
  - Computes aligned ROI, resolves chunk shape, loads slab through `slice_snapshot` helpers, and writes `PlaneState.applied_*`.
  - Emits pose once per intent and updates ledger metadata with `viewer_stage.apply_plane_metadata`.
- Volume path uses the same `_resolve_snapshot_ops` / `_apply_snapshot_ops` pairing:
  - Selects level (with downgrades), stages metadata, loads volume, applies camera pose, and updates `VolumeState`.
- Viewer metadata adjustments live in `viewer_stage.apply_plane_metadata` / `viewer_stage.apply_volume_metadata`, responsible for napari dims updates, camera range, and layer logging.

### 2.3 Controller alignment
- Intents extend `LevelSwitchIntent` to carry mode-aware payloads (or introduce explicit `PlaneLevelIntent` / `VolumeLevelIntent`).
- Resume tokens transmit serialized `ViewportState` so clients can restore both plane and volume contexts accurately.
- Control reducers consume the new intent schema; plane restore and level transactions no longer rely on worker private attributes.

### 2.4 Tests
- Unit coverage for state structs, plane/volume snapshot helpers, viewer staging, loader interactions, and volume state updates.
- Integration regression: state-channel ingest ensures single ROI/pose on toggles, render snapshot flow remains deterministic.

## 3. Migration Checkpoints (with tests per stage)

### Stage A — State migration (runner + worker)
1. Introduce `ViewportState`, `PlaneState`, `VolumeState`, and a `RenderMode` enum.
   ✅ Completed (`fixtures live in src/napari_cuda/server/runtime/viewport/state.py`).
2. Port `ViewportRunner` to use `ViewportState.plane`, deleting its private state dataclass.
   ✅ Completed (runner now mutates the shared plane state and tests remain green).
3. Seed `ViewportState` in `EGLRendererWorker.__init__` (`src/napari_cuda/server/runtime/worker/egl.py:128`). Shim properties:
   - `@property use_volume` → `self.viewport_state.mode is RenderMode.VOLUME`
   - `_active_ms_level` → proxy to `viewport_state.plane.applied_level`.
   ✅ Implemented along with shims for `_level_downgraded` and `_volume_scale`.
4. Update worker helpers (`_ensure_scene_source`, `render_loop.ticks.viewport.run`, `_build_scene_state_context`) to read/write through the new struct, guarding mutations with `_state_lock` where required.
   ✅ Worker shims now funnel level/volume updates, ROI tracking, and napari camera snapshots into `ViewportState`.
5. Extend existing tests to assert the struct mirrors legacy values (e.g., `test_render_snapshot` confirms `PlaneState.applied_level` changes on apply; harness checks mode flag).
   ✅ Render snapshot tests and viewport runner suite cover mode/level/pose updates; state-channel harness adoption remains as a follow-up.

### Stage B — Snapshot/application split
1. Factor plane and volume helpers into `slice_snapshot.py`, `volume_snapshot.py`, and `viewer_stage.py`.
2. Inline the former `SceneStateApplier` logic so slice/volume helpers and the worker read from `ViewportState` directly (no intermediate context).
3. Keep `apply_render_snapshot` delegating through the new modules while preserving the existing logging and fit suppression order.
4. Add targeted unit tests for the new helpers (`test_slice_snapshot.py`, `test_volume_snapshot.py`, `test_plane_ops.py`, `test_viewport_layers.py`).
5. Isolate shared viewport utilities under `runtime/viewport/` (`layers.py`, `roi.py`, `plane_ops.py`, `volume_ops.py`) so worker and snapshot code consume a single surface.

### Stage C — Controller/resume alignment
1. Expand intent schema and transactions to consume `ViewportState` (update `runtime/ipc/messages/level_switch.py`, `control/transactions/level_switch.py`, `plane_restore.py`, `control/state_reducers.py`). ✅ Completed.
2. Update `ipc/mailboxes/worker_intent.py`, `egl_headless_server`, and the snapshot helpers to pull state from `ViewportState` rather than private worker fields. ✅ Completed.
3. Propagate state snapshots through render mailboxes and resume tokens; update tests in `src/napari_cuda/server/tests/test_state_channel_ingest.py`, resume helpers, and history store. ✅ Completed.

### Stage D — Shim removal
1. Remove temporary properties (`worker.use_volume`, `_active_ms_level`) once all call sites consume the new state. ✅ Harnesses now mutate `ViewportState` directly.
2. Delete legacy ROI/logging flags that are redundant (`_level_downgraded`, `_roi_cache` duplicates) after verifying new plane loader handles them. ✅ Completed.
3. Run full test suite (`uv run pytest src/napari_cuda/server -q`), `make pre`, and `tox -e mypy`.
4. Drop the `preserve_view_on_switch` policy flag; cached `PlaneState.pose` now drives every restore.

## 4. Checklists

### 4.1 State fields to migrate into `ViewportState`
- Controller targets: `target_level`, `target_ndisplay`, `target_step`, `awaiting_level_confirm`, `snapshot_level`.
- Plane pose: `pose.rect`, `pose.center`, `pose.zoom`, `zoom_hint`, `camera_pose_dirty`.
- Applied plane state: `applied_level`, `applied_step`, `applied_roi`, `applied_roi_signature`.
- Plane reload bookkeeping: `camera_dirty`, `zoom_hint`, `_last_roi`.
- Volume state: `current_level`, `downgraded`, `pose` (center, angles, distance, fov), `scale`, `world_extents`.
- Shared metadata: `mode` (`PLANE` | `VOLUME`), `last_pose_reason`, `last_zoom_hint`.

### 4.2 Side-effects to preserve during refactors
- Napari fit suppression in `apply_render_snapshot`.
- Ledger staging via viewer metadata helpers
- ROI logging and `chunk_shape` alignment for slice snapshot helpers.
- Camera pose emission (`_emit_current_camera_pose`) once per intent.
- Policy evaluation triggers (`_evaluate_level_policy`) when ROI reload marks render.

### 4.3 Tests to expand
- Existing: `test_render_snapshot.py`, `test_scene_state_applier.py`, `test_egl_worker_camera.py`, `test_state_channel_ingest.py`.
- New: `test_state_structs.py`, `test_slice_snapshot.py`, `test_volume_snapshot.py`, `test_volume_state.py`.

Keep this document updated as we advance; it is the working contract between code and desired architecture.

## Appendix A – Target Runtime Layout

```
src/napari_cuda/server/runtime/
├── __init__.py
├── api.py
├── camera/
│   ├── __init__.py
│   ├── animator.py
│   ├── controller.py
│   └── service.py
├── core/
│   ├── __init__.py
│   ├── bootstrap.py         # worker bootstrap context (prev renderer_context)
│   ├── roi_utils.py         # shared ROI math helpers (prev roi_math)
├── data/
│   ├── __init__.py
│   ├── roi_math.py
│   └── scene_types.py
├── ipc/
│   ├── __init__.py
│   ├── mailboxes/
│   │   ├── __init__.py
│   │   ├── render_update.py
│   │   └── worker_intent.py
│   └── messages/
│       ├── __init__.py
│       └── level_switch.py
├── snapshots/
│   ├── __init__.py
│   ├── apply.py
│   ├── build.py
│   ├── plane.py
│   ├── volume.py
│   └── viewer.py
├── viewport/
│   ├── __init__.py
│   ├── layers.py
│   ├── plane_ops.py
│   ├── roi.py               # worker ROI resolver shim
│   ├── runner.py
│   ├── state.py
│   └── updates.py
├── worker/
│   ├── __init__.py
│   ├── capture.py
│   ├── core.py
│   ├── lifecycle.py
│   ├── loop.py
│   └── stage.py
```
