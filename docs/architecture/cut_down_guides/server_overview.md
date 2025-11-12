Server Overview (Cut-Down, Lean Path)

Scope: server stack entry points and primary flows. Goal: minimal indirection, few flags/branches, direct data paths.

Top-Level Entry
- `EGLHeadlessServer` (src/napari_cuda/server/app/egl_headless_server.py)
  - Starts metrics, websocket servers, and the EGL worker.
  - Holds `ServerStateLedger` and publishes notify payloads.
  - Key callbacks:
    - `_apply_worker_camera_pose(pose)` → reduce_camera_update → broadcast notify.camera.
    - `_handle_worker_level_intents(intent)` → reduce_level_update → enqueue snapshot.
    - `_refresh_scene_snapshot()` → `snapshot_render_state(ledger)` → cache + notify.scene.

Control Channel (State Updates)
- Handlers (src/napari_cuda/server/control/state_update_handlers/)
  - dims: `handle_dims_update` → `reduce_dims_update` (index/step/margins only).
  - view: `handle_view_ndisplay` → `reduce_view_update` (switch 2D/3D).
  - camera: pan/orbit/zoom/reset enqueue `CameraDeltaCommand` (no ledger write). `camera.set` invokes `reduce_camera_update` (rare).
  - layer: `handle_layer_property` → `reduce_layer_property`.

Reducers (src/napari_cuda/server/control/state_reducers.py)
- Build canonical blocks and call scoped transactions:
  - dims/view/level → write `('dims','main','dims_spec')` (+ current_index —
    legacy scope `'current_step'`); persist restore caches for plane/volume pose.
  - camera → write `camera.main.state` (CameraBlock) plus restore caches on worker pose.
  - layer → write per-layer visuals.

Transactions (src/napari_cuda/server/control/transactions/*)
- Strictly scoped writes with versions; no side effects beyond ledger.

Two Parallel Flows
- State channel (notify.*): reducers/txns commit to the ledger; mirrors/broadcasters build notify payloads (dims/camera/layers/scene.level) and send to clients immediately. No render/apply involvement.
- Pixel channel (render): worker pulls `RenderLedgerSnapshot` from the ledger → mailbox → `drain_scene_updates` → apply/plan. No notify emission on this path.

Runtime Pipeline
- Worker loop per tick:
  - Observe `scene.main.op_seq` bump, read `{view, axes, index, lod, camera, layers}` scopes directly from the ledger.
  - Apply dims/camera/layers blocks selectively (per-block signatures).
  - Emit camera poses + level intents through reducers (camera callbacks + worker intent mailbox).

Lean Design Targets (final)
- Factored ledger scopes (`view / axes / index / lod / camera / layers`) replace
  `dims_spec` for runtime consumers; legacy `dims_spec` remains only for compat.
- Camera ledger writes only on applied pose; worker emits pose via reducers.
- Per-block signatures → dims/camera/layers reapply only when necessary; no planner/mailbox.
- ActiveView remains reducer-owned; notify.level and the worker both read the same ledger key.
