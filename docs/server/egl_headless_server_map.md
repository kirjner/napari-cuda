# EGL Headless Server Responsibility Map

This note catalogs the surface area of `EGLHeadlessServer`
(`src/napari_cuda/server/app/egl_headless_server.py`) before we begin carving
it into focused modules. Each section lists the public entry points, the shared
state touched, and key collaborators. Understanding these seams will help us
extract testable helpers without introducing a heavy class hierarchy.

For the complementary worker breakdown, see `runtime_worker.md`.

## Lifecycle Overview
- Class construction pulls environment variables, resolves `ServerCtx`, wires
  pixel channel state, mirrors, metrics, worker intents, camera queues, and the
  thumbnail tracker (`__init__`, lines 150-342).
- `start` boots logging, enters idle or loads the configured dataset, starts
  websocket servers, launches the metrics dashboard, kicks off the pixel loop,
  and keeps the async loop alive (`start`, lines 1139-1246).
- Shutdown flows through the `finally` block in `start`, closing servers,
  stopping metrics, and tearing down the worker (`start`, lines 1215-1225).

## State Surfaces (current single-class layout)
- **Environment & context** – `_ctx_env`, `_ctx`, `_data_root`, `_browse_root`,
  and encoder bitstream policy (lines 150-206).
- **Pixel / encode state** – `Metrics`, `_metrics_runner`, `_pixel_channel`,
  `_pixel_config`, `_param_cache`, `_dump_*`, `_seq`, `_dump_remaining`
  (lines 205-265, 513-540).
- **Worker & runtime hooks** – `_worker_lifecycle`, `_runtime_handle`,
  `_worker_intents`, `_camera_queue`, `_camera_command_seq`,
  `_state_lock`, `_bootstrap_snapshot` (lines 233-354).
- **Mirrors & control-plane** – `_dims_mirror`, `_layer_mirror`, inline mirror
  callbacks, `_state_clients`, `_resumable_store`, `_log_*` flags
  (lines 266-339).
- **Scene cache** – `_scene_snapshot`, thumbnail state, default layer helpers
  (lines 320-373, 567-776).
- **Dataset configuration** – `_zarr_*`, volume budget caps, idle bootstrap
  (lines 176-208, 376-452, 881-1010).
- **Filesystem access** – `_require_data_root`, `_resolve_*`, `_list_directory`
  (lines 1030-1117).
- **Websocket hosting** – `_state_send`, `_broadcast_loop`, `_ingest_pixel`,
  inline `state_handler` definitions (lines 1139-1254, 1333-1365).

## Responsibility Clusters

### 1. Env & Configuration Bootstrap
- **Entry points:** `__init__`, `_resolve_env_path`.
- **Shared state:** `_ctx_env`, `_ctx`, `_data_root`, `_browse_root`,
  `_volume_max_bytes_cfg`, `_hw_volume_max_bytes`, `_codec_name`, `cfg`.
- **Dependencies:** `load_server_ctx`, `configure_bitstream`, `get_hw_limits`.
- **Notes:** All env parsing, bitstream policy, and codec selection are embedded
  in `__init__`, making it hard to test context resolution independently.

### 2. Pixel Channel & Encoder Policy
- **Entry points:** `__init__` (pixel state), `_current_avcc_bytes`,
  `_try_reset_encoder`, `_try_force_idr`, `_ensure_keyframe`, `_start_kf_watchdog`.
- **Shared state:** `_pixel_channel`, `_pixel_config`, `_param_cache`,
  `_dump_*`, `_seq`, `_metrics_runner`, `_metrics`.
- **Dependencies:** `Metrics`, `build_avcc_config`, `ensure_keyframe`,
  `mark_stream_config_dirty`, `broadcast_stream_config`.
- **Notes:** Encoder resets and keyframe triggers are sprinkled across methods,
  tightly coupled to request scheduling and worker hooks.

### 3. Worker Lifecycle & Mailboxes
- **Entry points:** `_start_worker`, `_stop_worker`, `_handle_worker_level_intents`,
  `_enqueue_camera_delta`, `_next_camera_command_seq`, `_apply_worker_camera_pose`.
- **Shared state:** `_worker_lifecycle`, `_runtime_handle`, `_worker_intents`,
  `_camera_queue`, `_state_lock`, `_camera_command_seq`.
- **Dependencies:** `WorkerLifecycleState`, `WorkerIntentMailbox`,
  `CameraCommandQueue`, `reduce_level_update`, `render_update.enqueue`.
- **Notes:** Level, camera, and render intent handling all live beside the mailbox
  polling logic, forcing the coordinator to know reducer semantics.

### 4. Dataset Lifecycle & Idle Flow
- **Entry points:** `_enter_idle_state`, `_switch_dataset`, `_handle_zarr_load`.
- **Shared state:** `_state_ledger`, `_bootstrap_snapshot`, `_layer_mirror`,
  `_dims_mirror`, `_zarr_*`, `_thumbnail_state`.
- **Dependencies:** `reduce_bootstrap_state`, `discover_dataset_root`,
  `pull_render_snapshot`, `probe_scene_bootstrap`, worker start/stop.
- **Notes:** Idle entry, dataset activation, ledger resets, worker orchestration,
  and stream reconfiguration are interwoven, making re-entry paths fragile.

### 5. Control-Plane Broadcasting & History
- **Entry points:** `_build_scene_payload`, `_cache_scene_history`,
  `_broadcast_state_baseline`, `_scene_snapshot_json`, `_state_send`.
- **Shared state:** `_scene_snapshot`, `_resumable_store`, `_state_clients`,
  `_state_ledger`, `_log_state_traces`.
- **Dependencies:** `build_notify_scene_payload`, `state_sequencer`,
  `orchestrate_connect`, `safe_send`.
- **Notes:** Snapshot caching, resumable history, websocket sequencing, and
  baseline orchestration share the same methods, so tests must spin up the
  entire class.

### 6. Thumbnail Pipeline (currently broken)
- **Entry points:** `_queue_thumbnail_refresh`, `_on_render_tick`,
  `_emit_worker_thumbnail`, `_handle_worker_thumbnails`, `_ingest_worker_thumbnail`,
  `_layer_thumbnail`.
- **Shared state:** `_thumbnail_state`, `_worker_intents`, `_scene_snapshot`,
  `_state_ledger`.
- **Dependencies:** `ThumbnailState` helpers, `ThumbnailIntent`, worker mark-tick
  methods, numpy transforms.
- **Notes:** Change detection, worker tick nudging, capture ingestion, and ledger
  writes are all inlined. The current pipeline suffers race conditions—render
  ticks queue thumbnails but ledger updates lag a frame—so this extraction must
  pay special attention to ordering and worker readiness.

### 7. Scene Snapshot & Mirrors
- **Entry points:** `_refresh_scene_snapshot`, `_start_mirrors_if_needed`,
  `_dims_metadata`, `_default_layer_id`.
- **Shared state:** `_scene_snapshot`, `_bootstrap_snapshot`, `_mirrors_started`.
- **Dependencies:** `snapshot_scene`, `snapshot_render_state`,
  `snapshot_multiscale_state`, `ServerDimsMirror`, `ServerLayerMirror`.
- **Notes:** Snapshot refresh logic blends worker-derived state with ledger reads
  and thumbnail providers, anchoring mirrors directly to server internals.

### 8. Filesystem Helpers
- **Entry points:** `_require_data_root`, `_resolve_data_path`,
  `_resolve_dataset_path`, `_list_directory`.
- **Shared state:** `_data_root`, `_browse_root`.
- **Dependencies:** `discover_dataset_root`, `inspect_zarr_directory`.
- **Notes:** User-facing directory browsing is embedded in the control class,
  mixing security checks with websocket handlers.

### 9. Websocket Hosting & Metrics
- **Entry points:** Inline `state_handler`, `_ingest_pixel`, `_broadcast_loop`,
  `_broadcast_stream_config`, `_update_client_gauges`.
- **Shared state:** `_state_clients`, `_pixel_channel.broadcast.clients`,
  `_metrics_runner`, `metrics`.
- **Dependencies:** `websockets.serve`, `ingest_state`, `ingest_client`,
  `run_channel_loop`, `metrics_server.start_metrics_dashboard`.
- **Notes:** Hosting concerns are tightly coupled to dataset bootstrap; there is
  no separation between serving and control coordination.

## Dominant Call Flows

### Startup Without Dataset
1. `start` initializes logging and locks (`start`, lines 1139-1163).
2. `_enter_idle_state` stops workers, resets ledgers/mirrors/thumbnails, seeds
   bootstrap snapshot (`_enter_idle_state`, lines 881-920).
3. Scene payload built and cached (`_build_scene_payload`, `_cache_scene_history`).
4. Websocket servers launch; pixel loop begins; server blocks forever.

### Dataset Load
1. `_switch_dataset` resolves and probes the dataset, applies bootstrap reducer,
   stores zarr metadata, refreshes snapshots (lines 922-1010).
2. Worker starts on the event loop; thumbnails queue for the default layer.
3. Scene payload cached, stream config marked dirty, baseline broadcast scheduled.

### Thumbnail Update (current behaviour)
1. Worker render tick invokes `_on_render_tick`, generating a signature.
2. `ThumbnailIntent` enqueued and handled on the control loop.
3. `_ingest_worker_thumbnail` normalizes the array, updates ledger metadata,
   and refreshes the scene snapshot.
4. Due to sequencing with worker readiness and ledger update timing, thumbnails
   often lag a frame; this is the key focus area for the upcoming refactor.

## Pain Points Observed
- Configuration, dataset policy, encoder tuning, and stream management live
  alongside control-plane handlers, making unit testing nearly impossible.
- Thumbnail pipeline is tightly coupled to worker lifecycle and ledger updates,
  hiding races under unrelated side effects (encoder resets, baseline churn).
- Worker orchestration and dataset lifecycle have no isolation; entering idle
  state clears mirrors, thumbnails, and pixel queues all in one method.
- Websocket hosting is intertwined with dataset switching, forcing integration
  tests to spin up full servers for simple broadcast assertions.

## Next Steps (High-Level)
1. Extract env/config helpers into a small module to shrink `__init__`.
2. Factor dataset/idle transitions into procedural helpers that operate on
   simple state bags, enabling isolated tests.
3. Separate control-plane broadcasting and resumable history from the server
   coordinator.
4. Design a focused thumbnail pipeline module that can be tested against worker
   tick ordering issues before re-integrating.
5. Gradually slim `EGLHeadlessServer` into a coordinator that wires helpers and
   schedules coroutines, leaving logic in dedicated modules.

This map should guide the incremental refactor: peel off independent surfaces,
add unit tests for the pure helpers, then revisit the thumbnail pipeline with
fresh ordering guarantees.
