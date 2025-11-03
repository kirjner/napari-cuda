# EGL Headless Server Responsibility Map

This note catalogs the surface area of `EGLHeadlessServer`
(`src/napari_cuda/server/app/egl_headless_server.py`) before we begin carving
it into focused modules. Each section lists the public entry points, the shared
state touched, and key collaborators. Understanding these seams will help us
extract testable helpers without introducing a heavy class hierarchy.

For the complementary worker breakdown, see `runtime_worker.md`.

## Lifecycle Overview
- Class construction now defers environment/context work to
  `context.capture_env`, `context.resolve_server_ctx`, and related helpers
  before wiring pixel channel state, mirrors, metrics, worker intents, camera
  queues, and the thumbnail tracker (`__init__`, lines 150-345).
- `start` boots logging, enters idle or loads the configured dataset, starts
  websocket servers, launches the metrics dashboard, kicks off the pixel loop,
  and keeps the async loop alive (`start`, lines 1139-1246).
- Shutdown flows through the `finally` block in `start`, closing servers,
  stopping metrics, and tearing down the worker (`start`, lines 1215-1225).

## State Surfaces (current single-class layout)
- **Environment & context** – `_ctx_env`, `_ctx`, `_data_root`, `_browse_root`,
  and encode policy resolved via `context.py` helpers (lines 150-214).
- **Pixel / encode state** – `Metrics`, `_metrics_runner`, `_pixel_channel`,
  `_pixel_config`, `_param_cache`, `_dump_*`, `_seq`, `_dump_remaining`
  (lines 205-265, 513-540).
- **Worker & runtime hooks** – `_worker_lifecycle`, `_runtime_handle`,
  `_worker_intents`, `_camera_queue`, `_camera_command_seq`,
  `_state_lock`, `_bootstrap_snapshot` (lines 233-354).
- **Mirrors & control-plane** – `_dims_mirror`, `_layer_mirror`, inline mirror
  callbacks, `_state_clients`, `_resumable_store`, `_log_*` flags
  (lines 266-339); publishing helpers now live in `scene_publisher.py`.
- **Scene cache** – `_scene_snapshot`, thumbnail state, default layer helpers
  (lines 320-373, 567-776); capture pipeline logic now routes through
  `thumbnail.py` for queuing, render ticks, and ingestion.
- **Dataset configuration** – `_zarr_*`, volume budget caps, idle bootstrap
  routed through `dataset_lifecycle.py` (lines 176-208, 376-452, 881-1010).
- **Filesystem access** – `_require_data_root`, `_resolve_*`, `_list_directory`
  (lines 1030-1117).
- **Websocket hosting** – `_state_send`, `_broadcast_loop`, `_ingest_pixel`,
  inline `state_handler` definitions (lines 1139-1254, 1333-1365).

## Responsibility Clusters

### 1. Env & Configuration Bootstrap
- **Entry points:** `__init__` delegating to `context.capture_env`,
  `context.resolve_server_ctx`, `context.resolve_data_roots`,
  `context.configure_bitstream_policy`, `context.resolve_volume_caps`,
  `context.resolve_encode_config`.
- **Shared state:** `_ctx_env`, `_ctx`, `_data_root`, `_browse_root`,
  `_volume_max_bytes_cfg`, `_hw_volume_max_bytes`, `_codec_name`, `cfg`.
- **Dependencies:** `context.py`, `get_hw_limits`.
- **Notes:** Env parsing and encode policy are now tested in isolation via the
  new helper module; the constructor simply stores results.

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
- **Entry points:** `_enter_idle_state`, `_switch_dataset`, `_handle_zarr_load`
  now delegate to `dataset_lifecycle.enter_idle_state` and
  `dataset_lifecycle.apply_dataset_bootstrap` via `ServerLifecycleHooks`.
- **Shared state:** `_state_ledger`, `_bootstrap_snapshot`, `_layer_mirror`,
  `_dims_mirror`, `_zarr_*`, `_thumbnail_state`.
- **Dependencies:** `dataset_lifecycle`, `discover_dataset_root`,
  `pull_render_snapshot`, `probe_scene_bootstrap`, worker start/stop.
- **Notes:** Lifecycle transitions are finally testable outside the server
  class; the hooks surface captures the minimal callbacks (worker stop/start,
  mirror resets, thumbnail reset, stream dirtiness) without passing the whole
  server.

### 5. Control-Plane Broadcasting & History
- **Entry points:** `_build_scene_payload`, `_scene_snapshot_json`, `_state_send`
  with resumable history handled by `scene_publisher.cache_scene_history` and
  baseline orchestration by `scene_publisher.broadcast_state_baseline`.
- **Shared state:** `_scene_snapshot`, `_resumable_store`, `_state_clients`,
  `_state_ledger`, `_log_state_traces`.
- **Dependencies:** `scene_publisher`, `build_notify_scene_payload`,
  `safe_send`.
- **Notes:** Publishing now lives in a focused helper; clearing client
  sequencers and scheduling baselines no longer require the monolith.

### 6. Thumbnail Pipeline
- **Entry points:** `_queue_thumbnail_refresh`, `_on_render_tick`,
  `_handle_worker_thumbnails`, `_ingest_worker_thumbnail`, `_layer_thumbnail`
  now delegate to `thumbnail.py` for state tracking, render-tick scheduling,
  and ledger writes.
- **Shared state:** `_thumbnail_state`, `_worker_intents`, `_scene_snapshot`,
  `_state_ledger`.
- **Dependencies:** `thumbnail.py`, `ThumbnailIntent`, worker mark-tick hooks,
  `ServerLayerMirror` (for notify.layers propagation).
- **Notes:** Thumbnail requests are recorded in the helper module, which nudges
  the worker, captures the napari thumbnail, and records both metadata and
  `thumbnail` ledger entries so mirrors can broadcast notify.layers deltas. The
  remaining glue ensures the default layer id is resolved whenever observers
  queue a refresh. Signature comparisons now rely on
  `server.utils.signatures.RenderSignature`, so the render tick, worker intent,
  and ledger ingestion agree on one canonical structure instead of ad-hoc tuples.
  TODO: add integration coverage once the control-plane harness exists so
  regressions in thumbnail refresh sequencing are caught automatically.

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
1. Validate end-to-end thumbnail refresh (render tick → ledger write →
   notify.layers) in integration tests or a lightweight harness to ensure the
   new helper surfaces behave under worker load.
2. Continue peeling camera/level reducers off the coordinator, following the
   hook-based approach to shrink `_handle_worker_level_intents` and related
   mailboxes.
3. Gradually slim `EGLHeadlessServer` into a coordinator that wires helpers and
   schedules coroutines, leaving logic in dedicated modules.

This map should guide the incremental refactor: peel off independent surfaces,
add unit tests for the pure helpers, then revisit the thumbnail pipeline with
fresh ordering guarantees.
