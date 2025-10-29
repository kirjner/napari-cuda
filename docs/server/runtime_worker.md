# EGL Render Worker Responsibility Map

This note captures the full surface area of `EGLRendererWorker` before we start
splitting the module. Every cluster below lists the public entry points, the
shared state they read or mutate, and the collaborating modules that depend on
that behaviour.

## Lifecycle Overview
- Module import side effects prime EGL headless operation by setting
  `PYOPENGL_PLATFORM=egl`, `QT_QPA_PLATFORM=offscreen`, and `XDG_RUNTIME_DIR`
  defaults (`src/napari_cuda/server/runtime/worker/egl.py:32-34`).
- `start_worker` (`runtime/worker/lifecycle.py:45`) constructs the worker, then
  drives `_init_cuda`, `_init_vispy_scene`, `_init_egl`, `_init_capture`,
  `_init_cuda_interop`, `_init_encoder`, and seeds the first snapshot.
- The render thread loops over `render_updates.consume_render_snapshot`, then
  `capture_and_encode_packet`; the latter calls `render_tick`, which runs
  `run_render_tick` (`runtime/worker/loop.py:9`) with worker callbacks.
- Shutdown runs `cleanup`, reclaiming CUDA/GL resources, encoder state, VisPy
  canvas, and the EGL context.

## State Surfaces (current single-class layout)
- **Dimensions & configuration** – Immutable constructor inputs (`width`,
  `height`, `fps`, volume defaults) plus policy name overrides stored on the
  instance (`src/napari_cuda/server/runtime/worker/egl.py:169`).
- **Render resources** – `_egl`, `_capture`, `_encoder`, `_enc_lock`,
  `_enc_input_fmt`, and CUDA handles initialised in `_init_render_components`
  and booted by `_init_cuda*` methods (`egl.py:787-1297`).
- **Viewport & ledger** – `_viewport_state`, `_viewport_runner`,
  `_render_mailbox`, `_ledger`, `_applied_versions`, `_last_*` snapshot
  signatures, and ndisplay metadata caches (`egl.py:205-2330`).
- **Queues & callbacks** – `_camera_queue`, `_camera_pose_callback`,
  `_level_intent_callback`, `_state_lock`, plus `RenderUpdateMailbox`
  interactions (`egl.py:205`, `egl.py:816-852`).
- **Debug & policy knobs** – `_debug_policy`, `_debug_config`, `_debug`,
  `_log_layer_debug`, `_level_threshold_*`, ROI flags, oversampling tables, raw
  dump budget (`egl.py:200-872`, `egl.py:934-1087`).
- **Telemetry & interim state** – `_layer_logger`, `_switch_logger`,
  `_render_tick_required`, `_render_loop_started`, timing hints, per-mode pose
  caches, `_viewport_runner` snapshots (`egl.py:811-1940`).

## Responsibility Clusters

### 1. Bootstrap & Resource Setup
- **Entry points:** `_configure_animation`, `_init_render_components`,
  `_init_scene_state`, `_init_locks`, `_configure_debug_flags`,
  `_configure_policy`, `_configure_roi_settings`, `_configure_budget_limits`,
  `_init_cuda`, `_init_egl`, `_init_capture`, `_init_cuda_interop`,
  `_init_encoder`, `_log_debug_policy_once`.
- **Shared state:** All render resources, locks, debug flags, policy thresholds,
  ROI settings, and ledger attachment.
- **Dependencies:** `CaptureFacade` (`rendering/capture.py`), `EglContext`,
  `Encoder`, `RenderUpdateMailbox`, `ServerCtx`, debug policy dataclasses, and
  `LayerAssignmentLogger` / `LevelSwitchLogger`.
- **Notes for extraction:** These helpers can be grouped into a
  `WorkerResources` façade that owns GL/CUDA/capture/encoder lifetimes and the
  associated locks. The remaining policy and ROI configuration reads directly
  from `ServerCtx`/`LevelPolicySettings` and can move with the level-selection
  logic.

### 2. Scene Source & Viewer Wiring
- **Entry points:** `_create_scene_source`, `_ensure_scene_source`,
  `_init_vispy_scene`, `_init_viewer_scene`, `_set_dims_range_for_level`,
  `_register_*_visual`, `_ensure_*_visual`, `_volume_shape_for_view`,
  `_volume_world_extents`.
- **Shared state:** `_scene_source`, `_viewer`, `_plane_visual_handle`,
  `_volume_visual_handle`, `_zarr_*` metadata, `_data_wh`, `_data_d`,
  `_viewport_state.volume.world_extents`.
- **Dependencies:** `ViewerBuilder` (`rendering/viewer_builder.py`),
  `ZarrSceneSource` (`data/zarr_source.py`), `ViewerModel`, VisPy cameras, the
  `_VisualHandle` wrapper (`egl.py:111-149`) that tracks node attachment, and
  plane ROI helpers (`runtime/data/roi_math.py`, `runtime/data/scene_types.py`).
- **Notes:** Anything touching VisPy canvas or napari `ViewerModel` should live
  together so we can preserve the adapter contract. Scene-source helpers are
  also used by level policy evaluation (`_ensure_scene_source`,
  `_set_dims_range_for_level`).

### 3. Camera Lifecycle & Viewport Pose
- **Entry points:** `_bootstrap_camera_pose`, `_configure_camera_for_mode`,
  `_frame_volume_camera`, `_enter_volume_mode`, `_exit_volume_mode`,
  `_apply_camera_reset`, `_emit_current_camera_pose`, `_emit_pose_from_camera`,
  `_pose_from_camera`, `_snapshot_camera_pose`, `_current_panzoom_rect`,
  `render_updates.apply_viewport_state_snapshot`, `process_camera_deltas`,
  `_drain_camera_ops_then_scene_updates`, `_record_zoom_hint`,
  `render_updates.drain_scene_updates`.
- **Shared state:** `_viewport_state`, `_viewport_runner`, `_camera_queue`,
  `_camera_pose_callback`, `_last_plane_pose`, `_last_volume_pose`,
  `_pose_seq`, `_max_camera_command_seq`, `_render_mailbox` zoom hints.
- **Dependencies:** `CameraCommandQueue`, camera controller helpers
  (`runtime/camera/controller.py`), `ViewportRunner`, ledger accessors,
  `napari.components.viewer_model.ViewerModel`, VisPy cameras.
- **Notes:** Camera helpers straddle both the render thread (actual camera
  mutation) and controller notifications (callbacks back to asyncio). They are
  natural candidates for a dedicated `viewer.py` module that takes explicit
  hooks for mailbox updates and ledger reads.

### 4. Snapshot Ingestion & Ledger Coordination
- **Entry points:** `enqueue_update`,
  `render_updates.consume_render_snapshot`,
  `render_updates.normalize_scene_state`,
  `render_updates.record_snapshot_versions`,
  `render_updates.extract_layer_changes`,
  `render_updates.apply_viewport_state_snapshot`,
  `render_updates.drain_scene_updates`,
  `_apply_dims_from_snapshot`, `_update_z_index_from_snapshot`,
  `snapshot_dims_metadata`, `_set_dims_range_for_level`, `_record_zoom_hint`.
- **Shared state:** `_render_mailbox`, `_viewport_state`, `_viewport_runner`,
  `_applied_versions`, `_last_snapshot_signature`, `_last_dims_signature`,
  `_z_index`, `_data_wh`, `_ledger`, ledger access helpers (`runtime.core.ledger_access`).
- **Dependencies:** `RenderUpdateMailbox`, `RenderLedgerSnapshot` (from
  `runtime.core.snapshot_build`), `runtime.worker.snapshots.*`,
  `runtime.worker.render_updates`, `viewport.updates`,
  ledger interfaces in `ServerStateLedger`.
- **Notes:** This block is where external updates enter the worker. The
  render-thread snapshot helpers now live under `runtime/worker/snapshots/`,
  while shared ledger readers reside in `runtime/core/ledger_access.py`.

### 5. Level Selection, ROI, and Policy Evaluation
- **Entry points:** `_configure_policy`, `set_policy`,
  `_build_policy_context`, `_estimate_level_bytes`, `_volume_budget_allows`,
  `_slice_budget_allows`, `_resolve_volume_intent_level`, `_apply_volume_level`,
  `_apply_slice_level`, `_format_level_roi`, `_aligned_roi_signature`,
  `_volume_world_extents`, `_evaluate_level_policy`, `_mark_render_tick_needed`,
  `_mark_render_tick_complete`, `_mark_render_loop_started`,
  `request_multiscale_level`, `_enter_volume_mode`, `_exit_volume_mode`,
  `_mark_render_tick_needed`, `_run_viewport_tick`.
- **Shared state:** Policy thresholds, `_viewport_runner`, `_viewport_state`,
  `_level_policy_suppressed`, `_last_level_switch_ts`,
  `_oversampling_thresholds`, `_oversampling_hysteresis`,
  `_roi_edge_threshold`, `_roi_align_chunks`, `_roi_pad_chunks`,
  `_roi_ensure_contains_viewport`, `_slice_max_bytes`, `_volume_*` limits,
  `_hw_limits`, `_log_layer_debug`, `_layer_logger`, `_switch_logger`.
- **Dependencies:** `level_policy` module (`runtime/worker/level_policy.py`), ROI helpers,
  `ViewportRunner`, `LevelSwitchIntent`, `lod.LevelDecision` contexts,
  `ServerStateLedger` (for step/order/axes hints), `viewport_roi_for_level`,
  `apply_plane_metadata`, `apply_volume_metadata`.
- **Notes:** These helpers blend policy selection with ROI alignment and level
  bookkeeping. They depend on both the scene source and the viewport runner,
  which makes them a good candidate for a `levels.py` module exposing pure
  functions that accept a worker façade.

### 6. Render Loop & Capture/Encode
- **Entry points:** `render_tick`, `_drain_camera_ops_then_scene_updates`,
  `_run_viewport_tick`, `capture_and_encode_packet`, `_capture_blit_gpu_ns`,
  `_mark_render_tick_needed`, `_mark_render_tick_complete`,
  `_mark_render_loop_started`.
- **Shared state:** `_capture`, `_encoder`, `_enc_lock`, `_render_tick_required`,
  `_render_loop_started`, `_animate`, `_animate_dps`, `_anim_start`,
  `_user_interaction_seen`, `_last_interaction_ts`, `_viewport_runner`
  checkpoints.
- **Dependencies:** `run_render_tick`, animation helper
  (`runtime/camera/animator.py`), `CaptureFacade` pipeline, `encode_frame`,
  `CameraCommandQueue`, `ViewportRunner.plan_tick` (indirect via
  `viewport.updates`), debug dumper.
- **Notes:** `capture_and_encode_packet` glues together GPU capture, encoder
  access, and metrics collection. Splitting this code requires preserving the
  locking discipline (`_enc_lock`) and the ordering guarantees around
  `_drain_camera_ops_then_scene_updates`.

### 7. Debugging, Metrics & Misc Utilities
- **Entry points:** `_log_debug_policy_once`, `snapshot_dims_metadata`,
  `_record_zoom_hint`, `_capture_blit_gpu_ns`, `force_idr`, `_request_encoder_idr`,
  `reset_encoder`.
- **Shared state:** `_debug`, `_debug_config`, `_debug_policy_logged`,
  `_raw_dump_budget`, `_render_mailbox` zoom hints, encoder format caches,
  `_applied_versions`.
- **Dependencies:** `DebugDumper`, encoder debug policy flags, capture pipeline.
- **Notes:** These helpers are invoked conditionally throughout the worker. When
  extracting modules, ensure they stay close to the resource management code so
  they continue to guard encoder and capture interactions.

### 8. Cleanup & Failure Handling
- **Entry points:** `cleanup`, `_mark_render_tick_complete` (on failure),
  `_mark_render_tick_needed` (retries), ledger accessor helpers (to resume),
  `_ensure_scene_source`.
- **Shared state:** Entire render resource surface plus ledger attachments.
- **Dependencies:** `CaptureFacade.cleanup`, CUDA context management, EGL
  teardown, VisPy canvas, debug dumper, logger.
- **Notes:** Cleanup is mirrored in the worker lifecycle `finally` block. Moving
  resource teardown into a dedicated façade keeps shutdown symmetrical with
  bootstrap.

## Dominant Call Flows

### Boot Sequence
1. Control thread instantiates `EGLRendererWorker`.
2. Worker thread calls `_init_cuda`, `_init_vispy_scene`, `_init_egl`,
   `_init_capture`, `_init_cuda_interop`, `_init_encoder`.
3. Initial snapshot pulled via `pull_render_snapshot`, handed to
   `render_updates.consume_render_snapshot`, then
   `render_updates.drain_scene_updates` hydrates the mailbox and viewport runner.

### Per-frame Loop
1. Controller snapshot fetched (`pull_render_snapshot`) and passed to
   `render_updates.consume_render_snapshot`.
2. `capture_and_encode_packet` performs:
   - `_drain_camera_ops_then_scene_updates` → `_process_camera_deltas` →
     `_viewport_runner.ingest_camera_deltas` and `render_updates.drain_scene_updates`.
   - `render_tick` → `run_render_tick` to render the VisPy canvas and mark tick
     completion.
   - `encode_frame` to map, copy, convert, and encode GPU output.
3. Metrics recorded; `pixel_channel` pipeline enqueues frames, keyframe config
   published when needed.

### Level Selection Flow
1. Incoming snapshot updates `_viewport_runner` intent and `_render_mailbox`.
2. Zoom hints from camera deltas feed `_evaluate_level_policy`.
3. Policy decision builds `lod.LevelContext`, applies metadata via
   `apply_plane_metadata`/`apply_volume_metadata`, records `LevelSwitchIntent`,
   and calls `_level_intent_callback`.
4. `_viewport_runner.request_level` tracks pending confirmation and pose events.

### Camera Delta Flow
1. `CameraCommandQueue.pop_all()` retrieves commands during
   `_drain_camera_ops_then_scene_updates`.
2. `_process_camera_deltas` (from `camera/controller.py`) mutates VisPy cameras,
   marks render ticks, and emits policy triggers.
3. `_viewport_runner.update_camera_rect` stores the rect for ROI planning;
   `_emit_current_camera_pose` notifies the ledger when volumes change.

## Extraction Guardrails
- Preserve `_state_lock` usage: `viewport.updates.drain_render_state` and
  `runtime/core/scene_setup.py` expect a re-entrant lock guarding worker state.
- Maintain `RenderUpdateMailbox` semantics: the mailbox currently coalesces
  viewport state and scene snapshots; extracted helpers should never bypass its
  `set_scene_state` / `drain` contract.
- Keep `ViewportRunner` mutations centralized: runner state must be updated in
  tandem with `_viewport_state` to keep level confirmations consistent.
- Ledger accessor helpers (`_ledger_*`) are consumed by snapshot bootstrap and
  level metadata; move them wholesale with the snapshot module to avoid partial
  imports.
- Encoder lock discipline (`_enc_lock`) protects NVENC calls and reset/IDR
  requests; new modules must continue to acquire the lock before touching the
  encoder.

This map should serve as the baseline for carving `egl.py` into focused modules
without losing sight of the shared state or cross-thread contracts.
