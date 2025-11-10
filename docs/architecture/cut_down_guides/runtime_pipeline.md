Runtime Pipeline (Cut-Down)

Mailbox & Signatures
- `RenderUpdateMailbox.set_scene_state(state)` coalesces snapshots per op_seq.
- `RenderUpdateMailbox.update_state_signature(state)` (current) hashes entire scene → change triggers full apply.
- Lean target: per-block signatures (dims, camera, layers, active) and dedicated update methods.

Worker Intent Mailbox (Special-Case Loopback)
- `WorkerIntentMailbox` carries worker → control intents that must pass through reducers to preserve the authoritative ledger path.
  - `LevelSwitchIntent`: emitted by the worker’s level policy (e.g., auto multiscale). Server handler `_handle_worker_level_intents` pops the intent and calls `reduce_level_update(...)`, then pulls a fresh `RenderLedgerSnapshot` and enqueues it to the mailbox. This ensures the ledger remains the single source of truth for mode/level changes before render consumes the update.
  - `ThumbnailCapture`: carries raw pixel arrays for thumbnails; control loop dedupes by content token and persists via `reduce_thumbnail_capture` if needed.

Drain & Apply (Pixel Channel Only)
- `drain_scene_updates(worker)`:
  - Reject stale op_seq snapshots.
  - Compute layer_changes by version; apply only changed layer visuals.
  - If mode changed or signature changed → `apply_render_snapshot`.
  - Update viewport_state; evaluate level policy in 2D only when not suppressed.

Apply
- `apply_render_snapshot` (current) suppresses fit callbacks; calls:
  - `apply_dims_from_snapshot` → update viewer dims.
  - Update z-index from snapshot.
  - Planner builds and applies viewport plan (ROI, level, camera pose) + layer data.

Lean target
- Split apply into:
  - `apply_dims_block` (only when dims sig changed),
  - `apply_camera_block` (only when camera sig changed),
  - `apply_layers_block` (only when layer sig changed).
- Planner consumes the just-applied dims/camera blocks; no ledger reads.

Performance Targets
- Camera-only path must avoid dims apply; end-to-end per tick < 1ms p50.

Note: notify.* is not produced from this path; it is emitted directly after ledger commits on the state channel.
