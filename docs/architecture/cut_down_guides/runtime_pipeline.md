Runtime Pipeline (Cut-Down)

Mailbox & Signatures
- Final state: RenderUpdateMailbox disappears. The worker watches `scene.main.op_seq`,
  reads `{view, axes, index, lod, camera, layers}` directly from the ledger, and
  re-applies only the blocks whose signatures changed.

Worker Intent Mailbox (Special-Case Loopback)
- `WorkerIntentMailbox` carries worker → control intents that must pass through reducers to preserve the authoritative ledger path.
  - `LevelSwitchIntent`: emitted by the worker’s level policy (auto multiscale). Server handler `_handle_worker_level_intents` pops the intent and calls `reduce_level_update(...)`, then bumps `scene.main.op_seq` so the worker re-pulls the ledger. This keeps the ledger as the sole source of truth for mode/level changes.
  - `ThumbnailCapture`: carries raw pixel arrays for thumbnails; control loop dedupes by content token and persists via `reduce_thumbnail_capture` if needed.

Drain & Apply (Pixel Channel Only)
- Worker loop per tick:
  1. Observe `scene.main.op_seq` change (no mailbox).
  2. Pull the latest `{view, axes, index, lod, camera, layers}` scopes.
  3. Compute per-block signatures to determine what changed.
  4. Call `apply_dims_block`, `apply_camera_block`, `apply_layers_block`
     directly—no planner, no cached viewport state.
  5. Emit pose/level intents back through reducers (camera pose callback +
     worker intent mailbox).
  6. Record the applied signatures locally for the next tick.

Performance Targets
- Camera-only path must avoid dims apply; per-block signatures ensure
  dims/layers never reapply when unchanged.

Note: notify.* is emitted immediately after ledger commits; the worker path
consumes the ledger directly and never re-emits notify payloads.
