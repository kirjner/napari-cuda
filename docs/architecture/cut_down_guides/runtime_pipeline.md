Runtime Pipeline (Cut-Down)

Mailbox & Signatures
- Flag-on (`NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1`): `RenderUpdateMailbox` is bypassed today. The worker
  watches `scene.main.op_seq`, reads `{view, axes, index, lod, camera, layers}` directly from the ledger,
  computes per-block signatures, and re-applies only the blocks whose signatures changed.
- Flag-off: legacy planner/mailbox pipeline remains in place until Phase 3 cleanup deletes it.

Worker Intent Mailbox (Special-Case Loopback)
- `WorkerIntentMailbox` carries worker → control intents that must pass through reducers to preserve the authoritative ledger path.
  - `LevelSwitchIntent`: emitted by the worker’s level policy (auto multiscale). Server handler `_handle_worker_level_intents` pops the intent and calls `reduce_level_update(...)`, then bumps `scene.main.op_seq` so the worker re-pulls the ledger. This keeps the ledger as the sole source of truth for mode/level changes.
  - `ThumbnailCapture`: carries raw pixel arrays for thumbnails; control loop dedupes by content token and persists via `reduce_thumbnail_capture` if needed.

Drain & Apply (Pixel Channel Only)
- Worker loop per tick:
  1. Observe `scene.main.op_seq` change (no render mailbox needed when the flag is on).
  2. Pull the latest `{view, axes, index, lod, camera, layers}` scopes via `fetch_scene_blocks(...)`.
  3. Compute per-block signatures to determine what changed.
  4. Call the existing apply helpers (`apply_render_snapshot`, layer drains, camera metadata) so the napari
     viewer stays in sync—no planner and no viewport cache mirrors.
  5. Emit pose/level intents back through reducers (camera pose callback +
     worker intent mailbox).
  6. Record the applied signatures locally for the next tick.

Performance Targets
- Camera-only path must avoid dims apply; per-block signatures ensure
  dims/layers never reapply when unchanged.

Note: notify.* is emitted immediately after ledger commits; the worker path
consumes the ledger directly and never re-emits notify payloads.
