# Server Streamlining & Renaming Plan

The goal of this phase is to make the server architecture self-explanatory by tightening the data
flow, reducing redundant state mirrors, and adopting names that reflect each module’s purpose. This
plan captures the remaining work so we can execute the refactor in deliberate, testable slices.

## Current Pain Points

- `state_channel_handler.py` mixes websocket plumbing, `state.update` handling, and legacy intent
  fallbacks with broadcast scheduling.
- `server_scene_queue.py` adds an extra layer between the dispatcher and worker just to coalesce
  updates.
- `render_worker.py` (formerly `egl_worker.py`) still owns viewer bootstrapping, mailbox draining, capture orchestration, and
  visual wiring in a single monolith.
- `rendering/viewer_builder.py` serves only as a one-time bootstrap; keeping it separate obscures the
  lifecycle.
- Layer control values continue to be mirrored into `extras`; clients should rely solely on the
  `controls` map.
- Architecture documentation still references legacy names; there is no single overview of the new
  flow.

## Target Architecture (Summary)

1. The **state channel handler** accepts `state.update` payloads, normalises them, mutates
   `ServerSceneData`, and
   hands each delta to the render worker’s update mailbox while broadcasting the canonical scene spec
   (`controls` included).
2. The **render worker** owns the napari viewer/VisPy visual, drains its mailbox, applies deltas via
   `SceneStateApplier`, renders, and pushes frames through capture/encode to the pixel channel.
3. A one-time **viewer builder** constructs the ViewerModel + VisPy node during worker start-up.
4. Capture/encode and pixel broadcasting stay as dedicated helpers, but naming/docstrings should make
   their roles explicit.

## Planned Work

### 1. State Channel Simplification
- [x] Rename `server_scene_control.py` → `state_channel_handler.py`; keep module-level docstring.
- [x] Replace the open-coded intent branching with a table-driven dispatcher that delegates to
      `server_scene_intents`.
- [x] Record deltas in `ServerSceneData` then call `render_worker.enqueue_update(delta)` directly.
- [x] Ensure acknowledgements/broadcasts are derived from the canonical store (no mirror dicts).
- [x] Track per-control client sequences + server acknowledgements so reconnect baselines and
      `state.update` broadcasts echo `{server_seq, source_client_seq, source_client_id}` metadata.
- [x] Introduce `state.update` handling, advertise the capability in reconnect payloads, and remove
      the legacy intent path once the client flips (complete).

### 2. Scene Update Mailbox
- [x] Move the coalescing logic from `server_scene_queue.py` into `render_mailbox.py` owned by
      the worker (API: `enqueue`, `drain`).
- [x] Update unit tests to cover the mailbox in isolation.
- [x] Delete the old queue module once callers were updated.

### 3. Render Worker Refactor
- [x] Rename `egl_worker.py` → `render_worker.py`; add a module docstring describing responsibilities.
- [ ] Inline viewer bootstrap at start-up (call the builder) and drop `_sync_visual_state` helpers.
- [ ] Adopt the new mailbox, exposing `enqueue_update` and `drain_updates` methods.
- [ ] Move capture/encode orchestration into a dedicated helper (`capture_encode.py`) only if we need
      swappable backends; otherwise keep it private to the worker.

### 4. Viewer Builder
- [x] Rename `rendering/adapter_scene.py` → `viewer_builder.py`; document that it runs once and
      returns `(canvas, viewbox, viewer)`.
- [ ] Slim the module to the minimal napari/VisPy bootstrap; remove event re-sync if not required.

### 5. Controls Map & Extras
- [ ] Remove control-key mirroring from `LayerSpec.extras`; rely on `LayerSpec.controls` only.
- [ ] Update `server_scene_spec.py` and `layer_manager.py` to write `controls` exclusively.
- [ ] Adjust client bridge once the new payload is in place (tracked separately).

### 6. Documentation & Tests
- [ ] Expand `docs/server_architecture.md` with the new module names once refactors land.
- [ ] Update mermaid diagram arrows and node labels to reflect renamed modules.
- [ ] Add regression tests:
  - `test_server_scene_intents` already covers colormap normalisation (keep).
  - Add state/worker integration test verifying mailbox coalesces multiple intents before render tick.
  - Add failure-path test ensuring invalid colormap is logged but does not crash worker.

### 7. Naming Cleanup
- [ ] Rename `SceneStateApplier` → `WorkerSceneApplier` (or similar) to clarify ownership.
- [ ] Rename `ViewerSceneManager` to emphasise it is the read-side builder (`scene_spec_builder`?).
- [ ] Update imports, docstrings, and typing aliases accordingly.

## Execution Notes

- Prefer incremental PRs (one bullet group at a time) with focused tests.
- While refactoring, keep the authoritative scene store as the only source of truth; worker should
  never mutate the payloads that the state loop will broadcast without feeding the change back.
- Once the client consumes the `controls` map, remove any remaining compatibility shims.
- After the rename wave, run `uv run pytest` for the server module subset and update the docs/diagrams
  in the same change set.

## Tracking

| Task                                          | Owner | Status | Notes |
|-----------------------------------------------|-------|--------|-------|
| Rename dispatcher + table-driven routing      |       | Done   | Completed rename + dispatcher refactor |
| Integrate render mailbox into worker          |       | Done   | render_mailbox replaces ServerSceneQueue |
| Worker rename & bootstrap inline              |       | To Do  |       |
| Viewer builder rename/trim                    |       | To Do  |       |
| Controls-only payload                         |       | To Do  |       |
| Docs/diagram refresh                          |       | To Do  |       |
| Additional regression tests                   |       | To Do  |       |
