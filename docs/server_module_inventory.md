# Server Module Inventory

Deep-dive catalogue of the server-side code base (Oct 2025). Focus is on the
streaming/control path analogous to the client inventory: identify modules,
responsibilities, sizes, and hotspots that complicate maintenance.

## Top-Level Server Modules

| Module | Lines | Purpose | Notes |
| --- | ---:| --- | --- |
| `server/egl_headless_server.py` | 753 | Entry point for running the headless GL renderer and control loop. | Handles CLI parsing, config loading, worker lifecycle. Mixed concerns (CLI + setup). |
| `server/config.py` | 539 | Environment/configuration context for server components. | Centralized but verbose; defines dataclasses for GPU/encoder limits, loop params, logging. |
| `server/render_worker.py` | 1 208 | Core render worker: GL context, capture, encoder integration. | Large file orchestrating capture pipelines, multi-threading, metrics. |
| `server/worker_runtime.py` | 537 | Manages background worker threads/process state. | Coordinates interop with render loop, pixel channel, control callbacks. |
| `server/worker_lifecycle.py` | 444 | Hooks to start/stop the worker, manage callbacks into render worker. |
| `server/layer_manager.py` | 566 | Maintains layer state on the server, builds SceneSnapshot blocks. | Emits control updates and notifies worker; tight coupling with napari viewer. |
| `server/scene_state_applier.py` | 444 | Applies control updates/incoming state to napari viewer/layer objects. |
| `server/control/control_channel_server.py` | 3 012 | WebSocket control server (state.update handling, resume tokens, heartbeats). | Single largest file; handles handshake, acknowledge queue, history store, command execution. |

Other key single-file modules:
- `render_mailbox.py` (140) – GPU frame hand-off queue.
- `roi.py` / `roi_applier.py` (502 / 123) – region-of-interest selection logic.
- `logging_policy.py` (257) – log suppression policy per stream.

## Control Package (`server/control`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `control_channel_server.py` | 3 012 | Accepts state channel connections, parses greenfield frames, issues acks and notify.* broadcasts. Contains embedded history store, command catalogue, resume logic. |
| `state_update_engine.py` | 505 | Implements reducers for state.update (dims, camera, settings, layers). Parses payloads, applies to server state, issues acks. |
| `resumable_history_store.py` | 249 | Stores mutable history for resumable topics (scene, layers, stream). |
| `command_registry.py` | 54 | Maps command names to callables (keyframe, etc.). |
| `control_payload_builder.py` | 194 | Builds ack/notify payloads from server state. |

Observations:
- `control_channel_server.py` is a kitchen sink (websocket server, message parsing, ack emission, state routing). Needs decomposition.
- `state_update_engine.py` mirrors client `state_update_actions.py`, but server-specific (applies to napari objects via `scene_state_applier`).

## Pixel / Rendering Modules

| Module | Lines | Purpose |
| --- | ---:| --- |
| `pixel_channel_server.py` | 312 | WebSocket for streaming pixel frames to clients. |
| `pixel_broadcaster.py` | 244 | Multi-client broadcaster for encoded frames. |
| `render_worker.py` | 1 208 | See above. |
| `rendering/*` | ~1 400 total | GL capture, CUDA interop, encoder wrappers (NVENC), viewer builder. |
| `bitstream.py` | 361 | AVC Annex-B/avcC bitstream helpers. |
| `capture.py` | 219 | Captures frames from OpenGL to GPU buffers. |
| `render_loop.py` | 33 | Legacy render loop stub (mostly superseded). |
| `render_mailbox.py` | 140 | Thread-safe mailbox for GPU frames. |

## Scene/Layer Management

| Module | Lines | Purpose |
| --- | ---:| --- |
| `server_scene.py` | 297 | Coordinates viewer, layer manager, and policies. Builds snapshots. |
| `layer_manager.py` | 566 | Maintains layer blocks, responds to notify.scene requests, handles deltas. |
| `scene_state_applier.py` | 444 | Applies state updates (from control channel) to napari viewer + layer manager. |
| `scene_state.py` | 23 | Minimal structure describing current scene metadata. |
| `plane_restore_state.py` | 23 | Stores slice plane state during transitions. |
| `zarr_source.py` | 538 | Access to Zarr-backed data sources (mip levels, caching). |

## Supporting Modules

- `camera_controller.py` (287), `camera_ops.py` (167), `camera_animator.py` (37) – server control of napari camera.
- `lod.py` (667), `level_budget.py` (74), `level_logging.py` (77) – level-of-detail management.
- `policy.py` (73), `policy_metrics.py` (67) – streaming policy decisions (quality, keyframes).
- `patterns.py` (206), `display_mode.py` (175) – UI/layout control hints.
- `metrics_core.py` (143), `metrics_server.py` (80) – capture metrics and expose web dashboard (with `dash_dashboard.py`).
- `debug_tools.py` (176) – utilities for dumping renderer state.
- `worker_notifications.py` (56) – send structured updates to clients about worker state.
- `hw_limits.py` (64) – GPU capability probes.

## Tests (`server/_tests`)

- Coverage for bitstream, camera, config, layer manager, LOD, logging policy, pixel channel, ROI, scene state applier, server state updates, worker lifecycle, Zarr source, etc. Largest test is `test_state_channel_updates.py` (964 lines) exercising greenfield protocol end-to-end.

## Pain Points / Observations

1. **Control channel size** – `control_channel_server.py` at 3k lines handles transport, routing, acking, history. Needs decomposition (e.g. websocket server vs message router vs resume/history).
2. **Duplicated logic** – Many helpers (state reducers, ack builders) mirror client code. Potential for shared utilities or protocol modules.
3. **Scene/Layer coupling** – Layer manager, scene state applier, server scene, and viewer manager each manipulate napari objects. Responsibilities overlap; projection logic could be centralized.
4. **Legacy artifacts** – Files like `render_loop.py`, `pixel_channel.py` (4 lines) hint at old architecture left in place to avoid breaking imports. Should be cleaned.
5. **Configuration sprawl** – Config flows from `config.py` into many modules via environment variables, sometimes read repeatedly (dash dashboard, metrics). Documenting a single config source would help.
6. **Concurrency complexity** – render worker, pixel channel, control server each spawn threads/loops; `render_worker.py` alone orchestrates GL, CUDA, NVENC, ROI, metrics. Hard to unit test.
7. **Telemetry** – `metrics_server.py`, `dash_dashboard.py`, `metrics_core.py` form a monitoring stack; architecture doc should decide whether this remains tightly coupled to render worker or separated.
8. **Testing** – Integration tests exist (`test_worker_integration`, `test_state_channel_updates`), but there’s no automated coverage for new snapshot-based layer manager flows except `test_layer_manager`. Worth verifying after refactor.

## Immediate Hygiene Targets

- Split `control_channel_server.py` into smaller modules: transport (websocket), session management (resume/history), message handling (state.update vs notify), command registry.
- Extract projection/apply logic from `scene_state_applier.py` into reusable components (align with upcoming client projections).
- Consolidate config reading: single config object passed through rather than environment reads everywhere.
- Remove or clearly archive legacy stubs (`render_loop.py`, `pixel_channel.py`).
- Document concurrency model (which threads own GL, control, pixel) to prepare for multi-client control.

This inventory complements the client map and should inform the upcoming architecture doc and cleanup plan.
