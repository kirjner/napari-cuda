# Server Module Inventory

Deep-dive catalogue of the server-side code base (Oct 2025). The tree now
matches the target layout: entry points under `server/app`, authoritative
state in `server/state`, data helpers in `server/data`, rendering pipeline in
`server/rendering`, transport in `server/control`, and colocated tests in
`server/tests`.

## App Package (`server/app`)

| Module | Lines | Purpose | Notes |
| --- | ---:| --- | --- |
| `app/egl_headless_server.py` | 753 | Headless EGL server bootstrap (CLI, config load, worker orchestration). | Still mixes CLI parsing, asyncio loop wiring, and worker lifecycle glue. |
| `app/config.py` | 539 | Resolves environment/config values into dataclasses (`ServerCtx`, `EncoderRuntime`, policy settings). | Central source of truth for downstream packages. |
| `app/dash_dashboard.py` | 389 | Optional Dash monitoring UI. | Reads metrics snapshots and exposes simple controls. |
| `app/metrics_core.py` | 143 | Shared metrics containers (frame timings, counters). | Consumed by `egl_headless_server`, render worker, and dashboard. |
| `app/metrics_server.py` | 80 | Writes metrics/policy snapshots to disk; optional HTTP hooks. |
| `app/experiments/*` | ~200 | Legacy benchmarking tools (baseline capture, vispy spike). | Consider archiving once profiling story stabilises. |

## Control Package (`server/control`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `control_channel_server.py` | 3 012 | WebSocket server handling state.update, notify.*, resume tokens, command execution. |
| `state_update_engine.py` | 505 | Reducers that apply state.update payloads to server state/scene. |
| `resumable_history_store.py` | 249 | Maintains topic history for resume/replay. |
| `control_payload_builder.py` | 194 | Builds ack/notify payloads from `ServerScene`. |
| `command_registry.py` | 54 | Maps greenfield command names to callables (e.g., `napari.pixel.request_keyframe`). |

## State Package (`server/state`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `layer_manager.py` | 566 | Maintains layer blocks, emits scene snapshots/deltas. |
| `scene_state_applier.py` | 444 | Applies reducer outputs to napari viewer/layer manager. |
| `camera_controller.py` | 287 | Interprets camera intents and delegates to `camera_ops`. |
| `server_scene.py` | 297 | Aggregates viewer state, metrics snapshots, policy metadata. |
| `scene_types.py` | 87 | Typed helpers describing scene payload structures. |
| `camera_ops.py` | 167 | Low-level napari camera mutations. |
| `camera_animator.py` | 37 | Optional auto-rotation animation hooks. |
| `plane_restore_state.py` | 23 | Stores plane widget state during transitions. |
| `scene_state.py` | 23 | Lightweight container for dims/camera metadata. |
| `server_state_updates.py` | 4 | Shims for legacy imports (to be removed once callsites migrate). |

## Data Package (`server/data`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `zarr_source.py` | 538 | Zarr-backed multiscale data access, caching, LOD descriptors. |
| `lod.py` | 667 | Level-of-detail calculations, oversampling, heuristics. |
| `roi.py` / `roi_applier.py` | 502 / 123 | ROI selection and application to layer data. |
| `logging_policy.py` | 257 | Controls stream logging verbosity based on events. |
| `policy.py` | 73 | Level selection policy (oversampling thresholds, hysteresis). |
| `level_budget.py` | 74 | Tracks level budgets for throttling. |
| `level_logging.py` | 77 | Debug logging helpers for LOD decisions. |
| `hw_limits.py` | 64 | GPU capability discovery. |
| `alignment/__init__.py` | 97 | Optional Allen CCF alignment profile loader (extra dependency). |

## Rendering Package (`server/rendering`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `runtime/egl_worker.py` | 1 208 | Core EGL-backed worker orchestrating GL capture, NVENC encoding, policy metrics. |
| `runtime/worker_runtime.py` | 537 | Manages worker thread/process state. |
| `runtime/worker_lifecycle.py` | 444 | Start/stop hooks tying worker, control callbacks, metrics. |
| `bitstream.py` | 361 | AVC Annex-B/avcC helpers, bitstream dumping. |
| `capture.py` | 219 | Captures frames from OpenGL into GPU buffers. |
| `encoder.py` | 389 | Encapsulates NVENC configuration and frame submission. |
| `display_mode.py` | 175 | Applies napari ndisplay transitions (2D↔3D). |
| `debug_tools.py` | 176 | Dumps renderer state / GL buffers for diagnostics. |
| `patterns.py` | 206 | Generates test patterns for validation runs. |
| `runtime/runtime_mailbox.py` | 140 | Thread-safe mailbox for frame hand-off. |
| `runtime/runtime_loop.py` | 33 | Legacy helper (largely superseded by worker). |
| `frame_pipeline.py` / `gl_capture.py` | 152 / 177 | Capture helpers (GPU copy, staging). |
| `vispy_intercept.py` | 267 | Hooks vispy GL calls to integrate with the worker. |
| `viewer_builder.py` | 347 | Constructs napari viewer for server-side rendering. |
| `policy_metrics.py` | 67 | Collects policy/LOD metrics for telemetry. |
| `pixel_broadcaster.py` | 244 | Multi-client broadcaster for encoded frames. |
| `control/pixel_channel.py` | 312 | WebSocket channel serving encoded frames to clients. |

## Tests (`server/tests`)

Moved from `server/_tests` to mirror the runtime structure. Coverage includes
bitstream helpers, camera controller, config context, layer manager, policy/LOD
logic, ROI application, state channel integration, worker lifecycle, and Zarr
source access.

## Pain Points / Observations

1. **Control channel size** – `control_channel_server.py` still folds transport,
   resume/history, acking, and command routing into one 3k-line module.
2. **Render worker complexity** – `runtime/egl_worker.py` coordinates capture,
   encoding, policy metrics, camera updates, and telemetry; refactoring into
   smaller components remains high priority.
3. **Policy/LOD split across packages** – `data/lod.py`, `data/policy.py`, and
   rendering-side `policy_metrics.py` share concepts; aligning interfaces would
   simplify debugging.
4. **Legacy shims** – Lightweight files like `runtime/runtime_loop.py` and
   `control/pixel_channel.py` exist purely for compatibility; consider
   removing once downstream imports are updated.
5. **Concurrency clarity** – Documenting which threads own GL contexts, pixel
   streaming, and control dispatch will help future refactors (and matches the
   checklist in `docs/repo_structure_future.md`).

This inventory mirrors the client-side map so forthcoming cleanup can follow a
consistent package structure.
