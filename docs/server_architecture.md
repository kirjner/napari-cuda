# napari-cuda Server Architecture

This document captures the current layout of the headless rendering server after the Phase B/C refactor work. It aligns with the "degodify" tenets by documenting which module owns each responsibility and how data moves between them.

## High-level Overview

```text
Websocket Clients (state + pixel)
            │
            ▼
+-------------------------+
| EGLHeadlessServer       |
| - Websocket I/O         |
| - Session + metrics     |
| - SceneSpec broadcast   |
| - State snapshot store  |
+-----------┬-------------+
            │ scene/dims
            ▼
+-------------------------+
| ViewerSceneManager      |
| - Builds SceneSpec      |
| - Maintains ViewerModel |
+-----------┬-------------+
            │ orchestration
            ▼
+-------------------------+
| EGLRendererWorker       |
| - Render loop           |
| - Schedules helpers     |
+-----------┼--------------------------+
            │                          │
            ▼                          ▼
   +------------------+      +----------------------------+
   | ServerSceneQueue  |      | WorkerSceneNotification   |
   | (control → worker)|      | Queue (worker → control)  |
   | + CameraController|      +----------------------------+
   | + SceneStateApplier|                 | drain
   +------------------+                  ▼
            │                 +-------------------------+
            ▼                 | CaptureFacade           |
   +------------------+       | + GLCapture             |
   | ROI helpers      |       | + CudaInterop           |
   | LOD policy       |       | + FramePipeline         |
   +------------------+       +-----------┬-------------+
                                          │ frame packets
                                          ▼
```

## Component Responsibilities

- **EGLHeadlessServer** (`src/napari_cuda/server/egl_headless_server.py`)
  - Owns websocket listeners for state and pixel channels.
  - Tracks authoritative scene state, metrics, and watchdogs.
  - Delegates SceneSpec construction to `ViewerSceneManager` and render work to `EGLRendererWorker`.
  - Maintains config (`ServerCtx`) and async orchestration (broadcasts, keyframe watchdog).

- **ServerSceneData** (`src/napari_cuda/server/server_scene.py`)
  - Mutable bag the server mutates for every state-channel request: carries the latest `ServerSceneState`, camera command deque, dims sequencing, volume/multiscale metadata, SceneSpec caches, and policy logging state.
  - Intent handlers, MCP tools, and broadcast helpers operate exclusively on this bag, emitting immutable snapshots for the worker when changes are ready.
  - Lives alongside `server_scene_queue.py` and `server_scene_spec.py`; the bag stays free of protocol helpers so the worker can import the queue without dragging in viewer/serialization code.
  - Worker-facing helpers (`scene_state_applier.py`, ROI/LOD modules) remain outside the namespace to keep render-thread code decoupled from headless-server orchestration.
- **Control Channel Helpers** (`server_scene_control.py`)
  - Orchestrate state-channel websocket flow: connection setup, intent dispatch, dims/spec broadcasting, and policy/metrics logging.
  - Operate on `ServerSceneData` bags directly while delegating worker hops back through the server object; `EGLHeadlessServer` now simply forwards events into these helpers.
  - Expose procedural APIs (`handle_state`, `broadcast_dims_update`, `rebroadcast_meta`, etc.) so other agents (CLI tools, MCP servers) can reuse the control surface without touching app globals.
  - Worker refresh notifications travel through a dedicated worker→control queue, so the control loop updates the scene manager before emitting dims/spec payloads; no deferred flushes or ad-hoc regeneration needed.


- **ViewerSceneManager** (`src/napari_cuda/server/layer_manager.py`)
  - Maintains a headless `ViewerModel` mirror for scene metadata.
  - Produces `SceneSpec`/`dims.update` payloads used by clients and the server HUD.

- **EGLRendererWorker** (`src/napari_cuda/server/egl_worker.py`)
  - Runs on a dedicated thread with EGL + CUDA contexts.
  - Coordinates render ticks, applies pending scene updates, and liaises with ROI/LOD helpers.
  - Delegates capture/encode to `CaptureFacade` and camera/scene logic to extracted helpers.

- **SceneState Helpers** (`scene_state_applier.py`, `server_scene_queue.py`, `server_scene_spec.py`, `camera_controller.py`)
  - `ServerSceneQueue` coalesces pending updates across threads before the worker drains them.
  - `WorkerSceneNotificationQueue` carries worker-driven refresh notices back to the control loop so metadata and broadcasts stay aligned.
  - `server_scene_spec.py` builds `scene.spec` and `dims.update` payloads from `ServerSceneData` for WebSocket and MCP callers.
  - `SceneStateApplier` applies dims/camera/volume changes to viewer + layer objects.
  - `CameraController` executes queued camera commands and reports policy triggers.

- **ROI & Level Helpers** (`roi.py`, `lod.py`, `level_budget.py`, `worker_runtime.py`)
  - Compute viewport ROI, manage multiscale level selection, and enforce per-level budgets.
  - Provide pure helpers that the worker invokes during render ticks.

- **Capture & Encoding** (`capture.py`, `rendering/*`)
  - `CaptureFacade` encapsulates VisPy FBO capture, CUDA interop, and the NVENC frame pipeline.
  - `encode_frame` returns packets + timing metadata consumed by the server for pacing.

- **Configuration & Policy** (`config.py`, `logging_policy.py`)
  - `ServerCtx` is resolved once at startup and bundles the structured `ServerConfig` with
    `DebugPolicy`, `EncoderRuntime`, and `BitstreamRuntime` dataclasses. Downstream modules use these
    immutable snapshots instead of reading environment variables directly.
  - `logging_policy.py` materialises every debug/logging toggle; worker, rendering, and bitstream
    code paths now consume the policy object exclusively.

## Data Flow Summary

1. Clients connect via websocket; `EGLHeadlessServer` registers them and pushes an initial `SceneSpec`.
2. Incoming intents mutate `ServerSceneData` (dims sequence, volume/multiscale hints, camera queue) and enqueue work in `ServerSceneQueue` when the worker needs to react.
3. The worker thread drains updates from `ServerSceneQueue`, applies them through `SceneStateApplier`, and kicks ROI/LOD recalculation as needed.
4. When the worker finishes applying a scene update, it enqueues a notification on the worker→control queue; the asyncio control loop drains it, refreshes `ViewerSceneManager`, and broadcasts the authoritative dims/spec payload.
5. `CaptureFacade` captures rendered frames, hands them to NVENC, and returns packet bytes to the server.
   - The NVENC helper (`rendering/encoder.py`) reads preset/bitrate/RC overrides from
     `ServerCtx.encoder_runtime` and honours `DebugPolicy.encoder` logging toggles.
6. The server broadcasts encoded frames over the pixel channel and sends authoritative dims/camera
   updates via the state channel when changes land.
   - Bitstream packing consults `ServerCtx.bitstream` to decide between the Cython fast path and
     Python fallback; SPS/NAL logging comes from `DebugPolicy.encoder`.
7. Metrics and watchdogs live in the server layer to detect stalled keyframes, drop counts, or policy anomalies.

## Future Refinements

- Phase D is underway: logging/debug toggles live in `logging_policy.py`, and rendering/bitstream
  modules rely on the `ServerCtx` snapshots. Remaining work focuses on the logging-guard audit and
  lint/test enforcement.
- Phase E targets further decomposition of `EGLHeadlessServer` into discrete broadcaster/state-manager components, with `ServerSceneData` acting as the authoritative state bag shared by future intent/MCP helpers.
- Capture resizing and dynamic canvas negotiation remain TODO items once client capabilities land.
