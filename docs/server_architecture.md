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
+-----------┼-------------------------+
            │                         │
            ▼                         ▼
   +------------------+     +-------------------------+
   | SceneStateQueue  |     | CaptureFacade           |
   | + CameraController|    | + GLCapture             |
   | + SceneStateApplier|   | + CudaInterop           |
   +------------------+     | + FramePipeline         |
            │               +-----------┬-------------+
            ▼                           │
   +------------------+                │ frame packets
   | ROI helpers      |<---------------┘
   | LOD policy       |
   +------------------+
```

## Component Responsibilities

- **EGLHeadlessServer** (`src/napari_cuda/server/egl_headless_server.py`)
  - Owns websocket listeners for state and pixel channels.
  - Tracks authoritative scene state, metrics, and watchdogs.
  - Delegates SceneSpec construction to `ViewerSceneManager` and render work to `EGLRendererWorker`.
  - Maintains config (`ServerCtx`) and async orchestration (broadcasts, keyframe watchdog).

- **ViewerSceneManager** (`src/napari_cuda/server/layer_manager.py`)
  - Maintains a headless `ViewerModel` mirror for scene metadata.
  - Produces `SceneSpec`/`dims.update` payloads used by clients and the server HUD.

- **EGLRendererWorker** (`src/napari_cuda/server/egl_worker.py`)
  - Runs on a dedicated thread with EGL + CUDA contexts.
  - Coordinates render ticks, applies pending scene updates, and liaises with ROI/LOD helpers.
  - Delegates capture/encode to `CaptureFacade` and camera/scene logic to extracted helpers.

- **SceneState Helpers** (`scene_state_applier.py`, `state_machine.py`, `camera_controller.py`)
  - `SceneStateQueue` coalesces pending updates across threads.
  - `SceneStateApplier` applies dims/camera/volume changes to viewer + layer objects.
  - `CameraController` executes queued camera commands and reports policy triggers.

- **ROI & Level Helpers** (`roi.py`, `lod.py`, `level_budget.py`, `worker_runtime.py`)
  - Compute viewport ROI, manage multiscale level selection, and enforce per-level budgets.
  - Provide pure helpers that the worker invokes during render ticks.

- **Capture & Encoding** (`capture.py`, `rendering/*`)
  - `CaptureFacade` encapsulates VisPy FBO capture, CUDA interop, and the NVENC frame pipeline.
  - `encode_frame` returns packets + timing metadata consumed by the server for pacing.

## Data Flow Summary

1. Clients connect via websocket; `EGLHeadlessServer` registers them and pushes an initial `SceneSpec`.
2. User interactions or policy decisions enqueue updates in `SceneStateQueue`.
3. The worker thread drains updates, applies them through `SceneStateApplier`, and kicks ROI/LOD recalculation as needed.
4. `CaptureFacade` captures rendered frames, hands them to NVENC, and returns packet bytes to the server.
5. The server broadcasts encoded frames over the pixel channel and sends authoritative dims/camera updates via the state channel when changes land.
6. Metrics and watchdogs live in the server layer to detect stalled keyframes, drop counts, or policy anomalies.

## Future Refinements

- Phase D will consolidate logging and env toggles into a dedicated policy module.
- Phase E targets further decomposition of `EGLHeadlessServer` into discrete broadcaster/state-manager components to mirror the worker split.
- Capture resizing and dynamic canvas negotiation remain TODO items once client capabilities land.
