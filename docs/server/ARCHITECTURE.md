# Server Architecture — Ledger‑Centric, Minimal, Fast

This document defines the server‑side architecture for napari‑cuda. It keeps the design streamlined (few layers of indirection), ledger‑centric (single source of truth), and performant.

## Principles

- Ledger is the only source of truth on the server.
- One intent path (reducers write ledger), one apply path (worker transaction), thin mirrors for broadcasts.
- Mode is driven solely by `view.ndisplay` (2D vs 3D). `dims.mode` is derived for notifications and never authored by clients.
- Camera is “applied‑only”: clients send deltas; the worker writes the applied pose back to the ledger.
- Deterministic order: worker applies dims/level/camera atomically so napari never observes partial state.

## Server Package Layout (target)

- `src/napari_cuda/server/app/`
  - `egl_headless_server.py` — orchestrator (WS, lifecycle, metrics, callback sinks)
- `src/napari_cuda/server/control/`
  - `state_ledger.py` — typed ledger API (subscriptions, batch writes, snapshots)
  - `state_reducers.py` — reducers (view, dims, level, volume, layer, camera ack)
  - `mirrors/`
    - `dims_mirror.py` — builds `notify.dims`; derives `mode` from `view.ndisplay`; op‑gated
    - `camera_mirror.py` — builds `notify.camera` from applied camera entries; op‑gated
    - `layer_mirror.py` — builds `notify.layers` (or fold into `notify.scene`)
- `src/napari_cuda/server/runtime/`
  - `egl_worker.py` — render thread (VisPy + CUDA/NVENC) and mailbox
  - `render_txn.py` — RenderTxn builder/apply (atomic dims/level/camera, fit suppression)
  - `camera_controller.py` — camera delta queue, rect↔ROI helpers, pose snapshots
  - `render_ledger_snapshot.py` — builds the worker snapshot from the ledger (no “scene bag” merge)
- `src/napari_cuda/server/data/`
  - `zarr_source.py`, `roi.py`, `lod.py`, `level_budget.py` — pure helpers
- `src/napari_cuda/server/rendering/`
  - `viewer_builder.py`, `egl_context.py`, `encoder.py`, `capture.py` — graphics/codec plumbing

## Ledger Schema (single source of truth)

- `view`:
  - `ndisplay:int` ∈ {2,3} — only driver for 2D/3D mode
  - `displayed:tuple[int]` — equals `dims.order[-ndisplay:]`
- `dims`:
  - `ndim:int`
  - `axis_labels:tuple[str]`
  - `order:tuple[int]`
  - `current_step:tuple[int]` (length == `ndim`, pad/truncate)
  - `level_shapes:tuple[tuple[int,...]]`, `levels:tuple[dict]` (descriptors)
- `multiscale`:
  - `level:int` (applied), `downgraded:bool`
- `camera` (worker‑applied only, current mode):
  - 2D: `center:tuple[float]`, `zoom:float`, `rect:(left,bottom,width,height)`
  - 3D: `center:tuple[float]`, `angles:(az,el,roll)`, `distance:float`, `fov:float`
- Per‑mode pose caches (worker‑applied only, used for deterministic restore):
  - `camera_plane.center`, `camera_plane.zoom`, `camera_plane.rect`
  - `camera_volume.center`, `camera_volume.angles`, `camera_volume.distance`, `camera_volume.fov`
- Plane view cache (worker‑applied only):
  - `view_cache.plane.level:int`, `view_cache.plane.step:tuple[int]`
- `layer/<id>`: normalized controls (opacity, colormap, etc.)
- `scene`:
  - `op_seq:int`, `op_state:"open"|"applied"`, `op_kind:"view-toggle"|"dims"|"policy"|"camera"`

Invariants:

- Mode is derived: `mode = ("volume" if view.ndisplay >= 3 else "plane")`.
- `view.displayed == dims.order[-ndisplay:]`.
- `len(dims.current_step) == dims.ndim` (pad/truncate on accept).
- `multiscale.level_shapes[current_level]` defines world sizes for apply.
- Camera keys are only written by the worker when a pose is applied.
- Per‑mode caches are authoritative for toggles: 3D→2D uses `view_cache.plane.*` + `camera_plane.rect`; 2D→3D uses `camera_volume.*` (or a canonical initialisation written once by the worker).

## Control Flow (server‑only)

Intents → Reducers → Ledger → Worker Txn → Ledger (applied) → Mirrors.

- View toggle (2D↔3D)
  - Reducer writes `view.ndisplay`, derives and writes `dims.order` + `view.displayed`, opens `scene.op_seq(kind="view-toggle")`.
  - Worker pulls snapshot, runs `RenderTxn`, commits applied `multiscale.level/current_step` + `camera.*`, closes op.
  - Dims and camera mirrors emit once (op‑gated).

- Dims step
  - Reducer writes normalized `dims.current_step` (pad/truncate); optional op fence `kind="dims"`.
  - Worker applies, commits applied step; mirrors emit.

- Level switch (policy or request)
  - Worker selects level (oversampling + hysteresis, budgeted), applies slab/volume, commits applied level/step; mirrors emit.

- Camera input
  - Client sends deltas → enqueued (no direct ledger writes).
  - Worker applies deltas, snapshots pose, writes `camera.*`; camera mirror emits.

- Bootstrap
  - `reduce_bootstrap_state(...)` seeds dims, levels, shapes, current_level, view.ndisplay (no `dims.mode`).
  - Start mirrors after bootstrap; start worker; first `RenderTxn` applies a consistent frame.

## Worker RenderTxn (atomic, fit‑safe)

Build from snapshot, apply atomically, then commit applied state:

1) Resolve mode: `target_volume = (snapshot.ndisplay >= 3)`.
2) Normalize dims: compute `ndim`; derive/normalize `order`, `displayed`, `current_step`.
3) Select level: requested or policy (oversampling + hysteresis), bounded by budgets; remap z proportionally unless restoring.
4) Determine 2D ROI from cache:
   - Project `camera_plane.rect` (world) to target level via `plane_scale_for_level`.
   - No fallbacks in steady‑state; the worker writes the caches on apply so they are always present.
5) Apply with fit suppression (napari sees a coherent state only):
   - Set `dims.ndim` → `dims.order` → `dims.current_step` → `dims.ndisplay` (last).
   - Apply level (slab/volume) and update layer.
   - Apply camera: 2D `set_range` then rect/center/zoom; 3D `set_range` then TT angles/dist.
   - Suppress napari auto‑fit during dims: block the `viewer.dims.events.ndisplay` and `viewer.dims.events.order` emitters while the transaction applies dims. The worker sets camera explicitly as part of the txn, so fit is not needed during toggles.
6) Commit applied:
   - `reduce_level_update(applied, downgraded)`, `reduce_camera_update(pose)`; update per‑mode caches (`camera_plane.*` + `view_cache.plane.*` for 2D, or `camera_volume.*` for 3D); close `scene.op_seq` as `applied`.

This ordering prevents napari’s auto‑fit from running on partial state and eliminates letterbox/flicker on 3D→2D.

## Mirrors (thin, op‑gated)

- Dims mirror
  - Build `notify.dims` from ledger; derive `mode` from `view.ndisplay`.
  - Gate emission: if `scene.op_state == "open"`, buffer; flush on `"applied"`.

- Camera mirror
  - Emit applied pose snapshots written by the worker; may coalesce within a frame.

- Layer mirror
  - Present normalized per‑layer control state; optionally fold into a scene summary.

## Performance

- Reducers batch related writes; mirrors coalesce per op.
- Worker avoids full encoder resets on mode toggle (request keyframe/IDR only).
- No extra client‑side dependencies; GPU pipeline unchanged.

## Error Handling & Observability

- Internal invariants enforced with assertions; subsystem boundaries log exceptions.
- Each op uses `scene.op_seq` and logs a single, correlated path: reducer write → worker apply → applied ledger writes → mirror emit.
- Metrics reflect policy decisions, level apply time, and mirror throughput.

## Camera Flow (Queue vs Ledger)

- Input plane (queue): clients never write `camera.*` to the ledger. They send `CameraDeltaCommand`s (zoom, pan, orbit, reset) to the server. The render worker consumes this queue and applies the deltas to the live camera.
- Output plane (ledger): after applying a camera change (or a render frame with a new pose), the worker snapshots the applied pose and writes it to the ledger via `reduce_camera_update`. This is the only way `camera.*` changes.
- Notifications: a camera mirror (or the existing broadcaster) emits `notify.camera` from the applied pose so clients can observe the canonical state.
- Mode caches: when the applied pose is 2D, the worker also writes `camera_plane.*` and `view_cache.plane.*`; when 3D, it writes `camera_volume.*`. These are used for deterministic plane/volume restore.

Sequence (zoom example): client intent → enqueue `CameraDeltaCommand` → worker applies deltas → worker snapshots pose → writes `camera.*` (and per‑mode caches) to ledger → camera notify emits.

## Render Loop vs Mailboxes

- Control loop (async): pulls ledger snapshots, decides whether to render this tick, and calls the worker’s consume/apply. It also sends the captured packets to clients.
- Worker (render thread): applies state and renders frames. RenderTxn is the single apply site (dims → level → camera) and runs on the worker thread.
- Mailboxes: the legacy “scene state” mailbox (`enqueue_scene_state`/`drain_scene_updates`) is being removed. RenderTxn applies state directly; only multiscale requests may keep a tiny queue or be applied directly.
- Cadence: continuous rendering for connected clients remains; adaptive throttling can be added later behind flags. Removing the scene mailbox does not change the control loop cadence.
