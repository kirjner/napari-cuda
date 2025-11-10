Current Dims/View/Cursor Stack
================================

This document catalogs the existing “dims + camera” pipeline so we can migrate it
to the new `view / axes / step / lod / camera` design without missing any touch
points.

Ledger State Today
------------------

* `dims.main.dims_spec` — monolithic snapshot built in
  `src/napari_cuda/server/scene/builders.py:678-920`. Carries:
  `ndisplay`, `order`, `displayed`, `current_step`, `current_level`,
  per-level shapes, and a rich `DimsSpecAxis` entry per axis (see
  `src/napari_cuda/shared/dims_spec.py`).
* `viewport.active.state` — “ActiveView” mirror source
  (`src/napari_cuda/server/control/mirrors/active_view_mirror.py`). Holds
  `{"mode": "plane"|"volume", "level": int}`. Every view toggle is reflected
  here so mirrors can emit `notify.level`.
* `viewport.state` (`plane`/`volume` scopes) — serialized copies of the worker
  `PlaneState` / `VolumeState` dataclasses (`state_reducers._store_plane_state`,
  `_store_volume_state`). Reducers rely on these cached blobs to determine
  whether a level/ROI reload is pending.
* `camera_plane` / `camera_volume` scopes — persisted by
  `reduce_camera_update` (`src/napari_cuda/server/control/state_reducers.py:1456+`).
  Keys: `center`, `zoom`, `rect` for plane; `center`, `angles`, `distance`, `fov`
  for volume. Each write is versioned so snapshot builders can tell whether
  anything changed.
* `scene.main` — sequencing fields updated by every reducer:
  `op_seq`, `op_kind`, `op_state`. Transactions mark `op_state='open'`; the
  worker flips it to `'applied'` via `_ack_scene_op_if_open`
  (`src/napari_cuda/server/app/egl_headless_server.py`).

Bootstrap / Restore Paths
-------------------------

* **`reduce_bootstrap_state`** (`state_reducers.py:450-832`) seeds the ledger
  from dataset metadata. It writes:
  - `dims_spec` (full document)
  - `dims.*.index` for the primary axis
  - `viewport.active.state` (mode + level)
  - `plane` / `volume` cached state blobs
  - `scene.main.{op_seq,op_kind,op_state}`
* **`reduce_plane_restore`** (`state_reducers.py:1185-1379`) rewrites plane
  pose + level metadata after the worker applies a plane snapshot. It commits
  new `camera_plane` values, updates `viewport.plane.state`, and records
  `scene.main` sequencing.
* **`reduce_volume_restore`** (`state_reducers.py:1381-1454`) mirrors the plane
  restore path for volume pose (`camera_volume`, `viewport.volume.state`).
* Runtime bootstrap (`src/napari_cuda/server/runtime/bootstrap/setup_camera.py`)
  uses `_bootstrap_camera_pose`, `_enter_volume_mode`, `_configure_camera_for_mode`,
  and `_frame_volume_camera` to build initial viewer state before the worker
  enters the render loop. These helpers all read `viewport_state` and the
  ledger’s `dims_spec` when deciding camera parameters.

Control-Plane Reducers
----------------------

All user intents hit a handler under
`src/napari_cuda/server/control/state_update_handlers/`. Each handler validates
payloads, calls into the reducers, and acks the client.

* **`reduce_dims_update`** (`state_reducers.py:833-1030`): mutates `dims_spec`
  by adjusting a single axis (step or margins). Also rewrites the cached plane
  state so the worker knows a new slice must be loaded.
* **`reduce_view_update`** (`state_reducers.py:1032-1183`): flips `ndisplay`,
  `order`, and `displayed`. Writes `viewport.active.state` to reflect the new
  mode/level, updates `dims_spec`, and emits a `Scene` transaction with
  `op_kind='view-toggle'`.
* **`reduce_plane_restore` / `reduce_volume_restore`** (see above): used when a
  worker reports back applied state (`worker.render.apply` origin), ensuring
  `scene.main.op_state` transitions to `'applied'`.
* **`reduce_level_update`** (`state_reducers.py:1462-1888`): records multiscale
  level/ROI decisions. Writes:
  - `multiscale.main.level`
  - `viewport.state` blobs
  - optional plane slice metadata
  - `scene.main` sequencing (`op_kind='level-update'`)
* **`reduce_camera_update`** (`state_reducers.py:1889-2148`): writes the
  `camera_plane` / `camera_volume` scopes and updates cached plane/volume poses.
  Called both from control (user camera intents) and from the worker callback
  (`origin="worker.state.camera"`).
* **Other reducers** (`reduce_layer_property`, `reduce_layer_assignment`,
  `reduce_stream_update`, etc.) are orthogonal but still bump `scene.main`.

Mirrors and Notifications
-------------------------

* `ServerDimsMirror`
  (`src/napari_cuda/server/control/mirrors/dims_mirror.py`) watches
  `scene.main.op_state`, `dims.main.dims_spec`, and `viewport.active.state`.
  Once `op_state='applied'`, it rebuilds a `NotifyDimsPayload` (which embeds
  the full `DimsSpec`) and publishes `notify.dims`.
* `ActiveViewMirror`
  (`src/napari_cuda/server/control/mirrors/active_view_mirror.py`) listens to
  `viewport.active.state` and emits `notify.level` whenever the mode/level pair
  changes.

Both mirrors re-read the ledger, so the legacy `dims_spec` + active view entries
must remain coherent.

Render Snapshot Composition
---------------------------

`pull_render_snapshot` (in `scene/builders.py`) merges ledger scopes into
`RenderLedgerSnapshot` (`src/napari_cuda/server/scene/models.py`). The snapshot
contains:

* Copy of `dims_spec` (including axes, levels, displayed axes, etc.).
* Active view metadata (`ndisplay`, `mode`, `current_level`).
* `camera_plane` / `camera_volume` pose values.
* Layer visual state and per-layer versions.
* LOD metadata (`multiscale.*`, `viewport.state`).
* `scene.main.op_seq` so stale snapshots can be discarded.

Worker Consumption Path
-----------------------

1. `worker_loop` pulls a snapshot per frame and enqueues it via
   `consume_render_snapshot` (`render_loop/planning/staging.py`), which marks a
   render tick as needed. (Future work: remove `RenderUpdate` entirely; right
   now we still push snapshots through the mailbox.)
2. `drain_scene_updates` drains the mailbox and applies the snapshot using
   `RenderApplyInterface`:
   * `apply_render_snapshot` dispatches to plane or volume paths in
     `render_loop/applying/apply.py`. Plane flows delegate to
     `plane.py` / `plane_ops.py`; volume flows go through
     `volume.py` / `volume_ops.py`.
   * `ViewportPlanner` (`render_loop/planning/viewport_planner.py`) consumes
     snapshots to decide whether a new slice/level needs to load. It relies on
     `PlaneState` (`request` vs `applied`), `snapshot.current_step`, and
     `snapshot.ndisplay`.
   * `snapshot_drain.drain_render_state` applies dims changes, layer visual
     updates, and (prior to the orbit fix) camera overrides.
3. After applying dims, the worker loads slices/volumes using the cached plane
   state (`viewport_state.plane`) and the ledger’s `current_step`. Camera deltas
   are re-applied via the queue (see below).

Camera Delta Flow
-----------------

* Control channel enqueues `CameraDeltaCommand` objects
  (`state_update_handlers/camera.py` → `_enqueue_camera_delta` in
  `egl_headless_server.py`).
* The worker drains `_camera_queue` during each render tick
  (`render_loop/planning/ticks/camera.py`). Applying the command updates the
  VisPy camera and triggers `_emit_current_camera_pose("camera-delta")`.
* `_emit_current_camera_pose` calls `reduce_camera_update` (origin
  `"worker.state.camera"`), so the ledger and future snapshots reflect the new
  pose. This is why we still persist plane/volume camera scopes even though the
  worker is now the source of truth each frame.

Bootstrap, Restore, and Runtime Helpers
---------------------------------------

* **Viewer bootstrap** (`runtime/bootstrap/setup_viewer.py`) wires up the napari
  viewer model, registers layer visuals, and calls into the camera helpers.
* **Camera helpers** (`runtime/bootstrap/setup_camera.py`):
  - `_configure_camera_for_mode` swaps between `PanZoomCamera` and
    `TurntableCamera`.
  - `_bootstrap_camera_pose`, `_enter_volume_mode`, `_exit_volume_mode` manage
    transitions during startup or when the worker toggles modes internally.
  These helpers read `worker.viewport_state` (seeded from ledger snapshots) and
  use `ViewerBootstrapInterface` to fetch multiscale metadata.
* **Render mailbox** (`runtime/ipc/mailboxes/render_update.py`) coalesces
  `RenderUpdate` structs. Even though we now feed the worker fresh snapshots,
  the mailbox still stores `mode`, `PlaneState`, and `VolumeState` copies for
  viewport runners to inspect. Removing it entirely is part of the longer-term
  plan.

Key Dependencies to Untangle
----------------------------

1. **`dims_spec` ubiquity** — used by reducers (`reduce_dims_update`,
   `reduce_view_update`), mirrors (`ServerDimsMirror`), worker apply logic
   (`render_loop/applying/*`), the `ViewportPlanner`, and notification payloads.
2. **ActiveView coupling** — worker, mirrors, and ledger all rely on
  `viewport.active.state` to answer “plane vs volume” and “current level”.
3. **Plane/volume cached state** — reducers stash `PlaneState`/`VolumeState`
   copies in the ledger so subsequent intents know whether a level/ROI reload is
   pending.
4. **Render snapshot shape** — everything downstream expects a
  `RenderLedgerSnapshot`. Changing the ledger schema implies updating that
  dataclass or replacing it entirely.
5. **RenderUpdate mailbox** — still sits between control + worker; we want to
  remove it once the worker snapshots directly from the ledger on each poke.

Migration Implications
----------------------

* Introducing `view` / `axes` / `step` / `lod` scopes means all existing ledger
  readers (`build_ledger_snapshot`, mirrors, worker apply code) must learn the
  new layout before we can delete `dims_spec`.
* Reducers become simpler once they no longer reconstruct the full spec, but we
  need new clamp/remap helpers to preserve current guarantees (see
  `reduce_dims_update`’s size clamping and
  `dims_spec_remap_step_for_level` in
  `src/napari_cuda/server/runtime/render_loop/applying/apply.py`).
* Notifications should switch from `NotifyDimsPayload` /
  `NotifyLevelPayload` to a single payload derived from the new ledger scopes,
  eliminating the ActiveView mirror entirely.
* The worker’s `ViewportState`/`ViewportPlanner` machinery becomes redundant
  once the snapshot directly provides `{mode, displayed axes, per-axis index,
  lod}`. Replacing it will simplify `render_loop/applying` and the level policy
  plumbing.
* Removing RenderUpdate requires a wake mechanism tied to `scene.main.op_seq`
  so the worker snapshots the ledger directly whenever the server pokes it.

Use this catalog as the authoritative list of files to touch as we introduce the
new schema. Every item above references the exact module that currently depends
on the legacy dims/camera model.
