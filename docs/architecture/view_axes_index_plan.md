View/Axes/Index/Lod/CAMERA Migration Plan
=========================================

Purpose
-------

Provide an execution-ready plan—diagrams plus task breakdown—for replacing the
legacy `dims_spec` + planner/mailbox stack with the factored ledger model
(`view / axes / index / lod / camera / layers`) and direct worker apply path.
Treat this as the canonical reference for sequencing, scope boundaries, and
temporary shims.

Target Call Stacks
------------------

### Control → Ledger → Notify

```mermaid
sequenceDiagram
    participant Client
    participant Handler
    participant Reducer
    participant Ledger
    participant Notify
    Client->>Handler: state.update (dims/view/camera/layer)
    Handler->>Reducer: validate + intent
    Reducer->>Ledger: write {view,axes,index,lod,camera,layers} + legacy dims_spec (flagged)
    Reducer-->>Ledger: bump scene.main.op_seq
    Ledger-->>Notify: mirrors/broadcasters pull new scopes, build notify payloads
    Notify-->>Client: notify.dims / notify.camera / notify.layers / notify.level
```

Key points:
- Reducers dual-write new scopes **and** legacy `dims_spec` until consumers flip.
- Notify builders read the new scopes first, falling back to `dims_spec`.
- No planner/mailbox involvement; ledger is the only truth source.

### Worker Tick (Ledger → Apply Blocks → Intents)

```mermaid
sequenceDiagram
    participant Poke as SceneOp
    participant Worker as EGL Worker
    participant Ledger
    participant Intent as WorkerIntentMailbox

    SceneOp--)Worker: poke (scene.main.op_seq increment)
    Worker->>Ledger: read view/axes/index/lod/camera/layers scopes
    Worker->>Worker: compute per-block signatures
    alt Dims changed
        Worker->>Worker: apply_dims_block(view, axes, index)
    end
    alt Camera changed
        Worker->>Worker: apply_camera_block(camera)
        Worker->>Reducer: reduce_camera_update (pose callback)
    end
    alt Layers changed
        Worker->>Worker: apply_layers_block(layers)
    end
    alt Level policy decision
        Worker->>Intent: enqueue LevelSwitchIntent
        Intent->>Reducer: reduce_level_update (server loopback)
    end
```

Key points:
- `RenderUpdateMailbox`, `PlaneViewportCache`, `VolumeViewportCache`, and the apply
  shims disappear once the block-native path is authoritative.
- `ViewportPlanner` survives temporarily as an ROI/level-policy helper but no longer
  owns snapshot staging; it will be deleted once those heuristics move into the
  ledger-backed restore caches.
- Only `WorkerIntentMailbox` remains (level intents, thumbnail captures).
- Ledger scopes are read directly each tick; per-block signatures prevent
  redundant reapply.

Issue Breakdown (Phased)
------------------------

### Phase 0 — Ledger Restructure (no behavioral change)
1. **Move ledger module**: rename `state_ledger/__init__.py` → `ledger.py`; fix imports.
2. **Create `scene/blocks/` package**:
   - `legacy_dims_spec.py` (current `DimsSpec` + helpers).
   - `view_state.py`, `axes_state.py`, `index_state.py`, `lod_state.py`,
     `camera_state.py`, `layer_state.py` (empty shells with dataclasses + serializers).
3. **Add feature flag plumbing** (`LEDGER_VIEW_AXES_V1`) shared by reducers/scene builders/worker.

### Phase 1 — Dual-write new scopes
1. Reducers/transactions write new blocks + legacy `dims_spec`.
2. Unit tests assert parity between `dims_spec` and the new scopes (indexes, modes, margins, level).
3. Scene builders encode the new blocks into `RenderLedgerSnapshot` but keep legacy fields populated.

Temporary patching:
- Worker/runtime still read only legacy fields.
- Notify payloads still emit `dims_spec` at top level.

### Phase 2 — Consumer flip (outside-in)
1. **Notify pipeline**: `notify.dims` emits both the new blocks and legacy spec;
   mirrors/clients start reading `{view,axes,index,lod}` first.
2. **Snapshot builder → worker**: expose the new blocks to the worker under a
   feature flag; start building per-block signatures.
3. **Runtime apply rewrite**:
   - Implement `apply_dims_block`, `apply_camera_block`, `apply_layers_block`
     that accept the new models.
   - Introduce a new worker tick that bypasses `RenderUpdateMailbox`.
   - Keep the legacy planner/mailbox path behind a fallback flag for staged rollout.

### Phase 3 — Legacy removal
1. Delete `PlaneViewportCache`, `VolumeViewportCache`, `RenderUpdateMailbox`,
   bootstrap camera helpers, and all planner/apply shims (as listed in
   `docs/architecture/dims_camera_legacy.md`). `ViewportPlanner` remains only as the
   worker’s ROI/level-policy helper until those heuristics move elsewhere.
2. Remove `dims_spec` writes and legacy notify fields; collapse `scene/blocks`
   to only the new schema.
3. Update docs/tests to treat the new blocks as the only source of truth.

Deliverables & Tracking
-----------------------

For each phase, file issues with the following template:
1. **Summary** (e.g., “Dual-write view/axes/index blocks in reducers”).
2. **Acceptance criteria** (tests, flags, docs updated).
3. **Dependencies** (must land before Phase N+1 tasks).
4. **Owner + estimated effort**.

This doc plus `docs/architecture/dims_camera_legacy.md` should be kept in sync
during implementation; mark items as completed and prune the legacy list as code
lands.

Current Status (2025-11-10)
---------------------------

- **Feature flag:** `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1` (see
  `scene/blocks/__init__.py::ENABLE_VIEW_AXES_INDEX_BLOCKS`) now switches on the
  Phase 1 dual-write path **and** the Phase 3 op_seq watcher runtime. With the
  flag enabled the worker bypasses `RenderUpdateMailbox`, watches `scene.main.op_seq`,
  and applies `{view, axes, index, lod, camera, layers}` blocks directly via the
  per-block signature cache introduced in `runtime/worker/lifecycle.py`.
  Reducers write:
  - `view.main.state` (`ViewBlock`)
  - `axes.main.state` (`AxesBlock`)
  - `index.main.cursor` (`IndexBlock`)
  - `lod.main.state` (`LodBlock`)
  - `camera.main.state` (`CameraBlock`)
  plus the plane/volume restore caches. Legacy `viewport.*` / `camera_*` scopes are gone.
- **Transactions:** All scene control transactions accept `extra_entries`, so
  block payloads land in the same atomic ledger batch as the legacy keys. This
  ensures notify/mirror code can start reading the new scopes without race
  conditions.
- **Baseline notify:** `orchestrate_connect` seeds `notify.level` directly from
  `viewport.active.state`. When the resumable history store is configured, the
  level baseline is recorded as a snapshot; otherwise we seed the per-client
  sequencer before emitting the first delta. This keeps `notify.level` cursors in
  sync with the new ledger scopes.
- **Parity guardrails:** targeted tests inside
  `src/napari_cuda/server/tests/test_state_channel_updates.py` still pass with
  the flag monkeypatched off/on. Remaining work is updating the docs/tests to drop
  references to the removed scopes entirely before flipping the flag on by default.
- **Restore cache helpers:** `state_reducers.py` now exposes `load_*_restore_cache`,
  `write_*_restore_cache`, and `_plane/_volume_cache_from_state`, and reducers
  (bootstrap, dims, camera, level, plane/volume restore) already dual-write
  cache payloads under the feature flag.
  - Today those helpers clone the just-updated `PlaneViewportCache` /
    `VolumeViewportCache`
    instances so cache writes share the same timestamp/intent metadata as the
    legacy `viewport.*` entries. Once `{view, axes, index, lod, camera}`
    blocks become authoritative (Phase 2/3), we will delete the legacy
    dataclasses and update the helpers to build caches straight from the new
    block payloads instead.
  - `reduce_view_update` / toggle handlers copy the target mode’s cache into
    `{lod,index,camera}` so toggles are single-pass, and the cache dataclasses
    are named `PlaneRestoreCacheBlock` / `VolumeRestoreCacheBlock` to keep
    terminology aligned with the other scene blocks.
  - Next focus: flip the consumers (`snapshot_scene_blocks`, notify builders,
    worker apply paths) to read the block scopes under the flag while legacy
    keys stay live for compatibility.

Outstanding Phase 1 work:
1. Add explicit unit tests that compare each block to the corresponding
   `DimsSpec` / cached pose fields (axes metadata, current index, margins, camera
   pose).
2. Teach `snapshot_scene_blocks` / `build_notify_scene_payload` to read the new
   ledger keys when the flag is set, while legacy scopes remain in place for the
   off-path. Once the block path is default, rename this helper to reflect that
   it snapshots scene blocks directly (e.g., `snapshot_scene_blocks`), move
  `viewport_state` metadata to the block payloads, and delete the legacy
  reconstruction path.
3. Treat the renamed worker-side caches (**PlaneViewportCache** /
   **VolumeViewportCache**) as worker-only snapshots for restore/autoframe
   toggles. They do not need dedicated ledger scopes once block consumers are
   authoritative, so plan the removal of their ledger mirrors after the
   consumer flip is complete.
4. Document + refactor the bootstrap defaults used in tests: both
   `src/napari_cuda/server/tests/test_state_channel_updates.py` and
   `src/napari_cuda/server/tests/_helpers/state_channel.py` now carry
   `_default_plane_pose` / `_default_volume_pose` helpers to seed camera +
   restore-cache blocks when reducers aren’t invoked directly. Once reducers
   own the entire bootstrap path we should expose a shared “seed ledger for
   tests” helper (or call the real reducers) so those defaults only live in one
   place.
4. Document the ledger schema (keys + payloads) in `dims_camera_legacy.md` and
   the protocol docs so consumers know where to look once the flag flips.

Phase 2 is now complete; all consumers read the block ledger under the flag.
Phase 3 (in progress) removed the planner/mailbox stack and the legacy ledger scopes.
What remains is flipping the feature flag on by default, updating docs/tests, and
finishing the render-loop cleanup.

### Phase 3 Direction (single snapshot + RenderInterface)

- Worker should pull the ledger once per tick, receive a `SceneBlockSnapshot`
  (view/axes/index/lod/camera + typed layer blocks + restore caches), and apply it
  directly. No redundant `pull_render_snapshot` calls, no reconstructing pose data
  from `dims_spec`.
- `RenderLedgerSnapshot` becomes an optional compatibility shim for legacy payloads.
  As soon as layer blocks replace `LayerVisualState`, the worker/render loop no longer
  depends on it.
- `RenderInterface` owns both the planning and apply APIs, mutating worker state directly from the block
  snapshot. Per-block signatures replace the current staging helpers.

### Phase 3 Direction (single snapshot + RenderInterface)

- Worker should pull the ledger once per tick, receive a `SceneBlockSnapshot`
  (view/axes/index/lod/camera + typed layer blocks + restore caches), and apply it
  directly. No redundant `pull_render_snapshot` calls, no reconstructing pose data
  from `dims_spec`.
- `RenderLedgerSnapshot` becomes an optional compatibility shim for legacy payloads.
  As soon as layer blocks replace `LayerVisualState`, the worker/render loop no longer
  depends on it.
- `RenderInterface` owns both the planning and apply APIs, mutating worker state directly from the block
  snapshot. Per-block signatures replace the current staging helpers.

Phase 3 work items (authoritative once this doc is updated again):
1. **✅ Render loop mailbox removed.** Worker ticks now hydrate the viewer directly
   from the block snapshots via the op_seq watcher. `RenderUpdateMailbox`/`RenderUpdate`
   no longer exist; `ViewportPlanner` sticks around only for ROI/level bookkeeping
   and can be simplified once the new block-native apply path lands.
   - **2025-11-12:** `_op_seq_watcher_apply_snapshot` now re-pulls a fresh ledger snapshot
     whenever a block signature changes and hands that snapshot back to the render loop.
     This prevents the worker from reapplying the previous frame’s camera pose (the “orbit
     jitter” observed when `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1`). The watcher, ledger,
     and notify paths now stay in lockstep for worker-origin camera poses without touching
     `scene.main.op_seq`.
   - **Planned follow-up:** collapse the render loop down to a single authoritative snapshot
     per tick (or operate directly on `SceneBlockSnapshot`) so we stop double-pulling the
     ledger entirely. Track this in the Phase 3 cleanup queue once the remaining legacy
     scopes are deleted.
2. **✅ Delete legacy ledger scopes.** Reducers/transactions now persist only the block
   payloads (plus restore caches). Notify/builders/worker bootstrap read those scopes exclusively.
3. Enable `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` by default, strip the flag-off
   branches from scene builders/notify/runtime, and refresh docs/tests/CI to
   assume the block schema is the sole path.
4. Finish the render-loop cleanup noted above (single snapshot per tick).

Restore Flow (Current → Target)
-------------------------------

Current (legacy planner/mailbox)
- Control: `reduce_view_update` toggles mode; handlers read `viewport.plane/volume.state`
  and call `reduce_*_restore` with cached pose/level/index.
- Worker: applies snapshots, emits intents back for applied camera (`origin="worker.state.camera"`)
  and level decisions (`origin="worker.state.level"`).
- Ledger: becomes authoritative only after those reducers run; notify/render consume ledger then.

Target (ledger‑driven, single‑pass on toggle)
- Persist per‑mode restore caches on the ledger so toggles don’t need a second pass:
  - `restore_cache.plane.state` → `{ level: int, index: tuple[int,...], pose: {rect,center,zoom}, optional: roi_signature, zoom_hint }`
  - `restore_cache.volume.state` → `{ level: int, index: tuple[int,...], pose: {center,angles,distance,fov}, optional: roi_signature }`
- Control (flag on): `reduce_view_update` writes `ViewBlock` and then copies the target mode’s
  restore cache into the authoritative blocks (`LodBlock.level`, `IndexBlock.value`, `CameraBlock.*`) in one transaction.
- Worker: still maintains minimal in‑memory caches for real‑time deltas, ROI hysteresis, and signature diffing,
  but no longer needs to “confirm” restore for toggles. It can continue to emit camera/level intents as telemetry.
- Notify/render: read the authoritative blocks immediately after the toggle and render atomically.

Notes
- If the restore caches are not persisted on the ledger, toggles become a two‑pass handshake
  (request → worker apply → worker intent back → control writes authoritative blocks). Persisting the caches
  avoids this roundtrip and keeps toggles single‑pass.
