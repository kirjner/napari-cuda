# 2025-11-13 — LayerBlock baseline ingest + shared diff helper

### Runtime
- Extracted a shared LayerBlock diff helper (`compute_layer_block_deltas`) so both the render loop and control consumers compute mutations directly from `{scene_layers.*.block}` records; the worker no longer owns bespoke delta logic.

### Control
- `_collect_default_visuals`, resumable history baselines, and state-channel tests now hydrate from `SceneBlockSnapshot.layers` via the LayerBlock adapter—`RenderLedgerSnapshot.layer_values` remains only for the mirror + payload builders.
- Multiscale reducer helpers tolerate dims specs without precomputed `downsample`, unblocking block snapshot writes during bootstrap.

### Next steps
- Rewire `ServerLayerMirror`, notify.layers deltas, and history replay to reuse the LayerBlock delta helper, then drop `LayerVisualState`/`RenderLedgerSnapshot.layer_values` as part of the protocol flip.

# Architecture Change Log

Tracks intentional updates to the architecture spec and the authoritative modules. Add new entries (date-descending) with links to the implemented sections/files so future sessions can pick up immediately.

# 2025-11-12 — SceneBlockSnapshot now authoritative

### Runtime
- `SceneBlockSnapshot` carries `{view, axes, index, lod, camera}` plus plane/volume restore caches;
  every `RenderLedgerSnapshot` attaches the typed block bundle. Worker op-seq watcher and apply paths
  now reuse that bundle instead of fetching/scattering pose data from legacy scopes.
- Block snapshots hydrate the worker’s plane/volume caches; legacy `_plane_cache_from_snapshot` /
  `_volume_cache_from_snapshot` remain only as fallbacks when a restore cache is incomplete during bootstrap.
- `RenderInterface.apply_scene_blocks` consumes the block bundle directly,
  eliminating the `_apply_snapshot` staging layer and paving the way to delete
  `RenderLedgerSnapshot` once the single-snapshot-per-tick path lands.

### Next steps
- Collapse `_apply_snapshot` + `_apply_render_snapshot` into a single `RenderInterface`
  entry point that consumes the block bundle directly.
- Remove `_plane_cache_from_snapshot`/`_volume_cache_from_snapshot` and flip
  `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` on by default once the consolidated apply path is stable.
- Delete `RenderLedgerSnapshot` and the staging module after the interfaces converge; update docs/tests to
  treat `SceneBlockSnapshot` as the only runtime payload.
- Replace `LayerVisualState` with a typed `LayerBlock` so layer deltas travel alongside the other scene blocks
  and the legacy snapshot data model can be removed entirely.

# 2025-11-12 — Remove legacy viewport/camera ledger scopes

### Runtime & Control
- `reduce_bootstrap_state`, `reduce_dims_update`, `reduce_level_update`, `reduce_plane_restore`, `reduce_volume_restore`, and `reduce_camera_update` now emit only `{view, axes, index, lod, camera}` plus the plane/volume restore caches. `viewport.plane/volume.state`, `camera_plane.*`, and `camera_volume.*` are no longer written anywhere.
- Plane/volume restore handlers, the EGL server bootstrap, and scene builders hydrate exclusively from the restore cache blocks and `camera.main.state`.
- `apply_plane_restore_transaction` / `apply_volume_restore_transaction` stopped writing the legacy scopes, so the ledger schema now matches the block-only runtime end-state.

### Tests
- Updated `test_scene_snapshot`, `test_scene_blocks_parity`, `test_state_channel_updates`, `test_state_channel_ingest`, and the signature tests to seed/assert against the camera block + restore caches instead of the removed ledger scopes.

### Docs
- AGENTS + Phase 3 plan docs now call out that the ledger cleanup is complete and the remaining work is docs/tests parity, flipping the feature flag on by default, and the render-loop refactor.

# 2025-11-12 — op_seq watcher snapshot refresh (camera jitter fix)

### Runtime
- `_op_seq_watcher_apply_snapshot` (src/napari_cuda/server/runtime/worker/lifecycle.py) now re-pulls the ledger snapshot whenever a block signature changes and returns that snapshot to the main render loop. The loop replaces its cached `frame_state` with the refreshed snapshot before acknowledging `scene.main.op_state`, so the worker applies the exact `{view, axes, index, lod, camera, layers}` payload that triggered the watcher instead of replaying the prior frame.
- The change eliminates the camera “boomerang” jitter that appeared when worker pose reducers wrote `camera.main.state` without bumping `scene.main.op_seq`. The block-ledger watcher is now authoritative even for worker-origin poses while keepings op_seq semantics intact.
- A follow-up task tracks the more holistic render-loop refactor (single authoritative snapshot per tick / block-only watcher inputs). Until that lands, this minimal fix keeps the runtime aligned with Phase 3 goals without reintroducing the mailbox.

### Docs
- Documented the snapshot refresh plus the planned render-loop cleanup in `docs/architecture/view_axes_index_plan.md` (Phase 3 checklist) and `docs/architecture/cut_down_guides/runtime_pipeline.md` (runtime overview) so future work picks up the remaining action item.

### Tests
- Targeted runtime-specific suites are already covered by the previous Phase 3 run; no behavior flags changed. Smoke-tested manually via client logs to confirm orbit deltas no longer interleave stale poses.

# 2025-11-11 — Render runtime now block-native (no RenderUpdate mailbox)

### Runtime
- Deleted `RenderUpdateMailbox`/`RenderUpdate` and their state-machine tests; the worker now samples `{view, axes, index, lod, camera, layers}` directly off the ledger via the op_seq watcher.
- Worker loop now calls `RenderInterface.apply_scene_blocks` directly; the legacy
  `drain_scene_updates` hook remains a no-op for compatibility only.
- Simplified the new `RenderInterface` + camera tick helpers to record zoom
  hints on the worker itself (`_record/_consume_zoom_hint`); level policy now reads those helpers instead of the mailbox.
- Control-plane code (`egl_headless_server`, state update handlers, runtime API, harnesses) no longer calls `worker.enqueue_update`—ledger bumps drive the render loop exclusively.
- Scene builders synthesize `CameraBlock` data directly from the new scopes; legacy `camera_plane` / `camera_volume` mirrors are no longer expected.

### Docs
- Updated `AGENTS.md`, `docs/architecture/system_design_spec.md`, `docs/architecture/dims_camera_legacy.md`, and `docs/architecture/view_axes_index_plan.md` to document the op_seq watcher as the only runtime path and mark the mailbox removal as complete.

### Tests
- Targeted pytest runs executed:  
  `uv run pytest -q src/napari_cuda/server/tests/test_egl_worker_camera.py`  
  `uv run pytest -q src/napari_cuda/server/tests/test_state_channel_updates.py`  
  `uv run pytest -q src/napari_cuda/server/tests/test_scene_blocks_parity.py`  
  `uv run pytest -q src/napari_cuda/server/utils/_tests/test_signatures.py`  
  `uv run pytest -q src/napari_cuda/server/runtime/_tests/test_op_seq_watcher_state.py`

## 2025-11-10 — View/Axes/Index/Lod/Camera docs refresh

### Docs
- Updated every active architecture doc to distinguish the legacy
  `dims_spec + ActiveView` stack from the upcoming `view / axes / index / lod / camera`
  ledger design.
- Standardised terminology: the new schema calls the multi-dimensional cursor
  `index`; legacy payloads/scopes that still emit `current_step` are explicitly
  marked as transitional.
- Added forward-looking sections covering dual-write expectations, notify payload
  changes, and worker/runtime implications across:
  - docs/architecture/system_design_spec.md (§3, §4, §8-10, §13)
  - docs/architecture/per_function_contracts.md (§1-3, §7-8)
  - docs/architecture/roadmap.md (Protocol + schema workstream)
  - docs/architecture/layer_parity_plan.md (Target End State + Workstreams)
  - docs/architecture/level_apply_activeview_contracts.md (Gaps + Steps)
  - docs/architecture/cut_down_guides/* (dims/index + render snapshots + notify)
- Added `docs/architecture/view_axes_index_plan.md` capturing call-stack diagrams and
  phased issue breakdowns for the migration.
- Clarified that all new ledger scopes and payloads must adopt the `index`
  terminology from day one; the docs now act as the cohesive reference for the
  migration plan before implementation starts.

### Code
- No code changes in this entry; documentation-only refresh to unblock design work.

## 2025-11-09 — ActiveView authority + notify.level pipeline

### Docs
- Centralised the **architecture tree** by moving every legacy doc into `docs/archive_local/`; only `docs/architecture/` stays live (see repo restructure in docs/architecture/repo_structure.md).
- Added this change log and a central roadmap (docs/architecture/roadmap.md), plus updates across:
  - docs/architecture/system_design_spec.md (§4 Ledger Model, §10 Active View, Notify Contracts)
  - docs/architecture/per_function_contracts.md (Control transactions + notify.level broadcaster)
  - docs/architecture/cut_down_guides/* references (`server_overview`, `scene_builders_notify`, `client_integration`)

### Server
- **Commit ActiveView via transaction**: `apply_active_view_transaction` (src/napari_cuda/server/control/transactions/active_view.py) ensures reducers never poke the ledger directly. Reducers (`reduce_view_update`, `reduce_level_update`, plane/volume restores) now call this helper.
- **Mirror + broadcaster**: `ActiveViewMirror` (src/napari_cuda/server/control/mirrors/active_view_mirror.py) listens to `("viewport","active","state")` and publishes `notify.level` through `broadcast_level` (src/napari_cuda/server/control/topics/notify/level.py).
- **Protocol rename**: `notify.scene.level` → `notify.level`. Constants/payloads/builders/parsers/tests renamed under src/napari_cuda/protocol/*.
- **Handshake/Resumable features**: server advertises `notify.level` as resumable; history store tracks it; state channel stores the resume cursor.
- **Baseline replay**: new `send_level_snapshot` and connect-time baseline now replay the stored ActiveView snapshot to every client, so HUDs see mode/level on connect (src/napari_cuda/server/control/topics/notify/level.py, `notify/baseline.py`).
- **ActiveView derives from intented state**: `reduce_view_update` now passes the plane/volume intent into `apply_view_toggle_transaction`, so the ledger records the target mode+level immediately (no stale dims_spec). Plane/volume restore reducers remain worker-driven; only level switches rewrite ActiveView afterward.
- **View toggles stop mutating cache state**: `reduce_view_update` now just rewrites the dims spec; plane/volume caches and ActiveView are updated only by worker restores. HUD no longer flashes stale levels. We still write ActiveView inside the toggle transaction to keep mode+ndisplay atomic until the worker path is fully migrated.
- **Next steps**: teach the runtime planner/restores to read `("viewport","active","state")` directly, then delete `DimsSpec.current_level` and the redundant `multiscale.level` entry.
- **Render + notify paths consume ActiveView**: `snapshot_multiscale_state`, `snapshot_scene_blocks`, `build_ledger_snapshot`, and the `notify.dims` mirror all read `("viewport","active","state")` for mode+level; dims specs only supply shapes/order metadata. `NotifyDimsPayload` now carries the authoritative ActiveView level/mode, and dedupe signatures honor that.

### Client
- **State channel** now negotiates and consumes `notify.level` (see `_REQUIRED_FEATURES`, `_RESUMABLE_TOPICS`, and `_ingest_notify_level` in src/napari_cuda/client/control/control_channel_client.py). Channel threads forward the callback into `ClientStreamLoop`.
- **Stream runtime**: `_ingest_notify_level` (src/napari_cuda/client/runtime/stream_runtime.py) hand-offs ActiveView updates to the presenter on the Qt thread.
- **Presenter HUD**: `PresenterFacade.apply_active_view` records mode+level for the HUD overlay; dims notifications only handle dims metadata now.

### Tests
- Targeted pytest runs executed: `uv run pytest -q src/napari_cuda/server/tests/test_active_view.py`, `uv run pytest -q src/napari_cuda/protocol/_tests/test_envelopes.py::test_notify_level_roundtrip`.

Open follow-ups (carry into next session):
1. Shift remaining “multiscale level” writes in transactions to rely solely on ActiveView.
2. Remove `current_level` from `DimsSpec` and `NotifyDimsPayload` entirely; consumers must read level from ActiveView/notify.level.
3. Remove the lingering state channel try/except around ingest callbacks (now done for level; repeat for others when steady).
4. Expand protocol tests to cover notify.level resume paths once the full suite is runnable in CI.
# 2025-11-13 — Drop LayerVisualState shim (protocol flip plan)

### Decision
- We confirmed no external clients depend on the legacy notify/state payloads, so the project can
  evolve the protocol alongside the server. Layer appearance should ride the `SceneBlockSnapshot`
  LayerBlocks everywhere (runtime, control, notify), then we delete `RenderLedgerSnapshot.layer_values`
  and the `LayerVisualState` dataclass entirely.

### Actions
- Updated `AGENTS.md`, `NEXT_STEPS.md`, and `docs/architecture/view_axes_index_plan.md` to spell out the
  new scope: migrate notify builders + mirrors to LayerBlocks, flip the protocol, and then remove the shim.
- Runtime/render loop already consumes `SceneBlockSnapshot` exclusively (see 2025-11-12 entry); remaining
  work is purely control/protocol + documentation.
- Future commits should treat LayerBlocks as the only source of truth; any new code that touches layer
  visuals must do so via `scene_layers.<id>.block` instead of `layer_values`.
# 2025-11-13 — ServerLayerMirror + notify.layers sourced from LayerBlocks

### Control
- `ServerLayerMirror` now subscribes to `scene_layers.*.block`, reuses the shared LayerBlock diff helper, and emits deltas derived from LayerBlocks instead of `RenderLedgerSnapshot.layer_values`.
- Notify baseline builders and `_split_layer_state` can ingest `LayerBlockDelta` structures, keeping the existing payload shape but sourcing values from the new blocks.

### Next steps
- Switch notify.layers history/replay to pass LayerBlock deltas end-to-end, then flip the protocol schema and delete `LayerVisualState`.
