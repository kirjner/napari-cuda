# Architecture Change Log

Tracks intentional updates to the architecture spec and the authoritative modules. Add new entries (date-descending) with links to the implemented sections/files so future sessions can pick up immediately.

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
