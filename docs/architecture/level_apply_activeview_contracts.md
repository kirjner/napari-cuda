Level Apply + ActiveView Contracts

Status: Proposed (to converge implementation and docs)
Owners: Server Runtime Team, Client Runtime Team
Last updated: 2025-11-10

Constraints (must)
- Apply-only-via-snapshot: Worker applies a level only when it arrives via a `RenderLedgerSnapshot` pulled from the ledger. Apply helpers do not re-decide or override snapshot level.
- Decision sources (only): A level decision can originate from exactly one of: (A) worker intent mailbox (policy), (B) a restore, or (C) bootstrap. Each source writes the level to the ledger first; render then consumes a snapshot.
- Derived ActiveView: `('viewport','active','state')` is derived from ledger state by a mirror. Reducers/transactions do not write ActiveView. Broadcasters build `notify.level` only from the ActiveView ledger key.
- Upcoming schema alignment: the new `view / axes / index / lod / camera` ledger scopes
  replace the monolithic dims spec as the authoritative source for level/index
  decisions. This document assumes any new helpers derive from those scopes
  while legacy reducers continue to dual-write `dims_spec` for compatibility.

Current adherence
- Policy → ledger → snapshot → render (2A, 1):
  - `EGLHeadlessServer._handle_worker_level_intents` commits level via `reduce_level_update(...)`, then builds a `RenderLedgerSnapshot` and enqueues it for render.
    - File: src/napari_cuda/server/app/egl_headless_server.py
- Plane restore → ledger → snapshot (2B, 1):
  - `reduce_plane_restore(...)` writes dims level and plane pose; server refreshes snapshot and render consumes it.
    - Files: src/napari_cuda/server/control/state_update_handlers/view.py, src/napari_cuda/server/control/state_reducers.py
- Bootstrap → ledger → snapshot (2C, 1):
  - Dataset activation records ledger state and refreshes a snapshot before rendering.
    - File: src/napari_cuda/server/app/dataset_lifecycle.py (via server app lifecycle hooks)
- `notify.level` broadcast from ActiveView:
  - `ActiveViewMirror` subscribes to the ledger and builds `notify.level` from `('viewport','active','state')`.
    - File: src/napari_cuda/server/control/mirrors/active_view_mirror.py

Gaps (breaks)
- Apply path overrides level (violates 1):
  - [FIXED] Apply now uses `snapshot_level=int(RenderLedgerSnapshot.current_level)` verbatim. Remaining work is listed below for other gaps.
- Enter-3D preloads level (violates 1 and weakens 2):
  - `_enter_volume_mode` loads a level and mutates current index before the ledger confirm/snapshot, then emits a LevelSwitchIntent. [PARTIAL] Preload removed; level selection still needs to be committed via viewport state level.
- Enter-3D preloads level (violates 1 and weakens 2):
  - `_enter_volume_mode` loads a level and mutates current index before the ledger confirm/snapshot, then emits a LevelSwitchIntent.
    - File: src/napari_cuda/server/runtime/bootstrap/setup_camera.py
- ActiveView written by reducers (violates 3):
  - Reducers/transactions write `('viewport','active','state')` during bootstrap, view toggles, and policy level updates.
    - Files: src/napari_cuda/server/control/state_reducers.py, src/napari_cuda/server/control/transactions/view_toggle.py
- Volume restore does not write a level (partial 2B):
  - `reduce_volume_restore(...)` persists volume pose/state but does not record a level under `('multiscale','main','level')`.
    - File: src/napari_cuda/server/control/state_reducers.py

Steps to comply 100%
1) Apply is read-only for level
   - In render apply helpers, do not re-resolve volume level; load the level specified by `RenderLedgerSnapshot.current_level`. If a requested level violates budgets, emit a `LevelSwitchIntent` first to commit an allowed level to the ledger; then consume the updated snapshot.
2) Gate enter/exit 3D through the ledger
   - Remove direct level loads and `_set_current_level_index(...)` from `_enter_volume_mode`. Entering volume should produce a `LevelSwitchIntent` (coarsest or policy-selected), commit via `reduce_level_update(...)`, refresh snapshot, and only then apply.
3) Restore is a decision source (2B)
   - Plane restore remains. Make volume restore persist a level under `('multiscale','main','level')` via the same transaction family (no ActiveView writes), then render consumes the snapshot.
4) Derive ActiveView from ledger (3)
   - Add/extend a mirror that subscribes to `('dims','main','dims_spec')` and `('multiscale','main','level')`, and writes `('viewport','active','state')` = {mode, level} where mode derives from dims.ndisplay and level from dims.current_level (plane) or multiscale.level (volume). Remove reducer/transaction writes to ActiveView.
5) Keep broadcasters simple
   - `ActiveViewMirror` broadcasts `notify.level` from the ActiveView ledger key; `ServerDimsMirror` mirrors mode+level from ActiveView into `notify.dims`. No apply-side notify.
6) Seed correctly on startup
   - On server start, mirrors compute initial ActiveView from the ledger snapshot; no bootstrap-time ActiveView writes are needed.
7) Tests/guards
   - Add tests that: (a) apply uses snapshot level verbatim, (b) only policy/restore/bootstrap mutate level, and (c) ActiveView is maintained solely by the derivation mirror.

Actionable TODO Checklist (authoritative)
- [ ] Apply path is read-only for level
  - Files: src/napari_cuda/server/runtime/render_loop/applying/apply.py
  - Remove level re-resolution for volume; use `snapshot.current_level` exactly.
  - Budgets: if a requested level violates caps, emit a LevelSwitchIntent first, commit allowed level, then apply from the next snapshot.
  - Acceptance: no calls that override level during apply; logs show levels applied match snapshot.
- [ ] Gate enter/exit 3D through ledger (no preload)
  - Files: src/napari_cuda/server/runtime/bootstrap/setup_camera.py
  - Remove direct level loads/current-index mutations in `_enter_volume_mode`; emit only a LevelSwitchIntent.
  - Acceptance: enter-3D sequence is intent → reducer commit → snapshot → apply; no pre-snapshot loads.
- [ ] Consolidate writers to `('viewport','state','level')`
  - Files: reducers/transactions touching level
  - Policy: server handler writes `viewport.state.level` on LevelSwitchIntent.
  - Restore: plane/volume restore write `viewport.state.level` (not multiscale/dims).
  - Bootstrap: seed `viewport.state.level` (coarsest for volume).
  - Acceptance: no writes to `multiscale.main.level` or `dims_spec.current_level` as authoritative sources.
- [ ] Derive ActiveView by mirror only
  - Files: new/extended mirror under src/napari_cuda/server/control/mirrors/
  - Subscribe to `dims_spec.ndisplay` and `viewport.state.level`; write `('viewport','active','state')={mode,level}`.
  - Remove ActiveView writes from reducers/transactions (bootstrap, view toggle, level update).
  - Acceptance: grep shows no state_reducers/transactions touching ActiveView; notify.level still broadcasts correctly.
- [ ] Enforce “volume = coarsest” in policy layer
  - Files: src/napari_cuda/server/runtime/lod/level_policy.py
  - Keep coarsest enforcement inside policy/intent selection (not in apply).
  - Acceptance: enter-3D intent selects coarsest; apply uses snapshot verbatim.
- [ ] Compatibility + removals
  - Optional shim mirror to backfill `multiscale.main.level` from `viewport.state.level` during transition.
  - Remove legacy readers/writers once call-sites rely on `viewport.state.level` and derived ActiveView.
  - Acceptance: codebase no longer writes legacy keys; doc notes updated.
- [ ] Tests and assertions
  - Apply uses snapshot level; only three sources mutate level; ActiveView derived only; enter-3D gated.
  - Add lightweight assertions in apply path to guard against future overrides.
