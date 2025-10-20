# Server Migration Plan — From Current State to Ledger‑Centric Design

This guide lays out a pragmatic path to reach the greenfield architecture in `docs/server/ARCHITECTURE.md`, with focused steps that keep changes minimal and measurable.

## Phase 0 — Lock in current stability ✅

- Apply dims coherently and set `ndisplay` last so napari’s fit sees consistent state.
  - Verify in `src/napari_cuda/server/runtime/egl_worker.py:938–1001`.
- Ignore `dims.mode` for mode decisions; decide 2D/3D from `view.ndisplay` only.
  - Ensure no remaining references affect mode flips in the worker.

Status: Both invariants are enforced in the worker apply path today.

## Phase 1 — Dims mirror correctness and gating ✅

- Derive `mode` from `view.ndisplay`; stop requiring `dims.mode` as a mandatory key.
- Add op gating: do not emit until `scene.op_state == "applied"` for the leading op in progress.
  - File: `src/napari_cuda/server/control/mirrors/dims_mirror.py`
  - Changes:
    - Relax required key set; compute `mode` from ledger `view.ndisplay`.
    - Maintain last payload, plus a small buffer for the open op.

Status: Mirror derives mode from the ledger and buffers notifications until the worker closes the op fence.

## Phase 2 — Camera is applied-only ✅

- Ensure only the render worker writes camera pose to the ledger (applied state).
- Client camera intents continue via queue (`CameraDeltaCommand`).
  - File: `src/napari_cuda/server/control/control_channel_server.py`
    - Reject direct `camera.*` ledger writes; keep ack + broadcast based on deltas.
  - File: `src/napari_cuda/server/app/egl_headless_server.py`
    - `_commit_applied_camera(...)` remains the only ledger writer for `camera.*`.

Status: Camera intents feed `CameraCommandQueue`; `_commit_applied_camera` (worker side) is the single writer for `camera.*` and per-mode caches.

## Phase 3 — RenderTxn scaffolding (atomic apply) ✅

- Introduce `src/napari_cuda/server/runtime/render_txn.py` with:
  - Builder from `RenderLedgerSnapshot` → (dims target, level target, camera hints) → plan.
  - Apply: dims.ndim → dims.order → dims.current_step → dims.ndisplay; then level → camera; commit applied state.
  - Temporary napari fit suppression within the apply scope so fit runs once at the end.
- Wire in `egl_worker.py` to use the txn for frame application.

De‑queue plan:
- Remove `enqueue_scene_state` calls and fold volume/layer property application into RenderTxn.
- Trim `drain_scene_updates` to handle only multiscale requests (or call level apply directly and remove the queue entirely).
- Control loop remains unchanged; it still drives ticks and camera deltas.

Status: `render_snapshot.apply_render_snapshot` is the single apply site; scene mailboxes are gone. Camera queue + worker ack close the loop.

## Phase 4 — Deterministic plane restore (no fallbacks) ✅

- Persist per‑mode snapshots in the ledger (worker‑applied only):
  - Plane: `camera_plane.center`, `camera_plane.zoom`, `camera_plane.rect`; plus `view_cache.plane.level` and `view_cache.plane.step`.
  - Volume: `camera_volume.center`, `camera_volume.angles`, `camera_volume.distance`, `camera_volume.fov`.
- Mode switches:
  - 2D → 3D: snapshot plane cache before switch; apply volume pose from `camera_volume.*` (if absent, compute canonical TT pose once and write it).
  - 3D → 2D: read `view_cache.plane.*` and `camera_plane.rect`; project rect (world→level) and apply slab, then apply plane camera pose.
- Remove private plane‑restore flags/caches and any “full slab” overrides.
  - Files: `src/napari_cuda/server/app/egl_headless_server.py` (cache writes in `_commit_applied_camera`), `src/napari_cuda/server/runtime/egl_worker.py` (read/apply on mode switch), `src/napari_cuda/server/runtime/worker_runtime.py` (world‑rect→ROI projection helper).

Status: Worker toggles reuse ledger-backed caches; `_restoring_plane` scaffolding removed. ROI helpers now live with `render_snapshot`.

## Phase 5 — Layer mirror

- Add `src/napari_cuda/server/control/mirrors/layer_mirror.py` to present normalized per‑layer control state.
- Optionally fold into `notify.scene` or keep separate `notify.layers` channel.

## Phase 6 — Remove scene bag merging

- Stop merging “pending layer updates” or volume hints into render snapshots.
- Render snapshots are built purely from the ledger; keep only minimal caches for IDs if required.
  - Files: `src/napari_cuda/server/runtime/render_ledger_snapshot.py`, `src/napari_cuda/server/scene/data.py`.

## Phase 7 — Tests & validation

- Unit tests:
  - Dims apply order (no IndexError; ndisplay last).
  - 2D↔3D toggle without flicker (dims mirror emits once, op‑gated).
  - 3D→2D plane restore: no letterbox; world rect preserved.
  - Level switch preserves 2D rect (no jolt to center when `preserve_view_on_switch=True`).
  - Camera via queue only; applied camera ledger matches the on‑screen pose.
  - No reliance on `enqueue_scene_state`/`drain_scene_updates` for scene state; RenderTxn is the single apply site.

Status: Harness exercises toggles/plane restore/level switches. Still need focused unit coverage for view toggle transaction, ROI counts, and layer mirror once implemented.

## Module‑by‑Module Worklist (current → target)

- `src/napari_cuda/server/control/mirrors/dims_mirror.py`
  - Derive mode from `view.ndisplay`; relax required keys.
  - Add simple op gating using `scene.op_seq/op_state` once available.

- `src/napari_cuda/server/runtime/egl_worker.py`
  - Ensure mode decisions only use `snapshot.ndisplay`.
  - Call into `render_txn.apply(...)` to batch dims/level/camera.
  - Keep `ndisplay` last when touching dims; remove private plane‑restore logic after Phase 4 and use ledger caches.

- `src/napari_cuda/server/runtime/worker_runtime.py`
  - Keep pure helpers for slab/volume application and ROI computation.
  - Use `camera_plane.rect` projection as the canonical ROI source in Phase 4.

- `src/napari_cuda/server/runtime/render_ledger_snapshot.py`
  - Build strictly from ledger; do not merge non‑ledger “scene bag” entries.

- `src/napari_cuda/server/app/egl_headless_server.py`
  - Continue to `_commit_level_snapshot(...)` and `_commit_applied_camera(...)` only from worker callbacks.
  - Introduce/propagate `scene.op_seq/op_state` with reducers for op gating.

- `src/napari_cuda/server/control/control_channel_server.py`
  - View toggle path unchanged (writes `view.ndisplay` only); keep keyframe/IDR request logic.
  - Enforce camera deltas via queue; no direct `camera.*` ledger writes.

- `src/napari_cuda/server/data/lod.py`
  - Stable; ensure `set_dims_range_for_level` + `apply_level` order remains viewer‑safe.

- `src/napari_cuda/server/rendering/viewer_builder.py`
  - Keep bootstrap minimal; do not over‑fit; worker txn becomes the authoritative applier.

## ROI follow-up

- Consolidate ROI helpers owned by `render_snapshot` (currently split with `worker_runtime.py`).
- Add regression tests that count ROI applications per transaction (toggle, level switch) to guard against over-apply.
- Ensure the harness paths call the public render snapshot entry point instead of legacy shims when asserting ROI behaviour.
## Order of Operations Reference (worker txn)

1) Decide mode from `view.ndisplay` only.
2) dims: `ndim` → `order` → `current_step` → `ndisplay` (last).
3) level: select/apply with budgets, update napari layer.
4) camera: set range then apply rect/center/zoom (2D) or TT (3D).
5) commit: write applied level/step and camera pose to ledger; close op.

## Notes

- Encoder resets and streaming policy are orthogonal; prefer IDR‑only on mode toggle to reduce flicker.
- Keep assertions for invariants; catch/log only at I/O and subsystem boundaries.

## Refactoring Principles (min indirection, shallow stacks)

- One public entry per module: reducers, mirrors, worker txn. Avoid nested dispatchers.
- Prefer free functions for stateless helpers; keep classes for stateful components only (worker, mirrors).
- Keep the call stack shallow on hot paths:
  - Control path: websocket handler → reducer (validate, normalize, batch) → ledger.
  - Apply path: worker mailbox drain → render_txn.apply (single function) → ledger commit.
  - Mirror path: ledger callback → payload build → broadcaster.
- No double indirection (e.g., no scene bag merging in snapshots). Ledger is the truth.
- Keep atomic scopes explicit: RenderTxn is the only place where dims/level/camera are applied together.

## Decomposition Plan (large modules → focused files)

- `runtime/egl_worker.py` → keep orchestration; move:
  - Atomic apply into `runtime/render_txn.py`.
  - Camera delta handling into `runtime/camera_controller.py`.
  - Pure ROI/level helpers remain in `server/data/` and `server/runtime/worker_runtime.py`.
- `control/control_channel_server.py` → keep websocket/routing; state changes go through `state_reducers.py` only.
- `control/state_reducers.py` → clarify sections: view, dims, multiscale, camera (applied ack only), layer, volume; avoid cross‑module imports.
- `rendering/viewer_builder.py` → bootstrap only; do not over‑apply dims/fit. Worker txn becomes authoritative applier.

## ROI follow-up

- Consolidate ROI helpers that moved alongside `render_snapshot` and drop any residual duplicates in `worker_runtime.py`.
- Add regression tests (state-channel harness is a good spot) that count ROI applies for toggles and level switches so we catch over-apply regressions.
- Ensure harness shortcuts call the public render snapshot entry point so ROI assertions match production code paths.

## Naming & API Conventions (server)

- Scopes/keys:
  - `view.ndisplay` (2|3), not `dims.mode`.
  - `camera.*` is the current applied pose; `camera_plane.*` and `camera_volume.*` are per‑mode caches.
  - `view_cache.plane.level/step` is the plane target on 3D→2D.
- Origins:
  - Client intents: `origin='client.state.view'|'client.state.dims'|...`.
  - Worker applied: `origin='worker.state.level'|'worker.state.camera'|'worker.state.camera_plane'|'worker.state.camera_volume'`.
- Functions:
  - Keep verbs clear: `apply_*`, `reduce_*`, `build_*`, `resolve_*`, `select_*`.

## Indirection & Call‑stack Budget

- Hot path targets:
  - Reducer depth ≤ 2 calls beyond handler.
  - Worker apply depth ≤ 3 calls (drain → render_txn → per‑op helpers).
  - Mirrors depth ≤ 2 (callback → payload → broadcast).
- No dynamic getattr/try/except in hot paths; assert invariants.

## Milestones & Checkpoints

- M1: Phase 1 complete (dims mirror derive+gate), tests for consistent notify.
- M2: Phase 3 scaffold (RenderTxn wired), 2D/3D toggles stable without flicker.
- M3: Phase 4 complete (caches, no fallbacks), 3D→2D restore exact.
- M4: Phase 6 complete (ledger‑only snapshots), scene bag retired.
- M5: Test suite coverage for dims/level/camera flows; perf budget validated.
