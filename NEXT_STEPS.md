# napari-cuda Roadmap (View/Axes/Index Ledger)

_Last updated: 2025-11-12_

## Where We Are
- ✅ Reducers, transactions, and notify builders emit and consume `{view, axes, index, lod, camera}` blocks plus restore caches.
- ✅ Worker op-seq watcher replays a fresh ledger snapshot whenever block signatures mutate, eliminating stale camera jitter when `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1`.
- ✅ Planner/mailbox stack is bypassed under the flag; worker now samples the ledger directly each tick and tracks per-block signatures locally.
- ⚠️ Render loop + apply helpers still depend on `RenderLedgerSnapshot`’s legacy pose fields (`plane_*`, `volume_*`, `current_step`, `dims_spec`), so we pull/transform the same data twice.
- ⚠️ `PlaneViewportCache` / `VolumeViewportCache` mirrors still exist in staging helpers and bootstrap code. They should become worker-only snapshots once block consumers are authoritative.

## Immediate Next Steps (Phase 3 Cleanup)
1. **Block-native render loop & worker apply**
   - Teach `render_loop/applying/*` to accept `SceneBlockSnapshot` + restore caches directly, instead of reconstructing pose data from `RenderLedgerSnapshot`.
   - Collapse `pull_render_snapshot` + `_op_seq_watcher_apply_snapshot` double-pull to a single authoritative snapshot per tick (or apply blocks directly) so the worker never renders stale pose data.
   - Update `_apply_snapshot` to treat `RenderLedgerSnapshot` as a thin view over the block payloads (no back-conversion to `dims_spec` for invariants already guaranteed by the block helpers).
2. **Eliminate ledger copies of viewport caches**
   - Remove `_plane_cache_from_snapshot` / `_volume_cache_from_snapshot` once the worker ingests restore caches straight from the block payloads.
   - Keep restore caches as worker-only dataclasses; reducers remain responsible for ledger persistence through `write_*_restore_cache`.
3. **Flag flip + cleanup**
   - Enable `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` by default after the render loop is block-native, then remove feature-flag branches and the legacy lerp path entirely.
   - Delete the remaining planner/mailbox artifacts (`ViewportPlanner`, `PlaneViewportCache`/`VolumeViewportCache` ledger mirrors, `RenderUpdate` shims) once the flag is default-on.
4. **Docs/tests refresh**
   - Update `docs/architecture/view_axes_index_plan.md`, `dims_camera_legacy.md`, and API docs to describe the block-only ledger pipeline (restore caches + `camera.main.state` as the only pose source).
   - Extend parity tests in `src/napari_cuda/server/tests/test_state_channel_updates.py` to assert block snapshots match the worker apply expectations (view mode, axis order, cursor, camera pose).

## Upcoming Passes
| Pass | Focus | Key Tasks | Exit Criteria |
| ---- | ----- | --------- | ------------- |
| **Pass A** | Render loop block ingestion | - Add block-aware staging helpers that operate on `SceneBlockSnapshot`<br>- Replace `snapshot_render_state` usages in worker apply path<br>- Remove redundant ledger pulls per tick | Worker applies dims/camera/lod directly from blocks behind the flag; watcher no longer re-pulls snapshots. |
| **Pass B** | Cache + bootstrap cleanup | - Drop ledger copies of `PlaneViewportCache` / `VolumeViewportCache`<br>- Update bootstrap + tests to source poses from `camera.main.state` + restore caches<br>- Delete legacy viewport scopes/doc sections | No references to `viewport.*` or `camera_plane/camera_volume`; restore caches seeded exclusively via typed helpers. |
| **Pass C** | Flag default + deletion | - Flip `ENABLE_VIEW_AXES_INDEX_BLOCKS` default to `True`<br>- Remove flag-guarded legacy branches in reducers, notify builders, worker runtime<br>- Prune planner/mailbox files and CI expectations | Block-only pipeline is the sole runtime path; docs/tests no longer mention the legacy planner/mailbox. |

## Supporting Work
- **Telemetry:** ensure metrics capturing render/apply timings still function once the render loop reads blocks directly; rename metrics where they reference “viewport” terminology.
- **State-channel helpers:** expose a shared “seed ledger for tests” helper that boots block + cache payloads through the reducers instead of duplicating defaults.
- **QA:** run `uv run pytest -q -m "not slow"` with `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1` and without to confirm zero-regression before flipping the flag.

## Risks & Mitigations
| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Residual snapshot consumers expect `dims_spec` fields | Medium | Introduce typed adapters that expose read-only views for legacy callers until they are updated; avoid duplicating ledger writes. |
| Worker cache removal breaks restore flow | Medium | Gate cache deletion behind a feature flag and prove parity tests using reducer-seeded restore caches before removing ledger scopes. |
| Flag flip causes unexpected client notify payloads | Low | Notify builders already emit block payloads; add release notes + protocol doc updates before enabling by default. |

## Test Plan Expectations
- `uv run pytest src/napari_cuda/server/tests/test_state_channel_updates.py`
- `uv run pytest src/napari_cuda/server/runtime/_tests -m "not slow"`
- `uv run pytest src/napari_cuda/server/runtime/render_loop/_tests -m "not slow"`
- GPU-integration smoke: `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1 uv run pytest tests/gpu -m "not slow and gpu"` (optional CI gate before flag flip).
