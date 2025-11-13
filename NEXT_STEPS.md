# napari-cuda Roadmap (View/Axes/Index Ledger)

_Last updated: 2025-11-12_

## Where We Are
- ✅ Reducers/transactions emit `{view, axes, index, lod, camera, layers}` blocks plus restore caches; worker runtime now ingests a single `SceneBlockSnapshot` per tick.
- ✅ Op-seq watcher + render interface reuse the block snapshot, so the render loop no longer reconstructs pose data from `RenderLedgerSnapshot`.
- ✅ Planner/mailbox stack is bypassed under the flag; worker tracks per-block signatures locally.
- ⚠️ Notify builders, the layer mirror, and snapshot helpers still hydrate from `RenderLedgerSnapshot.layer_values` instead of `SceneBlockSnapshot.layers`.
- ⚠️ `LayerVisualState` / `RenderLedgerSnapshot` remain as shims for the notify protocol even though no external clients require backward compatibility.

## Immediate Next Steps (Phase 3 Cleanup)
1. **Control/notify consumers → LayerBlocks**
   - Teach `_collect_default_visuals`, notify builders, and state-channel tests to read `SceneBlockSnapshot.layers` (or the ledger `scene_layers.*.block`) directly.
   - Mirror/cache modules should keep emitting the same payload shape for now, but populate it from LayerBlocks instead of `layer_values`.
2. **Protocol flip (no compat clients)**
   - Update notify payload builders + resumable history to speak LayerBlocks natively (TypedDict mirroring the block schema) and remove `LayerVisualState`.
   - Delete `RenderLedgerSnapshot.layer_values` + the dual-write shim once the new payload lands; adjust stub clients/tests alongside the server.
3. **Flag flip + cleanup**
   - Enable `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` by default, prune planner/mailbox artifacts, and remove `_plane/_volume_cache_from_snapshot` fallbacks.
4. **Docs/tests refresh**
   - Document the protocol change + block-only pipeline (`docs/architecture/view_axes_index_plan.md`, `dims_camera_legacy.md`).
   - Extend parity tests (`test_state_channel_updates`, notify history tests) so LayerBlock serialization/deserialization is covered end-to-end.

## Upcoming Passes
| Pass | Focus | Key Tasks | Exit Criteria |
| ---- | ----- | --------- | ------------- |
| **Pass A** | Control/notify block ingestion | - Consume `SceneBlockSnapshot.layers` (or `scene_layers.*.block`) everywhere in notify builders, mirrors, and snapshot helpers<br>- Keep emitting existing payloads via the shim until the protocol flip lands | Server-side state/notify code no longer reads `RenderLedgerSnapshot.layer_values`. |
| **Pass B** | Protocol flip + shim removal | - Update notify payload schema + clients/tests to carry LayerBlocks directly<br>- Delete `LayerVisualState`, `RenderLedgerSnapshot.layer_values`, and the dual-write reducer paths | Layer appearance travels exclusively as LayerBlocks end-to-end. |
| **Pass C** | Flag default + pipeline cleanup | - Enable `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` by default<br>- Remove planner/mailbox artifacts and snapshot fallbacks<br>- Refresh docs/tests for the block-only runtime | Block-only pipeline is the sole runtime path; docs/tests treat it as the contract. |

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
