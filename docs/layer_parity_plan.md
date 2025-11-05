`layer-parity` Roadmap
======================

This branch is the vehicle for closing the parity gap between napari’s
layer feature set and the napari-cuda remote stack. Below is the
authoritative plan that describes:

1. What napari’s `Image` layer exposes.
2. How that compares to the current remote implementation.
3. The concrete work required on the client and server.
4. A staged execution plan so we can land changes incrementally without
   breaking the pipeline.

------------------------------------------------------------------------

1. Feature Inventory vs. Current Support
----------------------------------------

### 1.1 Core visual controls

| napari control                | Client today                                           | Server today                                                   | Notes                                                            |
|-------------------------------|--------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------|
| `visible`                     | intents emitted, applied                               | reducer + apply path in place                                  | parity ✓                                                         |
| `opacity`                     | intents emitted, applied                               | reducer + apply path in place                                  | parity ✓                                                         |
| `blending`                    | **now emitted (new)**                                  | reducer + apply path in place                                  | verify downstream visuals update                                 |
| `interpolation2d/3d`          | not emitted, not applied                               | reducer normalizes but worker ignores                          | needs full wiring                                                |
| `rendering`                   | intents emitted, applied                               | reducer + worker setter                                        | parity ✓                                                         |
| `colormap`                    | intents emitted, applied                               | reducer + worker setter                                        | parity ✓                                                         |
| `gamma`                       | intents emitted, applied                               | reducer + worker setter                                        | parity ✓                                                         |
| `contrast_limits`             | intents emitted, applied                               | reducer + worker setter                                        | parity ✓                                                         |
| `depiction` (volume)          | not emitted, worker ignores                            | reducer stores but apply skips                                 | needs plumbing                                                   |
| `attenuation` (volume)        | not emitted, worker setter exists                      | reducer stores                                                 | only wire client + tests                                         |
| `iso_threshold` (volume)      | not emitted, worker setter exists                      | reducer stores                                                 | only wire client + tests                                         |
| `metadata`                    | accepted, not editable                                 | stored + round-tripped                                         | decide edit policy                                               |
| `thumbnail`                   | worker pushes; client read-only                        | handled                                                        | parity ✓                                                         |

### 1.2 Geometry & transform controls

| napari control            | Client today                | Server today                                   | Notes                  |
|---------------------------|-----------------------------|------------------------------------------------|------------------------|
| `scale` / `translate`     | ignored                     | stored only via scene snapshot                  | need ledger+apply      |
| `rotate` / `shear`        | ignored                     | ignored                                        | same as above          |
| `affine`                  | ignored                     | ignored                                        | same as above          |
| slicing plane (position, thickness, normal) | ignored | snapshot only (render mode volume)             | expose if needed       |
| experimental clipping planes | ignored                 | ignored                                        | optional, low priority |

### 1.3 Multilayer / data plumbing

- Client: `NapariLayerIntentEmitter` mirrors exactly one `RemoteImageLayer`.
- Server: worker runtime tracks a single napari layer (`_napari_layer`).
- Thumbnail capture chooses “first layer” heuristically.
- Ledger schema supports multiple IDs (`layer-*`), but runtime + mirrors do
  not.

------------------------------------------------------------------------

2. Required Changes
-------------------

### 2.1 Client (napari_cuda/client)

1. **Intent emission parity**
   - Extend `PROPERTY_CONFIGS` with:
     - `interpolation` (2D & 3D events).
     - `depiction`, `attenuation`, `iso_threshold`.
     - Optional: `custom_interpolation_kernel_*` if we expose it.
   - Update `_block_*` helpers so baseline payloads seed all properties.
   - Add unit coverage mirroring existing tests for each property.

2. **Remote layer application**
   - Update `RemoteImageLayer._apply_controls` to set new properties and
     emit napari events (matching napari expectations).
   - Ensure transform updates (`scale`, `translate`, etc.) trigger
     updates when received from the server.
   - Decide on metadata editability (probably read-only for now).

3. **Layer registry groundwork**
   - Build a registry that maps layer IDs → `RemoteImageLayer` instances.
   - Make sure new layers are created/removed on `notify.scene`/`notify.layers`.
   - Ensure per-layer thumbnails and controls route through the registry.

### 2.2 Server (napari_cuda/server)

1. **Ledger reducers**
   - Confirm `reduce_layer_property` normalization covers the new keys
     (interpolation, depiction, etc.). Add cases if missing.
   - Wire transform reducers (scale/translate/rotate/shear/affine) or
     decide to keep them derived from ROI geometry.

2. **Render-loop application**
   - Enhance `apply_layer_visual_state` to apply:
     - Interpolation updates (both 2D & 3D) to the active `VisPyImageLayer`.
     - Depiction, attenuation, iso threshold (volume visual).
     - Transform values (`scale`, `translate`, `rotate`, `shear`, full `affine`).
   - Ensure transform changes trigger `_mark_render_tick_needed`.

3. **Multi-layer runtime support**
   - Store multiple napari layer handles in `RenderApplyInterface`.
   - Route per-layer updates via `LayerVisualState.layer_id`.
   - Update thumbnail capture to iterate over all registered layers
     (or targeted subsets).
   - Adjust notify dedupe to operate per-layer signature.

4. **Scene/notify payloads**
   - Include new controls in baseline `snapshot_scene` & notify deltas.
   - Ensure volume/plane metadata (plane slicing, clipping) are
     serialized when relevant.

5. **Testing**
   - Extend unit tests for reducers and apply helpers to cover the new keys.
   - Add integration tests (or manual scenarios) for multi-layer updates.

------------------------------------------------------------------------

3. Staged Execution Plan
------------------------

### Stage 1 – Control parity (single layer)

1. Client: add emission + tests (`interpolation`, `depiction`,
   `attenuation`, `iso_threshold`).
2. Server: confirm reducers; extend worker apply path for these fields.
3. Verify: logging or manual test toggling each control updates the
   worker and round-trips via `notify.layers`.

### Stage 2 – Transform & advanced controls

1. Decide transform story (do we fully support user-driven transforms?).
2. Implement ledger storage and worker apply for scale/translate/rotate/shear/affine.
3. Expose plane/clipping controls if desired (optional).
4. Update tests and docs.

### Stage 3 – Multi-layer runtime

1. Introduce layer registry on client & server.
2. Allow `notify.scene` to instantiate additional layers.
3. Update worker to apply per-layer `LayerVisualState`.
4. Refresh thumbnail logic + dedupe per-layer.

### Stage 4 – Polish & validation

1. Add telemetry/logging (debug-level toggles) for new paths.
2. Run end-to-end validations (UI toggles, streaming parity, volume vs plane).
3. Update documentation, developer guides, and branch checklist.

------------------------------------------------------------------------

4. Open Questions / Follow-Ups
------------------------------

- Do we need to expose metadata editing, or leave it read-only?
- Should transforms be fully round-tripped or remain server-driven?
- How aggressively do we support experimental clipping planes?
- Multi-layer ordering / z-order semantics (tie-breaks when render stacking).

Document owners: `layer-parity` branch team. Update as implementation
progresses.

------------------------------------------------------------------------

Appendix A – Axis Semantics Consolidation
-----------------------------------------

### A.1 Objective

Before adding any further layer features we must eliminate the ad-hoc axis
heuristics that accumulated across server control, runtime, data loaders, and
the thin client. All axis interpretation (labels, order, displayed set, level
extents, world scale, margins) will flow through a single immutable data
structure that is produced exactly once per notify/update cycle. This keeps
plane/volume switching, multiscale level changes, thick slicing, and margin +
projection semantics aligned across the stack.

### A.2 Data Model

1. Introduce a shared module `napari_cuda/shared/axis_spec.py` housing the
   canonical schema:

   ```python
   AxisRole = Literal["x", "y", "z", "channel", "time", "depth", "unknown"]

   @dataclass(frozen=True)
   class WorldSpan:
        start: float
        stop: float
        step: float
        scale: float | None = None
        def steps_to_world(self, steps: float) -> float: ...
        def clamp_step(self, value: float, *, step_count: int) -> int: ...

   @dataclass(frozen=True)
   class AxisExtent:
       index: int
       label: str
       role: AxisRole
       displayed: bool
       order_pos: int
       current_step: int
       margin_left_world: float
       margin_right_world: float
       margin_left_steps: float
       margin_right_steps: float
       per_level_steps: tuple[int, ...]
       per_level_world: tuple[WorldSpan | None, ...]
       def world_span(self, level: int) -> WorldSpan | None: ...
       def step_count(self, level: int) -> int: ...

   @dataclass(frozen=True)
   class AxisSpec:
       axes: tuple[AxisExtent, ...]
       ndim: int
       ndisplay: int
       displayed: tuple[int, ...]
       order: tuple[int, ...]
       current_level: int
       level_shapes: tuple[tuple[int, ...], ...]
       plane_mode: bool
       version: int = 1
       def axis_by_index(self, idx: int) -> AxisExtent: ...
       def axis_by_label(self, target: str) -> AxisExtent: ...
   ```

   - `AxisRole` carries a lossy semantic tag, defaulting to `"unknown"` until we
     plumb viewer metadata.
   - Margins are stored in both world and step units to avoid recomputation in
     hot paths such as projection slabs.
   - `WorldSpan.scale` remains optional because not all loaders expose physical
     spacing; callers fall back to `step`.

2. Hard invariants for every `AxisSpec` instance:

   - `axes` covers the full `[0, ndim)` range with no gaps.
   - `order` and `displayed` are strictly derived from `order_pos` and
     `ndisplay`; consumers must never reconstruct tuples independently.
   - Per-level metadata (`per_level_steps`, `per_level_world`,
     `level_shapes`) share identical lengths, and `current_level` is always a
     valid index.
   - Missing labels are replaced with the deterministic `"axis-{index}"` string;
     aliases such as `"z"`/`"depth"` are translated to the `AxisRole` without
     mutating the label.

3. Module helpers expose the only supported manipulations:

   - lookup: `axis_by_index`, `axis_by_label`, `axis_role`
   - bounds: `clamp_index`, `axis_world_span`, `margin_span`
   - transforms: `world_to_steps`, `steps_to_world`, `with_updated_margins`
   - derivations: `derive_axis_labels`, `derive_order`, `derive_displayed`,
     `derive_margins`
   - serialization: `axis_spec_to_payload`, `axis_spec_from_payload`

4. Provide `fabricate_axis_spec(...)` for bootstrap paths that lack upstream
   metadata (e.g., server warm start). The helper emits canonical defaults:
   dense `order`, `ndisplay`-sized `displayed`, `axis-{idx}` labels, and
   unit-scale world spans whenever shape data exists.

5. `RenderLedgerSnapshot.axes` becomes mandatory and replaces the legacy tuple
   fields (`axis_labels`, `order`, `displayed`, `margin_*`). Any consumer asking
   for the old tuples must go via `derive_*` helpers during the transition window
   and be deleted once call sites migrate.

### A.3 Control Plane Changes

1. **Snapshot producer** (`state_reducers._record_axis_spec`):
   - Fail fast if the ledger snapshot cannot produce a spec; a missing entry is
     now a bug, not a soft fallback.
   - Feed prior specs only to reuse world spans or roles, never to resurrect
     tuples. Once the spec lands, write it to `("dims", "main", "axes")` and
     delete redundant scalar entries in the same transaction.
   - Drop `resolve_axis_index` entirely; reducers should read from
     `axis_spec_to_payload(...)` + helpers, and any caller that still wants the
     legacy tuple must migrate or be removed.

2. **State ledger entries / transactions**:
   - `apply_dims_step_transaction`, `apply_dims_margins_transaction`, and the
     bootstrap path must stop writing `axis_labels`, `order`, `displayed`,
     `margin_left`, and `margin_right` once the runtime switches over.
   - Tighten validation: reject client updates whose targets are not present in
     the spec instead of silently clamping.
   - Audit every `try/except` around axis resolution (e.g. the previous
     `KeyError` swallow in `_axis_to_index`) and replace them with assertions so
     we fail loudly when the contract is broken.

3. **Client control actions**:
   - `ControlStateContext.axis_spec` is non-optional after the first
     `notify.dims`; disconnect/reset paths must clear state explicitly.
   - `_axis_to_index`, `_axis_target_label`, and `_compute_primary_axis_index`
     should operate solely on the spec. No fallbacks to `state.dims_meta`
     tuples, no `None` returns—raise developer errors if callers provide invalid
     axis labels.

### A.4 Runtime Changes

1. **Render snapshot ingest**:
   - `build_ledger_snapshot` must read the spec first, derive `ndisplay`,
     `displayed`, and margins from it, and drop the remaining `*_tuple_or_none`
     calls that hit legacy ledger keys. Treat a missing spec as an assertion.
   - Ensure `RenderLedgerSnapshot.axes` is non-null and eliminate duplicated
     fields (`displayed`, `order`, `margin_*`, etc.) once callers switch to the
     spec.

2. **Render loop**:
   - `RenderApplyInterface` exposes the spec, and `apply_dims_from_snapshot`,
     `apply_slice_level`, `apply_slice_roi`, `viewport_roi_for_level`, and cache
     signatures rely exclusively on helper functions (`margin_span`,
     `axis_world_span`, `clamp_index`).
   - The render cache key includes axis-spec hashes (order, displayed, z-span),
     guaranteeing that projections invalidate when margins or plane mode flip.
   - Remove any secondary guards that attempt to recompute margins from tuples;
     treat divergence as programmer error.

3. **Data loaders**:
   - `ZarrSceneSource.slice` consumes `(z_start, z_stop)` exclusively from the
     spec; if the spec lacks world spans we project using step counts and log at
     debug level once.
   - `_axis_index` fallbacks in ROI helpers disappear; the spec is canonical and
     required for every remote scene.

### A.5 Client Runtime Changes

1. `ProxyViewer` deserializes `payload.axes_spec` into an `AxisSpec` and refuses
   to proceed if it is absent—partial payloads are protocol violations now.
   Order, displayed, and slider ranges flow directly from the spec.
2. `napari_dims_intent_emitter` diffs margins and indices using axis labels
   resolved via the spec. Drop position-based guesses and allow intent emission
   to raise if an invalid axis key is provided.
3. Remove the `dims_meta` cache entries for tuples once the emitter and mirrors
   no longer read them.

### A.6 Removal of Legacy Helpers

After the migration, delete or deprecate:

- `protocol/axis_labels.default_axis_labels` and `normalize_axis_labels`.
- `NotifyDimsPayload.axis_labels/order/displayed/current_step`—`axes_spec`
  becomes the sole carrier of axis semantics on the wire.
- `ControlStateContext.dims_meta['order'|'axis_labels'|'displayed']` plus all
  call sites.
- `_axis_index_from_target`, `_compute_primary_axis_index`, and any mirrored
  helpers that branch on tuple shapes.
- `server/data/roi_math._axis_index` and other ROI utilities that reach for the
  removed tuples.
- Any `try/except` or ternary fallback in axis resolution code paths; replace
  them with assertions so regressions surface immediately.

### A.7 Testing & Verification

1. Unit tests for the spec module covering:
   - Construction with/without labels.
   - Level clamping and span computation.
   - Margin conversion after level switches.
   - Serialization round-trips (payload ↔ dataclass) including optional
     `WorldSpan.scale`.
2. Update reducer, mirror, and runtime tests to assert they read from the spec
   and to fail if legacy tuples are accessed.
3. Add negative coverage: ensure clients/server raise when an axis target is
   missing from the spec or when payloads omit `axes_spec`.
4. End-to-end validation scenarios:
   - Axis reorder + notify propagation.
   - Margin updates producing multi-slice projections.
   - Multiscale level change (downsampled) with consistent ranges/margins.
   - Rendering cache invalidation when margins or plane/volume mode toggles.

### A.8 Execution Notes

- Migrate in small PRs: (1) write spec + serialization, (2) update control +
  runtime builders, (3) cut over runtime logic, (4) remove legacy helpers,
  (5) update documentation/tests.
- Because this is a dev branch, aggressively delete tuple writes as soon as the
  render path compiles; do not maintain feature flags.
- Coordinate with branch owners: block any new axis-related features until the
  consolidation merges, otherwise we risk reintroducing tuple dependencies.
- Run the full `pytest -q` suite after each removal step to catch latent users.

### A.9 Aggressive Cutover Plan

1. **Server ledger (Step 1)**:
   - Make `_record_axis_spec` the single producer, delete the tuple writes, and
     fix `build_ledger_snapshot` to depend on `AxisSpec` only.
   - Remove `state_reducers.resolve_axis_index` and ensure every reducer/mirror
     loads the spec before touching axis data.
2. **Protocol surface (Step 2)**:
   - Update `NotifyDimsPayload` and the associated JSON schema to require
     `axes_spec` and drop legacy fields. Update client + server deserializers in
     tandem.
3. **Client control/runtime (Step 3)**:
   - Require `axis_spec` in `ControlStateContext`, refactor intent emitters to
     raise on invalid axes, and delete `dims_meta` tuple caching.
4. **Render/runtime (Step 4)**:
   - Make `RenderLedgerSnapshot.axes` non-optional, drop tuple fields, and
     convert render helpers to rely on spec helpers exclusively.
5. **Final sweep (Step 5)**:
   - Rip out unused helpers (`protocol/axis_labels.py`, ROI tuple lookups),
     rerun the suite, and land follow-up cleanups (docs, examples) so no new code
     reaches for tuples again.
