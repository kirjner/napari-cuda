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

We are migrating viewer axis handling to a single canonical structure while
keeping the existing tuple fields available for callers that still depend on
them. The work lands in two aggressive phases so we can validate each step
without stalling the stream pipeline.

### Phase 1 – AxesSpec alongside ledger tuples (current)

- Introduce immutable `AxesSpec` / `AxesSpecAxis` dataclasses in
  `src/napari_cuda/shared/axis_spec.py` with a strict builder that derives the
  spec from the confirmed ledger snapshot. Assertions guard against missing or
  divergent tuple data.
- Extend `RenderLedgerSnapshot` and `NotifyDimsPayload` so both tuples and the
  derived spec travel together end-to-end (reducers → snapshot builders → notify
  payloads → runtime apply code).
- Update the EGL apply path and client mirrors to prefer the `AxesSpec` values
  while still asserting that historical tuples match. No fallbacks, no
  try/except.
- Tests: shared spec round-trip coverage, server notify/state-channel checks,
  and client control suites (`state_update_actions`, mirrors) run with the spec
  present.

### Phase 2 – Cut tuple legacy paths (planned)

- Once all runtime/client consumers are spec-native and coverage passes, stop
  writing tuple keys from reducers and emit them only for any remaining legacy
  mirror. Validate that all downstream paths rely purely on `AxesSpec`.
- Remove tuple plumbing and associated helpers (`_ledger_dims_payload`
  fallbacks, `_tuple_or_none` helpers, etc.). Update docs/tests to treat the
  spec as the single source of truth.
- Only after Phase 2 goes green do we delete tuple serialization from
  `NotifyDimsPayload`.

### Trackers / Follow-ups

- [ ] Audit remaining tuple reads (`snapshot.displayed`, `dims_meta['order']`,
      margin helpers) and convert them to use `AxesSpec`.
- [ ] Add end-to-end regression exercising view toggles and margin updates with
      tuples removed.
- [ ] Update external integration docs once tuple removal is complete.
