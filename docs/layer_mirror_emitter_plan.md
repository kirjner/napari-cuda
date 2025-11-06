# Layer Refactor Plan (Dims-Parity)

## 1. Baseline Audit *(dims analogue: none – dims already migrated)*
- Read `src/napari_cuda/client/state/bridges/layer_state_bridge.py` end-to-end to catalogue responsibilities (layer property wiring, state.update dispatch, presenter sync, registry maintenance, ACK handling). Dims parity: we previously performed this audit for `NapariDimsMirror` and `NapariDimsIntentEmitter` when they replaced the legacy dims bridge.
- Inventory runtime entry points that currently call the bridge (`_ingest_notify_scene_snapshot`, `_ingest_notify_layers`, `_ingest_ack_state`, `_abort_pending_commands`). Dims counterpart: `_initialize_mirrors_and_emitters()` and `_replay_last_dims_spec()` already route through the dims mirror, and multiscale data now arrives with the enriched `notify.dims` frame.
- Record how the bridge interacts with the ledger (confirmed vs pending values) so the new mirror/emitter keep the same projections—mirroring dims behaviour where `_record_dims_snapshot` and `_record_dims_delta` manage confirmed entries.

## 2. NapariLayerIntentEmitter *(dims analogue: `NapariDimsIntentEmitter`)*
- Create `src/napari_cuda/client/control/emitters/napari_layer_intent_emitter.py` with constructor signature matching the dims emitter (`ledger`, `state`, `loop_state`, `dispatch_state_update`, `ui_call`, `log_layers_info`, `tx_interval_ms`).
- Port property definitions from the bridge into emitter methods exactly like dims’ axis methods:
  - On `attach_layer(remote_layer)` (dims equivalent: `attach_viewer`), connect `RemoteImageLayer` event emitters. Each handler either calls `_queue_coalesced_update` (dims uses this for slider coalescing) or dispatches immediately via `_emit_state_update`.
  - Implement suppression guards identical to dims (`suppress_forward`, `resume_forward`, `suppressing()` context manager) so mirrored updates do not boomerang.
  - Reuse bridge equality helpers (`_isclose`, `_tuples_close`) similar to dims’ `_detect_changed_axis` and `_last_step_ui` checks, ensuring we only emit state updates when values change meaningfully.
- Provide explicit intent APIs mirroring bridge entry points: `set_opacity`, `set_visibility`, `set_colormap`, `set_rendering`, `set_contrast_limits`, `set_gamma`, `set_metadata`, layer reorder operations, etc. Analogous to dims’ `dims_step`, `dims_set_index`, `view_set_ndisplay`, each should call `_emit_state_update` with metadata describing `layer_id`, `property`, `update_kind`.
- Support `set_logging` and `set_tx_interval_ms` like dims. For rate gating, reuse `_rate_gate_settings` or create a layer-specific equivalent following dims’ `dims_set_index` gating.
- Manage lifecycle (`attach_layer`, `detach_layer`, `shutdown`) and ensure any Qt timers created for coalescing are cleaned up, paralleling dims emitter cleanup.

## 3. NapariLayerMirror *(dims analogue: `NapariDimsMirror`)*
- Add `src/napari_cuda/client/control/mirrors/napari_layer_mirror.py` with constructor parameters parallel to dims but extended for layer specifics: `ledger`, `state`, `loop_state`, `registry`, `presenter`, optional `viewer_ref`, `ui_call`, `log_layers_info`.
- Subscribe to ledger events via `ledger.subscribe_all`, just as dims mirror does, but handle scopes:
  - `layer`/`layers`: update `RemoteLayerRegistry`, notify presenter façade, and apply confirmed state to the viewer within `with emitter.suppressing()` (dims mirror’s `_mirror_confirmed_dims`).
  - `scene` notifications (`notify.scene`) should update registry snapshots, enrich control-state metadata, and record confirmed values (dims uses `_record_dims_snapshot` and `_record_multiscale_metadata`). Multiscale level changes now piggyback on the dims snapshot instead of `notify.scene.level`.
- Implement runtime ingest methods mirroring dims’ `ingest_dims_notify`:
  - `ingest_scene_snapshot(frame: NotifySceneFrame)` – adapt existing bridge logic to populate ledger and registry before mirroring.
  - Record multiscale metadata triggered by dims snapshots (see `NapariDimsMirror.ingest_dims_notify`) so layer HUDs stay consistent without a separate `notify.scene.level` frame.
  - `ingest_layer_delta(frame: NotifyLayersFrame)` – handle incremental updates using registry helpers.
- Provide `replay_last_payload()` similar to dims mirror to seed a newly attached viewer; leverage cached registry data to reconstruct layer state.
- Add `attach_emitter(napari_layer_intent_emitter)` so the mirror can suppress emitter callbacks during remote apply, matching dims’ `attach_emitter` contract.
- Expose an ACK reconciliation helper (`handle_layer_ack`) analogous to dims’ ledger handling so runtime can route state acknowledgements consistently.

## 4. ClientStreamLoop Integration *(dims analogue: `_initialize_mirrors_and_emitters` usage)*
- Extend `_initialize_mirrors_and_emitters()` (or split into layered functions) to construct both dims and layer components:
  - Instantiate `self._layer_mirror = NapariLayerMirror(...)` and `self._layer_emitter = NapariLayerIntentEmitter(...)` immediately after dims creation.
  - Pass shared logging flags and tx intervals, and call `self._layer_mirror.attach_emitter(self._layer_emitter)` akin to the dims setup.
- On `attach_viewer_proxy`, after dims wiring, attach the layer emitter to each `RemoteImageLayer` held by `RemoteLayerRegistry` (emitter may register as a listener for future additions). Then call `self._layer_mirror.replay_last_payload()` to mirror confirmed layer state, mirroring `dims`’ replay path.
- Rename runtime handlers from `_handle_*` to `_ingest_*` for parity with dims’ `ingest_dims_notify`, and update state-channel wiring in `loop_lifecycle.start_loop` to reference the new names.
- Replace direct `LayerStateBridge` usage:
  - UI-facing layer intent helpers in the runtime should delegate to `self._layer_emitter.*` (mirroring dims’ delegation).
  - ACK reconciliation should call a new `self._layer_mirror.handle_ack(outcome)` (dims equivalent: ledger-driven updates inside the mirror).

## 5. Registry & Presenter Coordination *(dims analogue: last dims spec cache + presenter.apply_dims_update)*
- Move registry seeding/delta application from the bridge into `NapariLayerMirror`, ensuring it remains the single source of truth for `RemoteLayerRegistry`.
- Allow the mirror to notify the emitter when layers are added/removed so the emitter can attach/detach listeners automatically—similar to how dims emitter attaches to the viewer once.
- Relocate shared utilities (debug logging toggles, value coercion helpers) from the bridge into the new emitter/mirror modules, keeping parity with dims’ helper placement.

## 6. Cleanup Legacy Bridge *(dims analogue: removal of old dims bridge)*
- Layer mirrors and emitters now replace the legacy bridge; `LayerStateBridge` and its package have been removed.
- Update fixtures/tests to instantiate `NapariLayerMirror` and `NapariLayerIntentEmitter` directly.

## 7. Testing Roadmap *(dims analogue: `test_napari_dims_intent_emitter.py`, `test_stream_runtime.py`)*
- Add `src/napari_cuda/client/control/_tests/test_napari_layer_intent_emitter.py` mirroring the dims emitter tests: verify property callbacks dispatch intents, coalescing behaviour, suppression guards.
- Create mirror tests to ensure ingest methods update the ledger, registry, presenter, and viewer when running under `with emitter.suppressing()`—paralleling `NapariDimsMirror` coverage.
- Update runtime stub tests (`src/napari_cuda/client/runtime/_tests/test_stream_runtime.py`) to reflect renamed ingest handlers and check that layer intents route through the new emitter.
- Port or retire legacy bridge tests in `test_state_update_actions.py`, ensuring equivalent scenarios are covered in the new suites.

## 8. Follow-up Tasks *(dims analogue: post-refactor sanity)*
- Run `uv run pytest -q` for client control/runtime suites and `make typecheck` to confirm parity with dims refactor practice.
- Document the new architecture inline and/or in developer docs, explicitly referencing the dims design so future refactors (camera, multiscale) follow the same template.
- Plan subsequent subsystem migrations (camera, multiscale, volume) leveraging the mirror/emitter pattern established for dims and layers.
