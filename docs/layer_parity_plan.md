`layer-parity` Reality & Dims Consolidation Plan
===============================================

This document now tracks the dims consolidation work that replaces the
legacy tuple metadata with the canonical `DimsSpec`. It reflects the
current code state *after* Phase 1 (spec stored alongside tuple fields)
and lays out the aggressive path to Phase 2 (spec-only) and the mirror /
metadata cleanup.

-----------------------------------------------------------------------

Current State Snapshot
----------------------

* Notify pipeline: `NotifyDimsPayload` is now spec-only; server and client
  paths rely on `dims_spec` for axis/order/displayed metadata.
* Ledger: reducers still write legacy mirrors (`view/main/ndisplay`,
  `multiscale/main/{level,level_shapes,levels,downgraded}`,
  `dims/main/current_step`) alongside the spec; bootstrap + level-switch
  transactions expect those tuples.
* Viewport persistence: `_record_viewport_state` and `_store_*` continue
  to emit `viewport/**` and `view_cache/**` rows even though the same
  pose data lives inside `PlaneState` / `VolumeState`.
* Client runtime: `NapariDimsMirror`, emitters, and control fixtures
  maintain tuple-heavy `dims_meta` caches and helper shims despite storing
  the canonical spec.
* Tests/fixtures: state-channel helpers fabricate the legacy tuple
  entries; integration tests assert on the redundant ledger scopes.
* Sanitizers everywhere: many reducers/mirrors still normalize tuples the
  spec already validates; helper proliferation makes reasoning harder.

-----------------------------------------------------------------------

Target End State
----------------

1. `DimsSpec` stored in the ledger remains the authoritative dims record;
   no auxiliary tuple metadata is written or broadcast alongside it.
2. Client, server, and worker code path derive everything from spec via
   shared helpers; tuple caches disappear.
3. Notify protocol treats `dims_spec` as required; compat fields are
   removed from schema, tests, and fixtures.
4. Legacy mirrors collapse into thin broadcasters that forward the spec;
   viewport metadata/cache rows are deleted (or replaced with spec-derived
   pose state if still needed).
5. Docs, helpers, and tests no longer reference tuple-era fields.

-----------------------------------------------------------------------

Workstreams & Sequencing
------------------------

### 1. Server Ledger Cleanup
* Rip out legacy tuple writes in `_dims_entries_from_payload`,
  `reduce_bootstrap_state`, `reduce_level_update`, `reduce_dims_update`,
  `reduce_plane_restore`, and the related transaction helpers; rely on
  `DimsSpec` plus the minimum multiscale metadata still required for
  restore flows.
* Collapse `_ledger_dims_payload` into a spec-first accessor (or inline
  where it is now trivial) so reducers stop rehydrating tuples.
* Simplify or delete transaction helpers that become one-line
  `batch_record_confirmed` calls once redundant entries disappear.

### 2. Viewport & Pose Simplification
* Remove `_record_viewport_state`, `view_cache/**`, and `viewport/**`
  ledger writes; persist pose information only inside `PlaneState /
  VolumeState` or the spec if absolutely necessary.
* Update restore handlers, scene snapshot builders, worker bootstrap,
  and tests to consume the streamlined pose sources.

### 3. Client & Fixture Lean-Out
* Prune tuple-heavy `dims_meta` caches in `NapariDimsMirror`,
  `state_update_actions`, and emitters; derive any needed metadata on the
  fly from the stored spec.
* Replace `_make_dims_snapshot`, `_record_dims_to_ledger`, and similar
  test helpers with spec-only factories; drop assertions that reference
  the removed ledger scopes.

### 4. Final Sweep
* Delete dead normalization helpers uncovered during the cleanup to keep
  the surface area small.
* Run the focused suites (`client/control`, `server/control`,
  `state_channel`) and then full `uv run pytest -q` to verify parity.
* Refresh this plan + release notes once the ledger is spec-only end to end.

-----------------------------------------------------------------------

Testing & Validation
--------------------

* Unit: helpers, reducers, mirrors, notify serializers (spec-only paths).
* Integration: server bootstrap → notify → client ingest, runtime render
  loop, level switches, plane/volume restore.
* Regression: ensure CUDA streaming + client control loops operate with
  spec-only payloads; add coverage for downgraded multiscale + plane mode.

-----------------------------------------------------------------------

Risks & Mitigations
-------------------

* **Residual tuple consumers**: enforce helper usage via targeted `grep`
  checks or warning asserts after each stage.
* **Protocol incompatibility**: if staggered rollout is required, gate
  spec-only payloads behind a feature flag until both ends are migrated.
* **Viewport restore**: evaluate whether removing `viewport/**` entries
  breaks reconnect flows; replace with spec-derived pose snapshots if
  necessary.

-----------------------------------------------------------------------

Ownership
---------

Dims consolidation working group (server + client). Update this document
as stages land; delete once the migration is complete.
