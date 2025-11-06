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

* Ledger: every reducer writes `dims_spec`; legacy tuple entries
  (`dims/main/order`, `axis_labels`, `displayed`, `labels`,
  `current_step`) are still emitted by `_dims_entries_from_payload`.
* Notify pipeline: `NotifyDimsPayload` carries both the spec and tuple
  fields; `ServerDimsMirror` reconstructs tuples from the spec before
  broadcasting.
* Client control/runtime: `NapariDimsMirror` and `state_update_actions`
  cache tuple metadata in `dims_meta` even though they store the spec.
* Render snapshot + worker application: snapshots persist tuple metadata
  (`order`, `displayed`, `axis_labels`) and the render loop asserts that
  those match the spec.
* Tests/fixtures: state-channel and runtime suites still manufacture
  tuple payloads; spec usage only supplements them.
* Ledger metadata: `view/main/ndisplay` and `multiscale/main/*` entries
  mirror spec data; viewport caches (`viewport/**`, `view_cache/**`)
  keep redundant plane/volume state.
* Sanitizers everywhere: reducers clamp/index/proportional helpers,
  mirror/client code re-normalises steps/orders/labels, runtime planners
  clamp again, and metadata only exists to ferry intent IDs/axis hints.
* Notify pipeline now emits spec-only payloads; axis/order/displayed metadata is derived from `DimsSpec` rather than
  duplicated in the wire format.

-----------------------------------------------------------------------

Target End State
----------------

1. `DimsSpec` is the sole source of dims truth: no tuple metadata written
   to the ledger or broadcast over notify channels.
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

### 1. Shared Helpers (Pure Functions)
* Add helper functions in `shared/dims_spec.py` for recurring derived
  values (axis labels, displayed axes, order, clamped levels/steps,
  primary axis, etc.).
* Replace ad-hoc tuple reconstruction and clamp/normalise helpers in
  reducers, mirrors, render builders, and client control/runtime code
  with calls to the helpers.
* Add unit coverage for the helpers (spec round-trips, remap/clamp).

### 2. Client Spec Adoption
* Update `NapariDimsMirror`, `state_update_actions`, emitters, and runtime
  planners to rely exclusively on helpers + the stored `DimsSpec`.
* Delete `dims_meta` tuple caches (`order`, `axis_labels`, `displayed`,
  `range`, etc.) once call sites fold into helpers.
* Refresh tests/fixtures to construct `DimsSpec` payloads directly.

### 3. Protocol Flip (DONE)
* `NotifyDimsPayload` now requires `dims_spec` and no longer transports the legacy tuple fields.
* Server/client serialization + tests (notify frames, state-channel replay, client runtime) assert spec-only payloads.
* Tuple parsing paths have been removed.

### 4. Server Reducer Cleanup
* Stop writing tuple keys from `_dims_entries_from_payload` and related
  reducers/transactions (`reduce_bootstrap_state`, `reduce_level_update`,
  `reduce_dims_update`, `reduce_plane_restore`).
* Replace `_ledger_dims_payload` with a spec-returning helper; migrate all
  callers to use the spec.
* Update render snapshot builders to source data through helpers and drop
  tuple assertions.
* Drop metadata arguments for dims-related ledger writes; rely on
  `ServerLedgerUpdate` fields + spec helpers for axis/intent routing.

### 5. Mirror & Metadata Pruning
* Simplify `ServerDimsMirror` and any client/runtime mirrors to forward
  spec-only payloads.
* Delete redundant viewport/view_cache ledger entries unless a surviving
  consumer needs pose persistence; document or replace them with a
  spec-derived representation if required.
* Strip dims metadata consumption (axis/intent) from client/server
  mirrors and ack logic once helpers cover resolution; delete the
  legacy clamp/normalize helpers uncovered in the audit.

### 6. Final Sweep
* Remove tuple-era compatibility helpers, docs, and fixtures.
* Run full CI (`uv run pytest -q`, relevant slow/integration suites, type
  checks) to confirm spec-only flow.
* Update release notes / consolidation log to capture the migration.

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
