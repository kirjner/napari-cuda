# Client State Architecture (Draft)

## Goals

- Establish a single, authoritative client-side state pipeline that mirrors the
  server’s greenfield protocol: snapshots/deltas flow into an authoritative
  store, are projected into napari, and UI/agent gestures emit intents in the
  opposite direction.
- Remove ambiguous data paths (napari layers mutated directly, proxy heuristics
  for dims) and define clear contracts for each component.
- Provide hooks for future multi-client/agent scenarios (provenance metadata,
  conflict resolution).
- Enable thorough testing: each layer (store, projection, bridge, runtime) can
  be unit-tested, with integration smoke covering the coordinator.

## High-Level Flow

```
notify.scene / notify.layers / notify.dims → State Store → Projection Layer →
Napari objects
Napari events (user input) → Intent Bridge → State Store pending → state.update
ack.state → State Store → Projection Layer (confirm) → Napari
```

### Core Components

1. **State Store** (authoritative model)
   - Tracks per-property records keyed by `(scope, target, key)`.
   - Stores confirmed value (last remote ack) and pending queue (local intents).
   - Maintains provenance metadata (origin, actor ID, version, timestamps).
   - Emits change notifications to projection layer subscribers.
   - Exposes read-only query API for intent bridge (to compare current vs
     confirmed values before emitting).

2. **Projection Layer**
   - Contains domain-specific projections (layer, dims, camera, settings, volume).
   - Each projection subscribes to the store and applies confirmed values to
     napari objects while holding provenance guard to suppress local event loops.
   - Replaces the current mix of logic inside `RemoteImageLayer`, `ProxyViewer`,
     `state_update_actions`. Explicit, testable class per domain.

3. **Intent Bridge**
   - Stateless translator that listens to napari events and converts them to
     `state.update` payloads.
   - Before emitting, compares requested value against `state_store.pending_projection_value` or confirmed value to skip duplicates.
   - Tags every pending update with provenance (origin, actor) and caches the
     metadata in the store for ack reconciliation.
   - Replaces ad-hoc rate limiting/coalescing in `ProxyViewer`; if throttling is
     needed, it lives here.

4. **Runtime Orchestration**
   - `stream_runtime.py` becomes an orchestrator: decode notify frames, seed the
     store (confirmed values), trigger projection updates, and route napari
     rebuilding.
   - Dims/camera/layer notifications no longer mutate UI directly; they feed the
     store which notifies projections.
   - Handles reconnect/heartbeat/resume by flushing pending state, replaying
     latest snapshot through the store.

5. **Provenance & Identity**
   - Store records include `origin` enum (`REMOTE`, `LOCAL`, `AGENT`, etc.),
     `actor_id`, `version`, and `intent_id`.*
   - Ack outcomes update the record; projections inspect `origin` to decide
     whether to apply or ignore.
   - Exposes metrics (pending depth per property, last confirmed version) for
     debugging/telemetry.

## Detailed Responsibilities

### State Store APIs

- `seed_remote(scope, target, key, value, *, version, metadata)` – called by
  runtime when notify frames arrive.
- `apply_local(scope, target, key, value, *, origin, actor_id, metadata)` – used
  by intent bridge to enqueue pending update; returns object containing payload,
  projection value, and metadata for dispatch.
- `apply_ack(ack_frame)` – reconciles pending entries, promotes confirmed value.
- `subscribe(scope, target, key, callback)` – projections register to get
  updates; ensures unidirectional application into napari.
- `pending_projection_value()` / `confirmed_value()` – read helpers for bridges.

### Projection Layer

Example: `LayerProjection` manages a `RemoteImageLayer` instance.
- On subscribe callback, receives `ProjectionEvent` describing new confirmed
  value; applies to napari while the bridge is muted.
- For multi-value controls (contrast_limits, opacity), ensures attribute writes
  are atomic and free of `try/except` scaffolding.
- Tracks napari event emitters to block feedback loops using explicit context
  managers (`with projection.guard('contrast_limits')`).

Same pattern for `DimsBridge` (drives `ViewerModel.dims` fields) and
`CameraProjection`.

### Intent Bridge

- Installer functions register handlers for napari events (layer events, dims
  sliders, camera changes). Each handler delegates to a specific `IntentEmitter`
  that knows the mapping (napari event + value) → protocol payload.
- Before dispatching, emitter queries store’s confirmed/pending values; if
  unchanged, it aborts.
- On success, emitter returns a structured `PendingUpdate` for runtime to send
  via `state_channel_client.send_frame`.
- Housekeeping: remove rate limiting from proxy; intent bridge keeps optional
  throttling per domain (if desired) using store metadata (last send timestamp).

### Runtime Orchestration Flow

1. Control channel receives `notify.scene`: decode using `protocol.snapshots`.
2. For each layer block: `state_store.seed_remote('layer', layer_id, key, value)`.
3. Projections receive store updates, apply to napari (muted to avoid local
   events), and update napari state.
4. `notify.layers` deltas call `seed_remote`; no direct napari mutation.
5. `notify.dims` / `notify.camera` follow the same path; dims projection updates
   napari sliders while `ProxyViewer` only serves UI scaffolding.
6. Ack frames (`ack.state`) trigger `state_store.apply_ack`, which notifies
   projections (confirm value), clears pending flag, and optionally re-projects
   to napari if the confirmed value differs from pending projection (e.g.
   server rejected or modified value).
7. Reconnect resets pending state; runtime replays latest snapshot through the
   store so projections rehydrate napari.

### Provenance & Conflict Handling

- Every store entry stores `version` (monotonic from notify payload). Pending
  entries carry proposed version (if known) or client-side counter.
- `seed_remote` discards stale versions (if server version ≤ current confirmed).
- `apply_ack` compares ack metadata; if remote value diverges from pending (e.g.
  another client won), projections receive the remote-confirmed value and update
  napari accordingly.
- Origin tagging ensures projections distinguish between remote updates and local
  projections (mute guard prevents bridging) without guessing.

### Agents / Multi-Client Readiness

- Expose store subscription API for agents to observe state changes.
- Intent bridge path accepts `actor_id` (UI vs agent). Pending queue tracks
  actor; ack handling surfaces winner/loser to agent code.
- Support conflict callbacks: when ack rejects an intent or remote overrides, we
  notify actor-specific hooks.

## Testing Strategy

- Unit tests for `StateStore` (cases: seed remote, apply local, ack accepted,
  ack rejected, version conflicts).
- Unit tests for each projection (given store events, ensure napari layer/dims
  attributes update exactly once and events are muted).
- Tests for intent bridge mapping (given napari event + store snapshot, ensure
  we produce correct `state.update` payloads and skip duplicates).
- Integration smoke: spin up state store + mock projections + mock bridge; feed
  notify frames and verify `state.update` path (similar to existing
  `test_state_channel_updates`, but fully client-side).
- UI-level manual smoke (already in plan): run streaming client with dim slider
  drag/click, layer adjustments, camera commands. Compare logs for acks.

## Migration Plan

We will land the refactor in four focused stages so each slice stays testable
and reviewable.

1. **Stage 1 – Store Contract & Dims Helper** *(shipped)*
   - Restore the richer `StateStore` API (provenance fields, subscriptions,
     pending snapshots) and port existing callers/tests.
   - Introduce `DimsUpdate`, subscribe it to the store inside
     `ClientStreamLoop`, and delete the legacy wobble guards in
     `DimsBridge`.
   - Update this document and the server counterpart so the design matches the
     implementation.

2. **Stage 2 – Layer/Camera Helpers & Auto-Config** *(in progress / next)*
   - Generate layer property configs and server `CONTROL_KEYS` from napari’s own
     enums (`Interpolation`, `layer._projectionclass`, etc.) so every UI control
     round-trips without hard-coded alias tables.
   - Add `LayerUpdate` and `CameraUpdate` helpers that subscribe to the store
     and own all napari mutations.
   - Update the intent bridge to consult `pending_state_snapshot` for all
     scopes, skipping duplicates automatically.
   - **Actionable next steps:**
     1. Implement per-axis monotonic sequencing for dims intents and gate
        projection mirroring on acknowledged sequence numbers so sliders stop
        jittering.
     2. Extend the layer intent bridge to read the new store helpers (pending
        vs confirmed) instead of hand-written guards; once that is in place,
        delete the manual napari event re-emission we just added and rely on the
        projection for UI sync.
     3. Move camera property application into a dedicated `CameraUpdate`
        projection so notify.camera follows the same store-driven flow.

3. **Stage 3 – Prefetch & Playback Smoothing** *(future)
   - Introduce predictive/pre-fetch helpers for dims playback so the presenter
     can run ahead of acknowledgements.
   - Wire the intent bridge to expose per-domain throttling policies
     (e.g. configurable slider debounce) that consult the store’s
     `last_dims_send` metadata.
   - Instrument telemetry (pending depth, ack latency) so we can diagnose jitter
     and adjust hysteresis.

4. **Stage 4 – Cleanup & Coverage**
   - Prune remaining defensive guards made obsolete by the helpers.
   - Beef up regression coverage (e.g., assert the supported-control set matches
     napari’s events) and polish diagnostics/logging.

Each stage must land with updated tests and documentation so we never diverge
from the stated contract.

## Open Questions

- How to integrate telemetry/metrics with the store (per-property stats?)
- Where to host agent hooks (separate API module or built into projections?).
- Do we need to support partial snapshots (e.g., dims-only)? Store design should
  allow per-scope seeding.
- Multi-client arbitration may require server-side versioning adjustments; keep
  store API flexible.
- Agent integration spans both client and server. This doc assumes agents use
  the client-side store/bridge API only; a companion server design must cover
  authentication, attribution, and conflict resolution semantics.
