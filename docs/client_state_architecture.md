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

1. **Hygiene pass** – (tracked separately) simplify modules, create placeholders
   for new projections, remove redundant logic.
2. **Introduce Store Contract** – refactor existing `pending_update_store` into
   new store, backfill projection notifications but keep old handlers.
3. **Implement Layer Projection** – detach napari writes from registry; ensure
   layer bridge uses store callbacks exclusively.
4. **Implement Dims/Camera Projection** – remove `_apply_remote_dims_update`
   heuristics; intent bridge moves out of proxy.
5. **Refactor Intent Bridge** – create new module that handles event wiring and
   state comparisons; proxy becomes thin event emitter.
6. **Runtime adjustments** – replace direct mutations with store seeding.
7. **Remove legacy streaming stack** – once new flow is proven, delete obsolete
   modules or migrate them under new architecture.

Each phase should land with unit tests + doc updates.

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
