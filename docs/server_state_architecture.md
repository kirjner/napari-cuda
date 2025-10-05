# Server State Architecture (Draft)
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Goals
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Provide an authoritative server-side model that mirrors the client’s state store.
- Decompose the monolithic control channel into transport, routing, and reducers.
- Define provenance/version semantics for multi-client and agent scenarios.
- Align layer/dims/camera update modules with the upcoming client helpers.
- Document threading/concurrency boundaries for render worker vs. control loop.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Control Pipeline Overview
Offensive coding tenet: make invariants explicit and crash on violation.

```
WebSocket Transport → Frame Parser → Router → State Update Engine →
Authoritative Scene Store → Broadcasters / Render Worker hooks
```

- **Transport Layer**: minimal asyncio WebSocket server handling handshake,
  heartbeat, compression. Raises on unexpected frames.
- **Router**: demultiplexes `state.update`, `session.*`, `notify.*` requests,
  forwarding to reducers or command handlers. No store mutations here.
- **State Update Engine**: validates payloads, applies to scene store, emits
  acks/notify deltas. Mirrors client intent bridge contract.
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Authoritative Scene Store
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Stores per-property records `(scope, target, key)` with fields:
  `confirmed_value`, `version`, `actor_id`, `origin`, `timestamp`.
- Maintains resumable history per topic (scene, layers, dims, camera, stream).
- Exposes APIs:
  - `seed_snapshot(topic, payload, version)` – baseline replay.
  - `apply_delta(topic, payload, version, actor)` – updates from reducers.
  - `replay_since(topic, version)` – resume support.
- Emits change notifications consumed by broadcasters (e.g., layer manager) and
  render worker.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Scene Update Modules
Offensive coding tenet: make invariants explicit and crash on violation.

- **LayerUpdateServer** – owns napari layer objects on the server, applies
  confirmed values from store, generates `LayerSnapshot` for clients.
- **DimsUpdateServer** – controls viewer dims sliders, ensures `current_step`
  and metadata stay consistent. Emits deltas via store.
- **CameraUpdateServer** – handles camera state (zoom, pan, orbit) and
  ensures render worker receives updates.
- Each projection:
  - Subscribes to store topics.
  - Applies mutations without try/except; failures surface immediately.
  - Produces notify deltas by comparing previous confirmed state vs. new.
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Control Channel Decomposition
Offensive coding tenet: assertions over guards unless interfacing external systems.

1. **Transport Module** – small file handling WebSocket lifecycle, handshake,
   ping/heartbeat, close semantics.
2. **Session Manager** – tracks sessions, resume tokens, actor IDs. Interacts
   with store to persist cursors.
3. **Message Router** – pure dispatcher mapping envelope types to handlers.
4. **Reducers** – existing logic from `state_update_engine.py`, refactored per
   scope (layers, dims, camera, settings, volume). Each reducer updates store and
   returns ack payload.
5. **Broadcaster** – publishes notify frames to connected clients based on store
   diffs.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Agent & Multi-Client Semantics
Offensive coding tenet: make invariants explicit and crash on violation.

- **Actor Identity**: handshake conveys `actor_id` for clients/agents.
- **Versioning**: each reducer increments per-scope version before seeding
  store. Versions flow to notify frames and client stores.
- **Conflict Resolution**: store compares incoming `state.update` version vs.
  current confirmed. If stale, reducer rejects and emits ack error; otherwise
  it applies and logs owning actor.
- **Audit Log**: append-only log (JSON or structured) capturing `(actor, scope,
  key, old_value, new_value, version)` for debugging.
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Resumable History
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Store maintains ring buffer per topic with `(version, payload)` tuples.
- On resume, session manager retrieves cursor from store, router streams
  deltas in order.
- History capped by configurable limits; when pruning, server emits
  `notify.scene` baseline.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Interaction with Render Worker & Pixel Channel
Offensive coding tenet: make invariants explicit and crash on violation.

- Render worker listens to store notifications (layers, dims, camera) via
  projection classes.
- Pixel broadcaster observes store for stream metadata (codec, fps) and restarts
  pipelines as needed.
- Worker lifecycle emits notify frames when GPU state changes (e.g., IDR,
  resolution), seeding store to keep clients consistent.
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Concurrency Model
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Event loop thread: WebSocket transport, message router, store mutations.
- Render worker thread: GL context, frame capture, encoder.
- Pixel broadcaster thread(s): frame distribution to clients.
- Store access synchronized through async primitives (no cross-thread mutations
  without explicit queues).
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Testing Strategy
Offensive coding tenet: make invariants explicit and crash on violation.

- Unit tests for store (seed, delta, resume, conflict rejection).
- Reducer unit tests (dims, layers, camera) feeding store and verifying acks.
- Session manager tests: resume cursors, heartbeat, actor identity handling.
- Integration smoke: spin up transport + router + store with fake worker to
  ensure notify/ack ordering.
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Migration Plan
Offensive coding tenet: assertions over guards unless interfacing external systems.

1. *(Shipped)* Extract store/history into dedicated module with unit tests.
2. *(In progress)* Refactor `state_update_engine` to use store API and return structured
   results.
3. Implement update modules for layers/camera, relocating code from layer
   manager and scene state applier.
4. Split `control_channel_server.py` into transport/session/router modules.
5. Update render worker to consume store notifications (instead of direct calls).
6. Remove legacy shims once new modules are stable.
Offensive coding tenet: remove try/except scaffolding that hides bugs.
