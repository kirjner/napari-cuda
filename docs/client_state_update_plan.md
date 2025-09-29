# Client Plan – Unified `state.update` Flow

## Overview
This document specifies the client-side refactor that aligns the streaming stack with the server's unified `state.update` protocol. The goal is to eliminate the legacy intents/commands split, consolidate optimistic + confirmed state into a reducer (`StateStore`), and simplify the Qt bridge under the new `LayerStateBridge` adapter. No backwards compatibility with the previous payloads is preserved.

Key objectives:
- One JSON contract for all property changes (layers + dims).
- Reducer-first design that owns optimistic projection and reconciles remote acknowledgements.
- Renamed adapter + context layers (`LayerStateBridge`, `ClientStateContext`) that expose a "dumb simple" API between Qt widgets and the transport.
- Explicit reconnect sequencing and quiet property setters to prevent oscillation during drags.
- Telemetry hooks and tests that guarantee smooth slider behaviour (gamma, contrast, dims).

## Message Contract
```
{
  "type": "state.update",
  "scope": "layer",           // or "dims"
  "target": "layer-0",        // layer id or axis label/index
  "key": "gamma",             // dims uses key="step"
  "value": 0.82,
  "client_id": "...",
  "client_seq": 42,
  "interaction_id": "...",    // optional
  "phase": "start|update|commit|reset",  // optional
  "timestamp": 1759097437.68,
  "server_seq": 981            // inbound only
}
```

Requirements:
- Constant `STATE_UPDATE_TYPE = "state.update"` lives in `napari_cuda.protocol.messages`.
- `StateUpdateMessage` dataclass extends `StateMessage` with the fields above; `server_seq` is mandatory on broadcasts.
- `StreamProtocol.parse_message` must emit `StateUpdateMessage` instances for the reducer.
- Server guarantees a single monotonically increasing `server_seq` across all updates.
- Dims naming is fixed: `scope="dims"`, `target=<axis label/index>`, `key="step"`.

## Components

### 1. `StateStore` Reducer (`src/napari_cuda/client/streaming/state_store.py`)
State keyed by `(scope, target, key)` with two slots:
- `confirmed`: `{value, server_seq, timestamp}`.
- `pending`: `OrderedDict[client_seq -> PendingEntry(value, phase, timestamp, interaction_id)]`.

API surface:
1. `apply_local(scope, target, key, value, phase, interaction_id=None)` → `(payload, projection_value)`.
2. `apply_remote(message: StateUpdateMessage)` → `ReconcileResult(projection_value, is_self, overridden, pending_len)`.
3. `seed_confirmed(scope, target, key, value, server_seq=None)` for baseline hydration.
4. `clear_pending_on_reconnect()` (first milestone drops all optimistic entries).

Rules:
- `phase="start"` flushes pending for the property before inserting the new entry.
- `phase="update"` overwrites the latest pending entry.
- `phase="commit"` marks completion and removes the matching pending record on ack.
- `phase="reset"` applies value + clears pending (baseline reset).
- Foreign updates (different `client_id`) replace `confirmed` and empty pending.
- Projection resolves to `pending[last].value` if pending exists; otherwise confirmed.
- Reducer stays pure (no Qt side effects).

### 2. `ClientStateContext` (rename from `IntentState`)
- Holds immutable `client_id`.
- Provides `next_client_seq()` counter for the reducer.
- Carries client-wide settings (e.g., rate limits) consumed by higher layers.
- Update all imports/call sites to the new name.

### 3. `LayerStateBridge` (rename from `LayerIntentBridge`)
Responsibilities:
1. **Scene Seeding**
   - On snapshot: invoke `state_store.seed_confirmed(...)` for every property.
   - Quietly apply projection values to the Qt layer (no signal emission).
   - After seeding, connect Qt signals to reducer callbacks.

2. **Outgoing Flow**
   - Determine phase (`start`, `update`, `commit`, `reset`).
   - Call `state_store.apply_local(...)` to obtain payload + projection.
   - Apply projection under the quiet setter helper.
   - Send payload via `dispatch_state_update` (see Transport below).

3. **Incoming Flow**
   - Deserialize `StateUpdateMessage` from the stream.
   - Feed into `state_store.apply_remote(...)`.
   - Quietly apply projection before third-party observers fire.
   - Forward reconciled metadata to presenters/overlays as needed.

### 4. Transport & Stream Loop
- Introduce `StateUpdateMessage` in `napari_cuda.protocol.messages` and wire parsing.
- `ClientStreamLoop._handle_layer_update` (and dims equivalents) now:
  1. Parse payload into `StateUpdateMessage`.
  2. Pass to `state_store.apply_remote` via the bridge.
  3. Use bridge quiet setter to update Qt layer.
  4. Notify presenters with reconciled state.
- Remove legacy helpers: `_send_control_command`, `_maybe_send_session`, `_apply_session_remote`, `_try_send_commit`, `control_sessions.py`, `CONTROL_COMMAND_TYPE`, `layer.intent.*` constants.
- Replace with a single `dispatch_state_update(payload)` that consults `ClientStateContext` for sequencing.

### 5. Quiet Property Application Helper
- Implement `apply_property(binding, config, value)` (exact path TBD, likely inside bridge or util module).
- Behaviour:
  - Add key to `binding.suspended` (prevents re-entrancy).
  - Use `QSignalBlocker` or existing `binding` context manager to suppress signals.
  - Call the property setter.
  - Remove key from suspended set.
- Must be nest-safe (track re-entry depth).

### 6. Reconnect Handling
- On state channel drop: bridge triggers `state_store.clear_pending_on_reconnect()`.
- Reconnect sequence:
  1. Wait for fresh `scene.spec` from server.
  2. Seed confirmed registers.
  3. Quietly apply projections.
  4. Reattach Qt signals (after seeding to avoid echo).
- Document ordering directly in code comments/docstrings.

### 7. Telemetry & Debug
- Counters to integrate with existing metrics plumbing:
  - `state_store_pending_total` (gauge).
  - `state_store_overrides_total`.
  - `state_store_foreign_updates`.
- Emit `DEBUG` logs for every `apply_local` / `apply_remote` path.
- Provide `state_store.dump_debug()` for inspector tooling.

### 8. Testing Strategy
1. **Unit Tests (StateStore)**
   - Self-ack reconciliation (start/update/commit cycle).
   - Server override drops pending.
   - Foreign client update resets queue.
   - `phase="reset"` path.
   - Reconnect clearing.

2. **Integration Tests**
   - Gamma drag: smooth projection without oscillation.
   - Dims update uses `scope="dims"`, `key="step"`.
   - Scene seeding ensures quiet setters before signal hookup.

3. **Regression Coverage**
   - Ensure no oscillation when ack values match optimistic state.
   - Telemetry counters increment/decrement as expected.

### 9. Cleanup Checklist
- Delete unused modules/helpers from the intents era.
- Update fixtures + mocks to expect `state.update` payloads.
- Rename files/tests to reference `LayerStateBridge` & `ClientStateContext`.
- Ensure docs/examples point at the new API.

### 10. Future Considerations
- Rate limiting / coalescing can migrate into `StateStore` later.
- Pending replay across reconnect is intentionally out of scope; document as known gap.
- Codify dims axis names (constant table) to keep client/server tests aligned.

## Next Steps
1. Land protocol changes (`StateUpdateMessage`, parser support).
2. Introduce `StateStore` module with tests.
3. Rename bridge/context classes and rewire the transport.
4. Implement quiet setter + reconnect flow.
5. Rip legacy intent plumbing and refresh docs/tests/telemetry.

Once complete, the client matches the server’s reducer-first architecture, guaranteeing smooth slider interactions and enabling richer collaborative features.
