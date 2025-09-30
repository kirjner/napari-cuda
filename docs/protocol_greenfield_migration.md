# Greenfield Control Protocol Migration Plan

This document lays out the engineering plan for bringing the CUDA streaming
stack from today’s mixed legacy/greenfield implementation to the fully
specified protocol in `docs/protocol_greenfield.md`. It enumerates the affected
modules, the sequencing strategy, removal checkpoints, validation requirements,
and risk mitigations. Treat this as the canonical playbook for the migration.

## Tenets

- **Fail fast at subsystem boundaries**: prefer assertions and explicit error
  propagation inside our code; reserve defensive `try/except` for I/O and
  third-party calls, and surface failures immediately with structured context.
- **Hostility to "just in case" code paths**: eliminate silent fallbacks and
  redundant guards—either the scenario is expected (handle it deliberately) or
  it is a bug (raise loudly).
- **Clean breaks over shims**: when migrating behaviour, isolate legacy support
  behind feature flags or archived modules so the greenfield path stays clear
  and easy to reason about.

## Migration Status (2024-09-29)

- ✅ **Field inventory** – Spec §§2–8 captured in `docs/protocol_greenfield_field_inventory.md` for quick reference.
- ✅ **Dataclass scaffolding** – `src/napari_cuda/protocol/greenfield/messages.py` now hosts spec-compliant envelopes/payloads with version guards.
- ✅ **Helper coverage & sequencing** – `protocol/greenfield/envelopes.py` now supplies builders for every session/notify lane plus shared `ResumableTopicSequencer`; parser helpers mirror the full matrix.
- ✅ **Schema tightening** – All greenfield payloads drop legacy `extras` bags; `from_dict` guards now reject unknown fields to match Appendix A.
- ✅ **Scene snapshot emission** – Server baselines now flow through `build_notify_scene_payload` → `build_notify_scene_snapshot`, replacing legacy `scene.spec` JSON broadcasts and wiring resumable sequencing.
- ✅ **Hot-path notify lanes** – Control channel emits `notify.scene`/`notify.layers`/`notify.dims`, and the pixel channel now issues `notify.stream` snapshots via `build_notify_stream`, advertising the stream sequencer cursor in `session.welcome`.
- ⏳ **Spec round-trip tests** – Initial handshake + notify scene/state round-trips live in `src/napari_cuda/protocol/_tests/test_envelopes.py`; expand to the remaining lanes and sequencing edge cases next.
- ✅ **Dual emission cleanup** – Legacy dual-emission shims (`legacy_dual_emitter`, `_broadcast_state_json`, `protocol_bridge`) removed; the server emits greenfield frames exclusively.

Pending follow-up:

1. Convert the ack/reply/error lanes to `build_ack_state`/`build_reply_command`/`build_error_command`, wiring `in_reply_to`/`intent_id` per Appendix A.

## 1. Scope & Objectives

- Re-platform the control channel so every client/server exchange uses the
  greenfield envelopes, schema versions, and acknowledgement semantics.
- Eliminate legacy payload types (`dims.update`, `video_config`, bespoke
  `camera.*` verbs) and the dual-emission bridge once compatibility is no
  longer required.
- Make the code layout reflect the protocol structure (control vs pixel vs
  command), reducing ambiguity for future maintenance.
- Validate reconnect, optimistic state-update, and command flows end-to-end with the
  new ack contracts and resumability rules.

## 2. Current Message Flow Inventory

### 2.1 Control Channel (Client)

- `client/control/control_channel_client.py` (archived original:
  `client/streaming/state.py`) maintains the WebSocket, issues a
  `SessionHello` built around `PROTO_VERSION = 1`, and still expects legacy
  payloads alongside `notify.*` envelopes.
- `client/control/pending_update_store.py` (archive: `state_store.py`) tracks
  optimistic state via `PendingEntry` queues keyed by `(scope, target, key)`,
  reconciling on inbound `state.update` echoes.
- `client/control/state_update_actions.py` (archive:
  `client/streaming/client_loop/intents.py`) centralises outgoing state-update
  helpers and local projection logic without the legacy "intent" terminology.
- `client/control/viewer_layer_adapter.py` (archive:
  `layer_state_bridge.py`) translates incoming layer/dims payloads into viewer
  mutations; today it still listens for legacy formats like `dims.update`.
- Legacy fallbacks remain: plain `video_config` dicts, `dims.update` diffs, and
  fire-and-forget keyframe requests to `/state` (`request_keyframe`).

### 2.2 Control Channel (Server)

- `server/control/control_channel_server.py` (archived original:
  `server/state_channel_handler.py`) accepts both greenfield and legacy verbs,
  dispatching via `MESSAGE_HANDLERS` (`set_camera`, `camera.pan_px`,
  `request_keyframe`, etc.).
- `server/control/state_update_engine.py` and
  `server/control/scene_snapshot_builder.py` (archive:
  `server/server_state_updates.py` and `server/server_scene_spec.py`) serialize
  authoritative `state.update` payloads before they reach the bridge.
- Legacy dual-emission bridge (`server/control/legacy_dual_emitter.py`,
  `server/protocol_bridge.py`) removed; only greenfield helpers remain on the
  server path.
- Baseline snapshots now emit through
  `scene_snapshot_builder.build_notify_scene_payload` →
  `build_notify_scene_snapshot`, so the state channel no longer broadcasts the
  legacy `scene.spec` JSON.

### 2.3 Pixel Channel

- `client/pixel/pixel_channel_client.py` / `client/pixel/frame_presenter.py` /
  `client/pixel/renderer.py` (archived under `client/streaming/`) handle pixel
  frames without protocol coupling.
- `server/pixel/pixel_channel_server.py` (archive: `server/pixel_channel.py`)
  now emits `notify.stream` payloads via `build_notify_stream`, feeding the
  control-channel sequencer instead of the legacy `video_config` JSON helper.

### 2.4 Command Lane

- Not wired yet. The feature matrix in the spec is advertised, but neither
  server nor client executes `call.command` / `reply.command` frames.

## 3. Target Architecture Alignment

- **Control envelope contract**: Every control frame uses the fields defined in
  Table 2 of the spec (`type`, `version`, `session`, `frame_id`, etc.).
- **State lane**: Client emits `state.update` (with mandatory `frame_id`),
  server responds with `ack.state` and then authoritative `notify.*` frames.
- **Notify coverage**: Only `notify.scene`, `notify.layers`, `notify.dims`,
  `notify.camera`, `notify.stream`, `notify.error`, `notify.telemetry` remain.
- **Command lane**: Routed over `call.command` / `reply.command` /
  `error.command` with the ack semantics described in §6 of the spec.
- **Session control**: `session.hello` → `session.welcome` handshake, periodic
  heartbeats, canonical shutdown contract.

## 4. Module Renames & Layout Changes

| Current Module | Proposed Path | Rationale |
|----------------|--------------|-----------|
| `protocol/envelopes.py` | `protocol/greenfield/envelopes.py` | Separate greenfield from legacy helpers. |
| `protocol/parser.py` | `protocol/greenfield/parser.py` | Explicitly names the parser domain. |
| `protocol/messages.py` | Split into `protocol/legacy/messages.py` and `protocol/greenfield/messages.py` (temporary) | Remove mixed concerns during migration; legacy module deleted once the legacy bridge is removed. |
| `client/streaming/state.py` | `client/control/control_channel_client.py` | Matches the control-lane responsibility. |
| `client/streaming/state_store.py` | `client/control/pending_update_store.py` | Tracks pending `state.update` emissions and ack reconciliation without legacy "intent" terminology. |
| `client/streaming/layer_state_bridge.py` | `client/control/viewer_layer_adapter.py` | Clarifies its bridging role. |
| `client/streaming/controllers.py` | `client/runtime/channel_threads.py` | Emphasizes thread orchestration. |
| `client/streaming/client_stream_loop.py` | `client/runtime/stream_runtime.py` | Marks the orchestrator that binds control + pixel paths. |
| `client/streaming/client_loop/intents.py` | `client/control/state_update_actions.py` | Emits state-update payloads and projection helpers, dropping "intent" naming. |
| `server/state_channel_handler.py` | `server/control/control_channel_server.py` | Symmetric naming with the client. |
| `server/server_state_updates.py` | `server/control/state_update_engine.py` | Makes the mutation responsibility explicit. |
| `server/server_scene_spec.py` | `server/control/scene_snapshot_builder.py` | Communicates “baseline snapshot” job. |
| — *(removed)* | — *(removed)* | Legacy dual-emission bridge deleted once stream lane went greenfield-only. |
| `server/pixel_channel.py` | `server/pixel/pixel_channel_server.py` | Aligns with the pixel-lane nomenclature. |

Renames happen early so the new directory tree mirrors the future ownership
model before logic changes land.

**Import transition note** – During the overlap window, keep lightweight shims
so existing callsites compile while the tree is renamed:

```python
# napari_cuda/protocol/__init__.py (temporary)
from .legacy import messages as legacy_messages
from .greenfield import messages as greenfield_messages

# Preserve historical import until dependants migrate explicitly.
messages = legacy_messages
```

Migration steps:

1. Update internal modules to use `legacy_messages`/`greenfield_messages`
   explicitly (no wildcard imports).
2. Once the repo is clean, remove the `messages = legacy_messages` re-export
   and delete the shim after downstream consumers migrate.

## 5. Migration Phases

### Phase 0 – Preparation & Renames

1. Apply the module renames in §4 without behaviour changes.
2. Update imports across client/server/protocol packages.
3. Add import shims (re-export modules from legacy paths) to keep downstream
   tooling functional until all callsites move.
4. Refresh developer docs to reference the new locations.

### Phase 1 – Greenfield Schema Library

1. Split `protocol/messages.py` into legacy vs greenfield modules; move the
   greenfield dataclasses (`SceneSpec`, `NotifyScene`, etc.) alongside the
   envelopes.
2. Generate JSON Schema stubs in `docs/protocol_greenfield/schemas/` and wire a
   CI task that validates example payloads.
3. Add runtime validators on both client and server (simple `jsonschema`
   wrapper) gated behind a feature flag (`NAPARI_CUDA_PROTOCOL_VALIDATE=1`).

Output: Both sides can construct/validate the new envelopes locally while still
translating to legacy formats.

### Phase 2 – Server Control Channel Rewrite

1. Introduce `ControlSession`/`ControlClient` abstractions in
   `server/control/control_channel_server.py` that encapsulate the handshake,
   feature advertisement, and heartbeat loop using only greenfield envelopes.
2. Replace `MESSAGE_HANDLERS` dispatch with:
   - `state.update` handler → `state_update_engine.apply_*` (still returns
     `StateUpdateResult`).
   - `call.command` handler → new command registry that calls into AppModel.
3. Emit `ack.state` after successful mutations (populate `applied_value` or
   rejection payloads).
4. Broadcast notified state strictly through `notify.scene` / `notify.dims` /
   `notify.camera` / `notify.layers`, validating payloads against
   `docs/protocol_greenfield/schemas/notify.scene.v1.schema.json`,
   `notify.dims.v1.schema.json`, `notify.camera.v1.schema.json`, and
   `notify.layers.v1.schema.json`; remove plain `dims.update` emission.
5. Keep the legacy WebSocket handler behind a feature flag (`LEGACY_STATE_WS`);
   route current deployments through the new handler in mirrored mode (emit
   both legacy + greenfield) until clients catch up.

### Phase 3 – Client Control Channel Rewrite

1. Update `control_channel_client.py` to emit `session.hello` envelopes with
   `frame_id` and per-topic `resume_tokens` (persisted via the state store).
2. Replace the legacy JSON switchboard with typed handlers driven by the
   parser (`parse_notify_scene`, `parse_notify_dims`, etc.), wiring validation
   against `docs/protocol_greenfield/schemas/notify.*.v1.schema.json`.
3. Teach `pending_update_store.py` to track outstanding `frame_id` → state
   update entries and reconcile on `ack.state`, verifying envelopes with
   `state.update.v1.schema.json` and `ack.state.v1.schema.json` (removing the
   old phase juggling).
4. Drop the `dims.update`/`video_config` fallback branches.
5. Replace the keyframe request helper with either:
   - A `call.command` to `napari.pixel.request_keyframe`, or
   - A `state.update` on a dedicated command scope.
6. Integrate viewer updates through the new notify payloads (ensure 3D toggle
   arises from `notify.scene`+`notify.dims` metadata).
7. Keep a temporary compatibility mode (`LEGACY_CONTROL_CLIENT`) that routes to
   the old logic for incremental rollout.

### Phase 4 – Command Lane Enablement

1. Define a command registry on the server (initial commands: `napari.viewer.fit_to_view`,
   `napari.widgets.features_table`, `napari.toggle_shape_measures`, keyframe).
2. Surface the registry via `session.welcome.payload.features.call.command`.
3. Implement client-side RPC helpers that queue `call.command` envelopes,
   await `reply.command` / `error.command`, and surface status/error to the UI.
4. Derive viewer menu actions from the advertised command catalog.

### Phase 5 – Legacy Cleanup

*Dual emission is removed during Phase 2 close-out; this phase sweeps any remaining compatibility artifacts.*

1. Remove `notify_state`/`notify_scene` legacy extras and the capability
   translation layer in `control_channel_client.py`.
2. Drop the compatibility feature flags and clean out re-export shims.
3. Remove log noise/artifacts tied to old payloads (e.g., `dims.update raw meta`).

### Phase 6 – Validation & Rollout

1. Extend the integration harness to:
   - Spin up server + client in greenfield-only mode.
   - Exercise reconnect with stale/valid tokens.
   - Drive optimistic state-updates at 60 Hz (orbit/pan), verifying `ack.state`
     timing and `notify.*` echo semantics.
   - Execute command lane RPCs and assert replies.
2. Capture metrics on `ack.state` latency, `notify.*` payload sizes, and
   heartbeat adherence.
3. Enable the new protocol in canary builds, collect telemetry, then roll
   broadly.

## 6. Redundant Handling to Remove

- Client:
  - `control_channel_client._handle_message` branches for `'video_config'` and
    `'dims.update'`.
  - Fire-and-forget keyframe connection in `request_keyframe_once`.
  - Phase juggling in `pending_update_store.apply_local` and
    `pending_update_store.apply_remote` tied to legacy ack semantics.
- Server:
  - `MESSAGE_HANDLERS` entries for bespoke verbs (`set_camera`, `camera.*`,
    `ping`, `request_keyframe`).
- Remaining references to
    `scene_snapshot_builder.build_scene_spec_json` once no callers require the
    cached legacy JSON representation.

## 7. Testing Matrix & Instrumentation

| Scenario | Checks |
|----------|--------|
| Initial connect | `session.welcome` contains advertised features, client stores `session`, baseline `notify.scene` arrives with `seq=0`. |
| Orbit drag loop | `state.update` → `ack.state(status="accepted", applied_value)` within `ack_timeout_ms`; subsequent `notify.camera(intent_id)` aligns. |
| Dims slider spam | Client optimistic projection matches authoritative `notify.dims`; pending queue drains on ack. |
| Command success/failure | `call.command` results in `reply.command` / `error.command` with `in_reply_to` = request `frame_id`. |
| Reconnect with valid token | `notify.layers` deltas replayed; `seq`/`delta_token` progress without forcing a snapshot. |
| Reconnect with stale token | Server sends `notify.scene(seq=0)` baseline, `notify.layers` issues fresh snapshot, client resets epoch. |
| Stream config refresh | Encoder reset triggers `notify.stream` reissue carrying updated extras (format/data/fps). |
| Heartbeat dropout | Missing two beats triggers `session.goodbye`, both sockets close (state/pixel). |
| Critical error | `notify.error(severity="critical")` precedes orderly socket teardown. |

Automate these using the existing pytest harness plus new fixtures for control
channel recording. Tie metrics (ack latency, notify size) into the monitoring
stack to validate SLAs from the spec.

## 8. Risks & Mitigations

- **Schema drift**: Keep JSON Schema validation on for all greenfield frames in
  CI to catch accidental field omissions.
- **Ack starvation**: Instrument `ack.state` latency; fail tests if responses
  exceed `ack_timeout_ms`. Provide backpressure (queue caps) so the server does
  not drop acks under load.
- **Mixed-client deployments**: Maintain the legacy compatibility flag through
  Phase 4; only remove once analytics show every client speaks greenfield.
- **Command side effects**: Document idempotency per command; log `in_reply_to`
  everywhere to triage mismatches quickly.
- **Reconnect storms**: Rate-limit heartbeats and resume requests; ensure
  resumability retention thresholds (512 deltas / 5 min) are configurable per
  deployment.

## 9. Deliverables Checklist

- [ ] Module rename PRs merged and docs updated.
- [ ] Schema split + validation landed with greenfield flag (CI runs `tox -e schema-validate` to lint `docs/protocol_greenfield/schemas/*.json` against representative payload fixtures and fails on deviation).
- [ ] New server control channel with ack semantics and command registry.
- [ ] Client control channel rewritten; optimistic state updates mapped to
      `ack.state`.
- [ ] Command lane MVP live (fit_to_view, features_table, toggle_measures,
      request_keyframe).
- [ ] Legacy bridges removed; greenfield-only deployment verified.
- [ ] Integration suite running greenfield scenarios in CI.
- [ ] Telemetry dashboards tracking ack latency, notify rates, heartbeat
      compliance.

Once every item clears, deprecate the legacy protocol documentation and treat
`docs/protocol_greenfield.md` plus this migration plan as the long-term source
of truth.
