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
- ✅ **Server ack lane** – `control_channel_server._handle_state_update` now parses greenfield envelopes, emits `ack.state` via `build_ack_state`, and no longer fabricates legacy `client_id`/`client_seq` metadata.
- ✅ **Client ack lane** – `control_channel_client`, `pending_update_store`, and the streaming loop now trade exclusively in greenfield `AckState`/`state.update` envelopes. `ControlStateContext` replaces the old "intent" shims, and tests under `client/streaming/_tests/` cover accepted/rejected ack flows end-to-end.

## Near-Term Execution Plan (2025-10-01)

- **Step 0 (complete)** – Keyframe requests now replay `notify.stream`, reset the encoder, and immediately issue `request_idr()`; `_ensure_keyframe` runs after the state baseline so VT always opens on a primed avcC. Keep the VT gate/keyframe diagnostics in the smoke checklist until the next cut.
1. **Finish the greenfield control channel**
   - Remove `meta_refresh` by emitting concrete `notify.dims` and scene payloads from the worker.
   - Replace volume/multiscale rebroadcast shims with direct `notify.layers` deltas driven by intents.
   - Implement `scope="camera"` on the server, emit `notify.camera`, migrate the client to the new lane, and delete the Alt+drag orbit fallback once the handlers land.
2. **Retire legacy protocol surface & scaffolding**
   - Delete `SceneSpecMessage`, `LayerUpdateMessage`, and related adapters once Step 1 ensures consumers hydrate from `notify.scene`/`notify.layers`.
   - Remove `protocol/messages.py` compatibility exports and the remaining control-channel shims. `client/minimal_client.py` has been archived under `archive/legacy_client/`.
   - Track the dependent dataclasses (`LayerSpec`, `LayerRenderHints`, `MultiscaleSpec`, etc.) called out in §6 so the layer registry falls alongside the scene/layer shims.
3. **Finish the command lane**
   - Wire `call.command`/`reply.command`/`error.command` end-to-end, including `napari.pixel.request_keyframe`.
   - Port UI hooks to command helpers so keyframe triggers and menu actions share one path.
4. **Expand protocol tests & smoke coverage**
   - Add round-trip tests for reconnect/resume tokens, notify sequencing, and command flows.
   - Script a smoke run that exercises dims toggles, camera pan/orbit, layer edits, and explicit keyframe requests via the reset path.
5. **Documentation & migration bookkeeping**
   - Keep `docs/protocol_greenfield_migration.md` in sync as drifts close, pruning entries instead of letting them rot.
 - Produce a streamlined smoke-test runbook covering encoder diagnostics and VT gate behaviour.

## Next Scoped Pass – Command Lane Cutover

| # | Task | Primary File Touches | Spec References |
|---|------|----------------------|-----------------|
| ✅ | Drop legacy keyframe verbs from the control dispatcher and route all keyframe triggers through `call.command`. Remove `'request_keyframe'` / `'force_idr'` entries and collapse `_handle_force_keyframe`. *(Completed 2025-10-03.)* | `src/napari_cuda/server/control/control_channel_server.py`, `src/napari_cuda/server/_tests/test_state_channel_updates.py`, `src/napari_cuda/client/streaming/pipelines/vt_pipeline.py` (ensure only `_request_keyframe_command` is used). | `docs/protocol_greenfield.md` §§6.1–6.3 (command request/reply/error contract). |
| ✅ | Retire the legacy `'ping' → 'pong'` shim on the control channel so liveness flows exclusively through `session.heartbeat` / `session.ack`. *(Completed 2025-10-03.)* | `src/napari_cuda/server/control/control_channel_server.py`, `docs/protocol_greenfield_migration.md`. | `docs/protocol_greenfield.md` §3.1 (session/heartbeat contract). |
| ✅ | Introduce a formal command registry that maps spec-qualified command names to server callbacks (initial catalogue limited to `napari.pixel.request_keyframe`; viewer fit/reset remain state updates). Advertise the catalogue in `session.welcome.features.call.command.commands`. *(Completed 2025-10-03.)* | `src/napari_cuda/server/control/control_channel_server.py`, `src/napari_cuda/server/server_scene.py`, new helper `src/napari_cuda/server/control/command_registry.py`, `src/napari_cuda/server/_tests/test_state_channel_updates.py`. | `docs/protocol_greenfield.md` §3.2 (feature advertisement in `session.welcome`), §§6.1–6.4. |
| ✅ | Surface the command catalogue in the client UI layer: bind menu/actions to `_issue_command`, remove any residual direct worker shims, and extend command future handling tests. *(Completed 2025-10-03.)* | `src/napari_cuda/client/runtime/stream_runtime.py`, `src/napari_cuda/client/streaming/client_loop/pipelines.py`, `src/napari_cuda/client/streaming/_tests/test_client_stream_loop.py`, viewer action wiring. | `docs/protocol_greenfield.md` §§6.1–6.4 (client behaviour), Appendix A table 6 (command capability matrix). |
| ✅ | Update documentation and smoke checklists to reflect the command-only path and archive status. *(Completed 2025-10-03.)* | `docs/protocol_greenfield_migration.md`, `docs/control_protocol_agent_notes.md`, `docs/consolidate_refactors_plan.md`. | `docs/protocol_greenfield.md` §§6.1–6.4, §8 (operational notes). |
| ✅ | Add regression coverage for the command lane (keyframe RPC success/error paths) and fold it into smoke automation. *(Completed 2025-10-03.)* | `src/napari_cuda/server/_tests/test_state_channel_updates.py`, `src/napari_cuda/client/streaming/_tests/test_client_stream_loop.py`, smoke scripts. | `docs/protocol_greenfield.md` §§6.1–6.4. |

## Documented Spec Drifts (2025-10-01)

The following deviations are now deliberate, documented behaviour until the corresponding migration tasks land. Each entry calls out the active code path, the target greenfield contract, and the planned remediation hook.

- **Volume / multiscale rebroadcast dependency** – Volume and multiscale `state.update` handlers normalise inputs but immediately schedule `rebroadcast_meta()` (`src/napari_cuda/server/control/control_channel_server.py:1320`). *Target:* emit layer-scoped `notify.layers` deltas directly from the update sites (no meta rebroadcast) once the worker path above is in place.
- ~~**Camera notify integration** – `notify.camera` now carries the authoritative deltas (`src/napari_cuda/server/control/control_channel_server.py:624`), and the client records them in the state store (`src/napari_cuda/client/control/state_update_actions.py:279`). The presenter/viewer mirror still ignores these updates, so local overlays can drift until the next full scene snapshot. *Target:* plumb `handle_notify_camera` into the presenter façade and thin-client HUD.~~
- ~~**Alt-drag orbit gating** – The streaming client continues to gate orbital camera drags on an `in_vol3d` flag that never flips in 2D mode (`src/napari_cuda/client/streaming/client_loop/camera.py:166`). Even though orbit updates now flow through `state.update(camera.orbit)`, Alt-drag still degrades to pan outside true volume sessions. *Target:* derive the volume flag from the negotiated viewer state (e.g., `notify.dims.mode`) and fall back to orbit when a TurntableCamera is active.~~
- ~~**Camera notify integration (resolved 2025-10-01)** – `notify.camera` deltas are now broadcast via the greenfield path, recorded in the state store, and mirrored into the presenter HUD (`src/napari_cuda/client/runtime/stream_runtime.py:591`, `src/napari_cuda/client/streaming/presenter_facade.py:212`).~~
- ~~**Alt-drag orbit gating (resolved 2025-10-01)** – `notify.dims.mode` drives the volume gate and Alt+drag emits `camera.orbit` intents with verified test coverage (`src/napari_cuda/client/runtime/stream_runtime.py:1246`, `_tests/test_camera.py`).~~
- **Dims baseline readiness (resolved 2025-10-xx)** – `notify.dims` now streams directly from the worker snapshot. The server persists the latest `snapshot_dims_metadata()` payload and reuses it for intent echoes instead of rebuilding from `SceneSpec`; the client consumes those metadata fields without touching the cached scene snapshot.
- ~~**Client state helper shim** – `src/napari_cuda/client/streaming/client_loop/control.py` still
  wildcard-imports the reducer helpers to preserve legacy imports. *Target:* drop once
  downstream call sites are updated. (Server compatibility shim removed 2025-10-03.)~~
- **Client state helper shim (resolved 2025-10-03)** – Legacy importers now target
  `src/napari_cuda/client/control/state_update_actions.py` directly; the
  compatibility shim has been deleted alongside the server handler.
- ~~**Minimal client legacy protocol** – `src/napari_cuda/client/minimal_client.py` continues to speak the deprecated control format (`{"type":"ping"}` etc.). *Target:* archive the minimal client alongside the legacy protocol or port it to the greenfield envelopes after the migration reaches Phase 5.~~
- **Minimal client archive (resolved 2025-10-xx)** – The legacy minimal client now lives in `archive/legacy_client/minimal_client.py` and is excluded from the active packaging targets.

### 3.1 Keyframe Handling Baseline

- **force_idr() limitation** – `src/napari_cuda/server/rendering/encoder.py:223` toggles NVENC reconfigure, but it still fails to guarantee an IDR frame. We now treat encoder reset plus an explicit `request_idr()` as the contract: `pixel_channel.ensure_keyframe()` replays the last avcC immediately before calling `_try_reset_encoder()` and marks the stream dirty (`src/napari_cuda/server/pixel/pixel_channel_server.py:100`), `_ensure_keyframe()` schedules the reset after the state baseline (`src/napari_cuda/server/control/control_channel_server.py:2815`) and follows up by asking the worker to request an IDR for the next frame (`src/napari_cuda/server/egl_headless_server.py:425`). This mirrors the initial startup behaviour, ensures the next frame after a request is produced from a clean encoder state, and leaves a TODO breadcrumb (`src/napari_cuda/server/pixel/pixel_channel_server.py:137`) to re-enable targeted IDR forcing once NVENC reliability is proven.
- **Client VT gate expectations** – The streaming runtime logs “Keyframe request skipped (vt gate pending)” (`src/napari_cuda/client/runtime/stream_runtime.py:1276`) while waiting for the first IDR after a request. This is expected when the server had to reset the encoder; gating logic should be revisited once `force_idr()` is reliable.

Cross-reference these items when planning migration work; each future change should remove its entry from this section.

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
- `client/control/state_update_actions.py` (legacy shim
  `client/streaming/client_loop/control.py` removed 2025-10-03) centralises
  outgoing state-update helpers and local projection logic without the legacy
  "intent" terminology.
- `client/control/viewer_layer_adapter.py` (archive:
  `layer_state_bridge.py`) translates incoming layer/dims payloads into viewer
  mutations using the greenfield envelopes only.
- Legacy fallbacks (`video_config`, raw `dims.update` diffs, and the
  fire-and-forget `/state` keyframe request) have been removed; the control
  channel relies exclusively on the builder-backed notify and ack frames.

### 2.2 Control Channel (Server)

- `server/control/control_channel_server.py` handles the state lane and command
  registry directly (legacy `server/state_channel_handler.py` shim removed
  2025-10-03), dispatching via `MESSAGE_HANDLERS` for the remaining verbs.
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
| ~~`client/streaming/client_loop/control.py`~~ | `client/control/state_update_actions.py` | Legacy shim deleted; reducer helpers now live solely in `ControlStateContext` and friends. |
| ~~`server/state_channel_handler.py`~~ | — *(removed 2025-10-03; callers use `server/control/control_channel_server.py` directly)* | Legacy compatibility shim no longer required. |
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
   rejection payloads). **Status: complete — the server now parses greenfield
   envelopes, emits builder-backed `ack.state` frames with `intent_id`
   / `in_reply_to`, and rejects malformed requests per Appendix 5.2.**
4. Broadcast notified state strictly through `notify.scene` / `notify.dims` /
   `notify.camera` / `notify.layers`, validating payloads against
   `docs/protocol_greenfield/schemas/notify.scene.v1.schema.json`,
   `notify.dims.v1.schema.json`, `notify.camera.v1.schema.json`, and
   `notify.layers.v1.schema.json`; remove plain `dims.update` emission.
5. Keep the legacy WebSocket handler behind a feature flag (`LEGACY_STATE_WS`);
   route current deployments through the new handler in mirrored mode (emit
   both legacy + greenfield) until clients catch up. **Status: greenfield-only;
   legacy emission for the state lane has been removed.**

### Phase 3 – Client Control Channel Rewrite

**Completed**

- `control_channel_client.py` emits builder-backed `session.hello`, parses
  greenfield notify/ack envelopes, and exposes callbacks for
  `ack.state`/`reply.command`/`error.command`. The legacy
  `_handle_legacy_notify_state` path has been deleted.
- `pending_update_store.py` reconciles optimistic updates on `AckOutcome`,
  keyed by `frame_id`/`intent_id`, without legacy phase juggling or
  `client_seq` gating.
- Streaming loop + viewer bridge consume the new `ControlStateContext`,
  rebuild multiscale metadata from `notify.scene` policies, and apply
  ack outcomes to keep the proxy viewer in sync.
- Client control stack consumes `notify.stream` and `notify.dims` frames
  directly; the `video_config` / `dims.update` fallbacks and the ad-hoc
  `request_keyframe` helper have been removed.
- Resume-aware handshake + heartbeat enforcement is live: the client reuses
  `session.welcome` resume tokens, responds with `session.ack`, and tears down
  the session when heartbeats lapse.

**Remaining**

1. Implement the command-lane replacement for manual keyframe requests once
   Phase 4 wiring lands (`call.command napari.pixel.request_keyframe`).
2. Remove the `StateUpdateMessage` compatibility export once command-lane tests
   confirm no modules rely on the legacy dataclass.
3. Expand client/server control suites to cover ack rejection paths and heartbeat
   dropout, matching the testing matrix in §7.

**Legacy plumbing scheduled for removal in Phase 4:**

- **Server:** `MESSAGE_HANDLERS` entries for `'request_keyframe'`, `'force_idr'`, and
  any ad-hoc ping/keyframe verbs will be replaced by `call.command`
  requests routed through the command registry. `_handle_force_keyframe` and the
  related manual keyframe helpers become dead code once `napari.pixel.request_keyframe`
  ships as a command. The scene snapshot path still caches `SceneSpecMessage`
  payloads for legacy consumers; migrating to greenfield notify snapshots must
  precede removal of that dataclass.
- **Client:** the optimistic keyframe helpers (`request_keyframe_once`,
  `_ensure_keyframe`) and direct `'request_keyframe'`/`'force_idr'` sends are slated
  for deletion. The streaming loop will instead issue `call.command` envelopes via
  the new RPC helper and wait on `reply.command` / `error.command` for completion.
- **Client (scene/layer shims):** `SceneSpecMessage`, `LayerUpdateMessage`, and the
  associated registry adapters remain as legacy carriers for scene/layer metadata.
  Phase 4/5 must replace them with greenfield notify envelopes and typed models so
  the client no longer consumes legacy dataclasses before we drop the compatibility
  exports.
- **Protocol extension:** add a resumable `notify.scene.level` lane carrying the
  active multiscale level (`current_level`, `downgraded`, optional `levels` metadata)
  so HUD/slider consumers stay in sync with server-driven LOD switches without
  forcing a full scene snapshot rebroadcast. Advertise the feature in
  `session.welcome.payload.features`, sequence it alongside `notify.scene`, and
  teach the client control channel to hydrate multiscale state from the new
  payload.

### Phase 4 – Command Lane Enablement

1. Define a command registry on the server (initial catalogue limited to
   `napari.pixel.request_keyframe`; fit-to-view and viewer reset continue as
   `state.update` interactions for now).
2. Surface the registry via `session.welcome.payload.features.call.command` and
   ensure it reflects the active command list.
3. Implement client-side RPC helpers that queue `call.command` envelopes,
   await `reply.command` / `error.command`, and surface status/error to the UI.
4. Derive any future command-driven viewer actions from the advertised catalog
 once additional commands migrate to the lane.

   *Decision:* The viewer “fit to view” behaviour already runs locally on the
   client and triggers the existing dims/camera state updates. Likewise the
   home/reset button issues `state.update(camera.reset)` and does not require a
   separate command verb. We will revisit additional commands only if a future
   client cannot express the interaction via the state lane.

### Phase 5 – Legacy Cleanup

*Dual emission is removed during Phase 2 close-out; this phase sweeps any remaining compatibility artifacts.*

1. Remove `notify_state`/`notify_scene` legacy extras and the capability
   translation layer in `control_channel_client.py`.
2. Drop the compatibility feature flags and clean out re-export shims.
3. Remove log noise/artifacts tied to old payloads (e.g., `dims.update raw meta`).
4. Align terminology: treat protocol messages as `control envelopes` (or `notify envelopes`) and reserve `video frame` / `packet` for the encoder stream; update docs/code accordingly.

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
  - `MESSAGE_HANDLERS` entries for bespoke verbs (`set_camera`, `camera.*`).
    ~~`ping`, `request_keyframe`~~ *(removed 2025-10-03)*.
 - Continue renaming residual `_handler` modules to match greenfield ownership;
   the client shim has been removed, so audit the remaining server helpers for
   similar cleanup.
  - Delete the legacy `StateUpdateMessage` module and re-export once both client
    and tests consume the greenfield dataclasses exclusively.
  - Retire legacy scene/layer messages still in circulation:
    - `SceneSpecMessage` (`client/control/control_channel_client.py`,
      `client/runtime/stream_runtime.py`, `server/control/scene_snapshot_builder.py`,
      `server/scene_spec.py`, layer registry/tests).
    - `LayerUpdateMessage` and `LayerRemoveMessage` (same modules + layer registry).
    - `LayerSpec`, `LayerRenderHints`, `MultiscaleSpec`, `MultiscaleLevelSpec`
      (used by the client layer registry/adapter and server layer manager).
    Map each consumer to the greenfield notify payloads (`notify.scene` /
    `notify.layers` dataclasses) so these compatibility shims can be deleted with
    `StateUpdateMessage`.
- `scene_snapshot_builder.build_scene_spec_json` retired; new notifier payloads
  (`build_notify_scene_payload`) back the remaining callers.

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
- [ ] Command lane MVP live (request_keyframe advertised/exercised via
      `call.command`).
- [ ] Legacy bridges removed; greenfield-only deployment verified.
- [ ] Integration suite running greenfield scenarios in CI.
- [ ] Telemetry dashboards tracking ack latency, notify rates, heartbeat
      compliance.

Once every item clears, deprecate the legacy protocol documentation and treat
`docs/protocol_greenfield.md` plus this migration plan as the long-term source
of truth.
