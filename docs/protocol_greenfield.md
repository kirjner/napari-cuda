# Greenfield Control Protocol Specification

This document is the authoritative reference for the next-generation control
protocol used by the CUDA streaming stack. The current implementation is
considered legacy; the naming, flow, and guarantees captured here define the
clean-room wire format we are building toward.

## 1. Purpose & Guiding Principles

**Purpose**

- Deliver a deterministic, resumable control channel while keeping the server as
  the single source of truth.
- Separate high-frequency view/state mutations from authenticated request/response
  commands.
- Provide a versioned, schema-backed wire format that remains evolvable and
  testable.

**Guiding Principles**

- **Single Source of Truth** – The server emits canonical state; clients cache
  and render but never become authoritative.
- **Deterministic Reconnects** – Every session starts with a full `notify.scene`
  snapshot; resumable topics expose monotonic sequences and resume tokens.
- **Lane Separation** –
  - `state.update` / `notify.*` handle low-latency, loss-tolerant updates.
  - `call.command` / `reply.command` handle discrete actions requiring
    confirmation or payloads.
- **Explicit Schema** – Every frame type has a JSON Schema definition in
  `docs/protocol_greenfield/schemas/`. CI validates both server and client
  payloads against these schemas.
- **Minimal Latency** – Orbit, zoom, dims sliders stay off the
  request/response path.
- **Observability** – Sequence numbers, tokens, and identifiers are first-class
  fields.

## 2. Vocabulary & Frame Grammar

All frames are UTF-8 JSON objects that follow the envelope contract below.

| Field       | Type   | Required                 | Notes                                                                    |
|-------------|--------|--------------------------|---------------------------------------------------------------------------|
| `type`      | string | ✓                        | Names the frame (e.g. `session.hello`, `notify.scene`, `state.update`).   |
| `version`   | int    | ✓                        | Schema version for the `type`.                                            |
| `session`   | str    | ✓ after welcome          | Session UUID returned in `session.welcome`.                               |
| `frame_id`  | str    | acked lanes¹             | UUID for observability; server/client emit by default (see note).         |
| `timestamp` | float  | ✓                        | Seconds since epoch (UTC), microsecond precision.                         |
| `seq`       | int    | resumable topics         | Monotonic per resumable topic; absent elsewhere.                          |
| `delta_token` | str  | resumable topics         | Resume token; opaque string updated with each frame.                      |
| `intent_id` | str    | optional                 | Client-specified correlation id (`state.*` + matching notifies).          |
| `payload`   | object | ✓                        | Type-specific schema.                                                     |

¹ Frames that require acknowledgements—`state.update`, `call.command`, and any
future acked lanes—MUST populate `frame_id` so servers can mirror it from
`payload.in_reply_to`. **Hot-path notify identifiers** – `notify.dims` and
`notify.camera` may omit `seq` for size but MUST include either `intent_id`
(when they originate from a local intent) or a short `frame_id` (8-character
UUID prefix emitted by the server) for duplicate detection and telemetry
correlation.

### 2.1 Naming Conventions

- Notify lanes: `notify.scene`, `notify.layers`, `notify.dims`, `notify.camera`,
  `notify.stream`, `notify.telemetry`, `notify.error`.
- State lane: `state.update`, `ack.state`.
- Command lane: `call.command`, `reply.command`, `error.command`.
- Session control: `session.hello`, `session.welcome`, `session.reject`,
  `session.heartbeat`, `session.ack`, `session.goodbye`.

Schemas for each `type` live under
`docs/protocol_greenfield/schemas/<type>.schema.json`.

## 3. Handshake

### 3.1 Client → Server `session.hello`

```json
{
  "type": "session.hello",
  "version": 1,
  "frame_id": "hello-6d5c...",
  "timestamp": 1759179000.123,
  "payload": {
    "protocols": [1],
    "client": {
      "name": "napari-cuda-desktop",
      "version": "0.7.0",
      "platform": "macOS-14.5-arm64"
    },
    "features": {
      "notify.scene": true,
      "notify.stream": true,
      "notify.telemetry": false,
      "call.command": true
    },
    "resume_tokens": {
      "notify.scene": "tok_scene_8fd2",
      "notify.layers": "tok_layers_17aa",
      "notify.stream": null
    },
    "auth": {
      "type": "bearer",
      "token": "eyJhbGciOi..."
    }
  }
}
```

### 3.2 Server Responses

- On failure, the server emits `session.reject` (with `code`/`message`) and
  closes the socket.
- On success, it emits `session.welcome`:

```json
{
  "type": "session.welcome",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "welcome-bf5...",
  "timestamp": 1759179000.200,
  "payload": {
    "protocol_version": 1,
    "session": {
      "id": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
      "heartbeat_s": 15.0,
      "ack_timeout_ms": 250
    },
    "features": {
      "notify.scene": { "enabled": true, "version": 1, "resume": true },
      "notify.layers": { "enabled": true, "version": 1, "resume": true },
      "notify.dims": { "enabled": true, "version": 1, "resume": false },
      "notify.camera": { "enabled": true, "version": 1, "resume": false },
      "notify.stream": { "enabled": true, "version": 1, "resume": true },
      "notify.telemetry": { "enabled": false },
      "call.command": {
        "enabled": true,
        "version": 1,
        "commands": [
          "napari.viewer.fit_to_view",
          "napari.widgets.features_table",
          "napari.toggle_shape_measures"
        ]
      }
    }
  }
}
```

**Feature downgrade** – If the client requests resume for a topic that is
advertised as `enabled: false`, the handshake still succeeds. The server ignores
submitted tokens for that topic and the client MUST drop local caches and treat
it as disabled.

### 3.3 Post-Welcome Baseline

Immediately after `session.welcome`, the server emits:

1. `notify.scene` (full snapshot, resumable; `seq = 0` and fresh `delta_token`).
2. `notify.layers` (if the snapshot references split layer state).
3. `notify.stream` (codec metadata).
4. Any optional channels enabled for the session (e.g. `notify.telemetry`).

Heartbeats begin once the baseline is sent.

## 4. Topics & Resumability

| Type             | Resume | Retention                             | Fallback if token stale                 | Typical Rate |
|------------------|--------|---------------------------------------|-----------------------------------------|--------------|
| `notify.scene`   | yes    | Current snapshot only                 | Resend full snapshot (seq reset to 0)   | bursts       |
| `notify.layers`  | yes    | Ring buffer: min(512 deltas, 5 min)   | Send fresh snapshot + current deltas    | low          |
| `notify.dims`    | no     | n/a                                   | Server emits fresh `notify.scene`; client waits for baseline | ≤ 60 Hz      |
| `notify.camera`  | no     | n/a                                   | Server emits fresh `notify.scene`; client waits for baseline | ≤ 60 Hz      |
| `notify.stream`  | yes    | Latest config only                    | Reissue full config                     | rare         |
| `notify.telemetry` | no  | n/a                                   | Drop                                    | high         |
| `notify.error`   | no     | n/a                                   | Drop                                    | sparse       |

- Resumable topics include both `seq` and `delta_token`. When `notify.scene`
  emits a new snapshot, every resumable topic resets its own `seq` counter to 0
  and receives a fresh token. Clients must treat this as a new epoch and discard
  cached sequence/tokens.
- Once the `notify.layers` buffer wraps, the next resume request with an expired
  token triggers a snapshot replay (no rejection).
- For non-resumable lanes (`notify.dims`, `notify.camera`), reconnecting clients
  wait for the server to emit a fresh `notify.scene` baseline. Servers SHOULD
  send that snapshot immediately after `session.welcome` and whenever they
  detect a client missing current view/camera state.

## 5. State Update Lane

### 5.1 Request (`state.update`, version 1)

```json
{
  "type": "state.update",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "state-88b4",
  "timestamp": 1759179010.314,
  "intent_id": "intent-88b4",
  "payload": {
    "scope": "view",
    "target": "main",
    "key": "ndisplay",
    "value": 3
  }
}
```

Payload schemas are scope-specific (e.g. `view/main`, `dims/<axis>`,
`camera/main`). All schemas live alongside the envelope definitions.

### 5.2 Acknowledgement (`ack.state`, version 1)

- The server should respond within 50 ms on LAN; the hard client timeout is
  `ack_timeout_ms` from `session.welcome` (default 250 ms).
- Accepted example:

```json
{
  "type": "ack.state",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "ack-88b4",
  "timestamp": 1759179010.330,
  "payload": {
    "intent_id": "intent-88b4",
    "in_reply_to": "state-88b4",
    "status": "accepted",
    "applied_value": 3
  }
}
```

- Rejected example:

```json
{
  "type": "ack.state",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "ack-88b4",
  "timestamp": 1759179010.335,
  "payload": {
    "intent_id": "intent-88b4",
    "in_reply_to": "state-88b4",
    "status": "rejected",
    "error": {
      "code": "state.forbidden",
      "message": "Viewer is read-only",
      "details": {"scope": "view", "key": "ndisplay"}
    }
  }
}
```

Clients treat missing acks after `ack_timeout_ms` as failures and MAY retry only
when the operation is documented as idempotent.

### 5.3 Resulting Notifies

- The server applies the mutation atomically and then broadcasts the resulting
  `notify.*` frame. When the notify is a direct consequence of the acknowledged
  intent, it includes the `intent_id` so the client reducer can reconcile.
- For externally sourced updates (other clients, automation), the notify omits
  `intent_id` and clients treat it as authoritative state.

## 6. Command Lane

### 6.1 Request (`call.command`, version 1)

```json
{
  "type": "call.command",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "cmd-02f9",
  "timestamp": 1759179020.14,
  "payload": {
    "command": "napari.viewer.fit_to_view",
    "args": [],
    "kwargs": {},
    "origin": "ui.menu"
  }
}
```

### 6.2 Success (`reply.command`)

```json
{
  "type": "reply.command",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "reply-cmd-02f9",
  "timestamp": 1759179020.15,
  "payload": {
    "in_reply_to": "cmd-02f9",
    "status": "ok",
    "result": null
  }
}
```

### 6.3 Failure (`error.command`)

```json
{
  "type": "error.command",
  "version": 1,
  "session": "bf52c9c6-4875-4280-9f00-6022cb5106d6",
  "frame_id": "err-cmd-02f9",
  "timestamp": 1759179020.16,
  "payload": {
    "in_reply_to": "cmd-02f9",
    "status": "error",
    "code": "command.forbidden",
    "message": "Fit to view unavailable",
    "details": { "reason": "viewer locked" }
  }
}
```

### 6.4 Idempotency & Permissions

- Commands are non-idempotent unless documented otherwise. If a command returns
  an `idempotency_key`, clients MAY retry when they receive
  `error.command.code = "command.retryable"`.
- `session.welcome.payload.features.call.command.commands` lists the commands
  enabled for the session. Clients should reflect this in the UI.
- The server rejects unknown/forbidden commands with `command.forbidden`.

## 7. Heartbeats & Shutdown

- The server emits `session.heartbeat` every `heartbeat_s` seconds.
- The client must respond with `session.ack` within the same window unless it
  sent a `state.update` or `call.command` during the interval (recent activity
  counts as liveness).
- Missing two consecutive heartbeats triggers `session.goodbye` and coordinated
  closure of state + pixel channels.
- Clients may send `session.goodbye` before closing voluntarily.

## 8. Error Handling

- Handshake failures: `session.reject` with `code`/`message`.
- State validation failures: `ack.state` with `status = "rejected"` and an
  `error` payload.
- Transport/runtime warnings: `notify.error` with `severity = "warning"` or
  `"info"`.
- Fatal faults: `notify.error` with `severity = "critical"` followed by
  `session.goodbye`. Before closing, the server flushes any pending
  `ack.state`, `reply.command`, or `error.command` frames. Both state and pixel
  sockets close together; the pixel channel uses close code `1011` (Internal
  Error).

## 9. Size & Performance Targets

- `notify.dims` and `notify.camera` payloads target ≤ 200 B at ≤ 60 Hz and
  should keep JSON shallow. Enable `permessage-deflate` on the WebSocket for
  additional compression as needed.
- `notify.scene` snapshots may be large; rely on transport compression (TLS
  compression or WebSocket deflate) for delivery.
- Encoding overhead for hot-path frames should remain below 0.1 ms.

## 10. Resume Tokens & Sequence Semantics

- Clients persist the latest `seq` and `delta_token` per resumable topic.
- On reconnect, they include these tokens in `session.hello`. The server either
  replays deltas within the retention window or sends a fresh snapshot and new
  token.
- When a new snapshot is issued, clients must treat it as a new epoch and reset
  stored sequences/tokens.

## 11. Security & Auth (Framework Placeholder)

- `session.hello.auth` supports bearer tokens today and leaves space for MTLS in
  the future (`{"type": "mtls"}` via out-of-band cert validation).
- Per-topic ACLs gate `state.update`; forbidden requests yield
  `ack.state.status = "rejected"` with `error.code = "state.forbidden"`.
- Command execution uses AppModel permissions. Authorization failures return
  `error.command` with `code = "command.forbidden"`.

## 12. Client Responsibilities

- Issue `session.hello` with supported protocol versions, features, and resume
  tokens.
- Maintain a single `ClientState` keyed by topic; selectors feed renderer/input
  layers (no reliance on legacy extras).
- Track outstanding intents with timeouts based on `ack_timeout_ms`.
- Persist resume tokens and cleanly close with `session.goodbye` when exiting.
- Respect the advertised command catalogue and disabled features.
- Handle `notify.error` gracefully (e.g. downgrade to PyAV on decode warnings).

## 13. Server Responsibilities

- Validate every inbound frame against the relevant schema.
- Apply state mutations atomically; emit `ack.state` before broadcasting
  resulting `notify.*` frames.
- Maintain per-topic `seq`/`delta_token` to honour resumability SLAs.
- Execute commands via AppModel under session permissions.
- Emit heartbeats, detect silent clients, and flush outstanding responses before
  teardown.
- Provide structured logging for handshake, notify, state, command, and error
  flows.

## 14. Migration Plan

1. Implement schemas and validators for the new envelope/types.
2. Add feature-flagged dual emission (legacy + greenfield) on the server.
3. Update the thin client to hydrate the new `ClientState` from `notify.*`.
4. Wire `state.update` + `ack.state` end-to-end with optimistic reducer support.
5. Bring up the command lane starting with
   `napari.viewer.fit_to_view`, `napari.widgets.features_table`,
   `napari.toggle_shape_measures`.
6. Persist resume tokens and validate reconnect flows (stale vs fresh snapshots).
7. Decommission legacy notify/state messages once all clients speak the new
   protocol.
8. Refresh documentation, examples, and monitoring tooling.

## 15. Testing Matrix

| Scenario                    | Expected Frames / Assertions                                                |
|-----------------------------|-----------------------------------------------------------------------------|
| Initial connect             | `session.welcome` → `notify.scene(seq=0)` → `notify.layers` → `notify.stream`. |
| Alt-drag orbit              | `state.update(camera.orbit)` → `ack.state(applied_value)` → `notify.camera(intent_id)`. |
| Toggle 2D/3D                | `state.update(view.ndisplay)` → `ack.state` → `notify.dims(intent_id)`.       |
| Command success             | `call.command` → `reply.command` → follow-up `notify.*` reflecting state.     |
| Command forbidden           | `call.command` → `error.command(code=command.forbidden)`.                     |
| Reconnect with valid token  | `notify.layers` deltas replayed (`seq` increments).                           |
| Reconnect with stale token  | `notify.scene` snapshot with `seq=0` then current deltas.                     |
| Missing heartbeat           | `session.heartbeat` ×2 without reply → `session.goodbye` + socket close.      |
| Critical server fault       | `notify.error(severity=critical)` → flushed outstanding replies → `session.goodbye`. |

Integration harnesses should replay golden snapshots, inject packet loss, and
assert that the client’s `ClientState` matches the server’s authoritative scene
throughout reconnects and command flows.

## Appendix A. Frame & Payload Reference

The tables below restate the envelope and payload requirements from §§2–14 so
implementations can translate code directly from these checklists without
re-parsing the prose.

### A.1 Session Frames

| Frame | Envelope fields | Payload schema | Notes |
|-------|-----------------|----------------|-------|
| `session.hello` | `type`, `version`, `frame_id`, `timestamp`; **no** `session` yet | `protocols` (list[int]); `client` (`name`, `version`, `platform`); `features` (topic→bool); `resume_tokens` (topic→str or null); optional `auth` (`type`, `token` or MTLS placeholder) | `frame_id` required for observability; clients may omit features they do not support. |
| `session.welcome` | `type`, `version`, `session`, `frame_id`, `timestamp` | `protocol_version`; `session` (`id`, `heartbeat_s`, optional `ack_timeout_ms`); `features` per topic (`enabled`, `version`, `resume`, optional `commands` list) | Server echoes negotiated capabilities; snapshot seq resets follow immediately after welcome. |
| `session.reject` | `type`, `version`, `frame_id`, `timestamp` | `code`, `message`, optional `details` object | Socket closes after reject; no `session` field because handshake failed. |
| `session.heartbeat` | `type`, `version`, `session`, `frame_id`, `timestamp` | *(empty object)* | Server emits every `heartbeat_s`. |
| `session.ack` | `type`, `version`, `session`, `frame_id`, `timestamp` | *(empty object)* | Client replies unless it sent another frame within the window. |
| `session.goodbye` | `type`, `version`, `session`, `frame_id`, `timestamp` | Optional `code`, `message`, `reason` | Emitted on graceful shutdown or after critical errors. |

### A.2 Notify Frames

| Frame | Envelope fields | Payload schema | Notes |
|-------|-----------------|----------------|-------|
| `notify.scene` | `type`, `version`, `session`, `frame_id` optional, **requires** `seq`, `delta_token`, `timestamp` | Full snapshot: `viewer` (dims/camera/settings), `layers[]` (id/type/name + metadata/render/multiscale), `policies`, optional ancillary sections | Snapshot baseline always uses `seq = 0`; issuing a snapshot resets seq/token for every resumable topic. |
| `notify.layers` | `type`, `version`, `session`, optional `frame_id`, **requires** `seq`, `delta_token`, `timestamp` | `layer_id`; `changes` dict describing delta fields | Ring buffer retention: min(512 deltas, 5 minutes). If token stale, server sends fresh snapshot + new deltas. |
| `notify.stream` | `type`, `version`, `session`, optional `frame_id`, **requires** `seq`, `delta_token`, `timestamp` | `codec`, `format`, `fps`, `frame_size`, `nal_length_size`, `avcc`, `latency_policy`, `vt_hint` | Latest config supersedes prior tokens; seq resets with every new snapshot epoch. |
| `notify.dims` | `type`, `version`, `session`, `timestamp`; omit `seq`/`delta_token` | `current_step`; `ndisplay`; `mode`; `source`; optional `intent_id` | If triggered by local intent, server echoes the `intent_id`; otherwise include an 8-char `frame_id` for observability. |
| `notify.camera` | `type`, `version`, `session`, `timestamp`; omit `seq`/`delta_token` | `mode`; `delta` (orbit/pan/zoom deltas); `origin`; optional `intent_id` | Hot-path frames target ≤ 200 B payloads. |
| `notify.telemetry` | `type`, `version`, `session`, `timestamp`; omit `seq`/`delta_token` | Rolling stats: `presenter`, `decode`, `queue_depth` numeric fields | High-rate diagnostics; clients may drop if disabled. |
| `notify.error` | `type`, `version`, `session`, `timestamp`; optional `frame_id` | `domain`; `code`; `message`; `severity` (`info`, `warning`, `critical`); optional `context` map | Severity `critical` must be followed by `session.goodbye` and coordinated channel teardown. |

### A.3 State Lane

| Frame | Envelope fields | Payload schema | Notes |
|-------|-----------------|----------------|-------|
| `state.update` | `type`, `version`, `session`, `frame_id`, `timestamp`, `intent_id` | `scope`; `target`; `key`; `value` (schema varies by scope) | No legacy `extras`/`controls`; clients obey `ack_timeout_ms` from welcome. |
| `ack.state` | `type`, `version`, `session`, `frame_id`, `timestamp` | `intent_id`; `in_reply_to`; `status` (`accepted`, `rejected`); optional `applied_value`; on rejection `error` (`code`, `message`, optional `details`) | Servers aim for 50 ms response; hard timeout is `ack_timeout_ms` (default 250 ms). |

### A.4 Command Lane

| Frame | Envelope fields | Payload schema | Notes |
|-------|-----------------|----------------|-------|
| `call.command` | `type`, `version`, `session`, `frame_id`, `timestamp` | `command` string; `args` list; `kwargs` dict; optional `origin` | Non-idempotent unless server returns explicit `idempotency_key`. |
| `reply.command` | `type`, `version`, `session`, `frame_id`, `timestamp` | `in_reply_to`; `status = "ok"`; optional `result`; optional `idempotency_key` | Positive acknowledgement; resulting state still flows via notify lanes. |
| `error.command` | `type`, `version`, `session`, `frame_id`, `timestamp` | `in_reply_to`; `status = "error"`; `code`; `message`; optional `details` map | Codes include `command.forbidden`, `command.not_found`, `command.retryable`. |

### A.5 Sequencing, Resumability, and Identifiers

- Resumable topics (`notify.scene`, `notify.layers`, `notify.stream`) require both
  `seq` and `delta_token`. `notify.scene` snapshots reset their own `seq` to 0 and
  start a new epoch for the other resumable topics (they must also reset `seq` and
  issue fresh tokens).
- Non-resumable topics (`notify.dims`, `notify.camera`, `notify.telemetry`,
  `notify.error`) omit `seq`/`delta_token`. `notify.dims` and `notify.camera` must
  include either the triggering `intent_id` or a short `frame_id` for duplicate
  detection.
- `frame_id` is mandatory for every lane that expects acknowledgements
  (`state.update`, `ack.state`, `call.command`, `reply.command`, `error.command`).
- Clients persist the latest `seq`/`delta_token` per resumable topic and send them
  in `session.hello.resume_tokens`. Servers either replay deltas or emit a new
  snapshot and tokens.
- `notify.error` with `severity = "critical"` obligates the server to flush any
  outstanding `ack.state` / `reply.command` / `error.command` frames before sending
  `session.goodbye` and closing both state and pixel sockets (pixel closes with
  code `1011`).

## Appendix B. Envelope Construction Checklist

Use this checklist when emitting or parsing frames so envelope fields never drift
from the spec. Think of it as the builder/parsing contract that the helpers in
`napari_cuda.protocol.greenfield.envelopes` must enforce.

### B.1 Core Rules (apply to every frame)

- Set `version = 1` everywhere. There is no shortcut even for legacy shim traffic.
- Always populate `timestamp` with `time.time()` (float seconds, microsecond
  precision). Helpers should never allow a missing timestamp.
- Only include `session` after the server emits `session.welcome`. Handshake
  frames prior to welcome (`session.hello`, `session.reject`) must omit it.
- `frame_id` is mandatory for every acked lane (`state.update`, `ack.state`,
  `call.command`, `reply.command`, `error.command`) and for heartbeat frames. For
  hot-path notify lanes (`notify.dims`, `notify.camera`) emit either the triggering
  `intent_id` or a short 8-char `frame_id`.
- `seq` and `delta_token` appear **only** on resumable topics: `notify.scene`,
  `notify.layers`, `notify.stream`. All other lanes must omit them.
- `intent_id` is permitted on notify lanes only when the frame is a direct echo of
  a client intent. Never invent an `intent_id` for server-originated deltas.
- Snapshots (`notify.scene`) reset their own `seq` to `0` and invalidate every
  cached resume token. Emit fresh tokens for the next frames.

### B.2 Envelope Field Matrix

| Lane / Type | Requires `session`? | Requires `frame_id`? | Requires `seq`/`delta_token`? | Allows `intent_id`? | Extra Notes |
|-------------|---------------------|-----------------------|-------------------------------|---------------------|-------------|
| `session.hello` | ✗ | ✓ | ✗ | ✗ | Pre-session handshake; only `frame_id` + `timestamp`.
| `session.welcome` | ✓ | ✓ | ✗ | ✗ | Sets the session id and advertises resume support per topic.
| `session.reject` | ✗ | ✓ | ✗ | ✗ | Last frame before closing on handshake failure.
| `session.heartbeat` | ✓ | ✓ | ✗ | ✗ | Server cadence = `heartbeat_s`.
| `session.ack` | ✓ | ✓ | ✗ | ✗ | Client replies when no other traffic proves liveness.
| `session.goodbye` | ✓ | ✓ | ✗ | ✗ | May include optional `code`/`message`.
| `notify.scene` | ✓ | optional (snapshot or echo) | ✓ | optional | Snapshot baseline; `seq = 0` on new epoch.
| `notify.layers` | ✓ | optional | ✓ | optional | Ring buffer (≥512 deltas or 5 min retention).
| `notify.stream` | ✓ | optional | ✓ | ✗ | Latest stream config replaces prior token.
| `notify.dims` | ✓ | optional* | ✗ | optional* | *Must send either `intent_id` or short `frame_id` (<=8 chars).
| `notify.camera` | ✓ | optional* | ✗ | optional* | Same rule as dims for identifiers.
| `notify.telemetry` | ✓ | optional | ✗ | ✗ | High-rate diagnostics; drop when disabled.
| `notify.error` | ✓ | optional | ✗ | ✗ | `severity="critical"` → send `session.goodbye` after flushing replies.
| `state.update` | ✓ | ✓ | ✗ | ✓ (required) | Payload drives authoritative state changes.
| `ack.state` | ✓ | ✓ | ✗ | ✗ (lives in payload) | Payload echoes `intent_id` + `in_reply_to`.
| `call.command` | ✓ | ✓ | ✗ | ✗ | `frame_id` correlates to reply/error.
| `reply.command` | ✓ | ✓ | ✗ | ✗ | Include optional `idempotency_key` when safe to retry.
| `error.command` | ✓ | ✓ | ✗ | ✗ | Provide `error.details` when the client can remediate.

### B.3 Helper Expectations

When rebuilding `protocol.greenfield.envelopes`:

1. **Builder functions** must inject the correct envelope defaults (version,
   timestamp, identifier strategy) and call the `validate()` methods on the new
   dataclasses before returning.
2. **Parser helpers** should fail fast with actionable errors (`ValueError`
   listing the missing or unexpected envelope fields) before attempting payload
   hydration.
3. **Resumable utilities** must centralise `seq` and `delta_token` bookkeeping so
   every emitter obeys the reset semantics listed above.
4. **Glue layers** (dual emitters / parser shims) should never mutate the
   envelopes by hand—call the helpers and trust them to enforce the matrices
   laid out here.

Treat this appendix as the final word on envelope structure; if runtime pressure
suggests a different optimisation, update the table and accompanying prose so the
contract remains unambiguous.
