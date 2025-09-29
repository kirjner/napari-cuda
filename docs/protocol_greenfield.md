# Greenfield Control Protocol

This document proposes a minimal, explicit control protocol that covers every
message the server currently emits (`scene.spec`, `state.update`,
`video.config`) while leaving room for future command/response flows (layer
creation, exports, etc.). The intent is to provide a single, easy-to-reason
about contract that works for desktop and mobile clients alike.

## Transport & Session

- Implementation scaffolding now lives alongside the legacy protocol but is not
  yet active on the wire:
  - `src/napari_cuda/protocol/envelopes.py` holds the new dataclasses.
  - `src/napari_cuda/protocol/parser.py` provides envelope parsing helpers.
  - Runtime still emits legacy payloads; the new shapes are test-covered only.

- **Wire**: single WebSocket connection (`wss://…/state`) per client.
- **Encoding**: UTF-8 JSON messages, each representing one envelope.
- **Handshake**: client sends `session.hello`, server replies with
  `session.welcome`. No other traffic is processed until the welcome lands.

```json
// client → server
{
  "type": "session.hello",
  "id": "0f6f…",
  "timestamp": 1759120000.123,
  "payload": {
    "protocol": 1,
    "client": {
      "name": "napari-ios",
      "version": "0.1.0"
    },
    "auth": {
      "token": "…"    // optional, application specific
    }
  }
}

// server → client
{
  "type": "session.welcome",
  "id": "0f6f…",              // echoes request id for correlation
  "timestamp": 1759120000.456,
  "payload": {
    "protocol": 1,
    "session": {
      "id": "srv-aa9d…",
      "capabilities": [
        "notify.state",
        "notify.scene",
        "notify.stream",
        "call.command"
      ]
    }
  }
}
```

If the server rejects a handshake it responds with `session.reject` describing
why (unsupported protocol, auth failure, etc.) and closes the socket.

## Envelope

Every subsequent message—whether it is a notification, a command, or a
response—uses the same top-level shape:

| Field       | Type   | Notes                                                     |
|-------------|--------|-----------------------------------------------------------|
| `type`      | string | Message topic (`notify.state`, `call.command`, …).        |
| `id`        | string | Optional UUID. Required for pairs of `call.*`/`reply.*`.  |
| `timestamp` | number | Seconds since epoch; allows ordering/diagnostics.         |
| `payload`   | object | Type-specific body.                                       |

All new message kinds should follow the `<category>.<name>` convention with
these reserved categories:

- `notify.*` – fire-and-forget broadcasts.
- `call.*` – client-issued commands that expect a response.
- `reply.*` – authoritative response to a `call.*` message (shares `id`).
- `error.*` – fatal errors. Can appear in response to either `call.*` or
  unexpected conditions detected by the server.

## Notifications (current capabilities)

### `notify.scene`

Full baseline describing the current viewer state. Supersedes today’s
`scene.spec` message.

```json
{
  "type": "notify.scene",
  "timestamp": 1759120012.0,
  "payload": {
    "version": 1,
    "scene": { … },          // same shape as SceneSpec.to_dict()
    "state": {
      "current_step": [206, 0, 0],
      "ndisplay": 2,
      "volume_mode": "slice"
    }
  }
}
```

### `notify.state`

Single property mutation (dims step, layer gamma, view zoom). This is a
direct evolution of the existing `state.update` payload; only the envelope
label changes.

```json
{
  "type": "notify.state",
  "timestamp": 1759120012.5,
  "payload": {
    "scope": "dims",            // "dims" | "layer" | "view" | …
    "target": "z",              // axis label, layer id, etc.
    "key": "step",              // property key
    "value": 206,
    "server_seq": 862,
    "axis_index": 0,
    "current_step": [206, 0, 0],
    "meta": {                    // dims metadata snapshot
      "ndim": 3,
      "order": ["z", "y", "x"],
      "axis_labels": ["z", "y", "x"],
      "displayed": [2, 1],
      "ndisplay": 2,
      "range": [[0, 679], [0, 1539], [0, 1156]],
      "controls": { "gamma": 1.3 }
    },
    "interaction_id": "drag-1",
    "phase": "update",
    "client": {
      "id": "ios-123",
      "seq": 14
    },
    "ack": true
  }
}
```

For layer and view scopes the same metadata fields remain optional—they are
omitted when not applicable.

### `notify.stream`

Configuration for the pixel/video channel (replacement for `video.config`).

```json
{
  "type": "notify.stream",
  "timestamp": 1759120015.2,
  "payload": {
    "codec": "h264",
    "fps": 60,
    "width": 1920,
    "height": 1080,
    "bitrate": 10000000,
    "idr_interval": 120
  }
}
```

## Commands & Responses (extensible surface)

The new protocol reserves a command channel but does not force current
clients to use it yet. When we migrate discrete operations (layer create,
remove, export, etc.) they will use this pattern.

### Command request: `call.command`

```json
{
  "type": "call.command",
  "id": "cmd-77f5…",
  "timestamp": 1759120020.0,
  "payload": {
    "name": "layer.create",
    "args": {
      "uri": "s3://…/image.zarr",
      "layer_type": "image",
      "display_name": "Sample"
    }
  }
}
```

### Command success: `reply.command`

```json
{
  "type": "reply.command",
  "id": "cmd-77f5…",
  "timestamp": 1759120020.3,
  "payload": {
    "status": "ok",
    "result": {
      "layer_id": "layer-7",
      "server_seq": 912
    }
  }
}
```

### Command failure: `error.command`

```json
{
  "type": "error.command",
  "id": "cmd-77f5…",
  "timestamp": 1759120020.3,
  "payload": {
    "status": "failed",
    "code": "LayerExists",
    "message": "Layer name already in use",
    "details": { "layer_id": "layer-7" }
  }
}
```

The `id` field allows the caller to match responses with in-flight commands.
Clients that only care about notifications can ignore this category entirely.

## Error Notifications

For transport-level or unrecoverable errors the server emits
`error.session` (missing auth, protocol mismatch) or `error.state`
(malformed payload). After sending an error the server decides whether to
close the connection based on severity.

```json
{
  "type": "error.state",
  "timestamp": 1759120025.4,
  "payload": {
    "code": "InvalidValue",
    "message": "gamma must be positive",
    "offending": {
      "scope": "layer",
      "target": "layer-0",
      "key": "gamma",
      "value": -2.0
    }
  }
}
```

## Mapping from Today’s Protocol

| Current message | Proposed message | Notes                                   |
|-----------------|------------------|-----------------------------------------|
| `scene.spec`    | `notify.scene`   | Body stays the same (SceneSpec dict).    |
| `state.update`  | `notify.state`   | Envelope rename; payload unchanged.     |
| `video.config`  | `notify.stream`  | Unified naming for stream metadata.     |
| *(none today)*  | `call.command`   | Future extension for discrete actions.  |
| *(none today)*  | `reply.command`  |                                          |
| *(none today)*  | `error.*`        | Structured error reporting.             |

Legacy messages (`layer.update`, `layer.remove`, `camera_update`, `dims.update`)
are preserved only for backward compatibility and should be removed once all
clients speak the new contract.

## Extensibility Guidelines

- **Namespacing**: new functionality should pick a clear namespace
  (`notify.telemetry`, `call.export`, etc.) so message intent is obvious.
- **Versioning**: bump the handshake `protocol` number when changing semantics
  or removing fields. Minor additions to payloads can use optional keys that
  default sensibly.
- **Streaming**: high-frequency interactions (sliders, camera drags) stay on
  `notify.state`. Commands must never be required for the hot path.
- **Auth**: the handshake gives us a single place to inject authentication and
  capability negotiation; future work can extend the `payload.auth` block.
- **Binary payloads**: leave the pixel/video stream on its dedicated channel.
  If another binary stream is needed we can add `notify.<name>` with out-of-band
  references (e.g., presigned URLs).

## Security Considerations

Refer to the security checklist in `docs/server_state_update_plan.md`. The new
handshake envelope introduces a natural anchor for bearer tokens, mutual TLS
proofs, or other auth mechanisms while keeping the data channel itself
unchanged.

## Migration Plan

1. **Author shared types**
   - Implement lightweight dataclasses/TypedDicts for the new envelopes
     (`SessionHello`, `NotifyState`, etc.) in `napari_cuda.protocol`. Maintain
     compatibility shims that can marshal to the legacy shapes until clients
     update.

2. **Dual-emit phase**
   - Modify the server state channel to emit both the legacy messages and the
     new envelopes. Gate the behavior behind a capability flag negotiated in
     `session.hello` so upgraded clients can opt in (`supports_notify_state`).
   - Ensure the worker baseline/orchestration emits `notify.scene` and
     `notify.stream` alongside the existing payloads.
   - For development/staging the dual emission is currently driven by the
     `NAPARI_CUDA_PROTOCOL_DUAL_EMIT=1` toggle, which causes the server to
     broadcast `notify.state`/`notify.scene`/`notify.stream` in addition to the
     legacy payloads.

3. **Client opt-in**
   - Extend the desktop client reducer to recognise the new envelope names
     while leaving the old `state.update` path intact.
   - In parallel, build the iOS client against the new contract to validate
     slider performance, ack handling, and handshake auth.

4. **Command channel scaffolding**
   - Add stubs on the server (`call.command` handler returning
     `error.command`/`reply.command` with `NotImplemented`) so clients can
     exercise the request/response flow without changing behaviour.

5. **Cut-over**
   - Once all clients speak the new protocol version, flip the default to send
     only the `notify.*` envelopes and retire the legacy names.
   - Clean up unused message classes (`LayerUpdateMessage`, `LayerRemoveMessage`,
     etc.) and remove compatibility branches.

6. **Post-migration**
   - Move discrete operations (layer create/remove, exports) onto
     `call.command` / `reply.command`.
   - Tighten the security posture by enforcing authenticated handshakes and
     per-capability authorisation.
