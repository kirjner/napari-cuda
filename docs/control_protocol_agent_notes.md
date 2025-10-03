# Control Protocol Agent & Thin-Client Notes

This note captures the current state of the greenfield control protocol in the
context of smarter clients and MCP-style agents (e.g., a connectomics
proofreading assistant). It summarizes what the protocol already enables and
the architectural enhancements we should plan to unlock more autonomous
collaborators.

## What the protocol already supports

- **Typed envelopes and sequencing** – All control traffic rides
  schema-backed envelopes (`session.*`, `state.update`, `ack.state`,
  `notify.*`) with explicit `frame_id`, `intent_id`, and replay tokens. That
  determinism is the backbone for thin clients and agents that need to reason
  about state transitions.
- **Optimistic intents with worker-confirmed acks** – Clients issue
  `state.update` intents and may project changes locally, but metadata-impacting
  scopes (dims, view mode, ROI) only receive
  `ack.state(status="accepted")` after the render worker applies the change.
  Pure control-side tweaks (camera pans, debug toggles) continue to ack
  immediately, keeping the loop snappy without sacrificing correctness.
- **Authoritative notify lanes** – Worker snapshots drive `notify.dims`,
  `notify.scene`, and related streams with a monotonically increasing
  `scene_seq`. Agents can consume these broadcasts to mirror the viewer
  precisely, confident each payload reflects committed worker state rather than
  an optimistic echo.
- **Resume tokens** – `session.welcome` advertises per-topic resume cursors so
  reconnecting clients (automated or human) can request only the missing
  deltas. This keeps bandwidth manageable and preserves context for long-lived
  proofreaders.
- **Command lane** – The server advertises a command catalogue via
  `session.welcome.features.call.command.commands` and currently exposes
  `napari.pixel.request_keyframe`. Agents can trigger keyframes without relying
  on legacy verbs, and additional commands will slot into the same lane.
- **Transport neutrality** – Because all semantics live in the envelopes,
  the same protocol can be tunneled over different transports or split across
  services without changing payload shapes.

### Thin client interaction loop (example)

1. User drags to pan; the client emits a `state.update(view)` with new camera
   parameters and immediately shifts the currently decoded frame to present the
   motion optimistically.
2. The server applies the update, sends `ack.state(status="accepted")`, and
   begins streaming fresh H.264/VT frames rendered from the new viewpoint.
3. `notify.camera` arrives with the authoritative pose, so the thin client and
   any agents stay synchronized.
4. The client clears the pending intent using the ack metadata; if the update
   were rejected, it would roll back using the cached state from the notify.

## Planned enhancements for agent workflows

We already have the building blocks, but richer automation will benefit from
the following protocol additions:

1. **Capability advertisement** – Extend `session.welcome` with a
   `control.capabilities` structure that declares which scopes, keys, or
   commands a client may exercise (plus rate limits). Agents and thin clients
   can then adjust behaviour without probing the server.
2. **Selective subscriptions** – Add subscribe/unsubscribe verbs or filtered
   resume tokens so agents can receive only the topics they need (e.g., a
   proofreading agent might subscribe to dims metadata and annotation updates,
   but skip stream or telemetry payloads).
3. **Richer ack semantics** – Expand `ack.state` to carry structured outcome
   fields (`status_detail`, `warnings`, `conflict_with`, queued/running/complete
   phases) so automated tools can branch on nuanced results. This mirrors MCP’s
   need for actionable tool feedback.
4. **Command lane maturity** – Finalize `call.command` / `reply.command` /
   `error.command` with request IDs, progress notifications, and result
   payloads. Agents can then trigger long-running analyses (e.g., segmentation
   cleanup) and track completion asynchronously.
5. **Conflict handling policy** – The protocol has the primitives (intent IDs,
   notify sequencing) needed for merge strategies, but we still owe an explicit
   policy: version vectors, CRDT-style merges, or at least consistent
   `notify.conflict` and rejection codes. This is the closest item to a
   “foundational” gap for multi-agent collaboration and should be designed
   deliberately.
6. **Domain metadata lanes** – Introduce `notify.annotation` or similar topics
   for structured annotation/proofreading data so agents aren’t forced to pack
   rich semantics into layer extras.
7. **Identity and audit** – Bind authenticated identities and per-role
   permissions to the handshake. A future agent framework will need to log who
   changed what, enforce approval workflows, and support supervised modes.
8. **Telemetry visibility** – Publish lightweight metrics (`notify.telemetry`)
   containing ack latency, render backlog, or stream drift so agents can adapt
   decisions based on system health.
9. **Data-plane hooks** – Provide a companion API or notify topic for cached
   tiles/meshes/results. Agents often need downsampled previews or analysis
   outputs that shouldn’t burden the main control channel.
10. **Batch/offline operations** – Allow clients to tag updates or commands as
    “batch” jobs and receive job tokens in the ack, enabling asynchronous
    proofreading pipelines.

## Why this foundation works for MCP agents

- **Deterministic control flow** – Every mutation is tied to an `intent_id` and
  reconciled via `ack.state`, making it simple for MCP agents to correlate
  tool executions with outcomes.
- **Typed context stream** – The notify lanes already map cleanly onto MCP
  context objects; the agent can mirror state without re-interpreting ad-hoc
  JSON.
- **Extensibility** – New scopes or commands slot into the existing envelope
  system; agents only need updated schema definitions. No bespoke integration
  per tool.
- **Graceful degradation** – Resume tokens and pairing of optimistic updates
  with authoritative notifies mean agents can reconnect, verify, and continue
  work mid-session without manual intervention.

## Open questions and next steps

1. Design and document the conflict-resolution policy (even if it is a
   first-fit reject approach) so multi-agent edits are predictable.
2. Specify the capability advertisement format and how the server enforces it.
3. Finalize the command lane contract, including streaming progress and result
   payload schemas.
4. Decide where domain-specific metadata lives (new notify topics vs. extended
   layer payloads) for workflows like connectomics proofreading.
5. Evaluate transport/service boundaries if parts of the control loop move to
   separate services or cloud agents, ensuring the existing envelopes remain
   portable.

Capturing these items now lets us extend the protocol methodically. The current
foundation already supports thin clients and lays the groundwork for
agent-driven experiences—our next work is layering richer capabilities,
conflict handling, and security so MCP-grade assistants can operate safely and
effectively.
