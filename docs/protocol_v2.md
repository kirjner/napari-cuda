Protocol v2 – Control Channel Tightening

Scope
- Single source of truth: server owns scene; client mirrors with intents.
- Deterministic ordering and resumability per topic.
- Typed frames and commands with strict schemas.
- Idempotent replies; explicit error taxonomy.

Envelope
- Fields: `type`, `version`, `session?`, `frame_id`, `timestamp`, `seq?`, `delta_token?`, `intent_id?`.
- Per-topic monotonic `seq`; resumable via `delta_token`.

Handshake
- Client `session.hello`: `{protocols:[int], client:{name,version,platform}, features:{name:bool}, resume_tokens:{topic?:token}, auth?}`.
- Server `session.welcome`: `{protocol_version:int, session:{id,heartbeat_s,ack_timeout_ms?}, features:{topic:{enabled,version?,resume?,commands?,resume_state?}}}`.
- Rejection: `{code,message,details?}`.

Notify: Layers (structured)
- `notify.layers` payload:
  - `layer_id: str`
  - Sections (all optional):
    - `controls: {visible,opacity,blending,interpolation,colormap,rendering,gamma,contrast_limits,iso_threshold,attenuation}`
    - `metadata: {...}`
    - `data: {...}` (structural/data-affecting)
    - `thumbnail: {array|bytes,dtype,shape,colorspace?,encoding?,version?}`
    - `removed: true` (mutually exclusive with other sections)

Notify: Dims/Scene/Camera
- Per-topic `seq`; no change in payload semantics. Stream advertises readiness explicitly.

Commands
- Request: `{command,args?,kwargs?,origin?,intent_id?,idempotency_key?,timeout_ms?}`
- Success: `reply.command {in_reply_to,status:'ok',result?,idempotency_key?}`
- Error: `error.command {in_reply_to,status:'error',code,message,details?,idempotency_key?}`
- Cancellation: `cancel.command {in_reply_to}` (optional future).

Filesystem RPC
- Request: `{path?,show_hidden?,only?,cursor?,page_size?,sort?}`
- Response: `{path,entries:[{name,path,type:'dir'|'file'|'zarr',size_bytes?,mtime_s?}],has_more:boolean,next_cursor?}`
- Server enforces chroot under data root; rejects traversal.

Backpressure/Rate Limits
- Server advertises per-topic caps; client coalesces redundant updates.

Observability
- Include `intent_id` through replies and acks. Server returns `duration_ms` (future).

Security
- Cap thumbnail size; allow compressed encodings.

Migration Notes
- No backward compatibility expected in this branch. All client/server code updated to v2.

