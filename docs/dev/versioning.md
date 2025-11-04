# Server Versioning and Signature Notes

This note documents how render snapshots carry version counters and how we dedupe outward traffic based on content signatures. The focus is on the shared helpers in `server/utils/signatures.py` and how the render pipeline consumes them.

## Ledger Versions
- `ServerStateLedger` assigns monotonically increasing integers to each `(scope, target, key)` entry when a confirmed write lands without an explicit `version=` override.
- `RenderLedgerSnapshot` exposes the counters that matter to the render pipeline: `dims_version`, `view_version`, `multiscale_level_version`, and `camera_versions` (split into `plane.*`, `volume.*`, or legacy camera keys).
- Call `snapshot_versions(snapshot)` to collect those counters. The function returns a `VersionGate` whose `apply(mapping)` method updates whatever cache (e.g., `applied_versions`) you are maintaining.

## Content Signatures
`server/utils/signatures.py` also provides canonical signatures that reflect *what the user sees*. They ignore version counters and instead hash pixel-affecting values.

- `scene_content_signature(snapshot)` drives `RenderUpdateMailbox` coalescing so the worker only replays meaningful scene changes.
- `layer_content_signature(layer_state)` lets `ServerLayerMirror` dedupe `notify.layers` payloads.
- `layer_inputs_signature(snapshot, layer_id)` produces the inputs-only token used by thumbnail capture on the worker.
- `dims_content_signature(...)` (and the convenience wrapper `dims_content_signature_from_payload`) power the dims mirror so `notify.dims` is only emitted when the payload would change.

## Runtime Touchpoints
- `runtime/render_loop/planning/staging.py` invokes `snapshot_versions(snapshot).apply(applied_versions)` before comparing layer deltas. Per-layer properties still rely on the `LayerVisualState.versions` map.
- `runtime/render_loop/applying/apply.py` prefixes its dedupe tokens with `(snapshot.dims_version, snapshot.view_version, snapshot.multiscale_level_version)` before stashing the final signature. That keeps the behaviour equivalent to the previous bundle helper.
- `ServerLayerMirror` and `ServerDimsMirror` now compute signatures before allocating protocol payloads, avoiding unnecessary work on no-op updates.

## Guidance for Future Changes
- Use the helpers above instead of rebuilding tuples by hand. If a new subsystem needs dedupe, add a function beside the existing ones rather than open-coding the logic.
- When adding ledger-backed properties, ensure the snapshot exposes the version counter and that `snapshot_versions` is extended if the property needs to participate in worker-side gating.
- Keep outward-facing notifications value-based: version counters are for ordering; signatures are for “pixels changed” decisions.
