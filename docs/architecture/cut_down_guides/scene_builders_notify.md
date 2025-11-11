Scene Builders & Notify (Cut-Down)

Scene Snapshot (Pixel Channel)
- `snapshot_render_state(ledger)` builds:
  - dims spec from `('dims','main','dims_spec')` (required today; the target
    is to emit the factored `view / axes / index / lod / camera` scopes and keep
    `dims_spec` only as a compatibility payload).
  - camera poses from `camera_plane/*` and `camera_volume/*` (optional per mode).
  - layer visuals from `('layer', id, *)`.
  - active view from `('viewport','active','state')` when present.

Notify Payloads (State Channel)
- `notify.dims`: must embed full `dims_spec` today. As the factored scopes land,
  dual-publish them so clients can switch to the `{view,axes,index,lod,camera}`
  structure before dropping the legacy blob. Top-level duplicated fields
  (order/displayed/etc.) must match.
- `notify.camera`: only pose deltas (plane + volume as applicable).
- `notify.level`: ActiveView only (mode+level); never infer from dims on the client.

Lean Rules
- One serializer per block shared by snapshot and notify; no ad-hoc payload assembly.
- No recomputation of derived values (use ActiveView for mode+level; do not derive from dims).
- Add serializers for the new `{view,axes,index,lod,camera}` blocks at the same
  time so snapshot/notify stay in lockstep during the migration.
