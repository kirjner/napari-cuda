Flags & Branches Audit (Trim Plan)

Server App
- `EGLHeadlessServer.__init__`:
  - `use_volume` (init mode) → keep.
  - `_animate`, `_animate_dps` (turntable) → keep behind debug/toggle; default off.
  - `_ctx.debug_policy.logging.log_sends_env` → keep as env toggle; default off.

Runtime Pipeline
- `level_policy_suppressed` toggles during mode changes → keep (prevents spurious policy eval); document transitions.

Notify/Resumable
- Resumable retention limits (counts/ages) → keep; move values to config; no dynamic toggles in hot path.

Control Handlers
- Dims: only support `index`, `step`, `margin_left/right`. Reject others early.
- Camera: accept only pan/orbit/zoom/reset/set; deltas enqueue only; set is rare.

Trim Targets (proposed)
- Remove synthetic intents and pose-overwrite helpers; rely on ActiveView + pose caches.
- Avoid duplicated fields in notify unless needed for backward compat; enforce equality with embedded spec when present.
- Gate the new `view / axes / index / lod / camera` ledger scopes behind a
  single feature flag while dual-writing so old clients keep working; remove the
  flag (and legacy `current_step` naming) once notify/snapshot consumers migrate.
