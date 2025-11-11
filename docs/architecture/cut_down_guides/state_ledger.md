State Ledger (Cut-Down)

Core
- `ServerStateLedger` (thread-safe): `record_confirmed`, `batch_record_confirmed`, `snapshot`, `get`, `clear_scope`.
- Entries are versioned per (scope, target, key); txns bump versions.

Logging (Lean)
- INFO: `ledger write: scope=... key=... origin=... value=<summary>`
  - Thumbnails summarized (shape/dtype/timestamp),
  - Viewport intents no longer logged (removed),
  - Dims spec summarized only (ndim, ndisplay, level, step, order, displayed, labels).
- DEBUG: dedupe, internal record traces.

Lean Rules
- Summary-only logging for large values; never emit large arrays or full `dims_spec` blobs.
- Use dedupe (`dedupe=True`) for idempotent writes to avoid version churn.
- No ledger writes for hot-path camera deltas; only write on applied pose.
- When introducing the `view / axes / index / lod / camera` scopes, wire them
  through the same logging/dedupe helpers on day one so legacy `dims_spec`
  logging can be removed entirely once consumers migrate.
