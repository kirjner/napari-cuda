Client Integration (Cut-Down)

Emitters (client/control/emitters)
- `napari_dims_intent_emitter`: listens to local napari dims events (index, margins) and emits `state.update` frames; compares against client state to avoid loops.
- `napari_camera_intent_emitter`: emits pan/orbit/zoom deltas only; no ledger writes on the server until pose applied.

Runtime (client/runtime/stream_runtime.py)
- Ingests notify messages (dims/camera/layers/scene.level), updates client-side ledger, and drives presenter/HUD.

Lean Rules
- Treat `notify.dims` as the single source of truth: today it embeds
  `dims_spec`, but the migration path replaces that payload with explicit
  `view / axes / index / lod / camera` blocks. Clients must be ready to consume
  the new structure as soon as it is advertised.
- Use `notify.level` to display active level in volume; use dims.current_level in plane.
