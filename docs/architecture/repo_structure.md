Repository Structure and Naming (Current → Proposed)

Goals
- Make core paths discoverable: handlers → reducers/txns → ledger → notify (state) and ledger → snapshot → apply/plan (pixel).
- Reduce indirection and deep nesting; one concept per module where possible.
- Consistent naming across control/runtime/scene/notify.

Current (condensed)
```
src/napari_cuda/server/
  app/                (egl_headless_server, publishers, metrics)
  control/            (state_update_handlers, reducers, mirrors, topics, transactions)
  runtime/            (bootstrap, ipc, render_loop/{applying,planning}, worker)
  scene/              (builders, viewport dataclasses)
  ledger.py
  data/, engine/, utils/, tests/
```

Proposed
```
src/napari_cuda/server/
  app/
    egl_headless_server.py
    scene_publisher.py
    dataset_lifecycle.py
    metrics_core.py
    context/ (env, encode config, hw limits)
  control/
    handlers/   (camera.py, dims.py, view.py, layer.py)
    reducers.py (reduce_dims_update, reduce_view_update, reduce_level_update, reduce_camera_update, ...)
    txns/       (dims.py, view.py, level.py, camera.py, layer.py)
    notify/     (dims.py, camera.py, layers.py, scene.py, stream.py)
  runtime/
    intent/     (WorkerIntentMailbox — only remaining mailbox; carries level/thumbnail intents)
    worker/     (EGL renderer)
    apply/      (dims.py, camera.py, layers.py)  # direct ledger → apply blocks
    bootstrap/  (setup_viewer.py, snapshot_helpers.py)
  scene/
    blocks.py   (DimsSpec + compat only); future view/axes/index/lod/camera records live here.
    builders.py (snapshot_render_state, build_notify_* payloads)
  ledger.py     (ServerStateLedger)
  utils/
  tests/
```

Naming Conventions
- Modules: snake_case nouns for state (builders.py, ledger.py), verbs for operations (reducers.py, apply/*.py).
- Public API: explicit __all__ exports in package __init__.py; avoid wildcard imports.
- One concept per file; split files > ~400 LOC by sub-concept (e.g., apply/dims.py vs apply/camera.py).
- Tests: colocated `_tests/` for unit tests; integration tests under server/tests/.

Import Guidance
- Control: `from napari_cuda.server.control.reducers import reduce_dims_update`
- Txns: internal to reducers (`napari_cuda.server.control.txns.*`), not part of public API.
- Runtime: `from napari_cuda.server.runtime.plan import drain_scene_updates`; `from napari_cuda.server.runtime.apply import apply_dims_block`
- Scene: `from napari_cuda.server.scene.blocks import DimsSpec, PlanePose, VolumePose`
- Notify: `from napari_cuda.server.control.notify import broadcast_*`
- Ledger: `from napari_cuda.server.ledger import ServerStateLedger`

Lean Refactor Plan (Safe Moves)
- (Done) `state_ledger/__init__.py` → `ledger.py`; keep this note until all callers settle.
- Flatten `control/topics/` → `control/notify/` with modules named by message domains.
- Move `runtime/render_loop/applying/*` → `runtime/apply/*`; `render_loop/planning/*` → `runtime/plan/*`.
- Consolidate `state_update_handlers/*` → `control/handlers/*` (`camera.py`, `dims.py`, `view.py`, `layer.py`).
- Combine scene dataclasses into `scene/blocks.py`; keep builders separate in `scene/builders.py`.
- Introduce explicit modules for the new view/axes/index/lod/camera scopes when they
  land so legacy `DimsSpec` types can be retired cleanly once consumers migrate.
- Delete the legacy `runtime/render_loop/planning/*`, `render_loop/applying/*`,
  `runtime/ipc/mailboxes/render_update.py`, and `scene/viewport.py` packages once
  the worker consumes the factored ledger scopes directly (worker pulls ledger
  → applies blocks → emits intents only via the worker intent mailbox).

Style Enforcement
- Add README.md per major package (control, runtime, scene) outlining roles and key entrypoints.
- Lint imports with ruff (ban relative imports); verify public API via `__all__`.
- Keep flags out of hot paths; configuration resolved at app startup.
