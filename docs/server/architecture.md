# napari-cuda Server Architecture

This document captures the desired architecture for the `napari_cuda.server`
package, the gaps in the current tree, and the plan for converging on a
contracted, well-partitioned layout.

## Current Layout (grown state)

| Domain | Location | Reality today |
| --- | --- | --- |
| Entry points | `server/app/` | Modules wire control, runtime, engine, and data directly in `egl_headless_server.py`. |
| Config | `server/config/` | Shared config/telemetry dataclasses; originally mixed with app and data helpers. |
| Control plane | `server/control/` | Imports runtime state types (`viewport.state`, IPC mailboxes) and engine helpers (`engine.pixel.channel`). |
| Runtime | `server/runtime/` | Still depends on control models and `server/data` helpers; circular imports previously blocked profiling. |
| Engine | `server/engine/` | GPU/NVENC code split across `capture/`, `encoding/`, and `pixel/`; consumers still import these submodules directly (façade TODO). |
| Scene | `server/scene/` | Centralised viewer DTOs/defaults consumed by control and runtime. |
| State ledger | `server/state_ledger/` | Thread-safe ledger shared across control/runtime/tests. |
| Shared data | `server/data/` | ROI/chunk math, hardware limits, LOD configs. (Former `runtime/data` helpers moved here.) |
| Tests | `server/tests/` | Flat bucket covering control, runtime, engine, and integration cases. |
| Docs | `docs/server/runtime_worker.md`, `docs/server/runtime_state_migration.md` | Reference prior module paths (`render_loop.applying.*`) and lack a top-level architecture contract. |

### Current Import Matrix (captured Oct 2025)

Direct intra-server imports discovered with `grimp` (`PYTHONPATH=src uv run python - <<'PY' ...`):

| From | `app` | `control` | `data` | `engine` | `external` | `runtime` | `scene` | `state_ledger` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `app` | 6 | 6 | 3 | 1 | 2 | 5 | 1 | 1 |
| `control` | 0 | 20 | 0 | 2 | 9 | 0 | 4 | 12 |
| `data` | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| `engine` | 3 | 0 | 2 | 15 | 2 | 0 | 0 | 0 |
| `external` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `runtime` | 2 | 0 | 38 | 4 | 0 | 154 | 19 | 2 |
| `scene` | 0 | 0 | 1 | 0 | 2 | 0 | 9 | 1 |
| `state_ledger` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(`scene` refers to `server/scene`; `state_ledger` covers the shared ledger package.)

This snapshot shows the control↔runtime edges collapsed via the new façade, while `engine → app` and `control → engine` remain to be tightened.

## Target Architecture (designed state)

- **server.app** – CLI and dashboard entry points only. Parses config/env, instantiates control, and exposes metrics. No direct runtime/engine logic.
- **server.config** – Shared configuration and debug-policy dataclasses used across app/runtime/engine.
- **server.control** – Websocket server, ledger, reducers, transaction scheduling. Consumes DTOs from `server.scene` and primitives from `server.data`. Talks to runtime through a narrow façade (`server.runtime.interface`), never via direct module imports.
- **server.runtime** – Worker bootstrap, viewport, render loop, LOD policy. Imports shared ROI math (`server.data`) and viewer DTOs from `server.scene`, exposes `runtime/api.py` for control interactions, and should call GPU code exclusively through an engine façade.
- **server.engine** – Renamed from `rendering/`. Hosts GL/EGL/CUDA/NVENC integration and compiled artifacts in `engine/encoding/compiled/`. We still need to formalise a thin API surface so runtime/control stop importing internals.
- **server.scene** – Authoritative DTOs, defaults, and builders for snapshots, layer schemas, and bootstrap metadata. Both control and runtime consume these types.
- **server.data** – Consolidated ROI/LOD math, hardware limits, logging helpers, env parsing. All pure helpers live here (merging the old `server/data/` and `server/runtime/data/`).
- **server.state_ledger** – Shared ledger primitives (`ServerStateLedger`, events, helpers) for control/runtime/tests.
- **server.tests** – Mirrors the package layout (`tests/control`, `tests/runtime`, `tests/engine`, `tests/integration`) with fixtures in `_helpers`.
- **docs/server/** – Contains this architecture doc plus updated worker/state references, showing module paths and dependency contracts.

## Dependency Contract

Allowed import edges (prefixes under `napari_cuda.server`):

```
app → config, control, runtime.interface, scene, data
control → config, scene, data, runtime.interface
control → state_ledger
runtime → config, scene, data, state_ledger, engine.api
engine → config
scene → data (helpers only when unavoidable)
data → (no intra-server dependencies)
tests/* → any domain as needed
```

Anything outside these edges should be considered a violation. A lightweight
AST-based check (see `tools/scripts/import_matrix.py`, to be added) will enforce
the contract in CI.

## Current Gaps vs Target

- **Engine façade discipline** – Control/runtime still import from `engine.encoding` and `engine.pixel` (two direct edges). Introduce an explicit `server/engine/api.py` (or similar façade) so engine internals stay private and control-side shims can be removed.
- **Runtime façade follow-through** – `runtime/api.py` removes direct control↔runtime imports; keep future control features on this surface and migrate lingering helpers (e.g., keyframe scheduling) where sensible.
- **Viewer DTO alignment** – Plane/volume/view snapshots now live in `server/scene`; keep new code on the shared models and trim remaining runtime-only shims.
- **Docs & tooling** – Update the worker/state docs plus CI checks to reference the new ledger package and façade.

## Roadmap

1. **Document & inventory**
   - Check in this architecture doc.
   - Generate import graphs (e.g., via `grimp` or the existing AST script) and
     embed the current matrix here for historical reference.
   - Update `docs/server/runtime_worker.md` and `runtime_state_migration.md`
     to reference the target packages.
2. **Unify data layer**
   - Move `server/runtime/data/*` into `server/data/`.
   - Update all imports (`rg "runtime\\.data"` → `server.data`).
   - Delete the orphaned `server/state/` package.
3. **Consolidate scene DTOs**
   - Move shared dataclasses/defaults into `server/scene/`.
   - Refactor `control/state_models.py`, `runtime/render_loop/apply/render_state`,
     and viewstate builders to use the shared models.
4. **Isolate engine**
   - Add `server/engine/api.py` (or equivalent) as the only public surface and migrate callers to it.
   - Keep compiled artifacts under `engine/encoding/compiled/` and document rebuild steps.
5. **Harden control/runtime façade**
   - Keep control-plane calls on `server/runtime/api.py`; move any remaining worker access (e.g., keyframe helpers) behind the façade.
   - Provide a focused adapter in control (proxy or service object) to centralise runtime interactions.
6. **Restructure tests & docs**
   - Mirror the package layout under `server/tests/`.
   - Add a top-level README section pointing to the architecture doc.
   - Record the import-contract check in developer tooling.
7. **Polish & verify**
   - Drop lingering `__pycache__` directories and ensure `.gitignore` coverage.
   - Run targeted pytest selections (`pytest server/tests/runtime`) and repeat
     the import matrix check to confirm compliance.

Each phase should land as its own pull request with updated tests and
documentation. By the end of Phase 7, the server package will have a clear,
enforced architecture that matches how we reason about the system.

### Implementation Checklist

Use this as you execute the roadmap:

- [ ] Commit architecture doc and import-matrix tooling to CI.
- [x] Merge data helpers into `server/data` and delete `server/state`.
- [x] Publish shared scene DTOs and update control/runtime call sites.
- [x] Move the ledger into `server/state_ledger` and repoint imports.
- [ ] Introduce `server/engine/api.py` and migrate runtime/control callers to the façade.
- [x] Add `runtime/api.py` and remove direct control ↔ runtime imports.
- [ ] Re-home tests under `server/tests/<domain>/` and refresh docs.
- [ ] Verify import contract compliance and rerun key pytest selections.
