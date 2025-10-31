# napari-cuda Server Architecture

This document captures the desired architecture for the `napari_cuda.server`
package, the gaps in the current tree, and the plan for converging on a
contracted, well-partitioned layout.

## Current Layout (grown state)

| Domain | Location | Reality today |
| --- | --- | --- |
| Entry points | `server/app/` | Modules wire control, runtime, engine, and data directly in `egl_headless_server.py`. |
| Control plane | `server/control/` | Imports runtime state types (`viewport.state`, IPC mailboxes) and engine helpers (`engine.pixel.channel`). |
| Runtime | `server/runtime/` | Still depends on control models and `server/data` helpers; circular imports previously blocked profiling. |
| Engine | `server/engine/` | GPU/NVENC code split across `capture/`, `encoding/`, and `pixel/`; consumers still import these submodules directly (façade TODO). |
| View state | `server/viewstate/` | Centralised viewer DTOs/defaults consumed by control and runtime. |
| Shared data | `server/data/` | ROI/chunk math, hardware limits, LOD configs. (Former `runtime/data` helpers moved here.) |
| Tests | `server/tests/` | Flat bucket covering control, runtime, engine, and integration cases. |
| Docs | `docs/server/runtime_worker.md`, `docs/server/runtime_state_migration.md` | Reference prior module paths (`render_loop.apply.render_state.*`) and lack a top-level architecture contract. |

### Current Import Matrix (captured Oct 2025)

Direct intra-server imports discovered with `grimp` (`PYTHONPATH=src uv run python - <<'PY' ...`):

| From | To | Direct imports |
| --- | --- | --- |
| `app` | `app` | 6 |
| `app` | `control` | 8 |
| `app` | `data` | 3 |
| `app` | `engine` | 1 |
| `app` | `external` | 2 |
| `app` | `runtime` | 6 |
| `app` | `scene` | 1 |
| `control` | `control` | 32 |
| `control` | `engine` | 1 |
| `control` | `external` | 9 |
| `control` | `runtime` | 3 |
| `control` | `scene` | 3 |
| `data` | `data` | 13 |
| `engine` | `app` | 3 |
| `engine` | `data` | 2 |
| `engine` | `engine` | 15 |
| `engine` | `external` | 2 |
| `runtime` | `app` | 2 |
| `runtime` | `control` | 3 |
| `runtime` | `data` | 39 |
| `runtime` | `engine` | 4 |
| `runtime` | `runtime` | 154 |
| `runtime` | `scene` | 16 |
| `scene` | `control` | 1 |
| `scene` | `external` | 2 |
| `scene` | `scene` | 5 |

(`scene` in the table represents modules under `server/viewstate`.)

This snapshot serves as the baseline to verify that later refactors remove the unwanted edges (e.g., `engine → app`, `control → engine`).

## Target Architecture (designed state)

- **server.app** – CLI and dashboard entry points only. Parses config/env, instantiates control, and exposes metrics. No direct runtime/engine logic.
- **server.control** – Websocket server, ledger, reducers, transaction scheduling. Consumes DTOs from `server.viewstate` and primitives from `server.data`. Talks to runtime through a narrow façade (`server.runtime.interface`), never via direct module imports.
- **server.runtime** – Worker bootstrap, viewport, render loop, LOD policy. Imports shared ROI math (`server.data`) and viewer DTOs from `server.viewstate`, and calls GPU code exclusively through `server.engine.api`.
- **server.engine** – Renamed from `rendering/`. Hosts GL/EGL/CUDA/NVENC integration and compiled artifacts in `engine/encoding/compiled/`. We still need to formalise a thin API surface so runtime/control stop importing internals.
- **server.viewstate** – Authoritative DTOs, defaults, and builders for snapshots, layer schemas, and bootstrap metadata. Both control and runtime consume these types.
- **server.data** – Consolidated ROI/LOD math, hardware limits, logging helpers, env parsing. All pure helpers live here (merging the old `server/data/` and `server/runtime/data/`).
- **server.tests** – Mirrors the package layout (`tests/control`, `tests/runtime`, `tests/engine`, `tests/integration`) with fixtures in `_helpers`.
- **docs/server/** – Contains this architecture doc plus updated worker/state references, showing module paths and dependency contracts.

## Dependency Contract

Allowed import edges (prefixes under `napari_cuda.server`):

```
app → control, runtime.interface, viewstate, data
control → viewstate, data, runtime.interface
runtime → viewstate, data, engine.api
engine → data
viewstate → data (helpers only when unavoidable)
data → (no intra-server dependencies)
tests/* → any domain as needed
```

Anything outside these edges should be considered a violation. A lightweight
AST-based check (see `tools/scripts/import_matrix.py`, to be added) will enforce
the contract in CI.

## Current Gaps vs Target

- **Duplicate data layers** – (Resolved) `runtime/data` helpers are now part of
  `server/data`; future changes should rely on that package exclusively.
- **Viewer DTO alignment** – Shared dataclasses/defaults live in `server/viewstate`; continue steering new code through that façade.
- **Engine façade discipline** – Runtime/control still reach directly into `engine.capture`, `engine.encoding`, and `engine.pixel`. Introduce an explicit `server/engine/api.py` (or similar façade) so engine internals stay private and control-side shims can be removed.
- **Control/runtime coupling** – Control reducers import runtime state classes,
  and runtime bootstrap imports control models. We need façades for cross-plane
  communication.
- **Flat tests & dated docs** – Test suite and docs do not mirror the desired
  package boundaries.

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
3. **Consolidate viewstate DTOs**
   - Move shared dataclasses/defaults into `server/viewstate/`.
   - Refactor `control/state_models.py`, `runtime/render_loop/apply/render_state`,
     and viewstate builders to use the shared models.
4. **Isolate engine**
   - Add `server/engine/api.py` (or equivalent) as the only public surface and migrate callers to it.
   - Keep compiled artifacts under `engine/encoding/compiled/` and document rebuild steps.
5. **Establish control/runtime façades**
   - Define `server/runtime/interface.py` (runtime entry points) and
     `server/control/runtime_proxy.py` (control-side adapter).
   - Remove direct `control → runtime.*` imports in reducers and channel code.
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
- [x] Publish shared viewstate DTOs and update control/runtime call sites.
- [ ] Introduce `server/engine/api.py` and migrate runtime/control callers to the façade.
- [ ] Add runtime/control façade modules and remove cross-plane imports.
- [ ] Re-home tests under `server/tests/<domain>/` and refresh docs.
- [ ] Verify import contract compliance and rerun key pytest selections.
