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
| Engine | `server/engine/` | GPU/NVENC code, capture pipeline, and pixel broadcaster; exported via `engine/api.py`. |
| View state | `server/viewstate/` | Centralised viewer DTOs/defaults consumed by control and runtime. |
| Shared data | `server/data/` | ROI/chunk math, hardware limits, LOD configs. (Former `runtime/data` helpers moved here.) |
| Tests | `server/tests/` | Flat bucket covering control, runtime, engine, and integration cases. |
| Docs | `docs/server/runtime_worker.md`, `docs/server/runtime_state_migration.md` | Reference prior module paths (`render_loop.apply.render_state.*`) and lack a top-level architecture contract. |

### Current Import Matrix (captured Oct 2025)

Direct intra-server imports discovered with `grimp` (`PYTHONPATH=src uv run python - <<'PY' ...`):

| From | To | Direct imports |
| --- | --- | --- |
| `app` | `app` | 6 |
| `app` | `control` | 9 |
| `app` | `data` | 3 |
| `app` | `rendering` | 2 |
| `app` | `runtime` | 6 |
| `app` | `viewstate` | 1 |
| `control` | `app` | 1 |
| `control` | `control` | 33 |
| `control` | `rendering` | 1 |
| `control` | `runtime` | 3 |
| `control` | `viewstate` | 3 |
| `data` | `data` | 13 |
| `rendering` | `app` | 2 |
| `rendering` | `control` | 1 |
| `rendering` | `data` | 2 |
| `rendering` | `rendering` | 6 |
| `runtime` | `app` | 2 |
| `runtime` | `control` | 4 |
| `runtime` | `data` | 39 |
| `runtime` | `rendering` | 8 |
| `runtime` | `runtime` | 154 |
| `runtime` | `viewstate` | 16 |
| `viewstate` | `control` | 1 |
| `viewstate` | `viewstate` | 5 |

This snapshot serves as the baseline to verify that later refactors remove the unwanted edges (e.g., `rendering → control`).

## Target Architecture (designed state)

- **server.app** – CLI and dashboard entry points only. Parses config/env, instantiates control, and exposes metrics. No direct runtime/engine logic.
- **server.control** – Websocket server, ledger, reducers, transaction scheduling. Consumes DTOs from `server.viewstate` and primitives from `server.data`. Talks to runtime through a narrow façade (`server.runtime.interface`), never via direct module imports.
- **server.runtime** – Worker bootstrap, viewport, render loop, LOD policy. Imports shared ROI math (`server.data`) and viewer DTOs from `server.viewstate`, and calls GPU code exclusively through `server.engine.api`.
- **server.engine** – Renamed `rendering/`. Hosts GL/EGL/CUDA/NVENC integration and compiled artifacts in `engine/_compiled/`. Exposes a thin API (capture, encode, reset) with no knowledge of control/runtime internals.
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
- **Engine façade discipline** – Keep runtime/control imports limited to `server/engine/api.py` to avoid leaking internal modules.
  and re-exports control pixel channels. The engine package must stand alone
  behind an API module.
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
   - Maintain `server/engine/api.py` as the only public surface.
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
- [ ] Introduce `server/engine/api.py`, migrate runtime, and rename the package.
- [ ] Add runtime/control façade modules and remove cross-plane imports.
- [ ] Re-home tests under `server/tests/<domain>/` and refresh docs.
- [ ] Verify import contract compliance and rerun key pytest selections.
