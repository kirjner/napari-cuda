# napari-cuda Server Architecture

This document captures the desired architecture for the `napari_cuda.server`
package, the gaps in the current tree, and the plan for converging on a
contracted, well-partitioned layout.

## Current Layout (grown state)

| Domain | Location | Reality today |
| --- | --- | --- |
| Entry points | `server/app/` | Modules wire control, runtime, engine, and data directly in `egl_headless_server.py`. |
| Control plane | `server/control/` | Imports runtime state types (`viewport.state`, IPC mailboxes) and engine helpers (`pixel_channel`). |
| Runtime | `server/runtime/` | Still depends on control models and `server/data` helpers; circular imports previously blocked profiling. |
| Engine | `server/rendering/` | GPU/NVENC code lives alongside control shims (`pixel_channel`) and depends on app config objects. Compiled `.so` files share the top-level directory. |
| Scene DTOs | `server/scene/`, `server/scene_defaults.py`, `control/state_models.py` | Defaults and data transfer objects are scattered; runtime builds its own ledger snapshots. |
| Shared data | `server/data/` | ROI/chunk math, hardware limits, LOD configs. (Former `runtime/data` helpers moved here.) |
| Tests | `server/tests/` | Flat bucket covering control, runtime, engine, and integration cases. |
| Docs | `docs/server/runtime_worker.md`, `docs/server/runtime_state_migration.md` | Reference old module paths (`render_loop.render_updates.*`) and lack a top-level architecture contract. |

### Current Import Matrix (captured Oct 2025)

Direct intra-server imports discovered with `grimp` (`PYTHONPATH=src uv run python - <<'PY' ...`):

| From | To | Direct imports |
| --- | --- | --- |
| `app` | `app` | 6 |
| `app` | `control` | 9 |
| `app` | `data` | 3 |
| `app` | `rendering` | 2 |
| `app` | `runtime` | 7 |
| `app` | `scene` | 1 |
| `control` | `app` | 1 |
| `control` | `control` | 33 |
| `control` | `rendering` | 1 |
| `control` | `runtime` | 3 |
| `control` | `scene` | 2 |
| `data` | `data` | 13 |
| `rendering` | `app` | 2 |
| `rendering` | `control` | 1 |
| `rendering` | `data` | 2 |
| `rendering` | `rendering` | 6 |
| `runtime` | `app` | 2 |
| `runtime` | `control` | 5 |
| `runtime` | `data` | 39 |
| `runtime` | `rendering` | 8 |
| `runtime` | `runtime` | 166 |
| `runtime` | `scene` | 4 |
| `runtime` | `scene_defaults` | 1 |
| `scene` | `control` | 1 |
| `scene` | `runtime` | 1 |
| `scene` | `scene` | 1 |
| `scene` | `scene_defaults` | 1 |

This snapshot serves as the baseline to verify that later refactors remove the unwanted edges (e.g., `rendering → control`).

## Target Architecture (designed state)

- **server.app** – CLI and dashboard entry points only. Parses config/env, instantiates control, and exposes metrics. No direct runtime/engine logic.
- **server.control** – Websocket server, ledger, reducers, transaction scheduling. Consumes DTOs from `server.scene` and primitives from `server.data`. Talks to runtime through a narrow façade (`server.runtime.interface`), never via direct module imports.
- **server.runtime** – Worker bootstrap, viewport, render loop, LOD policy. Imports shared ROI math (`server.data`) and scene DTOs, and calls GPU code exclusively through `server.engine.api`.
- **server.engine** – Renamed `rendering/`. Hosts GL/EGL/CUDA/NVENC integration and compiled artifacts in `engine/_compiled/`. Exposes a thin API (capture, encode, reset) with no knowledge of control/runtime internals.
- **server.scene** – Authoritative DTOs and defaults for snapshots, layer schemas, and bootstrap metadata. Both control and runtime consume these types.
- **server.data** – Consolidated ROI/LOD math, hardware limits, logging helpers, env parsing. All pure helpers live here (merging the old `server/data/` and `server/runtime/data/`).
- **server.tests** – Mirrors the package layout (`tests/control`, `tests/runtime`, `tests/engine`, `tests/integration`) with fixtures in `_helpers`.
- **docs/server/** – Contains this architecture doc plus updated worker/state references, showing module paths and dependency contracts.

## Dependency Contract

Allowed import edges (prefixes under `napari_cuda.server`):

```
app → control, runtime.interface, scene, data
control → scene, data, runtime.interface
runtime → scene, data, engine.api
engine → data
scene → data (helpers only when unavoidable)
data → (no intra-server dependencies)
tests/* → any domain as needed
```

Anything outside these edges should be considered a violation. A lightweight
AST-based check (see `tools/scripts/import_matrix.py`, to be added) will enforce
the contract in CI.

## Current Gaps vs Target

- **Duplicate data layers** – (Resolved) `runtime/data` helpers are now part of
  `server/data`; future changes should rely on that package exclusively.
- **Scene DTO scatter** – Defaults live in `scene_defaults.py`; runtime snapshot
  builders define their own immutable structures. All DTOs should move under
  `server/scene/`.
- **Engine bleed-through** – `server/rendering/encoder.py` imports app config
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
3. **Consolidate scene DTOs**
   - Create `server/scene/models.py` with shared dataclasses/defaults.
   - Refactor `control/state_models.py`, `runtime/render_loop/apply/snapshots/build.py`,
     and `scene_defaults.py` to use the shared models.
4. **Isolate engine**
   - Introduce `server/engine/api.py` and update runtime to call it.
   - Move compiled artifacts under `engine/_compiled/`.
   - Rename `server/rendering/` to `server/engine/` once imports are adjusted.
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
- [ ] Publish shared scene DTOs and update control/runtime call sites.
- [ ] Introduce `server/engine/api.py`, migrate runtime, and rename the package.
- [ ] Add runtime/control façade modules and remove cross-plane imports.
- [ ] Re-home tests under `server/tests/<domain>/` and refresh docs.
- [ ] Verify import contract compliance and rerun key pytest selections.
