# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/napari` (core app), `src/napari_cuda` (server/client/protocol), `src/napari_engine` (MVP engine, profiling), `src/napari_builtins` (built-in plugins).
- Tests: colocated under `src/**/_tests` and `src/napari_engine/tests/`.
- Examples & Docs: `examples/`, `docs/`; scripts and dev tooling in `scripts/`, `dev/`, `tools/`.

## Build, Test, and Development Commands
- Install (uv-managed): `uv sync` (base), `uv sync --extra server`, `uv sync --extra client`, or `uv sync --extra cuda-dev`.
- Run app: `uv run napari` or `uv run python -m napari`.
- CUDA streaming:
  - Server (GPU host): `uv run napari-cuda-server data.npy`
  - Client (local): `uv run napari-cuda-client`
- Tests: `uv run pytest -q` (skip slow: `-m "not slow"`).
- Type check: `make typecheck` (runs mypy) or `tox -e mypy`.
- Lint/format: `make pre` or `pre-commit run -a` (ruff & format).
- Build dist: `make dist` or `python -m build`.

## Coding Style & Invariants
- Python ≥ 3.10, 4-space indentation, UTF-8; ruff-enforced formatting.
- Absolute imports only. Modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- No defensive programming in core paths:
  - No `try/except` unless crossing subsystem boundaries (I/O, external libs). Inside our code we assert invariants instead.
  - No `getattr`/`hasattr`, `tuple(...)`, `float(...)`, or “just in case” `None` checks when the caller already guarantees shape. Use direct assignment. If a value is wrong, let it explode.
  - No `# type: ignore` band-aids in core modules; design the types (TypedDicts, dataclasses) so the compiler knows what we’re doing. If mypy complains and the schema is correct, prefer removing the annotation over sprinkling ignores.
- Logging: use `logger.exception(...)` at subsystem boundaries only; avoid chatty logging in hot paths.
- Restore/cache schema hygiene (applies to all new ledger-backed scopes):
  - Serialize to plain JSON-friendly payloads, deserialize immediately via TypedDict-aware helpers that return dataclasses; no inline `# type: ignore`, no `typing.cast`, and no "maybe" checks after the helper returns.
  - Encode invariants once (shape, dimensionality, dtype) inside the helper and raise/assert there; all other call sites treat the dataclass values as truth.
  - When writing cache data, keep the schema symmetric: tuples → lists on the way out, lists → tuples on the way in, performed centrally so call sites just assign fields.
  - Feature-flagged reducers/handlers must branch at the highest level (e.g., `if ENABLE_VIEW_AXES_INDEX_BLOCKS:`) and use the typed helpers for the new path; the legacy path stays untouched for compatibility.
  - Treat cache blocks as immutable snapshots: rebuild the entire dataclass from authoritative state changes and write it via `write_*_restore_cache`; never mutate cache payloads piecemeal.

## Testing Guidelines
- Framework: `pytest` with config in `pyproject.toml` (strict markers, warnings-as-errors for napari).
- Location: place tests next to code in `_tests/` packages; name files `test_*.py`.
- Marks: prefer `-m "not slow"` locally; add focused unit tests and GPU-mocked paths where CUDA is optional.
- Coverage: run via tox factors with `cov` or `coverage run -m pytest` for reports.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., "Add CUDA frame encoder").
- PRs: include purpose, summary of changes, test plan (`pytest` cmd), linked issues (e.g., `Fixes #123`), and screenshots for UI-affecting changes.
- Gates: CI green, `pre-commit` clean, tests pass locally; update docs/examples when behavior changes.

## Security & Configuration Tips
- Copy env: `cp .env.example .env`. On headless GPU hosts: set `QT_QPA_PLATFORM=offscreen` and `PYOPENGL_PLATFORM=egl`; configure `CUDA_VISIBLE_DEVICES` as needed.
- Prefer `uv` for all commands to ensure consistent environments.

## Current Focus (2025-11-10)
- Migrating the legacy `dims_spec + planner/mailbox` pipeline to the factored
  `view / axes / index / lod / camera / layers` ledger design.
- Reference plan: `docs/architecture/view_axes_index_plan.md` (call-stack
  diagrams + phased issue breakdown). Use it as the source of truth for scope
  and sequencing.
- Phase 0 progress:
  - Ledger module moved to `src/napari_cuda/server/ledger.py` (single module;
    update imports here going forward).
- Upcoming phases:
  1. **Scene blocks scaffolding** — add `scene/blocks/` dataclasses +
     serializers for the new scopes; dual-write from reducers behind a flag.
  2. **Consumer flip** — teach notify builders, snapshots, and worker apply
     code to consume the new blocks with per-block signatures.
  3. **Legacy removal** — delete `ViewportPlanner`, `RenderUpdateMailbox`,
     `PlaneViewportCache`/`VolumeViewportCache`, and redundant notify/multiscale fields once
     the new scopes are authoritative everywhere.

## IMMEDIATE NEXT GOAL

Implement ledger‑driven restore and block consumption under the feature flag:

- Restore cache helpers (`load_*`, `write_*`, `_plane/_volume_cache_from_state`) now dual-write from bootstrap + every reducer when the flag is on; legacy writers still run for compatibility.
- `reduce_view_update` now copies the restore cache payloads into `{lod,index,camera}` in a single transaction, and the cache dataclasses are `PlaneRestoreCacheBlock` / `VolumeRestoreCacheBlock` to match the rest of the scene block schema.
- Persist per‑mode RestoreCaches on the ledger so view toggles are single-pass:
  - `restore_cache.plane.state`: `{ level: int, index: tuple[int,...], pose: {rect,center,zoom} }`
  - `restore_cache.volume.state`: `{ level: int, index: tuple[int,...], pose: {center,angles,distance,fov} }`
  - Define TypedDicts/dataclasses in `src/napari_cuda/server/scene/blocks/restore.py` and dual-write them from current reducer/worker paths.
- View toggle transactions (flag on) write `ViewBlock` and copy the target mode’s RestoreCache into authoritative blocks (`LodBlock.level`, `IndexBlock.value`, `CameraBlock.*`) in a single pass.
- Worker-issued camera poses now refresh the viewport caches without bumping `scene.op_seq`; next step is to flip `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX=1` in dev runs and confirm the block-backed camera consumers stay jitter-free before cutting over permanently.
- Next step: begin the Phase 2 consumer flip—teach `snapshot_scene_blocks` (the block-focused successor to `snapshot_viewport_state`), notify builders, and worker apply paths to read the block scopes when the flag is enabled while legacy keys remain for compatibility.
- Once block-backed consumers are proven, plan the Phase 3 rename where `viewport_state`
  metadata is replaced entirely by the scene block payloads (view/index/lod/camera) so we can
  delete the legacy `viewport.*` ledger scopes without changing downstream behavior.
- Add parity tests to assert toggles set blocks from caches and notify payloads are unchanged; keep legacy scopes for compatibility until Phase 3.
- Keep minimal worker‑side caches (PlaneViewportCache/VolumeViewportCache) only for real‑time deltas, ROI hysteresis, and per‑block signature diffing.
