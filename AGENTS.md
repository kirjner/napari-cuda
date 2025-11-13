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

## Current Focus (2025-11-12)
- Migrating the legacy `dims_spec + planner/mailbox` pipeline to the factored
  `view / axes / index / lod / camera / layers` ledger design.
- Reference plan: `docs/architecture/view_axes_index_plan.md` (call-stack
  diagrams + phased issue breakdown). Use it as the source of truth for scope
  and sequencing.
- No external clients are depending on the notify/state protocol yet, so we can
  evolve the payloads (LayerBlock-native) in lockstep with the server without
  maintaining backward compatibility once Phase 3 lands.
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

Finish Phase 3 cleanup by treating `SceneBlockSnapshot` as the only runtime+protocol
payload:

1. **Runtime/render loop** – DONE: worker ingests a single block snapshot per tick. Keep
   shrinking `RenderLedgerSnapshot` to a shim until everything else flips.
2. **Notify + mirror consumers** – teach state-channel builders, the layer mirror, and
   snapshot helpers to read `SceneBlockSnapshot.layers` directly and emit the same payloads.
3. **Protocol flip** – once server-side consumers rely on LayerBlocks, update the notify
   payload schema (and any stub clients) to speak LayerBlocks natively, then delete
   `LayerVisualState` / `RenderLedgerSnapshot.layer_values`.
4. **Flag flip & cleanup** – enable `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` by default, remove
   the legacy planner/mailbox code, and document the block-only pipeline as the contract.
