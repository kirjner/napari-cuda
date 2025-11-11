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
     `PlaneState`/`VolumeState`, and redundant notify/multiscale fields once
     the new scopes are authoritative everywhere.
