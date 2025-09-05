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

## Coding Style & Naming Conventions
- Python ≥ 3.10, 4-space indentation, UTF-8.
- Ruff enforces lint/format; run `pre-commit run -a` before pushing.
- Imports: absolute only (relative imports are banned by config).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Docstrings use double quotes; keep public APIs typed.

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
