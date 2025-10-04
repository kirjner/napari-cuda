# Consolidate Refactors Integration Plan

## Context
Server work on `server-refactor` rebuilt the headless EGL worker, state channel, and protocol surface around the enriched `state.update` envelope. The `client-refactor` branch (tip `b068480c`) introduces a reducer-driven streaming loop, new state store, and removes the old intent bridge. We need a clean integration branch that preserves the server architecture while adopting the client changes and keeps the protocol aligned with the upcoming greenfield contract in `docs/protocol_greenfield.md`.

## Goals
- Merge `origin/client-refactor` into `server-refactor` without regressing the refactored server stack or the new client reducer pipeline. ✅
- Align protocol definitions with the greenfield handshake/notify envelopes while maintaining stability across clients. ✅
- Ensure the integrated branch builds, runs, and passes both client and server unit suites. ✅

## Deliverable Branch
- Branch name: `consolidate-refactors` (created from `server-refactor`).
- Integration complete; branch now carries the reducer-driven client, refactored server, and the enforced handshake/notify protocol.
- Experimental work outstanding: command-lane expansion beyond the
  `napari.pixel.request_keyframe` RPC and naming cleanups (tracked below).

## Prerequisites
1. `git fetch origin client-refactor` to ensure tip `b068480c` is present.
2. Record merge base: `036a76ef9e52d9b46f6665d52def186c25ad560d` (useful for selective diffs).
3. Confirm clean worktree except for `docs/protocol_greenfield.md` (keep staged if needed).
4. Run `uv run pytest -q -k "napari_cuda and not slow"` on `server-refactor` and capture results for comparison.

## High-Level Sequence
1. `git switch server-refactor`
2. `git pull --ff-only origin server-refactor`
3. `git switch -c consolidate-refactors`
4. Optional: re-stage `docs/protocol_greenfield.md` if we want it in the integration commit.
5. `git merge --no-ff origin/client-refactor`
6. Resolve conflicts following the per-area guidance below.
7. Run formatting/ruff if merge markers touched linted code (e.g., `uv run ruff format <files>` as needed).
8. Execute validation commands (see Testing section).
9. Commit with message `Merge client-refactor into server-refactor`.

## Conflict Resolution Playbook

### Protocol (`src/napari_cuda/protocol`)
- **messages.py**: retain the enriched `StateUpdateMessage` from `server-refactor` (fields like `meta`, `axis_index`, `current_step`, `ack`) and introduce an optional `extras` mapping so the new client reducer can consume additional context without breaking existing payload shapes.
- Drop the obsolete `ControlCommand` helper—future command traffic will use the greenfield `call.command` envelopes.
- Ensure `StreamProtocol.parse_message` continues to hydrate `StateUpdateMessage` instances; no other message categories need to change yet.

### Server (`src/napari_cuda/server/**`)
- Keep the `server-refactor` modules (`render_worker.py`, `worker_lifecycle.py`,
  `server_scene/*.py`, control channel helpers, etc.). When resolving conflicts
  with `client-refactor`, favor the refactored architecture and delete
  references to the old `egl_worker` in client code.
- Restore deleted test suites from `server-refactor` (state updates, worker lifecycle). If client branch removed them, bring them back unless they rely on removed components.
- In `egl_headless_server.py`, ensure imports match retained modules (no `egl_worker`). Wire any new client-introduced hooks (e.g., metrics toggles) to the refactored API.
- Keep server config docs (`docs/server_*`). Merge client docs by adding, not replacing.

### Client (`src/napari_cuda/client/**`)
- Accept the new reducer pipeline: files like `client_stream_loop.py`, `state_store.py`, `state/bridges/layer_state_bridge.py`, and client-loop utilities should come from `origin/client-refactor`.
- Update any imports that expect removed server helpers (e.g., `control_sessions`) to the merged protocol API.
- Review `launcher.py`, `proxy_viewer.py`, and `streaming_canvas.py` for references to the old intent bridge; ensure they now instantiate `ClientStreamLoop` and `LayerStateBridge` correctly.
- Align `StateStore` with the merged `StateUpdateMessage`: handle optional `meta`/`current_step` fields that the server still sends.

### Shared Docs & Tooling
- Keep `.gitignore` entries from both branches (ensure `docs/archive_local/` remains ignored).
- Merge `pyproject.toml`: keep `napari_cuda` under `[tool.pytest.ini_options] testpaths` so server tests still run and avoid adding `pytest` to the base dependency set (existing extras already cover it).
- Defer to `docs/protocol_greenfield.md` for the canonical control-plane contract; legacy protocol/doc plans now live under `docs/archive_local/`.

## Testing Checklist
1. `uv run pytest -q -m "not slow" src/napari_cuda/server/tests` (server suite).
2. `uv run pytest -q src/napari_cuda/client/runtime/_tests src/napari_cuda/client/control/_tests src/napari_cuda/client/state/_tests` (client runtime + reducer suites).
3. `uv run pytest -q src/napari_cuda/protocol/_tests` (protocol serialization).
4. Manual smoke: `uv run napari-cuda-server --state-port ...` and
   `uv run napari-cuda-client` to verify pan/zoom/dims adjust correctly and
   confirm the Home/reset action issues `call.command napari.pixel.request_keyframe`
   (check server logs for the command acknowledgement).

## Post-Merge Follow-Ups
- Extend the command lane (`call.command` / `reply.command`) beyond the
  already-wired `napari.pixel.request_keyframe` command and update docs/tests as
  new verbs land.
- Sweep client code to remove remaining “intent” terminology and keep modules (`client/control/state_update_actions.py`, etc.) aligned with the reducer architecture.
- Continue converging naming so server control helpers all live under
  `server/control/` with notify-centric terminology per
  `docs/server_streamlining_plan.md`.
- Refresh onboarding/docs to drop references to dual emission and legacy command paths.
- Expand CI to include the reducer suites and any new command-channel coverage once available.

## Notes on command envelopes
- The integration drops the legacy `control.command` scaffolding. When we tackle the command lane, add the greenfield `call.command`/`reply.command` helpers directly instead of refactoring an intermediate type.

## Sign-off Criteria
- Branch `consolidate-refactors` builds, tests, and manually validates streaming interactions.
- All documentation from both branches present and cross-linked.
- Known follow-up tickets filed for handshake implementation and notify envelope migration.
