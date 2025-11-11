# Architecture Roadmap

A living roadmap that ties high‑level goals to concrete specs and code. Treat this as the index for the architecture folder; each item links to the authoritative doc(s) and canonical modules.

## Vision & Guardrails
- Single source of truth per concern (ledger for committed state; ActiveView for mode/level).
- Explicit contracts (typed schemas + serializers) and deterministic apply.
- Transition legacy `dims_spec` toward the factored `view / axes / index / lod / camera`
  ledger scopes; new payloads adopt the `index` terminology from day one.
- Offensive coding: assertions over guards; no try/except in hot paths; boundary logging only.
- Final runtime contract: server pokes the worker (via `scene.main.op_seq`),
  worker pulls `{view, axes, index, lod, camera, layers}` directly from the
  ledger, applies block-by-block, and only sends intents back via the worker
  intent mailbox (no intermediate planner/mailbox state on the worker).

References:
- docs/architecture/system_design_spec.md
- docs/architecture/per_function_contracts.md
- docs/architecture/repo_structure.md

## Current State (2025‑11‑10)
- ActiveView authority in ledger; server emits `notify.level` (resumable).
- View toggles avoid touching cache state; reducers just rewrite dims while worker restores keep ActiveView authoritative.
- Render snapshot + scene builders read ActiveView for mode/level; dims specs only provide shapes/order metadata.
- Client consumes `notify.level`; HUD renders mode/level from ActiveView.
- Legacy `dims_spec` + `current_step` naming still powers reducers and notify;
  the factored `view / axes / index / lod / camera` scopes are not landed yet.
- Render mailbox still uses unified signature (per‑block migration planned).

References:
- Server: src/napari_cuda/server/control/transactions/active_view.py, control/mirrors/active_view_mirror.py, control/topics/notify/level.py
- Client: src/napari_cuda/client/control/control_channel_client.py, client/runtime/stream_runtime.py, client/rendering/presenter_facade.py
- Protocol: src/napari_cuda/protocol/messages.py, envelopes.py, parser.py
- Change log: docs/architecture/change_log.md

## Workstreams & Milestones

1) Protocol & Schemas
- Define the new `view / axes / index / lod / camera` ledger scopes alongside
  legacy `dims_spec`, dual-write them in reducers, and teach notify payloads +
  snapshots to consume the factored data. When complete, the worker call stack is
  simply: ledger poke (scene.op_seq) → worker pulls scopes → applies dims/camera/
  layers in place (no planner/mailbox intermediates).
- Remove `current_level` from DimsSpec and `NotifyDimsPayload` (source level from ActiveView once the new scopes are live).
  - Spec: system_design_spec.md (§3 DimsSpec, new §3.x View/Axes/Lod blocks, §9 Notify Path)
  - Code: shared/dims_spec.py, new shared models, protocol/messages.py (NotifyDimsPayload), server/client consumers.

2) Control Plane (Reducers/Transactions)
- Eliminate writes to `("multiscale","main","level")`; rely solely on ActiveView and the new view/index/lod scopes.
  - Code: server/control/transactions/level_switch.py, plane_restore.py; server/scene/builders.py reads ActiveView/new scopes.
- Remove reducer‑internal try/except (handlers catch/map only).
  - Code: server/control/state_reducers.py, state_update_handlers/*

3) Render Runtime
- Per‑block signatures in mailbox; drop unified scene signature once all call sites updated.
  - Code: server/utils/signatures.py, server/runtime/ipc/mailboxes/render_update.py
- Name alignment: apply_dims_block/apply_camera_block/apply_layers_block (no wrappers).
  - Code: server/runtime/render_loop/applying/*, apply.py

4) Client Integration
- Dims mirror no longer mirrors multiscale level; presenter HUD strictly from `notify.level`.
  - Code: client/control/mirrors/napari_dims_mirror.py, presenter_facade.py
- Expand state channel tests for `notify.level` (resume paths).

5) Tests & CI
- Extend unit + integration tests to lock invariants (ActiveView writes, mailbox dedupe, apply contracts).
  - Code: server/tests/*, protocol/_tests/*, client/control/_tests/*

## Document Map (Index)
- System Design Spec: docs/architecture/system_design_spec.md (canonical contracts & data models)
- Function Contracts: docs/architecture/per_function_contracts.md (pre/postconditions, side‑effects)
- Change Log: docs/architecture/change_log.md (session‑by‑session trail)
- Repo Structure: docs/architecture/repo_structure.md (module ownership)
- Cut‑Down Guides:
  - server_overview.md (entry flows, mirrors, notify)
  - reducers_txns.md (reducers vs txns split)
  - runtime_pipeline.md (render path, apply order)
  - scene_builders_notify.md (snapshot/notify builders)
  - state_ledger.md (ledger semantics)
  - client_integration.md (state channel, HUD)
- Layer Parity Plan: docs/architecture/layer_parity_plan.md (scope & milestones for multi‑layer parity)

## Status Tags
- Implemented: ActiveView + notify.level; HUD integration; server/client wiring; protocol rename + tests.
- Next: remove `current_level` in DimsSpec/notify.dims; remove multiscale writes; per‑block signatures; apply API rename.
