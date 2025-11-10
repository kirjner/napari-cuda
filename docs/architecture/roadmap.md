# Architecture Roadmap

A living roadmap that ties high‑level goals to concrete specs and code. Treat this as the index for the architecture folder; each item links to the authoritative doc(s) and canonical modules.

## Vision & Guardrails
- Single source of truth per concern (ledger for committed state; ActiveView for mode/level).
- Explicit contracts (typed schemas + serializers) and deterministic apply.
- Offensive coding: assertions over guards; no try/except in hot paths; boundary logging only.

References:
- docs/architecture/system_design_spec.md
- docs/architecture/per_function_contracts.md
- docs/architecture/repo_structure.md

## Current State (2025‑11‑09)
- ActiveView authority in ledger; server emits `notify.level` (resumable).
- Client consumes `notify.level`; HUD renders mode/level from ActiveView.
- DimsSpec-first notify for dims metadata; render mailbox with unified signature (per‑block planned).

References:
- Server: src/napari_cuda/server/control/transactions/active_view.py, control/mirrors/active_view_mirror.py, control/topics/notify/level.py
- Client: src/napari_cuda/client/control/control_channel_client.py, client/runtime/stream_runtime.py, client/rendering/presenter_facade.py
- Protocol: src/napari_cuda/protocol/messages.py, envelopes.py, parser.py
- Change log: docs/architecture/change_log.md

## Workstreams & Milestones

1) Protocol & Schemas
- Remove `current_level` from DimsSpec and `NotifyDimsPayload` (source level from ActiveView).
  - Spec: system_design_spec.md (§3 DimsSpec, §9 Notify Path)
  - Code: shared/dims_spec.py, protocol/messages.py (NotifyDimsPayload), server/client consumers.

2) Control Plane (Reducers/Transactions)
- Eliminate writes to `("multiscale","main","level")`; rely solely on ActiveView and DimsSpec.
  - Code: server/control/transactions/level_switch.py, plane_restore.py; server/scene/builders.py reads ActiveView.
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

