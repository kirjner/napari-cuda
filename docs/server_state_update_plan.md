# Server State Update Migration Plan

## Purpose
- Establish the server side of the unified `state.update` protocol that the new client reducer consumes.
- Replace legacy `control.command` / intent plumbing with a single authoritative envelope that carries sequencing, interaction, and scope metadata. ✅
- Guarantee deterministic reconciliation between server and client so sliders and other controls stay stable during collaborative sessions.

## Context
- Client has already implemented `StateUpdateMessage` and is rebuilding around a reducer that owns optimistic + confirmed registers per `(scope, target, key)`.
- Prior to this effort the server processed `control.command` messages and emitted heterogeneous payloads for layers, dims, and baselines; the new implementation replaces those paths with the unified `state.update` message shape.
- There is no backward-compatible path once both sides ship—mixed deployments will fail—so we must land this work in lockstep with the client branch.

## Objectives
- Accept, validate, and apply inbound `state.update` payloads for both layer and dims scopes.
- Maintain a single global `server_seq` counter and per-property metadata keyed by `(scope, target, key)`.
- Broadcast the authoritative value using the exact `state.update` shape (plus sequencing metadata) for live updates and reconnect baselines.
- Advertise the new capability in scene specs and remove legacy intent/command handlers once the new path is in place.

## Deliverables
- Updated protocol handling in `state_channel_handler.py` exposing `_handle_state_update` as the only control entry point.
- Refactored scene helpers (`server_scene.py`, `server_state_updates.py`) producing `StateUpdateResult` objects with deterministic sequencing metadata.
- Payload builders (`build_state_update_payload`, baseline emitters) that emit the unified message for layers, dims, and reconnects.
- Comprehensive unit/integration tests covering state application, sequencing, and broadcast behaviour.
- Documentation describing the new control flow and deployment expectations.

## Scope
- In scope: state channel control messages, layer/dims sequencing metadata, scene spec capabilities, docs/tests for the server.
- Out of scope: client reducer implementation (already in flight), worker rendering semantics, CUDA capture, and unrelated refactors flagged in the broader server plan.

## Assumptions
- Client reducer relies on a global monotonically increasing `server_seq` (32-bit wrap acceptable) and will drop any message lacking that field.
- Dims scope uses `{scope: "dims", target: <axis label>, key: "step"}` naming; any deviation must be negotiated with the client team before coding.
- Legacy clients will not connect once the flag flips; we coordinate deployment so only new clients speak to the updated server.
- Existing tests for intents/control commands were repurposed to verify the new flow; we have adequate fixtures to simulate state channel traffic.

## Risks
- **Mixed-version deployments**: A legacy client connecting to the new server will fail; mitigation is coordinated rollout and capability gating in ops.
- **Sequencing regressions**: Bugs in the global counter or metadata storage could break client reconciliation; mitigate with targeted unit tests and smoke logging (`NAPARI_CUDA_DEBUG=state`).
- **Dims naming mismatch**: If axis identifiers diverge, dims sliders will stop moving; mitigate by validating constants with client team before merge.
- **Baseline omissions**: Failing to emit `state.update` during reconnect would leave the client reducer without confirmed state; cover with reconnect tests.

## High-Level Timeline
1. **Day 0** – Finalise plan, align constants with client, prep branch.
2. **Day 1** – Implement protocol/dataclass adjustments, global `server_seq`, metadata map.
3. **Day 2** – Replace handler/dispatch path, update payload builders, emit new baselines.
4. **Day 3** – Update tests, run targeted pytest suites, add DEBUG logging knobs.
5. **Day 4** – Refresh docs, review with client team, schedule coordinated deploy.

## Work Breakdown Structure
1. **Protocol Wiring**
   - Export `StateUpdateMessage` (already on client) and update the server parser.
   - Deprecate `CONTROL_COMMAND_TYPE` usage.
2. **Sequencing Core**
   - Add `next_server_seq` to `ServerSceneData` and replace per-layer counters.
   - Store metadata in a dict keyed by `(scope, target, key)`; provide helpers to get/clear entries.
3. **State Application Helpers**
   - Rename intent helpers to `apply_layer_state_update` / `apply_dims_state_update` returning `StateUpdateResult`.
   - Enforce stale detection, value normalisation, and metadata updates inside those helpers.
4. **Handler Refactor**
   - Implement `_handle_state_update` and register it as the sole control handler.
   - Remove legacy intent/command handlers after confirming no call sites remain.
5. **Payload/Baseline Builders**
   - Author `build_state_update_payload` and reuse it for live broadcasts and `_send_layer_baseline` / `_send_dims_baseline`.
   - Ensure control version metadata mirrors the client reducer’s expectations (`server_seq`, `source_client_id`, `source_client_seq`, `interaction_id`, `phase`).
6. **Capabilities & Manager Updates**
   - Advertise `state.update` in capabilities, update any handshake logging accordingly.
7. **Testing & Validation**
   - Update unit tests, add coverage for dims, run `uv run pytest -q src/napari_cuda/server/_tests`.
   - Add smoke logs and perform manual run with `NAPARI_CUDA_DEBUG=state,dims`.
8. **Documentation & Cleanup**
   - Refresh architecture and plan docs; remove references to intents/commands.
   - Delete unused helpers/constants.

## Testing Strategy
- **Unit**: Focused tests for sequencing and stale rejection in `test_server_state_updates.py` (renamed from intent tests).
- **Integration**: State channel tests verifying inbound/outbound payload shapes, live broadcasts, and reconnect baselines using the new message type.
- **Regression Suite**: `uv run pytest -q src/napari_cuda/server/_tests` as gating command before merge.
- **Smoke**: Manual run of `uv run napari-cuda-server` with debug flags to confirm logs show monotonic `server_seq` and correct metadata.

## Documentation Updates
- `docs/server_architecture.md`: Describe the unified state.update pipeline and metadata semantics.
- `docs/server_streamlining_plan.md`: Mark the control-channel rewrite milestone complete and adjust future tasks.
- New doc (`docs/server_state_update_plan.md` – this file) maintained until rollout is finished.
- Update any developer onboarding or README references to intent/control command terminology. ✅

## Open Questions
- Do we need to persist `next_server_seq` across process restarts for audit trails, or is per-session sequencing sufficient?
- Should dims target identifiers be axis labels ("z") or numeric indices? We align with the client decision before coding.
- Do we include server-generated updates (no `client_id`) in the same stream immediately, or reserve a separate scope for automation?
- What logging granularity do we keep post-rollout to aid debugging without flooding production logs?
