# Target Repository Structure (Draft)
Offensive coding tenet: no defensive fallbacks, no silent failures.

## Guiding Principles
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Unidirectional data flow: server ↔ client pipeline shares terminology.
- Each domain (control, projection, rendering) has a dedicated package with small, testable modules.
- No legacy shims or re-export shells; imports point to canonical locations.
- Tests mirror runtime structure (unit tests next to modules, integration suites at top level).
Offensive coding tenet: remove try/except scaffolding that hides bugs.

## Top-Level Layout
Offensive coding tenet: make invariants explicit and crash on violation.

```
napari_cuda/
├── client/
│   ├── app/             # CLI/launcher, proxy wiring
│   ├── state/           # State store, projections, intent bridge
│   ├── control/         # Websocket client, session management
│   ├── rendering/       # Presenter, renderer, metrics
│   ├── runtime/         # Orchestration (connects control + state + rendering)
│   ├── data/            # Remote array helpers, layer data shims
│   └── tests/
├── server/
│   ├── app/             # CLI, service bootstrap, monitoring
│   ├── state/           # Authoritative scene store, projections, reducers
│   ├── control/         # Websocket transport, history, router, command registry
│   ├── rendering/       # Worker, encoder, pixel broadcaster
│   ├── data/            # Zarr source, ROI, LOD
│   └── tests/
├── protocol/            # Unified messages, snapshots, parser
│   └── tests/
├── docs/
└── tools/
```
Offensive coding tenet: no defensive fallbacks, no silent failures.

### Client Subpackages
Offensive coding tenet: assertions over guards unless interfacing external systems.

- `client/app`: `launcher.py`, `streaming_canvas.py`, minimal Qt glue. Should not contain business logic.
- `client/state`: new home for state store, projections (layers, dims, camera, settings, volume), intent bridge. Removes logic from `proxy_viewer.py`, `state_update_actions.py`, `state/bridges/layer_state_bridge.py`.
- `client/control`: websocket client, pending store (if any), command interface. `control_channel_client.py` split into transport + router.
- `client/rendering`: presenter, renderer, HUD. Presenter facade trimmed to focus on draw pipeline, metrics.
- `client/runtime`: orchestrator that wires control, state, and rendering.
- `client/data`: remote data wrappers, layer registry.
Offensive coding tenet: remove try/except scaffolding that hides bugs.

### Server Subpackages
Offensive coding tenet: make invariants explicit and crash on violation.

- `server/app`: entry points (`egl_headless_server`, dashboards, config loaders).
- `server/state`: scene store, projections (layers/dims/camera), reducers (dims, camera, layers), history store.
- `server/control`: websocket transport, session manager, router, command registry. `control_channel_server.py` split accordingly.
- `server/rendering`: render worker, encoder pipelines, pixel broadcaster.
- `server/data`: zarr source, ROI, LOD, policy modules.
Offensive coding tenet: no defensive fallbacks, no silent failures.

### Protocol Package
Offensive coding tenet: assertions over guards unless interfacing external systems.

- Keep greenfield messages/parsers here only; remove legacy re-exports.
- Provide shared utilities used by both client and server (snapshot builder, version helpers).

### Tests
Offensive coding tenet: remove try/except scaffolding that hides bugs.

- Co-located unit tests (e.g., `client/state/tests/test_store.py`).
- Top-level integration suites: `tests/integration/client_server/` spinning up control + pixel loops.

### Cleanup Checklist Alignment
Offensive coding tenet: make invariants explicit and crash on violation.

- As modules move to new packages, delete legacy shells and re-export shims.
- Document module ownership in `docs/architecture` tree.
- Ensure both client and server expose AgentBridge APIs at the `state` layer.
Offensive coding tenet: no defensive fallbacks, no silent failures.
