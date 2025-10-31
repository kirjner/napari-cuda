# Repository Structure

Offensive coding tenet: no defensive fallbacks, no silent failures.

## Current Layout Snapshot

```
napari_cuda/
├── client/          # App launcher, state projections, rendering stack, runtime loop
├── server/          # Headless service (control, state, rendering, data)
├── protocol/        # Message schema, snapshots, parser
├── codec/           # Format helpers (AnnexB/AVCC parsing, H.264 utilities)
├── cpu/, cuda/, _vt/ # Platform shims (CPU capture, CUDA bindings, VT integration)
├── utils/           # Cross-cutting helpers (env parsing, etc.)
├── web/             # Dash/Flask dashboards and static assets
└── docs/, tools/, scripts/
```

Recent reorganisations landed the top-level split called out in the future-structure draft, and both client/server trees now expose the expected `app`, `control`, `data`, `rendering`, `runtime` (client-only today), and `state` packages.

## Alignment Highlights

- Client runtime/app/rendering separation matches the target layout; `proxy_viewer` is now thin and bridges live in `client/state`.
- Server engine owns encoder + worker code after moving the PyAV encoder from the shared codec layer (`src/napari_cuda/server/engine/encoding/h264_encoder.py`).
- Protocol, docs, and tools directories match the published draft (no more greenfield shims).
- Server runtime package introduced (`server/runtime/worker/egl.py`, `runtime/worker/loop.py`, `runtime/ipc/mailboxes/render_update.py`, `runtime/core/*`), isolating orchestration from rendering helpers.

## Outstanding Organisation Work

### Server package

- **Document engine façade expectations.** Add guidance in the architecture doc covering `server/engine/api.py` so downstream contributors rely on the façade rather than internal capture/encoding modules.
- **Audit pixel channel responsibilities.** With the channel logic now in `server/engine/pixel/channel.py`, decide whether any remaining control-plane glue should migrate or stay transport-side (e.g., heartbeat vs broadcaster orchestration).
- **Split control channel orchestration.** `control_channel_server.py` currently mixes handshake, resume journal, command routing, and heartbeat/watchdog logic. Break it into `server/control/handshake.py`, `server/control/resume.py`, `server/control/dispatch.py`, etc., so future hoists (e.g., state sync, metrics taps) have explicit homes.
- **Clarify GL/canvas helpers.** Split `napari_viewer/bootstrap.py`, `egl_context.py`, `gl_capture.py`, `vispy_intercept.py`, and `cuda_interop.py` into a `rendering/gl/` namespace so rendering vs orchestration vs transport responsibilities are obvious.
- **Audit `app/experiments`.** Decide whether experiments remain in-tree or move to `tools/` to avoid diluting the production app surface.

### Client package

- **Rename `client/runtime/client_loop/` to `client/runtime/loop/`.** The folder exports lifecycle helpers (`loop_state.py`, `scheduler.py`, `warmup.py`); giving it a neutral name mirrors the proposed `server/runtime` and removes the legacy “streaming” vocabulary.
- **Move `state_update_actions.py` out of `client/control`.** Those reducers are state projections and should live next to the bridge/projection stack in `client/state/`.
- **Clarify rendering pipelines.** Consider `client/rendering/pipelines/` → `client/runtime/pipelines/` to keep draw-loop wiring in runtime, reserving `rendering/` for presenter/renderer/decoders only.
- **Audit naming for parity.** Align `client/runtime/stream_runtime.py` with the eventual `server/runtime` entry point (e.g., `client/runtime/stream_loop.py`) once the server move lands.
- **Decompose `control_channel_client.py`.** Extract handshake/session metadata, outbound command queueing, heartbeat watchdog, and notification dispatch into separate modules (`client/control/handshake.py`, `client/control/session.py`, etc.) so new control-plane features can land without swelling a single file.

### Shared / top level

- **Collapse dormant packages.** `src/napari_cuda/cuda` currently contains only `__pycache__`; remove or repurpose once CUDA helpers move under rendering/runtime modules.
- **Document ownership of platform shims.** Clarify in this doc where `_vt/`, `cpu/`, and `web/` fit into the architecture, and whether they should gain subpackages (e.g., `_vt/shim/`, `cpu/capture/`).
- **Ensure tests mirror the final layout.** As packages move, migrate test modules (e.g., `server/tests/test_worker_confirmations.py`) to new `_tests` packages inside the destination modules to keep co-location consistent with the guiding principles.

## Next Steps

1. Draft RFC / PR plan for the server runtime extraction (worker modules + pixel channel relocation) so the package split happens atomically.
2. Rename the client loop package and move `state_update_actions.py`, updating imports/tests in one pass.
3. After both sides settle, refresh this doc with the new directory tree and link to any architectural overviews living in `docs/architecture/`.

Offensive coding tenet: assertions over guards unless interfacing external systems.
