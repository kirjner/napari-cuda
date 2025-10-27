# Headless MCP Integration Plan for the EGL Server

## Goals
- Embed the FastMCP tool surface directly into the EGL headless server so agents can automate layer and camera workflows without a GUI napari viewer.
- Reuse existing headless primitives (the ledger-backed `RenderLedgerSnapshot`, `snapshot_scene`, `RenderUpdateMailbox`) instead of instantiating Qt objects.
- Provide a secure, minimal tool set (layer CRUD, camera/dims controls, code execution hook) suitable for research automation.
- Leave screenshots and other GUI-only behaviors out of scope for this phase.

## Deployment Model
- **Process**: run the MCP service inside the `EGLHeadlessServer` process; the server already owns the asyncio loop that coordinates WebSocket traffic.
- **Event loop**: attach the FastMCP application to the main asyncio loop during `EGLHeadlessServer.start()` and keep it running until shutdown.
- **Threading**: continue to run the EGL renderer in its dedicated worker thread; the MCP handlers operate on ledger-backed scene snapshots (`RenderLedgerSnapshot`, `snapshot_scene`) through existing locks.
- **Ports**: expose MCP on a configurable HTTP port (default disabled) to avoid interfering with existing state/pixel sockets.

## Architectural Touchpoints
- **Scene state**: the authoritative scene lives in the server state ledger + `RenderLedgerSnapshot` plus the worker viewer model. MCP commands should route state changes through the same reducers and mailboxes (`RenderUpdateMailbox`, ledger-driven dims/camera updates) that the WebSocket state channel uses.
- **Scene metadata**: the `snapshot_scene` helpers can generate layer listings, dims metadata, and camera snapshots without Qt. The MCP `list_layers`, `session_information`, and related tools should call into them instead of reading napari Viewer objects.
- **Layer storage**: data layers originate from the worker's scene source (e.g., NGFF datasets). For MCP-driven layer CRUD, surface translation helpers in `napari_cuda.server.scene.snapshot` (or a dedicated adapter) should create/remove server-side layer specs and notify clients via the existing state broadcast pipeline.
- **Camera/dims controls**: reuse `_enqueue_camera_delta` and `_rebroadcast_meta` from `EGLHeadlessServer` so MCP camera operations coalesce with WebSocket-sourced commands.
- **Code execution**: limit the `execute_code` tool to a constrained namespace that mirrors what the worker already exposes (e.g., viewer metadata snapshot, metrics hooks). Avoid letting the tool mutate global module state directly.

## Tool Surface Redesign
- **Keep**: `detect_viewers`, `init_viewer` (headless), `list_layers`, `add_image` (using NGFF/zarr loader or numpy upload), `add_points`, `remove_layer`, `set_layer_properties`, `reorder_layer`, `set_active_layer`, `set_camera`, `set_ndisplay`, `set_dims_current_step`, `set_grid`, `execute_code`, `read_output`.
- **Drop**: GUI-dependent features (`screenshot`, `timelapse_screenshot`) and Qt lifecycle helpers.
- **Adapt**:
  - `init_viewer`: return metadata about the headless viewer model (canvas size, layer names) and ensure the worker thread is running; avoid creating a Qt application.
  - `add_image`/`add_labels`: accept either file paths or serialized numpy arrays. For on-server data, hand off to the `ZarrSceneSource` helper; for ad-hoc uploads, register temporary assets inside the server cache.
  - `install_packages`: optional; gate behind a feature flag because production deployments might forbid runtime package installs.
- `execute_code`: expose a curated set of bindings such as `snapshot=RenderLedgerSnapshot`, `metrics=Metrics`, `ctx=ServerCtx` to keep behavior predictable.

## Implementation Steps
1. **Service scaffolding**
   - Create `napari_cuda/server/mcp_service.py` with a trimmed `HeadlessMCPTools` class that mirrors the FastMCP registration seen in `napari_mcp_server.py` but rewired to server primitives.
   - Provide a `start_mcp_service(server: EGLHeadlessServer, loop: asyncio.AbstractEventLoop, port: int)` helper that spins up `FastMCP` with the adapted tools and returns a shutdown coroutine.
2. **Headless tool bindings**
   - Implement tool methods using existing server helpers:
     - Layer queries: call `server._refresh_scene_snapshot()` and consume `server._scene_snapshot` to build listings.
     - Mutations: enqueue updates through the ledger-driven reducers, `_enqueue_camera_delta`, and the worker's request queues.
     - Outputs: use the existing `_store_output` utility or a simplified variant for command transcripts.
3. **Server wiring**
   - Add optional configuration in `ServerConfig` / `ServerCtx` for `mcp_enabled` and `mcp_port` (default off).
   - When enabled, call `start_mcp_service` inside `EGLHeadlessServer.start()` and ensure the returned shutdown coroutine executes during `stop()`.
4. **State synchronization**
   - Ensure MCP-triggered mutations trigger the same broadcast path as WebSocket state changes. For example, after `add_points`, schedule `_schedule_scene_broadcast("mcp")`.
   - Guard long operations with the server's existing locks (`_viewer_lock` equivalent is `_state_lock`) to avoid cross-thread races.
5. **Testing**
   - Add targeted async tests under `src/napari_cuda/server/tests/test_mcp_service.py` exercising: layer listing, camera adjustments, dims update propagation, and execute_code sandboxing.

## Security & Access Control
- Default to disabled in production builds; require an explicit config or environment variable to expose MCP.
- Gate high-risk tools (`execute_code`, `install_packages`) behind a capability flag served in the MCP session info; downstream clients can opt out.
- Reuse the server's auth handshake (if any) by wrapping the FastMCP ASGI app with the same token validator used for WebSocket upgrades.
- Log every MCP invocation along with requester identity for audit.

## Operational Considerations
- **Resource limits**: enforce per-call timeouts (e.g., 5 seconds) and max payload sizes, mirroring WebSocket safeguards.
- **Backpressure**: because FastMCP is HTTP-based, ensure the handler rejects concurrent long-running requests if the worker thread is busy; keep interactions short to avoid stalling the render loop.
- **Metrics**: extend `Metrics` to emit counters for MCP usage (e.g., `mcp.calls_total`, `mcp.failures_total`) to Grafana along with existing frame metrics.
- **Observability**: add structured logs when MCP mutates the scene so client operators can trace automated changes.

## Agent Integration Pattern
1. Agent discovers the MCP endpoint (`detect_viewers`) and confirms the service advertises `viewer_type="egl_headless"`.
2. Agent calls `session_information` to obtain dims, camera state, and layer inventory from the headless viewer model.
3. Agent submits layer operations (`add_points`, `set_layer_properties`, etc.) and observes resulting `scene.update` broadcasts over the existing WebSocket client.
4. Agent uses `execute_code` sparingly for algorithmic entry points (e.g., run a pre-registered processing pipeline) and retrieves any bulk output via `read_output`.

## Future Enhancements
- Support remote data uploads (e.g., chunked numpy blobs) for agents that need to push intermediate results without touching disk.
- Add a declarative job interface on top of MCP so agents can submit high-level “apply algorithm X to ROI Y” commands rather than orchestrating low-level layer mutations.
- Consider a lightweight permission model where MCP capabilities map to server roles (viewer-only, editor, admin).
- Evaluate exposing MCP actions over the existing WebSocket control channel to reduce the number of endpoints a client must coordinate.
