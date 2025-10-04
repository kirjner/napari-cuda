# Client Module Inventory

This document catalogues the Python modules that implement the streaming client.
It is a snapshot of the current layout (Oct 2025) and highlights the size,
responsibilities, and pain points that surfaced during the recent debugging
pass. It is intentionally descriptive; architectural recommendations will come
in a follow-up design doc.

## Top-Level Entry Points

| Module | Lines | Role | Notes |
| --- | ---:| --- | --- |
| `client/launcher.py` | ~120 | CLI bootstrap used by `uv run napari-cuda-client`. | Constructs `ProxyViewer`, `Window`, and hands control to `StreamingCanvas`. |
| `client/streaming_canvas.py` | 217 | Qt-facing wrapper that instantiates `ClientStreamLoop`, defers window show, and binds presenter hooks. | Still performs config work (env parsing, latency defaults) that probably belongs in the runtime. |
| `client/proxy_viewer.py` | 541 | Napari `ViewerModel` subclass acting as thin client mirror. | Owns slider forwarding, suppress guards, and layer syncing. Contains a lot of logic (axis detection, play-state probing, metadata application). |
| `client/runtime/stream_runtime.py` | 1420 | The main coordinator: wires control channel, presenter facade, decoders, and layer registry. | 1400+ lines with mixed responsibilities (connecting websockets, handling acks, applying deltas, issuing commands). |

Other single-module entry points:
- `client/avcc_shim.py`, `client/vt_shim.py`, `client/h264_util.py`: small helpers for video decoding quirks.

## Control Package (`client/control`)

| Module | Lines | Purpose | Observations |
| --- | ---:| --- | --- |
| `control_channel_client.py` | 820 | Async state-channel client (websockets, resume tokens, message dispatch). | Monolithic; mixes networking, logging, heartbeats, and callbacks wiring. |
| `state_update_actions.py` | 1 687 | All state intent helpers (dims, camera, layers, volume, HUD). | Largest single file; contains rate limiting, ack handling, viewer mirroring, policy application. |
| `viewer_layer_adapter.py` | 549 | Layer bridge between napari events and `state.update`. | Recently gained mute guard; still handles store seeding, projection, ack reconciliation. |
| `pending_update_store.py` | 308 | Tracks confirmed/pending values and ack reconciliation. | Clean but still lacks provenance metadata. |

## Runtime Package (`client/runtime`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `stream_runtime.py` | 1420 | See above; orchestrates everything.
| `channel_threads.py` | 96 | Thread bootstrappers for control/pixel channels.
| `__init__.py` | 2 | Re-export helper.

## Streaming Package (`client/streaming`)

This package is the legacy coordinator. Many files are thin wrappers or older
abstractions that `stream_runtime.py` partially supersedes.

### Core modules
- `client_stream_loop.py` (4 lines) re-exports `ClientStreamLoop` from the
  nested `client_loop` package.
- `config.py` (98) and `client_loop/client_loop_config.py` (132) parse
  environment for the streaming loop.
- `presenter.py` (430), `renderer.py` (786), `presenter_facade.py` (595): renderer
  and HUD control layer.
- `receiver.py` (121) handles pixel-channel networking.
- `state_store.py`, `state.py` (both 4 lines) are vestigial placeholders that
  redirect to newer structures.

### `client_loop` subpackage (1 478 lines total)
- `camera.py` (453) – camera intent staging and smoothing helpers.
- `loop_lifecycle.py` (208) – start/stop of the old loop threads.
- `loop_state.py` (137) – data container for pipelines, presenters, metrics.
- `pipelines.py`, `warmup.py`, `scheduler.py`, `scheduler_helpers.py`,
  `renderer_fallbacks.py`, `input_helpers.py`, `telemetry.py` – per-feature
  helpers that the legacy `ClientStreamLoop` used. Many are still referenced
  indirectly from `stream_runtime.py` or the presenter.

### Pipelines and decoders
- `pipelines/pyav_pipeline.py`, `pipelines/vt_pipeline.py`, `pipelines/jitter_*`
  describe video decoding queues.
- `decoders/pyav.py`, `decoders/vt.py` implement actual decoding.

### Misc
- `display_loop.py`, `eventloop_monitor.py`, `metrics.py`, `minimal_presenter.py`,
  `vt_frame.py`, `dims_payload.py`, `types.py` – support utilities for playback,
  diagnostics, and data shaping.

### Tests
- `_tests/test_state_store.py`, `_tests/test_state_update_actions.py`, etc.
  exercise individual helpers but do not cover the new runtime wiring.

## Layers Package (`client/layers`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `remote_image_layer.py` | 354 | Implements read-only `Image` layer using remote metadata. Does both projection and data coercion. |
| `remote_data.py` | 339 | Builds `RemoteArray` / `RemoteMultiscale` wrappers around streamed numpy data. |
| `registry.py` | 216 | Thread-safe registry for remote layers; emits snapshots to the bridge. |
| `__init__.py` | 34 | Exposes layer classes for import convenience. |

## Miscellaneous Client Modules

- `client/input.py`, `client/controllers.py`, `client/config.py` – wrappers around
  legacy streaming stack.
- `_tests/test_renderer_vt.py` – smoke test for VT renderer shim.

## Pain Points and Clean-up Targets

1. **Giant modules** – `state_update_actions.py` (1.6k lines),
   `stream_runtime.py` (1.4k), `control_channel_client.py` (820),
   `presenter_facade.py` (595), `proxy_viewer.py` (541). Each mixes multiple
   concerns and is hard to reason about. Splitting by domain (dims/layers/camera)
   or responsibility (networking vs orchestration) is overdue.
2. **Legacy streaming stack** – `client/streaming` still contains the old
   `ClientStreamLoop` machinery. The new runtime reuses pieces piecemeal,
   but there is no clear separation between “current” and “deprecated”.
3. **Projection logic scattered** – layer projection lives inside
   `RemoteImageLayer`; dims projection is in `ProxyViewer`; camera projection is
   mixed into `state_update_actions`. There is no unified interface.
4. **Control flow duplicates** – both `viewer_layer_adapter.py` and
   `streaming/layer_state_bridge.py` exist; only the former is actively used.
   Similarly, there are two `state_store` implementations (legacy streaming vs
   current control).
5. **Event wiring** – `ProxyViewer` manually inspects napari internals
   (`_qt_viewer`, play state) and probes `_state_sender` with `getattr`. This
   should move into a dedicated projection/bridge layer.
6. **Testing gaps** – unit tests focus on legacy streaming helpers; there are no
   integration tests covering `stream_runtime.py` or the new `LayerStateBridge`
   mute behavior.

## Immediate Hygiene Candidates

- Extract dim/camera helpers from `proxy_viewer.py` into dedicated modules that
  can be unit-tested and shared with the runtime.
- Introduce small projection classes (layer, dims, camera) to encapsulate napari
  mutations instead of coding them inside `RemoteImageLayer` or the proxy.
- Trim legacy exports (`client_stream_loop`, `streaming/state.py`) once the new
  runtime owns the code path.
- Replace `getattr`/`try/except` guardrails with assertions so divergences fail
  fast and are easy to track.
- Document the new state-store contract (pending, confirmed, provenance) so
  future work can build on it without reverse engineering 1.6k-line modules.

This inventory gives us a concrete starting point for the upcoming architecture
spec. It functions as both a map of the current code base and a list of cleanup
hotspots.
