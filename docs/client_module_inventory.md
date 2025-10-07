# Client Module Inventory

This document catalogues the Python modules that implement the streaming client.
It is a snapshot of the current layout (Oct 2025) and highlights the size,
responsibilities, and pain points that surfaced during the recent debugging
pass. It is intentionally descriptive; architectural recommendations will come
in a follow-up design doc.

## Top-Level Entry Points

| Module | Lines | Role | Notes |
| --- | ---:| --- | --- |
| `client/app/launcher.py` | ~120 | CLI bootstrap used by `uv run napari-cuda-client`. | Constructs `ProxyViewer`, `Window`, and hands control to `StreamingCanvas`. |
| `client/app/streaming_canvas.py` | 217 | Qt-facing wrapper that instantiates `ClientStreamLoop`, defers window show, and binds presenter hooks. | Still performs config work (env parsing, latency defaults) that probably belongs in the runtime. |
| `client/app/proxy_viewer.py` | 541 | Napari `ViewerModel` subclass acting as thin client mirror. | Owns slider forwarding, suppress guards, and layer syncing. Contains a lot of logic (axis detection, play-state probing, metadata application). |
| `client/runtime/stream_runtime.py` | 1420 | The main coordinator: wires control channel, presenter facade, decoders, and layer registry. | 1400+ lines with mixed responsibilities (connecting websockets, handling acks, applying deltas, issuing commands). |

Other single-module entry points:
- `client/rendering/avcc_shim.py`, `client/rendering/vt_shim.py`, `client/rendering/h264_util.py`: small helpers for video decoding quirks.

## Control Package (`client/control`)

| Module | Lines | Purpose | Observations |
| --- | ---:| --- | --- |
| `control_channel_client.py` | 820 | Async state-channel client (websockets, resume tokens, message dispatch). | Monolithic; mixes networking, logging, heartbeats, and callbacks wiring. |
| `state_update_actions.py` | 1 687 | All state intent helpers (dims, camera, layers, volume, HUD). | Largest single file; contains rate limiting, ack handling, viewer mirroring, policy application. |
| `control/mirrors/napari_layer_mirror.py`<br>`control/emitters/napari_layer_intent_emitter.py` | ~520 combined | Layer mirror/emitter pair replacing the legacy bridge. | Own registry sync, per-property intents, and ack reconciliation; keeps `ProxyViewer` passive. |
| `client_state_ledger.py` | 308 | Tracks confirmed/pending values and ack reconciliation. | Clean but still lacks provenance metadata. |

## Runtime Package (`client/runtime`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `stream_runtime.py` | 1420 | Orchestrates control channel, presenter facade, decoders, and layer registry. |
| `channel_threads.py` | 96 | Starts/stops state and pixel threads. |
| `config.py` | 98 | Normalizes environment knobs into `ClientConfig`. |
| `receiver.py` | 121 | Pixel websocket client with reconnect/backoff policy. |
| `eventloop_monitor.py` | 85 | Qt event-loop watchdog used by the loop and presenter. |
| `input.py` | 308 | Qt event filter translating pointer/keyboard events into intents. |

### Client loop helpers (`client/runtime/client_loop`)

- `camera.py` (453): camera intent staging and smoothing helpers.
- `client_loop_config.py` (132): parses loop-specific env settings.
- `input_helpers.py` (74): attaches the input sender to the Qt canvas.
- `loop_lifecycle.py` (208): start/stop wiring for state/pixel threads and timers.
- `loop_state.py` (137): data container for pipelines, presenter, metrics, and threads.
- `pipelines.py`, `scheduler.py`, `scheduler_helpers.py`, `renderer_fallbacks.py`,
  `telemetry.py`, `warmup.py`, `qt_bridge.py`: per-feature helpers reused by the
  new runtime (wake scheduling, backlog handling, telemetry, Qt adapters).

## Rendering Package (`client/rendering`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `presenter.py` | 430 | Fixed-latency presenter coordinating frame selection. |
| `presenter_facade.py` | 476 | Facade consumed by the runtime; wires presenter, renderer, HUD. |
| `renderer.py` | 786 | OpenGL renderer and VT texture lifecycle management. |
| `minimal_presenter.py` | 326 | Headless/minimal presenter used in diagnostics. |
| `display_loop.py` | 138 | Optional Qt display loop for steady frame pacing. |
| `metrics.py` | 199 | Client metrics collectors and reporting helpers. |
| `types.py` | 52 | Shared enums/data classes (`Source`, `SubmittedFrame`, ...). |
| `vt_frame.py` | 134 | VT buffer retain/release helpers exposed to the renderer. |

### Pipelines and decoders

- `pipelines/pyav_pipeline.py`, `pipelines/vt_pipeline.py`, `pipelines/jitter_*` describe the
  decode/present queues.
- `decoders/pyav.py`, `decoders/vt.py` implement actual decoding.

## Data Package (`client/data`)

| Module | Lines | Purpose |
| --- | ---:| --- |
| `remote_image_layer.py` | 354 | Implements read-only `Image` layer using remote metadata. Does both projection and data coercion. |
| `remote_data.py` | 339 | Builds `RemoteArray` / `RemoteMultiscale` wrappers around streamed numpy data. |
| `registry.py` | 216 | Thread-safe registry for remote layers; emits snapshots to the bridge. |
| `__init__.py` | 34 | Exposes layer classes for import convenience. |

## Miscellaneous Client Modules

- `client/runtime/input.py`, `client/runtime/channel_threads.py`, `client/runtime/config.py` – runtime glue for input handling, thread 
  startup, and environment normalization.
- `_tests/test_renderer_vt.py` – smoke test for VT renderer shim.

## Pain Points and Clean-up Targets

1. **Giant modules** – `state_update_actions.py` (1.6k lines),
   `stream_runtime.py` (1.4k), `control_channel_client.py` (820),
   `presenter_facade.py` (595), `proxy_viewer.py` (541). Each mixes multiple
   concerns and is hard to reason about. Splitting by domain (dims/layers/camera)
   or responsibility (networking vs orchestration) is overdue.
2. **Legacy loop scaffolding** – much of the runtime still leans on helpers
   that originated in the old loop (e.g., wake scheduling, warmup, telemetry).
   They now live under `client/runtime/client_loop`, but responsibilities overlap
   with `stream_runtime.py` and should be trimmed or merged.
3. **Projection logic scattered** – layer projection lives inside
   `RemoteImageLayer`; dims projection is in `ProxyViewer`; camera projection is
   mixed into `state_update_actions`. There is no unified interface.
4. **Layer projection alignment** – `NapariLayerMirror` and
   `NapariLayerIntentEmitter` now own registry sync and per-property intents. We
   should keep `ProxyViewer` passive by routing all layer mutations through these
   components.
5. **Event wiring** – `ProxyViewer` manually inspects napari internals
   (`_qt_viewer`, play state) and probes `_state_sender` with `getattr`. This
   should move into a dedicated projection/bridge layer.
6. **Testing gaps** – unit tests now cover the dims/layer/camera mirrors,
   but we still lack integration coverage for `stream_runtime.py` wiring the
   new emitters/mirrors end-to-end.

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
