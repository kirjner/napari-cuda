# Client Streaming Architecture (Current)

This snapshot captures the current layout of the client streaming stack after
the VT lease migration. It focuses on how data, control, and threading glue the
streaming client together today.

## Refactor Status (2025-09-27)

- FrameLease lifetime fixes removed the VT zero-copy crash; keep `NAPARI_CUDA_VT_GL_SAFE=1` during the current soak.
- InputSender now asserts on Qt invariants (position(), modifiers(), etc.) and only guards subsystem boundaries; no generic "safe call" fallback remains in the event handlers.
- GLRenderer keeps `_safe_call` for OpenGL / VT boundary calls but asserts for internal state (texture caches, lease handling). Guard snapshot lives in `docs/client_refactor_plan.md` and should be kept in sync when the topology changes.

## High-Level Flow

```
                              ┌────────────────────────────────────────┐
                              │ Qt / GUI Thread                        │
                              │ ───────────────────────────────────── │
                              │ • ClientStreamLoop.draw()              │
                              │ • GLRenderer (VT zero-copy + fallback)│
                              │ • InputSender shortcuts & callbacks    │
                              │ • EventLoopMonitor sampling            │
                              └──────────────┬────────────────────────┘
                                             │ FrameLease capsules or RGB numpy
                                             │
┌────────────────┐                 ┌─────────▼─────────┐
│ PixelReceiver  │  H.264 packets  │ SourceMux         │
│ (Receiver thread) ─────────────▶ │ FixedLatencyPresenter buffers       │
└────────┬───────┘                 │ (leases + due times) │
         │                         └─────────┬─────────┘
         │ SubmittedFrame               │ due frames / release_cb
         │                               │
┌────────▼────────┐    FrameLease    ┌────▼────────────┐
│ VT Pipeline     │────────────────▶│ Submitted queue │
│ • Submit worker │                 │ (Priority by due time) │
│ • Drain worker  │◀─────────────── │ (Presenter-owned)     │
└────────────────┘  decoder flush   └──────────────────┘

┌────────────────┐  numpy arrays    ┌──────────────────┐
│ PyAV Pipeline   │───────────────▶│ Presenter buffers │
│ (worker thread) │                │ (PyAV fallback)   │
└────────────────┘                 └──────────────────┘

```

### Control & Metadata Paths

- **State channel thread** (`StateController` + `StateChannel.run`) maintains a
  websocket connection for layer metadata, dimension updates, and intent acks.
  It feeds `ClientStreamLoop` to gate intents until the first `dims.update`
  arrives and to request keyframes when discontinuities are observed.
- **ReceiveController** owns the network thread that drives `PixelReceiver` and
  pushes Annex-B/avcC payloads into the VT/PyAV pipelines.
- **Wake scheduling**: pipelines call `schedule_next_wake()` so the loop only
  repaints when frames are due instead of relying on a fixed timer.

## Thread Model (Current)

| Thread / Timer                | Owner                                   | Responsibilities |
|------------------------------|-----------------------------------------|------------------|
| Qt GUI thread                | `ClientStreamLoop.draw`, `GLRenderer`    | Present frames, drive warmup, bind shortcuts, update metrics |
| Pixel receiver thread        | `ReceiveController`                      | Read pixel WebSocket packets, hand to pipelines |
| VT submit worker             | `VTPipeline.start`                       | Normalize H.264 payloads, push into decoder |
| VT drain worker              | `VTPipeline.start`                       | Pop decoded frames, create `FrameLease`, enqueue presenter submissions |
| PyAV worker                  | `PyAVPipeline.start`                     | Decode software frames and emit numpy RGB payloads |
| State channel thread         | `StateController.start`                  | Maintain control-channel websocket, publish dim/layer intents |
| Metrics timer (Qt)           | `start_metrics_timer` / `start_stats_timer` | Emit periodic Prometheus-style metrics snapshots |
| EventLoopMonitor timer (Qt)  | `EventLoopMonitor`                       | Watch for GUI stalls, trigger kicks |

All VT capsules now travel through `FrameLease`, so Refcount management lives
within the lease (decoder → cache → presenter → renderer). Pipelines shed their
legacy retain/release diagnostics, and the renderer automatically releases
leases once the draw call finishes.

## Component Snapshot

### ClientStreamLoop

- Coordinates lifecycle for pipelines, presenter, renderer, input, telemetry.
- Holds a `ClientLoopState` data bag (`src/napari_cuda/client/streaming/client_loop/loop_state.py`)
  that owns the render queue, wake scheduling handles, timers, metrics facade,
  and VT/PyAV gating flags so helpers no longer poke private attributes.
- Tracks cached VT/PyAV fallbacks through the state bag instead of ad-hoc
  `_last_*` attributes; presenter façade simply pulls from the state.
- Wake scheduling still flows through `WakeProxy`/`CallProxy`, but helpers now
  dereference proxies and timers through `ClientLoopState`, keeping draw/update
  routines pure.

#### Hoisted modules (2025-10)

- `client_loop/warmup.py`: the `WarmupPolicy` helper computes the VT latency
  boost on gate lift, drives the timer-free ramp-down in `draw`, and resets
  presenter latency during shutdown.
- `client_loop/control.py`: `ControlStateContext` captures dims/settings metadata
  while pure helpers normalise payloads, reconcile ACKs, mirror viewer state,
  and expose the public control senders used by input bindings.
- `client_loop/camera.py`: `CameraState` tracks pan/orbit accumulators, zoom
  anchors, and cadence limits; helper functions emit `camera.*` payloads while
  keeping HUD snapshots in sync.
- `client_loop/loop_lifecycle.py`: centralises startup/shutdown wiring for the
  loop (state/pixel controllers, metrics timers, watchdog, event-loop monitor).
  `ClientStreamLoop.start/stop` now delegate to these helpers so lifecycle
  changes stay localised.

`ClientStreamLoop` now focuses on collaborator wiring and high-level control
flow; hot-path logic lives in compact helper modules with targeted tests.

### Presenter / SourceMux

- `FixedLatencyPresenter` buffers `SubmittedFrame` objects for VT and PyAV,
  ordering by due time using stream latency offsets.
- `SourceMux` selects the active pipeline (VT vs PyAV) while still allowing the
  non-active pipeline to warm caches.
- Presenter release callbacks now delegate to `FrameLease.release_presenter`
  so ownership is explicit.
- `PresenterFacade` (scaffolded) is instantiated by `ClientStreamLoop` so the
  upcoming Phase C hoists can take over draw/HUD wiring without modifying the
  canvas again. The façade currently records collaborators and caches dims
  payloads while behaviour stays routed through `StreamingCanvas`.

### Pipelines

- **VTPipeline**: submit worker repacks Annex-B to avcC, primes `VTLiveDecoder`,
  and signals the drain worker to pull decoded frames. Drain worker creates
  `FrameLease` instances, acquires cache/presenter roles, and dispatches
  `SubmittedFrame` entries to the presenter along with the renderer release
  callbacks.
- **PyAVPipeline**: simpler worker that decodes into numpy RGB arrays for the
  fallback path; shares metrics hooks and backlog gating logic with the VT
  pipeline.

### Renderer

- Consumes `(FrameLease, release_cb)` tuples for VT frames or bare numpy arrays
  for PyAV fallback.
- Tracks OpenGL context changes, rebuilds caches, and, when VT zero-copy draws
  succeed, defers lease release until the GPU fence signals (when safe mode is
  enabled).

### Supporting Services

- **StateController / ReceiveController** gate metadata and pixel streams.
- **InputSender** mirrors user input (pointer, wheel, resize) back to the
  server.
- **Telemetry** (metrics + stats timers) expose client health to Prometheus.
- **Layer Control Bridge (planned)** will sit alongside `InputSender` to
  translate Qt layer-control mutations (opacity sliders, contrast ranges,
  colormap pickers, etc.) into remote intents. Rather than letting napari’s
  widgets mutate the `RemoteImageLayer` directly, we’ll override the thin
  layer’s setters to dispatch `image.intent.*` messages through
  `ClientStreamLoop`, then wait for the server’s `layer.update` mirror to land
  before the UI reflects the change. The bridge will target the new
  `server_scene_intents.py` helpers introduced on the server `server-refactor`
  branch (commit `695ccf3c`), so each client-side `image.intent.*` payload has a
  corresponding handler on the coordinator.
  - PresenterFacade already exposes `set_intent_dispatcher`; the bridge will
    register a dispatcher that emits structured `layer.intent.*` messages keyed
    by `RemoteImageLayer._remote_id`.
  - Layer widgets will observe server acks by listening for
    `layer.update`/`scene.update` broadcasts; reentrancy is avoided by gating
    local widget updates until those mirrors land.
  - Contrast and gamma payloads are forwarded in the exact units advertised by
    the server (`LayerSpec.contrast_limits`, `LayerRenderHints.gamma`). Avoid
    introducing local re-normalisation—past bugs stemmed from clients rescaling
    slider values before emitting intents, which desynchronised the server
    mirror. If a new control needs custom mapping, document the conversion next
    to its payload builder and confirm the server helper expects the same
    units.
  - Initial scope covers volume properties (opacity, clim, colormap, sample
    step) and dims/ndisplay toggles; follow-up passes can fold in blending,
    interpolation, and future scene intents.

This diagram acts as the baseline for the ongoing client refactor. Future
iterations can evolve it as modules split out from `ClientStreamLoop` and the
presenter gains a dedicated facade.

## ClientLoopState Data Bag

`ClientStreamLoop` now instantiates `ClientLoopState` from
`src/napari_cuda/client/streaming/client_loop/loop_state.py`. The bag groups
threads, channels, render queue, pipelines, timers, and warmup/gating flags so
helpers manipulate explicit state instead of mutating `self` attributes. The
core fields look like this:

```python
from dataclasses import dataclass, field
from threading import Thread
from queue import Queue

@dataclass
class ClientLoopState:
    threads: list[Thread] = field(default_factory=list)
    state_channel: StateChannel | None = None
    pixel_receiver: PixelReceiver | None = None
    presenter: FixedLatencyPresenter | None = None

    next_due_pending_until: float = 0.0
    in_present: bool = False
    wake_proxy: WakeProxy | None = None
    wake_timer: QtCore.QTimer | None = None
    gui_thread: QtCore.QThread | None = None

    frame_queue: Queue[object] | None = None
    fallbacks: RendererFallbacks | None = None

    vt_pipeline: VTPipeline | None = None
    pyav_pipeline: PyAVPipeline | None = None
    vt_wait_keyframe: bool = False
    pyav_wait_keyframe: bool = False

    metrics: ClientMetrics | None = None
    stats_timer: QtCore.QTimer | None = None
    metrics_timer: QtCore.QTimer | None = None
    warmup_reset_timer: QtCore.QTimer | None = None
    watchdog_timer: QtCore.QTimer | None = None
    evloop_monitor: EventLoopMonitor | None = None

    pending_intents: dict[int, dict[str, object]] = field(default_factory=dict)
    last_dims_seq: int | None = None
    last_dims_payload: dict[str, object] | None = None

    sync: LoopSyncState = field(default_factory=LoopSyncState)
    disco_gated: bool = False
    last_draw_pc: float = 0.0
    last_present_mono: float = 0.0
    warmup_until: float = 0.0
    warmup_extra_active_s: float = 0.0
```

`LoopSyncState` replaces the inline `_SyncState` helper and continues to emit a
rate-limited discontinuity log when sequence numbers jump.

Once the state object exists, helper routines (draw loop, gating logic,
shutdown) can be rewritten to accept `ClientLoopState` explicitly, reducing the
need for cross-cutting `self.` attributes and untangling responsibilities.
