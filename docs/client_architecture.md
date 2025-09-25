# Client Streaming Architecture (Current)

This snapshot captures the current layout of the client streaming stack after
the VT lease migration. It focuses on how data, control, and threading glue the
streaming client together today.

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
- Maintains `_frame_q` (render queue), `_last_vt_lease` (cached lease for VT
  fallback), and `_last_pyav_frame` (software fallback).
- Handles gating on keyframes via `_vt_wait_keyframe` and schedules redraws
  through `WakeProxy`/`CallProxy` onto the GUI thread.
- **Future direction:** replace the growing set of ad hoc attributes with a
  `ClientLoopState` dataclass (see sketch below) so routines operate on an
  explicit data bag instead of `self`. This aligns with the Casey Muratori
  “data + procedures” style and will make it easier to unit-test helpers.

### Presenter / SourceMux

- `FixedLatencyPresenter` buffers `SubmittedFrame` objects for VT and PyAV,
  ordering by due time using stream latency offsets.
- `SourceMux` selects the active pipeline (VT vs PyAV) while still allowing the
  non-active pipeline to warm caches.
- Presenter release callbacks now delegate to `FrameLease.release_presenter`
  so ownership is explicit.

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

This diagram acts as the baseline for the ongoing client refactor. Future
iterations can evolve it as modules split out from `ClientStreamLoop` and the
presenter gains a dedicated facade.

## `ClientLoopState` Sketch (Future Work)

```python
from dataclasses import dataclass, field
from typing import Optional, Deque
from threading import Thread

@dataclass
class ClientLoopState:
    # Threads and channels
    threads: list[Thread] = field(default_factory=list)
    state_channel: "StateChannel | None" = None
    pixel_receiver: "PixelReceiver | None" = None
    presenter: "FixedLatencyPresenter | None" = None

    # Scheduling / wake coordination
    next_due_pending_until: float = 0.0
    in_present: bool = False
    wake_proxy: "WakeProxy | None" = None

    # Renderer queue + fallbacks
    frame_queue: "queue.Queue[object] | None" = None
    fallbacks: "RendererFallbacks | None" = None

    # Smoke / pipelines
    smoke: "SmokePipeline | None" = None
    vt_pipeline: "VTPipeline | None" = None
    pyav_pipeline: "PyAVPipeline | None" = None

    # Metrics / telemetry
    metrics_timer: "QtCore.QTimer | None" = None
    stats_timer: "QtCore.QTimer | None" = None
    metrics: "Metrics | None" = None

    # Input + shortcuts
    input_sender: "InputSender | None" = None
    shortcuts: list[object] = field(default_factory=list)

    # Stream continuity / gating
    vt_wait_keyframe: bool = True
    pyro_wait_keyframe: bool = False
    sync: _SyncState = field(default_factory=_SyncState)

    # Misc client telemetry
    last_draw_pc: float = 0.0
    last_present_mono: float = 0.0
    warmup_until: float = 0.0

```

Once the state object exists, helper routines (draw loop, gating logic,
shutdown) can be rewritten to accept `ClientLoopState` explicitly, reducing the
need for cross-cutting `self.` attributes and untangling responsibilities.
