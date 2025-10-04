"""Client stream loop state aggregates."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING

logger = logging.getLogger("napari_cuda.client.runtime.stream_runtime")


@dataclass
class LoopSyncState:
    """Track pixel sequence continuity for the streaming loop."""

    last_seq: int | None = None
    last_disco_log: float = 0.0
    keyframes_seen: int = 0

    def update_and_check(self, cur: int) -> bool:
        """Return ``True`` when a sequence discontinuity is detected."""

        if self.last_seq is None:
            self.last_seq = int(cur)
            return False

        expected = (int(self.last_seq) + 1) & 0xFFFFFFFF
        if int(cur) != expected:
            now = time.time()
            if (now - self.last_disco_log) > 0.2:
                logger.warning(
                    "Pixel stream discontinuity: expected=%d got=%d; gating until keyframe",
                    expected,
                    int(cur),
                )
                self.last_disco_log = now
            self.last_seq = int(cur)
            return True

        self.last_seq = int(cur)
        return False

    def reset_sequence(self) -> None:
        """Forget the last seen sequence so the next frame sets the baseline."""

        self.last_seq = None
        self.last_disco_log = 0.0


@dataclass
class ClientLoopState:
    """Aggregate mutable runtime state for ``ClientStreamLoop``."""

    # Thread coordination
    threads: list[Thread] = field(default_factory=list)
    state_thread: Thread | None = None
    pixel_thread: Thread | None = None

    # Channels and receivers
    state_channel: "StateChannel | None" = None
    pixel_receiver: "PixelReceiver | None" = None
    presenter: "FixedLatencyPresenter | None" = None

    # Scheduling
    next_due_pending_until: float = 0.0
    in_present: bool = False
    wake_proxy: "WakeProxy | None" = None
    wake_timer: "QtCore.QTimer | None" = None
    gui_thread: "QtCore.QThread | None" = None

    # Renderer queue and fallbacks
    frame_queue: "queue.Queue[object] | None" = None
    fallbacks: "RendererFallbacks | None" = None

    # Pipelines
    vt_pipeline: "VTPipeline | None" = None
    pyav_pipeline: "PyAVPipeline | None" = None
    vt_wait_keyframe: bool = False
    pyav_wait_keyframe: bool = False

    # Telemetry and timers
    metrics: "ClientMetrics | None" = None
    stats_timer: "QtCore.QTimer | None" = None
    metrics_timer: "QtCore.QTimer | None" = None
    warmup_reset_timer: "QtCore.QTimer | None" = None
    watchdog_timer: "QtCore.QTimer | None" = None
    evloop_monitor: "EventLoopMonitor | None" = None

    # Input handling
    input_sender: "InputSender | None" = None
    shortcuts: list[object] = field(default_factory=list)

    # Control channel handling
    control_state: "ControlStateContext | None" = None

    # Camera handling
    camera: "CameraState | None" = None

    # Intent caches
    pending_intents: dict[int, dict[str, object]] = field(default_factory=dict)
    last_dims_payload: dict[str, object] | None = None
    state_session_metadata: "SessionMetadata | None" = None

    # Stream continuity
    sync: LoopSyncState = field(default_factory=LoopSyncState)
    disco_gated: bool = False
    last_key_logged: int | None = None

    # Draw timing diagnostics
    last_draw_pc: float = 0.0
    last_present_mono: float = 0.0

    # Warmup behaviour
    warmup_policy: "WarmupPolicy | None" = None
    warmup_until: float = 0.0
    warmup_extra_active_s: float = 0.0


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import queue

    from qtpy import QtCore

    from napari_cuda.client.runtime.client_loop.renderer_fallbacks import RendererFallbacks
    from napari_cuda.client.runtime.client_loop.scheduler import WakeProxy
    from napari_cuda.client.runtime.client_loop.pipelines import VTPipeline, PyAVPipeline
    from napari_cuda.client.runtime.eventloop_monitor import EventLoopMonitor
    from napari_cuda.client.runtime.input import InputSender
    from napari_cuda.client.rendering.metrics import ClientMetrics
    from napari_cuda.client.rendering.presenter import FixedLatencyPresenter
    from napari_cuda.client.runtime.receiver import PixelReceiver
    from napari_cuda.client.control.control_channel_client import SessionMetadata, StateChannel
    from .warmup import WarmupPolicy
    from .control import ControlStateContext
    from .camera import CameraState
