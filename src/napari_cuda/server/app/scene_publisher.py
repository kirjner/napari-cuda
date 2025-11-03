"""Helpers for scene payload publishing and resumable history management."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Iterable, Optional

from napari_cuda.protocol import (
    NOTIFY_LAYERS_TYPE,
    NOTIFY_SCENE_TYPE,
    NOTIFY_STREAM_TYPE,
    NotifyScenePayload,
)
from napari_cuda.protocol.snapshots import SceneSnapshot
from napari_cuda.server.control.control_payload_builder import (
    build_notify_scene_payload,
)
from napari_cuda.server.control.protocol.runtime import state_sequencer
from napari_cuda.server.control.resumable_history_store import ResumableHistoryStore
from napari_cuda.server.control.topics.notify.baseline import orchestrate_connect


def build_scene_payload(
    scene_snapshot: SceneSnapshot,
    ledger_snapshot: Any,
    viewer_settings: Optional[dict[str, Any]] = None,
) -> NotifyScenePayload:
    """Construct a notify.scene payload for the current snapshot."""

    return build_notify_scene_payload(
        scene_snapshot=scene_snapshot,
        ledger_snapshot=ledger_snapshot,
        viewer_settings=viewer_settings,
    )


def cache_scene_history(
    store: Optional[ResumableHistoryStore],
    state_clients: Iterable[Any],
    payload: NotifyScenePayload,
    *,
    now: Callable[[], float] = time.time,
) -> None:
    """Persist a scene payload into the resumable store and clear client sequencers."""

    if store is None:
        return

    timestamp = float(now())
    store.snapshot_envelope(
        NOTIFY_SCENE_TYPE,
        payload=payload.to_dict(),
        timestamp=timestamp,
    )
    store.reset_epoch(NOTIFY_LAYERS_TYPE, timestamp=timestamp)
    store.reset_epoch(NOTIFY_STREAM_TYPE, timestamp=timestamp)

    for ws in list(state_clients):
        state_sequencer(ws, NOTIFY_SCENE_TYPE).clear()
        state_sequencer(ws, NOTIFY_LAYERS_TYPE).clear()
        state_sequencer(ws, NOTIFY_STREAM_TYPE).clear()


def broadcast_state_baseline(
    server: Any,
    state_clients: Iterable[Any],
    *,
    schedule_coro: Callable[[Awaitable[None], str], None],
    reason: str,
) -> None:
    """Schedule baseline broadcasts for every connected state client."""

    for ws in list(state_clients):
        resume_map = getattr(ws, "_napari_cuda_resume_plan", {}) or {}
        coro = orchestrate_connect(server, ws, resume_map)
        schedule_coro(coro, f"baseline-{reason}")


__all__ = ["broadcast_state_baseline", "build_scene_payload", "cache_scene_history"]
