"""Thumbnail state helpers and capture pipeline functions for the server.

TODO(#thumbnail-followup): Build an integration test that exercises the full
render_tick → ledger write → notify.layers path to ensure client thumbnails
stay in sync after dims/level changes and dataset switches.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from napari_cuda.server.runtime.ipc.mailboxes.worker_intent import ThumbnailIntent
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.state_ledger import ServerStateLedger

logger = logging.getLogger(__name__)

RenderSignature = Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, int], ...]]


@dataclass
class ThumbnailState:
    pending_signature: Optional[RenderSignature] = None
    last_signature: Optional[RenderSignature] = None
    pending_layer_id: Optional[str] = None
    force_refresh: bool = False


def build_render_signature(snapshot: RenderLedgerSnapshot) -> RenderSignature:
    op_seq = int(snapshot.op_seq or 0)
    level = int(snapshot.current_level or 0)
    current_step = tuple(int(v) for v in (snapshot.current_step or ()))
    displayed = tuple(int(v) for v in (snapshot.displayed or ()))
    camera_versions = snapshot.camera_versions or {}
    camera_items = tuple(sorted((str(k), int(v)) for k, v in camera_versions.items()))
    return (op_seq, level, current_step, displayed, camera_items)


def flag_render_change(
    state: ThumbnailState,
    signature: RenderSignature,
    layer_id: Optional[str],
) -> bool:
    if not layer_id:
        return False
    if state.last_signature == signature and not state.force_refresh:
        return False
    state.pending_signature = signature
    state.pending_layer_id = layer_id
    return True


def request_thumbnail_refresh(state: ThumbnailState, layer_id: Optional[str]) -> bool:
    if not layer_id:
        return False
    state.pending_layer_id = layer_id
    state.pending_signature = None
    state.force_refresh = True
    return True


def record_thumbnail_success(state: ThumbnailState, signature: Optional[RenderSignature]) -> None:
    if signature is not None:
        state.last_signature = signature
    state.pending_signature = None
    state.pending_layer_id = None
    state.force_refresh = False


def record_thumbnail_failure(state: ThumbnailState) -> None:
    state.pending_signature = None
    if not state.force_refresh:
        state.pending_layer_id = None
    # Preserve force_refresh so manual requests retry on next tick


def reset_thumbnail_state(state: ThumbnailState) -> None:
    state.pending_signature = None
    state.last_signature = None
    state.pending_layer_id = None
    state.force_refresh = False


def queue_thumbnail_refresh(
    state: ThumbnailState,
    layer_id: Optional[str],
    *,
    mark_render_tick: Optional[Callable[[], None]] = None,
) -> bool:
    """Track a manual refresh request and optionally nudge the worker."""

    if not request_thumbnail_refresh(state, layer_id):
        return False

    if mark_render_tick is not None:
        mark_render_tick()
    return True


def handle_render_tick(
    state: ThumbnailState,
    snapshot: RenderLedgerSnapshot,
    *,
    worker_ready: bool,
    fetch_thumbnail: Callable[[str, RenderSignature], Optional[np.ndarray]],
    request_render_tick: Optional[Callable[[], None]] = None,
) -> Optional[ThumbnailIntent]:
    """Return a ThumbnailIntent when a pending refresh is satisfied."""

    target_layer = state.pending_layer_id
    if not target_layer:
        return None

    signature = build_render_signature(snapshot)
    state.pending_signature = signature

    if not worker_ready:
        record_thumbnail_failure(state)
        if request_render_tick is not None:
            request_render_tick()
        return None

    array = fetch_thumbnail(target_layer, signature)
    if array is None:
        record_thumbnail_failure(state)
        if request_render_tick is not None:
            request_render_tick()
        return None

    arr = np.asarray(array)
    if arr.size == 0:
        record_thumbnail_failure(state)
        if request_render_tick is not None:
            request_render_tick()
        return None

    return ThumbnailIntent(
        layer_id=str(target_layer),
        signature=signature,
        array=arr.copy(),
    )


def ingest_thumbnail_payload(
    state: ThumbnailState,
    payload: ThumbnailIntent,
    ledger: ServerStateLedger,
    *,
    origin: str = "server.thumbnail",
    now: Callable[[], float] = time.time,
) -> bool:
    """Normalise a thumbnail payload, persist it to the ledger, and update state.

    Returns True when the ledger is updated, False if the payload was ignored
    (e.g., identical to the cached metadata).
    """

    array = np.asarray(payload.array)
    if array.size == 0:
        record_thumbnail_failure(state)
        return False

    arr = _prepare_thumbnail(array)
    timestamp = float(now())
    metadata_update = {
        "thumbnail": arr.tolist(),
        "thumbnail_shape": list(arr.shape),
        "thumbnail_dtype": "float32",
        "thumbnail_version": timestamp,
    }
    thumbnail_payload = {
        "array": arr.tolist(),
        "shape": list(arr.shape),
        "dtype": "float32",
        "version": timestamp,
    }

    layer_id = payload.layer_id
    signature = payload.signature

    existing = ledger.get("layer", layer_id, "metadata")
    if existing is not None and isinstance(existing.value, dict):
        merged = dict(existing.value)
        prev = merged.get("thumbnail")
        if prev is not None:
            prev_arr = np.asarray(prev, dtype=np.float32)
            if prev_arr.shape == arr.shape and np.allclose(prev_arr, arr, atol=1e-3):
                record_thumbnail_success(state, signature)
                return False
        merged.update(metadata_update)
        metadata = merged
    else:
        metadata = metadata_update

    ledger.record_confirmed(
        "layer",
        layer_id,
        "metadata",
        metadata,
        origin=origin,
    )
    ledger.record_confirmed(
        "layer",
        layer_id,
        "thumbnail",
        thumbnail_payload,
        origin=origin,
    )
    record_thumbnail_success(state, signature)
    return True


def _prepare_thumbnail(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size > 0:
            max_val = float(np.nanmax(arr))
            if max_val > 0:
                arr /= max_val
    np.clip(arr, 0.0, 1.0, out=arr)
    return np.flip(arr, axis=0)


__all__ = [
    "RenderSignature",
    "ThumbnailState",
    "build_render_signature",
    "flag_render_change",
    "handle_render_tick",
    "ingest_thumbnail_payload",
    "queue_thumbnail_refresh",
    "record_thumbnail_failure",
    "record_thumbnail_success",
    "request_thumbnail_refresh",
    "reset_thumbnail_state",
]
