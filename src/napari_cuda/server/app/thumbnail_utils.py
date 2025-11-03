"""State helpers for server-side thumbnail scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from napari_cuda.server.scene import RenderLedgerSnapshot

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
    state.pending_layer_id = None
    # Preserve force_refresh so manual requests retry on next tick


def reset_thumbnail_state(state: ThumbnailState) -> None:
    state.pending_signature = None
    state.last_signature = None
    state.pending_layer_id = None
    state.force_refresh = False
