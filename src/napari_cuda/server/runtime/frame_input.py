"""
Frame input builder: merge LatestIntent onto the applied baseline snapshot.

No classes, no try/except. Called from the render thread per tick.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

from napari_cuda.server.control.latest_intent import get_all as latest_get_all
from napari_cuda.server.runtime.scene_ingest import build_render_snapshot, RenderSceneSnapshot


def _clamp_step(
    step: Sequence[int],
    level_shapes: Sequence[Sequence[int]] | None,
    current_level: Optional[int],
) -> Tuple[int, ...]:
    values = [int(v) for v in step]
    if level_shapes is None or current_level is None:
        return tuple(int(v) for v in values)
    li = int(current_level)
    if li < 0 or li >= len(level_shapes):
        return tuple(int(v) for v in values)
    shape = level_shapes[li]
    for i in range(min(len(values), len(shape))):
        size = int(shape[i])
        if size > 0:
            if values[i] < 0:
                values[i] = 0
            elif values[i] >= size:
                values[i] = size - 1
    return tuple(int(v) for v in values)


def build_frame_input(server: Any) -> tuple[RenderSceneSnapshot, Dict[str, int], Optional[int]]:
    """Construct the frame snapshot and return desired seqs and view target.

    Returns
    -------
    snapshot : RenderSceneSnapshot
        Baseline ledger snapshot with latest desired dims merged (clamped).
    desired_seqs : dict
        Mapping with keys 'dims' and 'view' holding max desired seq per scope (or -1).
    desired_ndisplay : Optional[int]
        Desired ndisplay (2 or 3) if provided via LatestIntent; otherwise None.
    """
    with server._state_lock:
        snapshot = build_render_snapshot(server._state_ledger, server._scene)
        ledger_snap = server._state_ledger.snapshot()

        level_shapes_entry = ledger_snap.get(("multiscale", "main", "level_shapes"))
        current_level_entry = ledger_snap.get(("multiscale", "main", "level"))
        level_shapes = None if level_shapes_entry is None else level_shapes_entry.value
        current_level = None if current_level_entry is None else int(current_level_entry.value)

        # dims latest-wins
        desired_dims_seq = -1
        desired_step: Optional[Tuple[int, ...]] = None
        latest_dims = latest_get_all("dims")
        for _k, (seq, value) in latest_dims.items():
            s = int(seq)
            if s >= desired_dims_seq and isinstance(value, (list, tuple)):
                candidate = tuple(int(v) for v in value)
                desired_step = candidate
                desired_dims_seq = s
        if desired_step is not None:
            clamped = _clamp_step(desired_step, level_shapes, current_level)
            if snapshot.current_step != clamped:
                snapshot = replace(snapshot, current_step=clamped)

        # view latest-wins
        desired_view_seq = -1
        desired_ndisplay: Optional[int] = None
        latest_view = latest_get_all("view")
        for _k, (seq, value) in latest_view.items():
            s = int(seq)
            if s >= desired_view_seq:
                if isinstance(value, (int, float)):
                    desired_view_seq = s
                    desired_ndisplay = 3 if int(value) >= 3 else 2

        desired_seqs = {"dims": int(desired_dims_seq), "view": int(desired_view_seq)}
        return (snapshot, desired_seqs, desired_ndisplay)


__all__ = ["build_frame_input"]
