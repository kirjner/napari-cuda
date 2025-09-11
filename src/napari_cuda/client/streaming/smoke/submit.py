from __future__ import annotations

"""
Minimal helpers to handle backlog and enqueue in smoke paths.
"""

from typing import Iterable, Optional

from napari_cuda.client.streaming.types import Source


def submit_vt(
    aus: Iterable["object"],
    vt_q: "object",
    presenter: "object",
    backlog_trigger: int,
    ts_fallback: Optional[float],
) -> int:
    """Submit VT AUs (payload, ts) with simple backlog handling.

    Returns number of submitted items.
    """
    submitted = 0
    for au in aus:
        payload = getattr(au, 'payload', None)
        if not payload:
            continue
        ts = getattr(au, 'pts', None)
        if ts is None:
            ts = ts_fallback
        if vt_q.qsize() >= max(2, int(backlog_trigger) - 1):
            while vt_q.qsize() > 0:
                _ = vt_q.get_nowait()
            presenter.clear(Source.VT)
        vt_q.put_nowait((payload, ts))
        submitted += 1
    return submitted


def submit_pyav(
    aus: Iterable["object"],
    pyav_q: "object",
    presenter: "object",
    backlog_trigger: int,
    ts_fallback: Optional[float],
) -> int:
    submitted = 0
    for au in aus:
        payload = getattr(au, 'payload', None)
        if not payload:
            continue
        ts = getattr(au, 'pts', None)
        if ts is None:
            ts = ts_fallback
        if pyav_q.qsize() >= max(2, int(backlog_trigger) - 1):
            while pyav_q.qsize() > 0:
                _ = pyav_q.get_nowait()
            presenter.clear(Source.PYAV)
        pyav_q.put_nowait((payload, ts))
        submitted += 1
    return submitted

