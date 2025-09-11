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
    """Submit to PyAV path supporting either a queue-like object or a pipeline.

    - If `pyav_q` has `enqueue()`/`clear()`/`qsize()`, treat it as a pipeline.
    - Otherwise, assume a legacy Queue with `put_nowait()`/`get_nowait()`/`qsize()`.
    """
    is_pipeline = hasattr(pyav_q, 'enqueue')
    submitted = 0
    for au in aus:
        payload = getattr(au, 'payload', None)
        if not payload:
            continue
        ts = getattr(au, 'pts', None)
        if ts is None:
            ts = ts_fallback
        try:
            qsz = int(pyav_q.qsize())
        except Exception:
            qsz = 0
        if qsz >= max(2, int(backlog_trigger) - 1):
            try:
                if is_pipeline and hasattr(pyav_q, 'clear'):
                    pyav_q.clear()
                else:
                    while pyav_q.qsize() > 0:
                        _ = pyav_q.get_nowait()
            except Exception:
                pass
            presenter.clear(Source.PYAV)
        try:
            if is_pipeline:
                pyav_q.enqueue(payload, ts)
            else:
                pyav_q.put_nowait((payload, ts))
            submitted += 1
        except Exception:
            # Drop on failure
            pass
    return submitted
