from __future__ import annotations

"""
Minimal helpers to handle backlog and enqueue in smoke paths.
"""

from typing import Iterable, Optional

import logging
from napari_cuda.client.streaming.types import Source

logger = logging.getLogger(__name__)


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
        qsz = int(vt_q.qsize())
        if qsz >= max(2, int(backlog_trigger) - 1):
            vt_q.clear()
            try:
                presenter.clear(Source.VT)
            except Exception:
                logger.debug("submit_vt: presenter.clear failed", exc_info=True)
        vt_q.enqueue(payload, ts)
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
        qsz = int(pyav_q.qsize())
        if qsz >= max(2, int(backlog_trigger) - 1):
            pyav_q.clear()
            try:
                presenter.clear(Source.PYAV)
            except Exception:
                logger.debug("submit_pyav: presenter.clear failed", exc_info=True)
        pyav_q.enqueue(payload, ts)
        submitted += 1
    return submitted
