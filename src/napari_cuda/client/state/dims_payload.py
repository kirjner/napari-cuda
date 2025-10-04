from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any


logger = logging.getLogger(__name__)


def normalize_meta(meta_raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with backwards-compatible aliases resolved."""
    meta = dict(meta_raw)
    axes = meta_raw.get("axes")
    if axes is not None and "axis_labels" not in meta:
        labels: list[str] = []
        order: list[int] = []
        if isinstance(axes, Mapping):
            label = axes.get("label") or axes.get("name")
            if label is not None:
                labels.append(str(label))
                order.append(_to_int(axes.get("index"), default=0))
        elif isinstance(axes, Sequence):
            for idx, entry in enumerate(axes):
                if isinstance(entry, Mapping):
                    label = entry.get("label") or entry.get("name") or entry.get("id")
                    labels.append(str(label) if label is not None else str(idx))
                    order.append(_to_int(entry.get("index"), default=idx))
                else:
                    labels.append(str(entry))
                    order.append(idx)
        if labels:
            meta["axis_labels"] = labels
        if order and "order" not in meta:
            meta["order"] = order
    displayed = meta_raw.get("displayed_axes")
    if displayed is not None and "displayed" not in meta:
        meta["displayed"] = displayed
    if "range" not in meta and "ranges" in meta_raw:
        meta["range"] = meta_raw["ranges"]
    return meta


def inflate_current_step(cur: object, meta: Mapping[str, Any]) -> object:
    """Pad and clamp ``current_step`` to match ndim and range in ``meta``."""
    if not _is_sequence(cur):
        return cur
    values = list(cur)
    ndim = _to_int(meta.get("ndim"), default=len(values))
    if ndim <= 0 or len(values) == ndim:
        return cur
    inflated = [0] * ndim
    order = meta.get("order")
    indices: list[int] | None = None
    if isinstance(order, Sequence) and len(order) == ndim:
        indices = [_to_int(val, default=i) for i, val in enumerate(order)]
    if indices is None:
        limit = min(ndim, len(values))
        for idx in range(limit):
            inflated[idx] = _to_int(values[idx], default=0)
    else:
        for src_idx, axis in enumerate(indices[: len(values)]):
            if 0 <= axis < ndim:
                inflated[axis] = _to_int(values[src_idx], default=0)
    rng = meta.get("range")
    if isinstance(rng, Sequence) and len(rng) >= ndim:
        for axis in range(ndim):
            bounds = rng[axis]
            if isinstance(bounds, Sequence) and len(bounds) >= 2:
                lo = _to_int(bounds[0], default=inflated[axis])
                hi = _to_int(bounds[1], default=inflated[axis])
                if lo > hi:
                    lo, hi = hi, lo
                inflated[axis] = min(max(inflated[axis], lo), hi)
    return inflated


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("_to_int fallback for %r", value)
        return default

def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

