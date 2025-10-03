"""Shared helpers for axis label canonicalisation."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional


def default_axis_labels(ndim: int) -> List[str]:
    """Return canonical axis labels for a given dimensionality."""

    if ndim <= 0:
        return []
    if ndim == 3:
        return ["z", "y", "x"]
    if ndim == 2:
        return ["y", "x"]
    if ndim == 1:
        return ["d0"]
    return [f"d{i}" for i in range(ndim)]


def normalize_axis_labels(labels: Optional[Iterable[Any]], ndim: int) -> List[str]:
    """Ensure axis labels are non-empty and cover ``ndim`` dimensions."""

    cleaned: List[str] = []
    if labels is not None:
        for label in labels:
            text = str(label).strip()
            if text:
                cleaned.append(text)

    if ndim <= 0:
        return []

    if cleaned:
        placeholder = True
        for text in cleaned[:ndim]:
            lowered = text.lower()
            if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
                continue
            if re.match(r"^axis[\s_-]*[-+]?\d+$", lowered):
                continue
            placeholder = False
            break
        if not placeholder:
            if len(cleaned) == ndim:
                return cleaned
            if len(cleaned) > ndim:
                return cleaned[:ndim]

    defaults = default_axis_labels(ndim)
    if len(defaults) >= ndim:
        return defaults[:ndim]

    result = list(defaults)
    for idx in range(len(result), ndim):
        result.append(f"d{idx}")
    return result
