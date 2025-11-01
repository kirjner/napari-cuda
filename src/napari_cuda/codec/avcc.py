from __future__ import annotations

"""
AVCC helpers: thin, stable import surface for H.264 AU handling.

This module re-exports the AVCC/AnnexB utilities from `codec.h264` to provide
an explicit place to depend on for bitstream normalization without pulling in
unrelated helpers. Future refactors can move implementations without breaking
callers that import from `napari_cuda.codec.avcc`.
"""

from dataclasses import dataclass
from typing import Optional

from .h264 import (
    annexb_to_avcc,
    avcc_to_annexb,
    build_avcc,
    find_sps_pps,
    is_annexb,
    normalize_to_annexb,
    parse_avcc,
    split_annexb,
    split_avcc_by_len,
)

__all__ = [
    "AccessUnit",
    "annexb_to_avcc",
    "avcc_to_annexb",
    "build_avcc",
    "find_sps_pps",
    "is_annexb",
    "normalize_to_annexb",
    "parse_avcc",
    "split_annexb",
    "split_avcc_by_len",
]


@dataclass(frozen=True)
class AccessUnit:
    """A single encoded access unit (frame) in AVCC form.

    - payload: AVCC-delimited bytes (NAL-length-prefixed), suitable for VT submission.
    - is_keyframe: True if AU contains an IDR or equivalent (used for decoder gating).
    - pts: Presentation timestamp in seconds (optional for demos/tools).
    """

    payload: bytes
    is_keyframe: bool
    pts: Optional[float] = None
