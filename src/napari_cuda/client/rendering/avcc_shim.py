from __future__ import annotations

"""Deprecated: use `napari_cuda.codec.h264` instead.

This module re-exports minimal helpers for backward compatibility.
"""

from typing import Union

from napari_cuda.codec.h264 import (
    avcc_to_annexb as _core_avcc_to_annexb,
    is_annexb as _core_is_annexb,
)

BytesLike = Union[bytes, bytearray, memoryview]


def avcc_to_annexb(avcc: BytesLike) -> bytes:  # pragma: no cover - shim
    return _core_avcc_to_annexb(bytes(avcc), 4)


def normalize_to_annexb(buf: BytesLike) -> tuple[bytes, bool]:  # pragma: no cover - shim
    b = bytes(buf)
    if _core_is_annexb(b):
        return b, False
    return avcc_to_annexb(b), True
