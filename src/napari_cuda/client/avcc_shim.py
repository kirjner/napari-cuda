from __future__ import annotations

"""
AVCC → Annex B shim for PyAV/FFmpeg.

Server emits AVCC (length‑prefixed NAL units) for hardware decoder
compatibility. PyAV typically expects Annex B (start‑code delimited)
when feeding raw H.264 access units. This shim converts a single AVCC
access unit to Annex B. If the buffer already looks like Annex B,
it is returned unmodified.
"""

from typing import Union, Tuple

BytesLike = Union[bytes, bytearray, memoryview]


def _is_annexb(buf: bytes) -> bool:
    # Common Annex B start codes: 0x000001 or 0x00000001
    return buf.startswith(b"\x00\x00\x01") or buf.startswith(b"\x00\x00\x00\x01")


def avcc_to_annexb(avcc: BytesLike) -> bytes:
    """Convert one AVCC access unit (length‑prefixed) to Annex B (start codes).

    Input must be a sequence: [len][nal][len][nal]... with 4‑byte big‑endian
    lengths. Returns a byte string with 4‑byte start codes (0x00000001)
    before each NAL unit. On malformed input, returns the original bytes.
    """
    b = bytes(avcc)
    out = bytearray()
    i = 0
    l = len(b)
    while i + 4 <= l:
        ln = int.from_bytes(b[i:i+4], 'big')
        i += 4
        if ln <= 0 or i + ln > l:
            return b
        out += b"\x00\x00\x00\x01"
        out += b[i:i+ln]
        i += ln
    return bytes(out if out else b)


def normalize_to_annexb(buf: BytesLike) -> Tuple[bytes, bool]:
    """Return (annexb_bytes, converted_flag).

    - If `buf` already looks like Annex B, returns (buf_bytes, False).
    - Otherwise treats it as AVCC and returns (converted_bytes, True).
    """
    b = bytes(buf)
    if _is_annexb(b):
        return b, False
    return avcc_to_annexb(b), True

