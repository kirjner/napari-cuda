from __future__ import annotations

"""
H.264 helpers for client-side parsing and conversions.

- Splitters for Annex B (start-code) and AVCC (length-prefixed) AUs
- AnnexB ↔ AVCC conversions for single access units
- avcC (AVCDecoderConfigurationRecord) builder from SPS/PPS
"""

from typing import List, Sequence


def is_annexb(buf: bytes) -> bool:
    if not buf:
        return False
    # Be tolerant of leading zeros/padding
    i = 0
    n = len(buf)
    limit = min(n, 32)
    while i < limit and buf[i] == 0:
        i += 1
    if i + 3 <= n and buf[i : i + 3] == b"\x00\x00\x01":
        return True
    if i + 4 <= n and buf[i : i + 4] == b"\x00\x00\x00\x01":
        return True
    return False


def split_annexb(data: bytes) -> List[bytes]:
    """Split an Annex B access unit into raw NAL payloads (without start codes)."""
    out: List[bytes] = []
    i = 0
    n = len(data)
    idx: List[int] = []
    while i + 3 <= n:
        if data[i : i + 3] == b"\x00\x00\x01":
            idx.append(i)
            i += 3
        elif i + 4 <= n and data[i : i + 4] == b"\x00\x00\x00\x01":
            idx.append(i)
            i += 4
        else:
            i += 1
    idx.append(n)
    for a, b in zip(idx, idx[1:]):
        j = a
        # Skip zeros and the start code itself
        while j < b and data[j] == 0:
            j += 1
        if j + 3 <= b and data[j : j + 3] == b"\x00\x00\x01":
            j += 3
        elif j + 4 <= b and data[j : j + 4] == b"\x00\x00\x00\x01":
            j += 4
        nal = data[j:b]
        if nal:
            out.append(nal)
    return out


def split_avcc_by_len(data: bytes, nal_len_size: int = 4) -> List[bytes]:
    """Split an AVCC AU (length-prefixed) into raw NAL payloads."""
    out: List[bytes] = []
    i = 0
    n = len(data)
    try:
        while i + nal_len_size <= n:
            ln = int.from_bytes(data[i : i + nal_len_size], "big", signed=False)
            i += nal_len_size
            if ln <= 0 or i + ln > n:
                break
            out.append(data[i : i + ln])
            i += ln
    except Exception:
        out = []
    return out


def annexb_to_avcc(data: bytes) -> bytes:
    """Convert a single Annex B access unit to AVCC (4-byte lengths).

    If `data` already looks like AVCC, returns it unmodified.
    """
    if not is_annexb(data):
        return data
    nals = split_annexb(data)
    out = bytearray()
    for n in nals:
        out.extend(len(n).to_bytes(4, "big"))
        out.extend(n)
    return bytes(out)


def avcc_to_annexb(avcc: bytes, nal_len_size: int = 4) -> bytes:
    """Convert a single AVCC AU to Annex B (4-byte start codes)."""
    out = bytearray()
    for n in split_avcc_by_len(avcc, nal_len_size) or []:
        out += b"\x00\x00\x00\x01" + n
    return bytes(out if out else avcc)


def build_avcc(sps: bytes, pps: bytes) -> bytes:
    """Build avcC (AVCDecoderConfigurationRecord) from SPS and PPS NALs."""
    if len(sps) < 4:
        raise ValueError("SPS too short for avcC")
    profile = sps[1]
    compat = sps[2]
    level = sps[3]
    avcc = bytearray()
    avcc.append(1)  # configurationVersion
    avcc.append(profile)
    avcc.append(compat)
    avcc.append(level)
    avcc.append(0xFF)  # lengthSizeMinusOne = 3 (4-byte lengths)
    avcc.append(0xE1 | 1)  # numOfSequenceParameterSets (low 5 bits)
    avcc.extend(len(sps).to_bytes(2, "big"))
    avcc.extend(sps)
    avcc.append(1)  # numOfPictureParameterSets
    avcc.extend(len(pps).to_bytes(2, "big"))
    avcc.extend(pps)
    return bytes(avcc)


def find_sps_pps(nals: Sequence[bytes]) -> tuple[bytes | None, bytes | None]:
    sps = next((n for n in nals if n and (n[0] & 0x1F) == 7), None)
    pps = next((n for n in nals if n and (n[0] & 0x1F) == 8), None)
    return sps, pps

