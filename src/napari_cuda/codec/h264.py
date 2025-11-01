from __future__ import annotations

"""
Shared H.264 helpers used across client and server.

Includes:
- AnnexB/AVCC detection and splitters
- Conversions between AnnexB and AVCC for single access units
- avcC parsing and building (SPS/PPS)
- NAL type helpers and simple IDR detection
"""

from collections.abc import Sequence
from typing import Union

BytesLike = Union[bytes, bytearray, memoryview]


# --- Detection & splitting ---


def is_annexb(buf: BytesLike) -> bool:
    if not buf:
        return False
    mv = memoryview(buf).cast('B')
    i = 0
    n = len(mv)
    limit = min(n, 32)
    while i < limit and buf[i] == 0:
        i += 1
    if i + 3 <= n and mv[i : i + 3].tobytes() == b"\x00\x00\x01":
        return True
    if i + 4 <= n and mv[i : i + 4].tobytes() == b"\x00\x00\x00\x01":
        return True
    return False


def split_annexb(data: BytesLike) -> list[bytes]:
    out: list[bytes] = []
    mv = memoryview(data).cast('B')
    i = 0
    n = len(mv)
    idx: list[int] = []
    while i + 3 <= n:
        if mv[i : i + 3].tobytes() == b"\x00\x00\x01":
            idx.append(i)
            i += 3
        elif i + 4 <= n and mv[i : i + 4].tobytes() == b"\x00\x00\x00\x01":
            idx.append(i)
            i += 4
        else:
            i += 1
    idx.append(n)
    for a, b in zip(idx, idx[1:], strict=False):
        j = a
        while j < b and mv[j] == 0:
            j += 1
        if j + 3 <= b and mv[j : j + 3].tobytes() == b"\x00\x00\x01":
            j += 3
        elif j + 4 <= b and mv[j : j + 4].tobytes() == b"\x00\x00\x00\x01":
            j += 4
        nal = mv[j:b].tobytes()
        if nal:
            out.append(nal)
    return out


def split_avcc_by_len(data: BytesLike, nal_len_size: int = 4) -> list[bytes]:
    out: list[bytes] = []
    i = 0
    mv = memoryview(data).cast('B')
    n = len(mv)
    try:
        while i + nal_len_size <= n:
            ln = int.from_bytes(mv[i : i + nal_len_size].tobytes(), "big", signed=False)
            i += nal_len_size
            if ln <= 0 or i + ln > n:
                break
            out.append(mv[i : i + ln].tobytes())
            i += ln
    except Exception:
        out = []
    return out


# --- Conversions ---


def annexb_to_avcc(data: BytesLike, out_len: int = 4) -> bytes:
    if not is_annexb(data):
        return bytes(data)
    nals = split_annexb(data)
    out = bytearray()
    for n in nals:
        out.extend(len(n).to_bytes(out_len, "big"))
        out.extend(n)
    return bytes(out)


def avcc_to_annexb(avcc: BytesLike, nal_len_size: int = 4) -> bytes:
    out = bytearray()
    for n in split_avcc_by_len(avcc, nal_len_size) or []:
        out += b"\x00\x00\x00\x01" + n
    return bytes(out if out else avcc)


def normalize_to_annexb(buf: BytesLike) -> tuple[bytes, bool]:
    if is_annexb(buf):
        return bytes(buf), False
    return avcc_to_annexb(buf, 4), True


# --- avcC helpers ---


def parse_avcc(avcc: BytesLike) -> tuple[list[bytes], list[bytes], int]:
    mv = memoryview(avcc).cast('B')
    if not mv or len(mv) < 7:
        raise ValueError("Invalid avcC: too short")
    i = 0
    _configuration_version = mv[i]
    i += 1
    _profile = mv[i]
    i += 1
    _compat = mv[i]
    i += 1
    _level = mv[i]
    i += 1
    length_size_minus_one = mv[i] & 0x03
    i += 1
    nal_length_size = int(length_size_minus_one) + 1
    num_sps = mv[i] & 0x1F
    i += 1
    sps_list: list[bytes] = []
    for _ in range(num_sps):
        if i + 2 > len(mv):
            raise ValueError("Invalid avcC: truncated SPS length")
        ln = int.from_bytes(mv[i : i + 2].tobytes(), "big")
        i += 2
        if i + ln > len(mv):
            raise ValueError("Invalid avcC: truncated SPS data")
        sps_list.append(mv[i : i + ln].tobytes())
        i += ln
    if i >= len(mv):
        raise ValueError("Invalid avcC: missing PPS count")
    num_pps = mv[i]
    i += 1
    pps_list: list[bytes] = []
    for _ in range(num_pps):
        if i + 2 > len(mv):
            raise ValueError("Invalid avcC: truncated PPS length")
        ln = int.from_bytes(mv[i : i + 2].tobytes(), "big")
        i += 2
        if i + ln > len(mv):
            raise ValueError("Invalid avcC: truncated PPS data")
        pps_list.append(mv[i : i + ln].tobytes())
        i += ln
    return sps_list, pps_list, nal_length_size


def build_avcc(sps: bytes, pps: bytes) -> bytes:
    if len(sps) < 4:
        raise ValueError("SPS too short for avcC")
    profile = sps[1]
    compat = sps[2]
    level = sps[3]
    avcc = bytearray()
    avcc.append(1)
    avcc.append(profile)
    avcc.append(compat)
    avcc.append(level)
    avcc.append(0xFF)  # lengthSizeMinusOne = 3 (4-byte lengths)
    avcc.append(0xE1 | 1)  # num SPS
    avcc.extend(len(sps).to_bytes(2, "big"))
    avcc.extend(sps)
    avcc.append(1)  # num PPS
    avcc.extend(len(pps).to_bytes(2, "big"))
    avcc.extend(pps)
    return bytes(avcc)


def find_sps_pps(nals: Sequence[bytes]) -> tuple[bytes | None, bytes | None]:
    sps = next((n for n in nals if n and (n[0] & 0x1F) == 7), None)
    pps = next((n for n in nals if n and (n[0] & 0x1F) == 8), None)
    return sps, pps


# --- NAL type helpers / keyframe checks ---


def nal_type_h264(b0: int) -> int:
    return b0 & 0x1F


def nal_type_h265(b0: int) -> int:
    return (b0 >> 1) & 0x3F


def contains_idr_annexb(data: BytesLike, hevc: bool = False) -> bool:
    for n in split_annexb(data):
        if not n:
            continue
        b0 = n[0]
        if hevc:
            t = nal_type_h265(b0)
            if 16 <= t <= 21:  # IRAP
                return True
        else:
            t = nal_type_h264(b0)
            if t == 5:
                return True
    return False


def contains_idr_avcc(data: BytesLike, nal_len_size: int = 4, hevc: bool = False) -> bool:
    for n in split_avcc_by_len(data, nal_len_size):
        if not n:
            continue
        b0 = n[0]
        if hevc:
            t = nal_type_h265(b0)
            if 16 <= t <= 21:
                return True
        else:
            t = nal_type_h264(b0)
            if t == 5:
                return True
    return False


# --- Extradata heuristics ---


def extract_sps_pps_from_blob(b: BytesLike) -> tuple[bytes | None, bytes | None]:
    """Best-effort SPS/PPS extraction from arbitrary H.264 blob.

    Tries AnnexB split, then AVCC split with multiple length sizes.
    If still not found, heuristically splits on first PPS (nal_type=8)
    following an SPS (nal_type=7).
    """
    nals = split_annexb(b)
    if not nals:
        for nsz in (4, 3, 2, 1):
            nals = split_avcc_by_len(b, nsz)
            if nals:
                break
    if nals:
        return find_sps_pps(nals)
    # Heuristic: locate SPS start and PPS start by byte-wise scan
    n = len(b)
    sps_i = -1
    pps_i = -1
    for i in range(n):
        t = b[i] & 0x1F
        if t == 7 and sps_i == -1:
            sps_i = i
        elif t == 8 and sps_i != -1:
            pps_i = i
            break
    if sps_i != -1 and pps_i != -1:
        sps = b[sps_i:pps_i]
        pps = b[pps_i:]
        return sps, pps
    return None, None
