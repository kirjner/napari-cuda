from __future__ import annotations

"""
Shared H.264 helpers used across client and server.

Includes:
- AnnexB/AVCC detection and splitters
- Conversions between AnnexB and AVCC for single access units
- avcC parsing and building (SPS/PPS)
- NAL type helpers and simple IDR detection
"""

from typing import List, Sequence, Tuple


# --- Detection & splitting ---


def is_annexb(buf: bytes) -> bool:
    if not buf:
        return False
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


# --- Conversions ---


def annexb_to_avcc(data: bytes, out_len: int = 4) -> bytes:
    if not is_annexb(data):
        return data
    nals = split_annexb(data)
    out = bytearray()
    for n in nals:
        out.extend(len(n).to_bytes(out_len, "big"))
        out.extend(n)
    return bytes(out)


def avcc_to_annexb(avcc: bytes, nal_len_size: int = 4) -> bytes:
    out = bytearray()
    for n in split_avcc_by_len(avcc, nal_len_size) or []:
        out += b"\x00\x00\x00\x01" + n
    return bytes(out if out else avcc)


def normalize_to_annexb(buf: bytes) -> Tuple[bytes, bool]:
    if is_annexb(buf):
        return buf, False
    return avcc_to_annexb(buf, 4), True


# --- avcC helpers ---


def parse_avcc(avcc: bytes) -> Tuple[List[bytes], List[bytes], int]:
    if not avcc or len(avcc) < 7:
        raise ValueError("Invalid avcC: too short")
    i = 0
    _configuration_version = avcc[i]
    i += 1
    _profile = avcc[i]
    i += 1
    _compat = avcc[i]
    i += 1
    _level = avcc[i]
    i += 1
    length_size_minus_one = avcc[i] & 0x03
    i += 1
    nal_length_size = int(length_size_minus_one) + 1
    num_sps = avcc[i] & 0x1F
    i += 1
    sps_list: List[bytes] = []
    for _ in range(num_sps):
        if i + 2 > len(avcc):
            raise ValueError("Invalid avcC: truncated SPS length")
        ln = int.from_bytes(avcc[i : i + 2], "big")
        i += 2
        if i + ln > len(avcc):
            raise ValueError("Invalid avcC: truncated SPS data")
        sps_list.append(avcc[i : i + ln])
        i += ln
    if i >= len(avcc):
        raise ValueError("Invalid avcC: missing PPS count")
    num_pps = avcc[i]
    i += 1
    pps_list: List[bytes] = []
    for _ in range(num_pps):
        if i + 2 > len(avcc):
            raise ValueError("Invalid avcC: truncated PPS length")
        ln = int.from_bytes(avcc[i : i + 2], "big")
        i += 2
        if i + ln > len(avcc):
            raise ValueError("Invalid avcC: truncated PPS data")
        pps_list.append(avcc[i : i + ln])
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


def contains_idr_annexb(data: bytes, hevc: bool = False) -> bool:
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


def contains_idr_avcc(data: bytes, nal_len_size: int = 4, hevc: bool = False) -> bool:
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

