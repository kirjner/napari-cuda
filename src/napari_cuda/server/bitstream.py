"""
Bitstream utilities for packaging encoder output into Annex B and detecting keyframes.

Independent of GL/CUDA/torch for easy testing.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import logging
from typing import List, Optional, Tuple, Sequence, Union


BytesLike = Union[bytes, bytearray, memoryview]


def _is_annexb(buf: bytes) -> bool:
    """Return True if `buf` starts with an Annex B start code."""
    return buf.startswith(b"\x00\x00\x01") or buf.startswith(b"\x00\x00\x00\x01")


@dataclass
class ParamCache:
    vps: Optional[bytes] = None
    sps: Optional[bytes] = None
    pps: Optional[bytes] = None


def pack_to_annexb(packets: Union[BytesLike, Sequence[BytesLike], None], cache: ParamCache) -> Tuple[Optional[bytes], bool]:
    """Normalize encoder output (bytes or list of bytes) to Annex B and detect keyframe.

    - Accepts bytes or list/tuple of byte-like objects
    - Detects Annex B vs AVCC and converts AVCC length prefixes to start codes
    - Caches VPS/SPS/PPS and prepends on keyframes if missing
    - Returns (payload bytes or None, is_keyframe)
    """
    if packets is None:
        return None, False
    # Fast path: single Annex B byte buffer → return as-is with quick keyframe detection
    if isinstance(packets, (bytes, bytearray, memoryview)):
        buf = bytes(packets)
        if _is_annexb(buf):
            # Quick scan for keyframe NAL types without rebuilding
            i = 0
            is_key = False
            l = len(buf)
            while i < l:
                p3 = buf.find(b"\x00\x00\x01", i)
                p4 = buf.find(b"\x00\x00\x00\x01", i)
                if p3 == -1 and p4 == -1:
                    break
                if p3 == -1 or (p4 != -1 and p4 < p3):
                    j = p4 + 4
                    i = j
                else:
                    j = p3 + 3
                    i = j
                if j < l:
                    b0 = buf[j]
                    n264 = b0 & 0x1F
                    n265 = (b0 >> 1) & 0x3F
                    if n264 == 5 or n265 in (19, 20, 21):
                        is_key = True
                        break
            return buf, is_key
        chunks = [buf]
    else:
        chunks: List[bytes] = []
        if isinstance(packets, (list, tuple)):
            for p in packets:
                if p is None:
                    continue
                chunks.append(bytes(p))
        else:
            try:
                chunks = [bytes(packets)]
            except Exception:
                return None, False

    def parse_nals(buf: bytes) -> List[bytes]:
        nals: List[bytes] = []
        if not buf:
            return nals
        if _is_annexb(buf):
            i = 0
            l = len(buf)
            idx: List[int] = []
            while i < l - 3:
                if buf[i:i+3] == b"\x00\x00\x01":
                    idx.append(i); i += 3
                elif i < l - 4 and buf[i:i+4] == b"\x00\x00\x00\x01":
                    idx.append(i); i += 4
                else:
                    i += 1
            if not idx:
                return [buf]
            idx.append(l)
            for a, b in zip(idx, idx[1:]):
                j = a
                while j < b and buf[j] == 0:
                    j += 1
                if j < b and buf[j] == 1:
                    j += 1
                nal = buf[j:b]
                if nal:
                    nals.append(nal)
            return nals
        else:
            # AVCC: 4-byte big-endian length prefixes
            i = 0
            l = len(buf)
            while i + 4 <= l:
                ln = int.from_bytes(buf[i:i+4], 'big'); i += 4
                if ln <= 0 or i + ln > l:
                    return [buf]
                nals.append(buf[i:i+ln]); i += ln
            return nals if nals else [buf]

    nal_units: List[bytes] = []
    for c in chunks:
        nal_units.extend(parse_nals(c))

    def t264(b0: int) -> int: return b0 & 0x1F
    def t265(b0: int) -> int: return (b0 >> 1) & 0x3F

    is_key = False
    saw_sps = saw_pps = saw_vps = False
    log_sps = bool(int(os.getenv('NAPARI_CUDA_LOG_SPS', '0') or '0'))
    logger = logging.getLogger(__name__)
    for nal in nal_units:
        if not nal:
            continue
        b0 = nal[0]
        n264 = t264(b0)
        n265 = t265(b0)
        if n264 == 7:  # SPS
            saw_sps = True; cache.sps = nal
            if log_sps:
                try:
                    logger.info("H264 SPS len=%d first16=%s", len(nal), nal[:16].hex())
                except Exception:
                    pass
        elif n264 == 8:  # PPS
            saw_pps = True; cache.pps = nal
        elif n264 == 5:  # IDR
            is_key = True
        if n265 == 32:  # VPS
            saw_vps = True; cache.vps = nal
        elif n265 == 33:  # SPS
            saw_sps = True; cache.sps = nal
            if log_sps:
                try:
                    logger.info("H265 SPS len=%d first16=%s", len(nal), nal[:16].hex())
                except Exception:
                    pass
        elif n265 == 34:  # PPS
            saw_pps = True; cache.pps = nal
        elif n265 in (19, 20, 21):  # IDR/CRA
            is_key = True

    out: List[bytes] = []
    if is_key:
        if cache.vps and not saw_vps:
            out.append(cache.vps)
        if cache.sps and not saw_sps:
            out.append(cache.sps)
        if cache.pps and not saw_pps:
            out.append(cache.pps)
    out.extend(nal_units)
    if not out:
        return None, False
    ba = bytearray()
    first = True
    for nal in out:
        ba.extend(b"\x00\x00\x00\x01" if first else b"\x00\x00\x01"); first = False
        ba.extend(nal)
    return bytes(ba), is_key
