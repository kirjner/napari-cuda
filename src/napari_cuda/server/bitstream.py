"""
Bitstream utilities for packaging encoder output and detecting keyframes.

Independent of GL/CUDA/torch for easy testing.

Provided helpers:
- parse_nals(buf): split Annex B or AVCC buffers into NAL units (without startcodes/lengths)
- pack_to_annexb(...): normalize to Annex B (existing path)
- pack_to_avcc(...): normalize to AVCC (4-byte length-prefixed)
- build_avcc_config(cache): build AVCDecoderConfigurationRecord (avcC) from cached SPS/PPS
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Sequence, Union

from napari_cuda.server.logging_policy import EncoderLogging
from napari_cuda.server.config import BitstreamRuntime

# Optional Cython fast packer; default is to require it on server
_BITSTREAM_RUNTIME = BitstreamRuntime()
_FAST_PACK = None


def _load_fast_packer(runtime: BitstreamRuntime):
    if runtime.disable_fast_pack:
        return None
    try:
        from ._avcc_packer import pack_to_avcc_fast as fast  # type: ignore
        return fast
    except Exception:
        if not runtime.build_cython:
            return None
        try:
            import pyximport  # type: ignore

            pyximport.install(language_level=3, inplace=True)  # type: ignore
            from ._avcc_packer import pack_to_avcc_fast as fast  # type: ignore
            return fast
        except Exception:
            return None


def configure_bitstream(runtime: BitstreamRuntime) -> None:
    """Configure bitstream packer runtime options from server context."""

    global _BITSTREAM_RUNTIME, _FAST_PACK
    _BITSTREAM_RUNTIME = runtime
    _FAST_PACK = _load_fast_packer(runtime)


_FAST_PACK = _load_fast_packer(_BITSTREAM_RUNTIME)


BytesLike = Union[bytes, bytearray, memoryview]


def _is_annexb(buf: bytes) -> bool:
    """Return True if `buf` starts with an Annex B start code."""
    return buf.startswith(b"\x00\x00\x01") or buf.startswith(b"\x00\x00\x00\x01")


@dataclass
class ParamCache:
    vps: Optional[bytes] = None
    sps: Optional[bytes] = None
    pps: Optional[bytes] = None


def parse_nals(buf: bytes) -> List[bytes]:
    """Split a buffer into raw NAL units (no start codes / no length prefixes).

    - If buffer uses Annex B start codes (0x000001 or 0x00000001), split on them.
    - Otherwise, treat as AVCC: a sequence of 4-byte big-endian lengths.
    - On malformed input, return the buffer as a single NAL for robustness.
    """
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


def pack_to_annexb(
    packets: Union[BytesLike, Sequence[BytesLike], None],
    cache: ParamCache,
    *,
    encoder_logging: Optional[EncoderLogging] = None,
) -> Tuple[Optional[bytes], bool]:
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

    nal_units: List[bytes] = []
    for c in chunks:
        nal_units.extend(parse_nals(c))

    def t264(b0: int) -> int: return b0 & 0x1F
    def t265(b0: int) -> int: return (b0 >> 1) & 0x3F

    is_key = False
    saw_sps = saw_pps = saw_vps = False
    if encoder_logging is not None:
        log_sps = bool(encoder_logging.log_sps)
    else:
        log_sps = False
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


def pack_to_avcc(
    packets: Union[BytesLike, Sequence[BytesLike], None],
    cache: ParamCache,
    *,
    encoder_logging: Optional[EncoderLogging] = None,
) -> Tuple[Optional[bytes], bool]:
    """Normalize encoder output to AVCC (4-byte length-prefixed) and detect keyframe.

    - Accepts bytes or list/tuple of byte-like objects (Annex B or AVCC)
    - Caches VPS/SPS/PPS when present in the AU
    - Does NOT inject cached SPS/PPS; rely on encoder's "repeat SPS/PPS on IDR"
    - Returns (payload bytes or None, is_keyframe)
    """
    # If Cython fast path is available, use it; by default, Python fallback is disabled
    allow_fallback = bool(_BITSTREAM_RUNTIME.allow_py_fallback)
    if _FAST_PACK is None and not allow_fallback:
        raise RuntimeError(
            "Cython packer not available. Install with 'napari-cuda[server]' (includes Cython) "
            "or enable the Python fallback via server context." 
        )
    if _FAST_PACK is not None:
        try:
            return _FAST_PACK(packets, cache)  # type: ignore[misc]
        except Exception as e:
            # If Cython fails and fallback is not allowed, fail fast
            if not allow_fallback:
                raise RuntimeError(
                    "Cython packer failed. Ensure 'napari-cuda[server]' is installed or enable "
                    "the Python fallback via server context."
                ) from e
            # else fall through to Python path
    if packets is None:
        return None, False
    # Gather chunks as bytes
    if isinstance(packets, (bytes, bytearray, memoryview)):
        chunks: List[bytes] = [bytes(packets)]
    elif isinstance(packets, (list, tuple)):
        chunks = [bytes(p) for p in packets if p is not None]
    else:
        try:
            chunks = [bytes(packets)]
        except Exception:
            return None, False

    # Flatten into NAL units
    nal_units: List[bytes] = []
    for c in chunks:
        nal_units.extend(parse_nals(c))

    def t264(b0: int) -> int: return b0 & 0x1F
    def t265(b0: int) -> int: return (b0 >> 1) & 0x3F

    is_key = False
    if encoder_logging is not None:
        log_sps = bool(encoder_logging.log_sps)
    else:
        log_sps = False
    logger = logging.getLogger(__name__)
    for nal in nal_units:
        if not nal:
            continue
        b0 = nal[0]
        n264 = t264(b0)
        n265 = t265(b0)
        if n264 == 7:  # SPS
            cache.sps = nal
            if log_sps:
                try:
                    logger.info("H264 SPS len=%d first16=%s", len(nal), nal[:16].hex())
                except Exception:
                    pass
        elif n264 == 8:  # PPS
            cache.pps = nal
        elif n264 == 5:  # IDR
            is_key = True
        if n265 == 32:  # VPS
            cache.vps = nal
        elif n265 == 33:  # SPS
            cache.sps = nal
            if log_sps:
                try:
                    logger.info("H265 SPS len=%d first16=%s", len(nal), nal[:16].hex())
                except Exception:
                    pass
        elif n265 == 34:  # PPS
            cache.pps = nal
        elif n265 in (19, 20, 21):  # IDR/CRA
            is_key = True

    if not nal_units:
        return None, False

    # Concatenate as 4-byte big-endian length prefixed NALs
    out = bytearray()
    for nal in nal_units:
        out.extend(len(nal).to_bytes(4, 'big'))
        out.extend(nal)
    return bytes(out), is_key


def build_avcc_config(cache: ParamCache) -> Optional[bytes]:
    """Build AVCDecoderConfigurationRecord (avcC) from cached SPS/PPS.

    Returns None if SPS/PPS are not yet available.

    Notes:
    - Profile/compat/level are read from SPS bytes [1], [2], [3] (after NAL header)
    - lengthSizeMinusOne is set to 3 (4-byte lengths)
    - Includes exactly one SPS and one PPS if available
    """
    sps = cache.sps
    pps = cache.pps
    if not sps or not pps or len(sps) < 4:
        return None
    profile_idc = sps[1]
    profile_compat = sps[2]
    level_idc = sps[3]
    # Build avcC
    ba = bytearray()
    ba.append(1)  # configurationVersion
    ba.append(profile_idc)  # AVCProfileIndication
    ba.append(profile_compat)  # profile_compatibility (constraint flags)
    ba.append(level_idc)  # AVCLevelIndication
    ba.append(0xFF)  # 6 bits reserved (111111) + lengthSizeMinusOne (3)
    # numOfSPS: 3 bits reserved (111) + 5-bit count
    num_sps = 1 if sps else 0
    ba.append(0xE0 | (num_sps & 0x1F))
    if sps:
        ba.extend(len(sps).to_bytes(2, 'big'))
        ba.extend(sps)
    # PPS count and data
    num_pps = 1 if pps else 0
    ba.append(num_pps & 0xFF)
    if pps:
        ba.extend(len(pps).to_bytes(2, 'big'))
        ba.extend(pps)
    return bytes(ba)
