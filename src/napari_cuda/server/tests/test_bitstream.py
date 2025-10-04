import base64

import pytest

from napari_cuda.server.rendering.bitstream import (
    parse_nals,
    pack_to_avcc,
    build_avcc_config,
    ParamCache,
)


def _annexb(*nals: bytes) -> bytes:
    out = bytearray()
    first = True
    for nal in nals:
        out.extend(b"\x00\x00\x00\x01" if first else b"\x00\x00\x01")
        out.extend(nal)
        first = False
    return bytes(out)


def _avcc(*nals: bytes) -> bytes:
    out = bytearray()
    for nal in nals:
        out.extend(len(nal).to_bytes(4, 'big'))
        out.extend(nal)
    return bytes(out)


def test_parse_nals_annexb_and_avcc_equivalence():
    # Minimal valid-looking SPS/PPS/IDR payloads
    sps = bytes([0x67, 100, 0, 31, 0xAA, 0xBB])  # nal type 7, profile=100, compat=0, level=31
    pps = bytes([0x68, 0xEF, 0x3C])  # nal type 8
    idr = bytes([0x65, 0x12, 0x34, 0x56])  # nal type 5 (key)

    b_annexb = _annexb(sps, pps, idr)
    b_avcc = _avcc(sps, pps, idr)

    n_annexb = parse_nals(b_annexb)
    n_avcc = parse_nals(b_avcc)

    assert n_annexb == [sps, pps, idr]
    assert n_avcc == [sps, pps, idr]


def test_pack_to_avcc_from_annexb_and_cache_and_keyflag():
    from napari_cuda.server.rendering import bitstream

    sps = bytes([0x67, 100, 0, 31, 0xAA, 0xBB])
    pps = bytes([0x68, 0xEF, 0x3C])
    idr = bytes([0x65, 0x12, 0x34, 0x56])
    buf = _annexb(sps, pps, idr)
    cache = ParamCache()

    from napari_cuda.server.app.config import BitstreamRuntime

    prev_runtime = bitstream._BITSTREAM_RUNTIME
    bitstream.configure_bitstream(BitstreamRuntime(
        build_cython=False,
        disable_fast_pack=True,
        allow_py_fallback=True,
    ))
    try:
        out, is_key = pack_to_avcc(buf, cache)
    finally:
        bitstream.configure_bitstream(prev_runtime)
    assert is_key is True
    assert cache.sps == sps and cache.pps == pps
    # Expect AVCC prefixing with lengths
    expected = _avcc(sps, pps, idr)
    assert out == expected


def test_build_avcc_config_structure():
    # Construct cache with SPS/PPS and validate avcC fields
    profile_idc = 100
    profile_compat = 0
    level_idc = 31
    sps = bytes([0x67, profile_idc, profile_compat, level_idc, 0xAA, 0xBB])
    pps = bytes([0x68, 0xEF, 0x3C])
    cache = ParamCache(sps=sps, pps=pps)

    avcc = build_avcc_config(cache)
    assert avcc is not None
    # Parse minimal structure
    assert avcc[0] == 1  # configurationVersion
    assert avcc[1] == profile_idc
    assert avcc[2] == profile_compat
    assert avcc[3] == level_idc
    assert avcc[4] & 0x03 == 3  # lengthSizeMinusOne
    # numOfSPS in low 5 bits; top 3 reserved must be 1s
    assert (avcc[5] & 0xE0) == 0xE0
    num_sps = avcc[5] & 0x1F
    assert num_sps == 1
    # Next 2 bytes are SPS length
    sps_len = int.from_bytes(avcc[6:8], 'big')
    assert sps_len == len(sps)
    sps_start = 8
    sps_end = sps_start + sps_len
    assert avcc[sps_start:sps_end] == sps
    # Next: PPS count
    pps_count = avcc[sps_end]
    assert pps_count == 1
    pps_len = int.from_bytes(avcc[sps_end+1:sps_end+3], 'big')
    assert pps_len == len(pps)
    assert avcc[sps_end+3:sps_end+3+pps_len] == pps

