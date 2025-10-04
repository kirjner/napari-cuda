import pytest

from napari_cuda.server.rendering.bitstream import pack_to_annexb, ParamCache


def start_code4():
    return b"\x00\x00\x00\x01"


def start_code3():
    return b"\x00\x00\x01"


def h264_nal(nal_type: int, payload: bytes = b"\x11\x22\x33") -> bytes:
    # forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=nal_type
    b0 = ((3 & 0x3) << 5) | (nal_type & 0x1F)
    return bytes([b0]) + payload


def avcc_pack(nals: list[bytes]) -> bytes:
    # 4-byte big-endian lengths
    out = bytearray()
    for n in nals:
        out.extend(len(n).to_bytes(4, 'big'))
        out.extend(n)
    return bytes(out)


def test_list_to_annexb_and_keyframe():
    cache = ParamCache()
    sps = h264_nal(7)
    pps = h264_nal(8)
    idr = h264_nal(5)
    payload, key = pack_to_annexb([sps, pps, idr], cache)
    assert key is True
    assert payload is not None and payload.startswith(start_code4())
    # Expect three start codes present
    assert payload.count(start_code3()) + payload.count(start_code4()) >= 3


def test_annexb_passthrough_like():
    cache = ParamCache()
    sps = h264_nal(7)
    pps = h264_nal(8)
    idr = h264_nal(5)
    annexb = start_code4() + sps + start_code3() + pps + start_code3() + idr
    payload, key = pack_to_annexb(annexb, cache)
    assert key is True
    assert payload is not None
    # Should remain Annex B with at least 3 start codes
    assert payload.count(start_code3()) + payload.count(start_code4()) >= 3


def test_avcc_conversion():
    cache = ParamCache()
    sps = h264_nal(7)
    pps = h264_nal(8)
    idr = h264_nal(5)
    avcc = avcc_pack([sps, pps, idr])
    payload, key = pack_to_annexb(avcc, cache)
    assert key is True
    assert payload is not None and payload.startswith(start_code4())


def test_param_cache_prepended():
    cache = ParamCache()
    # First pass: fill cache
    _ = pack_to_annexb([h264_nal(7), h264_nal(8)], cache)
    # Second: only IDR, expect SPS/PPS inserted
    idr = h264_nal(5)
    payload, key = pack_to_annexb([idr], cache)
    assert key is True
    assert payload is not None
    # Expect at least two start codes before IDR
    assert payload.count(start_code3()) + payload.count(start_code4()) >= 3


def test_non_keyframe():
    cache = ParamCache()
    pframe = h264_nal(1)
    payload, key = pack_to_annexb([pframe], cache)
    assert key is False
    assert payload is not None and payload.startswith(start_code4())

