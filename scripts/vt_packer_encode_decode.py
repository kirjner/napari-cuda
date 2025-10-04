#!/usr/bin/env python3
from __future__ import annotations

"""
Encode all frames up front, assemble AVCC access units via the server packer,
then decode them with the VT shim. This mirrors server behavior and avoids
"raw" encoder quirks.

Notes:
- Uses pack_to_avcc + build_avcc_config to create avcC and AU bytes.
- Gates VT decode until a keyframe AU is available.
- Precomputes all frames before decoding; this is not a throughput test.

Usage:
  uv run python scripts/vt_packer_encode_decode.py --encoder h264_videotoolbox --w 1280 --h 720 --fps 60 --frames 60
  uv run python scripts/vt_packer_encode_decode.py --encoder libx264 --w 640 --h 360 --fps 30 --frames 30
"""

import argparse
import os
import time
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from napari_cuda.server.rendering.bitstream import ParamCache, pack_to_avcc, build_avcc_config
from napari_cuda.codec.avcc import split_avcc_by_len, AccessUnit


def _pkt_bytes(pkt) -> bytes:
    try:
        return pkt.to_bytes()
    except Exception:
        try:
            return bytes(pkt)
        except Exception:
            return memoryview(pkt).tobytes()


def encode_all(
    encoder: str,
    w: int,
    h: int,
    fps: float,
    frames: int,
    pix_fmt: Optional[str],
) -> Tuple[bytes, List[AccessUnit]]:
    import av

    # Ensure Python fallback if Cython packer isn't built
    os.environ.setdefault('NAPARI_CUDA_ALLOW_PY_FALLBACK', '1')

    enc = av.CodecContext.create(encoder, 'w')
    enc.width = int(w)
    enc.height = int(h)
    enc.pix_fmt = pix_fmt or ('nv12' if encoder == 'h264_videotoolbox' else 'yuv420p')
    enc.time_base = Fraction(1, int(round(fps)))
    # Low-latency / immediate IDR hints
    try:
        if encoder == 'h264_videotoolbox':
            enc.options = {'realtime': '1'}
            try:
                enc.max_b_frames = 0  # type: ignore[attr-defined]
            except Exception:
                pass
        elif encoder == 'libx264':
            enc.options = {
                'tune': 'zerolatency',
                'preset': 'veryfast',
                'bf': '0',
                'x264-params': 'keyint=1:min-keyint=1:scenecut=0:repeat-headers=1',
                'annexb': '0',
            }
    except Exception:
        pass
    enc.open()

    # Synthetic frame data
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    r = np.broadcast_to(x, (h, w))
    g = np.broadcast_to(y, (h, w))
    b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)

    cache = ParamCache()
    avcc_cfg: Optional[bytes] = None
    aus: List[AccessUnit] = []

    # Encode all frames; pack each set of output packets into a single AVCC AU
    for i in range(max(1, int(frames))):
        rgb = np.dstack([r, g, b])
        vf = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        try:
            vf = vf.reformat(w, h, format=str(enc.pix_fmt))
        except Exception:
            pass
        # nudge first frame to I
        try:
            if i == 0:
                vf.pict_type = 'I'  # type: ignore[attr-defined]
        except Exception:
            pass
        pkts = enc.encode(vf)
        payload_obj: List[bytes] = []
        for p in pkts:
            payload_obj.append(_pkt_bytes(p))
        au, is_key = pack_to_avcc(payload_obj, cache)
        if au is not None:
            ts = (i + 1) / float(max(1.0, fps))  # start at >0 to avoid 0.0
            aus.append(AccessUnit(payload=au, is_keyframe=bool(is_key), pts=float(ts)))
        if avcc_cfg is None:
            avcc_cfg = build_avcc_config(cache)

    # Flush delayed packets
    try:
        pkts = enc.encode(None)
        payload_obj = [_pkt_bytes(p) for p in pkts]
        au, is_key = pack_to_avcc(payload_obj, cache)
        if au is not None:
            ts = (len(aus) + 1) / float(max(1.0, fps))
            aus.append(AccessUnit(payload=au, is_keyframe=bool(is_key), pts=float(ts)))
    except Exception:
        pass

    # Ensure we have avcC
    avcc_cfg = avcc_cfg or build_avcc_config(cache)
    if not avcc_cfg:
        raise RuntimeError('Failed to build avcC (SPS/PPS not observed)')

    return avcc_cfg, aus


def decode_with_vt(avcc: bytes, aus: List[AccessUnit], min_decoded: int = 0) -> int:
    from napari_cuda.client.rendering.decoders.vt import VTLiveDecoder

    vt = VTLiveDecoder(avcc, 0, 0)  # width/height are not used by shim mapping in this demo
    # Gate decode until keyframe
    started = False
    decoded = 0

    # Print first two AU NAL types for visibility
    for i, au in enumerate(aus[:2]):
        nals = split_avcc_by_len(au.payload, 4)
        nts = [(n[0] & 0x1F) for n in nals if n]
        print(f'AU{i} types={nts} key={au.is_keyframe} len={len(au.payload)}')

    for au in aus:
        if not started and not au.is_keyframe:
            continue
        started = True
        _ = vt.decode(au.payload, au.pts)
        # brief poll for output
        t0 = time.time()
        while True:
            item = vt.get_frame_nowait()
            if item:
                decoded += 1
                break
            if (time.time() - t0) > 0.02:
                break
            time.sleep(0.001)

    # Drain
    try:
        vt.flush()
    except Exception:
        pass
    extra = 0
    t_end = time.time() + 1.0
    while time.time() < t_end:
        if not vt.get_frame_nowait():
            time.sleep(0.005)
            continue
        extra += 1
    decoded += extra
    sub, out, qlen = vt.counts()
    print(f'done: decoded={decoded} vt_counts submit={sub} output={out} qlen={qlen}')
    if min_decoded > 0 and decoded < min_decoded:
        print(f'FAIL: decoded {decoded} < required {min_decoded}')
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description='VT decode demo using server packer (AVCC)')
    ap.add_argument('--encoder', default='h264_videotoolbox')
    ap.add_argument('--w', type=int, default=640)
    ap.add_argument('--h', type=int, default=360)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--pixfmt', default=None)
    ap.add_argument('--min-decoded', type=int, default=0, help='If >0, exit nonzero if decoded frames < this')
    args = ap.parse_args()

    # Print packer fast-path availability
    try:
        from napari_cuda.server.rendering.bitstream import _FAST_PACK  # type: ignore
        print(f'FAST_PACK available: {bool(_FAST_PACK)}')
    except Exception:
        pass
    avcc, aus = encode_all(args.encoder, int(args.w), int(args.h), float(args.fps), int(args.frames), args.pixfmt)
    # Decode using VT
    rc = decode_with_vt(avcc, aus, args.min_decoded)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
