#!/usr/bin/env python3
from __future__ import annotations

"""
Sweep H.264 encoder options to find configs that yield SPS+PPS (and ideally IDR/AUD)
either in extradata (avcC) or in the first N frame access-units.

Prints a compact line per config with findings.

Usage:
  uv run python scripts/sweep_h264_encoders.py --w 1920 --h 1080 --fps 60 --frames 60
"""

import argparse
from fractions import Fraction

import numpy as np

from napari_cuda.codec.avcc import (
    extract_sps_pps_from_blob,
    is_annexb,
    parse_avcc,
    split_annexb,
    split_avcc_by_len,
)


def nal_types(data: bytes) -> list[int]:
    if is_annexb(data):
        nals = split_annexb(data)
    else:
        nals = []
        for nsz in (4, 3, 2, 1):
            nals = split_avcc_by_len(data, nsz)
            if nals:
                break
    return [(n[0] & 0x1F) for n in nals if n]


def names(ts: list[int]) -> list[str]:
    mp = {1: "nonIDR", 5: "IDR", 6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD"}
    return [mp.get(t, str(t)) for t in ts]


def test_config(enc_name: str, pixfmt: str, opts: dict[str, str], w: int, h: int, fps: float, frames: int) -> tuple[dict[str, bool], list[list[str]], int, bool]:
    import av
    enc = av.CodecContext.create(enc_name, 'w')
    enc.width = int(w); enc.height = int(h)
    enc.pix_fmt = pixfmt
    enc.time_base = Fraction(1, int(round(fps)))
    if opts:
        enc.options = opts
    enc.open()
    # Capture GLOBAL_HEADER flag (when set, headers often not repeated in-stream)
    try:
        from av.codec.context import Flags as CFlags  # type: ignore
        global_header = bool(int(enc.flags) & int(CFlags.GLOBAL_HEADER))  # type: ignore[attr-defined]
    except Exception:
        global_header = False
    ed = getattr(enc, 'extradata', None)
    edb = bytes(ed) if ed else b''
    has_avcc = (len(edb) >= 5 and edb[0] == 1)
    # Some encoders attach avcC via NEW_EXTRADATA on the first packet; capture it.
    side_edb: bytes | None = None
    have_sps_avcc = have_pps_avcc = False
    if has_avcc:
        try:
            sps, pps, n = parse_avcc(edb)
            have_sps_avcc = bool(sps)
            have_pps_avcc = bool(pps)
        except Exception:
            pass
    # Generate frames
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    r = np.broadcast_to(x, (h, w))
    g = np.broadcast_to(y, (h, w))
    b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
    any_sps = any_pps = any_idr = any_aud = False
    au_summaries: list[list[str]] = []
    for i in range(frames):
        rgb = np.dstack([r, g, b])
        vf = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        try:
            vf = vf.reformat(w, h, format=pixfmt)
        except Exception:
            pass
        out = enc.encode(vf)
        # Combine packets for this frame
        buf_parts: list[bytes] = []
        for p in out:
            # Collect payload
            try:
                buf_parts.append(p.to_bytes())
            except Exception:
                try:
                    buf_parts.append(bytes(p))
                except Exception:
                    buf_parts.append(memoryview(p).tobytes())
            # Grab NEW_EXTRADATA if present
            if side_edb is None:
                try:
                    sd_list = getattr(p, 'side_data', None)
                    if sd_list:
                        for sd in sd_list:
                            t = getattr(sd, 'type', None)
                            tname = str(t).lower() if t is not None else ''
                            if 'new_extradata' in tname:
                                try:
                                    side_edb = sd.to_bytes()  # type: ignore[attr-defined]
                                except Exception:
                                    try:
                                        side_edb = bytes(sd)  # type: ignore[arg-type]
                                    except Exception:
                                        side_edb = memoryview(sd).tobytes()  # type: ignore[arg-type]
                                break
                except Exception:
                    pass
        buf = b''.join(buf_parts)
        if not buf:
            au_summaries.append([])
            continue
        nts = nal_types(buf)
        any_sps |= 7 in nts
        any_pps |= 8 in nts
        any_idr |= 5 in nts
        any_aud |= 9 in nts
        au_summaries.append(names(nts))
    # Try a flush encode to catch side-data on delayed packets
    try:
        out = enc.encode(None)
        for p in out:
            if side_edb is None:
                try:
                    sd_list = getattr(p, 'side_data', None)
                    if sd_list:
                        for sd in sd_list:
                            t = getattr(sd, 'type', None)
                            tname = str(t).lower() if t is not None else ''
                            if 'new_extradata' in tname or 'parameter_sets' in tname:
                                try:
                                    side_edb = sd.to_bytes()  # type: ignore[attr-defined]
                                except Exception:
                                    try:
                                        side_edb = bytes(sd)  # type: ignore[arg-type]
                                    except Exception:
                                        side_edb = memoryview(sd).tobytes()  # type: ignore[arg-type]
                                break
                except Exception:
                    pass
    except Exception:
        pass
    # Prefer side-data extradata when present
    if side_edb:
        if not has_avcc and len(side_edb) >= 5 and side_edb[0] == 1:
            has_avcc = True
            try:
                sps, pps, _n = parse_avcc(side_edb)
                have_sps_avcc = bool(sps)
                have_pps_avcc = bool(pps)
            except Exception:
                pass
        else:
            try:
                sps, pps = extract_sps_pps_from_blob(side_edb)
                have_sps_avcc = have_sps_avcc or bool(sps)
                have_pps_avcc = have_pps_avcc or bool(pps)
            except Exception:
                pass

    return (
        {
            'avcc': has_avcc,
            'avcc_sps': have_sps_avcc,
            'avcc_pps': have_pps_avcc,
            'stream_sps': any_sps,
            'stream_pps': any_pps,
            'stream_idr': any_idr,
            'stream_aud': any_aud,
        },
        au_summaries,
        len(edb),
        global_header,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, default=1280)
    ap.add_argument('--h', type=int, default=720)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--frames', type=int, default=60)
    args = ap.parse_args()

    variants = []
    # VT encoder variants
    variants.append(('h264_videotoolbox', 'nv12', {'realtime': '1'}))
    variants.append(('h264_videotoolbox', 'nv12', {}))
    # x264 variants
    x264_sets = [
        {'x264-params': 'keyint=1:min-keyint=1:scenecut=0:repeat-headers=1', 'annexb': '1'},
        {'x264-params': 'keyint=1:min-keyint=1:open-gop=0:scenecut=0:repeat-headers=1', 'annexb': '1'},
        {'x264-params': 'keyint=30:scenecut=0:repeat-headers=1', 'annexb': '1'},
        {'x264-params': 'keyint=30:scenecut=0:repeat-headers=1', 'annexb': '0'},
    ]
    for opts in x264_sets:
        variants.append(('libx264', 'yuv420p', opts))

    print('w h fps frames encoder pixfmt options | global_header extradata_len avcc avcc_sps avcc_pps | stream_sps stream_pps stream_idr stream_aud')
    for enc_name, pixfmt, opts in variants:
        try:
            res, summaries, edlen, gh = test_config(enc_name, pixfmt, opts, args.w, args.h, args.fps, args.frames)
            print(args.w, args.h, args.fps, args.frames, enc_name, pixfmt, opts, '|', gh, edlen, res['avcc'], res['avcc_sps'], res['avcc_pps'], '|', res['stream_sps'], res['stream_pps'], res['stream_idr'], res['stream_aud'])
            # Print first two AU summaries for quick view
            for i, s in enumerate(summaries[:2]):
                print('  ', f'frame{i}:', s)
        except Exception as e:
            print('ERROR', enc_name, pixfmt, opts, e)


if __name__ == '__main__':
    main()
