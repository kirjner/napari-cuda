#!/usr/bin/env python3
"""
Encode a short synthetic sequence with PyNvVideoCodec and verify IDR cadence.

This uses CPU input buffers with ARGB frames for simplicity.
Settings enforce a strict GOP: gopLength=60, idrPeriod=60, no B-frames,
strictGOPTarget=1, and no lookahead/intra-refresh.

It forces IDR at frame 0 and frame 60 (and 120 if long enough) and writes
an Annex B H.264 bitstream to nvenc_idr_test.h264.
"""
import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore

try:
    import PyNvVideoCodec as pnvc
except Exception as e:
    print(f"ERROR: Failed to import PyNvVideoCodec: {e}")
    sys.exit(1)


def make_argb_frame_cpu(w: int, h: int, t: int) -> np.ndarray:
    """Create a simple ARGB uint8 test frame that changes over time."""
    # A = 255 constant, RGB vary with gradients and time
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    # Time-varying phase to avoid duplicates
    phase = np.uint8((t * 13) % 256)
    r = (x.astype(np.uint16) + int(phase)) % 256
    g = (y.astype(np.uint16) + int(phase) // 2) % 256
    b = ((x.astype(np.uint16) // 2 + y.astype(np.uint16) // 2 + int(phase)) % 256)
    r = np.broadcast_to(r.astype(np.uint8), (h, w))
    g = np.broadcast_to(g.astype(np.uint8), (h, w))
    b = b.astype(np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)
    # ARGB ordering: A,R,G, B
    argb = np.stack([a, r, g, b], axis=-1)
    return argb


def make_argb_frame_gpu(w: int, h: int, t: int):
    if cp is None:
        raise RuntimeError("CuPy not available for GPU mode")
    yy = cp.linspace(0, 255, h, dtype=cp.uint8)[:, None]
    xx = cp.linspace(0, 255, w, dtype=cp.uint8)[None, :]
    phase = int((t * 13) % 256)
    r = (xx.astype(cp.uint16) + phase) % 256
    g = (yy.astype(cp.uint16) + phase // 2) % 256
    b = ((xx.astype(cp.uint16) // 2 + yy.astype(cp.uint16) // 2 + phase) % 256)
    r = cp.broadcast_to(r.astype(cp.uint8), (h, w))
    g = cp.broadcast_to(g.astype(cp.uint8), (h, w))
    b = b.astype(cp.uint8)
    a = cp.full((h, w), 255, dtype=cp.uint8)
    argb = cp.stack([a, r, g, b], axis=-1)
    return argb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frames", type=int, default=180)  # 3 seconds @ 60fps
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--bitrate", type=int, default=4_000_000)
    ap.add_argument("--outfile", type=Path, default=Path("nvenc_idr_test.h264"))
    ap.add_argument("--preset", type=str, default="ultra_low_latency", help="Encoder preset string (e.g., ultra_low_latency); leave empty for default")
    ap.add_argument("--gpu", action="store_true", default=True, help="Use GPU input buffer (CuPy)")
    ap.add_argument("--cpu", action="store_true", default=False, help="Force CPU input buffer")
    args = ap.parse_args()

    w, h, n = args.width, args.height, args.frames
    out_path: Path = args.outfile
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoder kwargs enforcing strict IDR cadence
    enc_kwargs = dict(
        codec="h264",
        preset=args.preset if args.preset else None,
        profile="high",
        rcMode="CBR",
        bitrate=args.bitrate,
        maxBitrate=max(args.bitrate, int(args.bitrate * 1.2)),
        frameRateNum=args.fps,
        frameRateDen=1,
        gopLength=args.fps,     # 60
        idrPeriod=args.fps,     # 60
        frameIntervalP=1,       # no B-frames
        maxNumRefFrames=1,
        enablePTD=1,
        repeatSPSPPS=1,
        enableIntraRefresh=0,
        enableLookahead=0,
        enableTemporalAQ=0,
        enableNonRefP=0,
        strictGOPTarget=1,
    )

    try:
        # Filter out None-valued kwargs (e.g., when no preset provided)
        kw = {k: v for k, v in enc_kwargs.items() if v is not None}
        use_cpu = bool(args.cpu and not args.gpu)
        enc = pnvc.CreateEncoder(
            width=w,
            height=h,
            fmt="ARGB",
            usecpuinputbuffer=use_cpu is True,
            **kw,
        )
    except Exception as e:
        print("ERROR: CreateEncoder failed:", e)
        sys.exit(2)

    # Open output raw bitstream file
    with open(out_path, "wb") as f:
        for i in range(n):
            if args.gpu and not use_cpu:
                frame = make_argb_frame_gpu(w, h, i)
            else:
                frame = make_argb_frame_cpu(w, h, i)
            # Force IDR on first frame and exactly at GOP boundary
            pic_flags = 0
            if i == 0 or i % enc_kwargs["gopLength"] == 0:
                pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.FORCEIDR)
                # Also ask to repeat SPS/PPS with keyframe for robust probing
                pic_flags |= int(pnvc.NV_ENC_PIC_FLAGS.OUTPUT_SPSPPS)

            try:
                if pic_flags:
                    pkts = enc.Encode(frame, pic_flags)
                else:
                    pkts = enc.Encode(frame)
            except Exception as e:
                print(f"ERROR: Encode failed at frame {i}: {e}")
                sys.exit(3)

            if pkts:
                for p in pkts:
                    # pkts may be a list of Python bytes-like objects
                    f.write(bytes(p))

        # Flush
        try:
            tail = enc.EndEncode()
            if tail:
                for p in tail:
                    f.write(bytes(p))
        except Exception as e:
            print("WARN: EndEncode raised:", e)

    print(f"Wrote: {out_path.resolve()} ({out_path.stat().st_size} bytes)")
    print("Done. You can probe with ffprobe, e.g.:")
    print("  ffprobe -v error -select_streams v:0 -show_frames -show_entries frame=key_frame,pict_type,pts_time -of csv=p=0 ", out_path)
    print("  ffprobe -v error -select_streams v:0 -show_packets -show_entries packet=pts_time,flags -of csv=p=0 ", out_path)


if __name__ == "__main__":
    main()
