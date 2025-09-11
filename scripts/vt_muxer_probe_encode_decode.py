#!/usr/bin/env python3
from __future__ import annotations

"""
Demonstrate H.264 encode -> decode using a muxer probe to obtain avcC (SPS/PPS).

Flow:
- Create a short MP4 with the chosen encoder to deterministically obtain avcC.
- Initialize the VT shim decoder from that avcC.
- Run a raw encoder to produce AVCC access units and feed them into VT.

Usage examples:
  uv run python scripts/vt_muxer_probe_encode_decode.py --encoder h264_videotoolbox --w 1280 --h 720 --fps 60 --frames 60
  uv run python scripts/vt_muxer_probe_encode_decode.py --encoder libx264 --w 640 --h 360 --fps 30 --frames 30
"""

import argparse
import os
import sys
import tempfile
from fractions import Fraction
import time
from typing import Optional, Tuple

import numpy as np

from napari_cuda.codec.avcc import (
    is_annexb,
    annexb_to_avcc,
    split_avcc_by_len,
    split_annexb,
    parse_avcc,
)


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def probe_avcc_via_muxer(
    encoder: str,
    w: int,
    h: int,
    fps: float,
    pix_fmt: Optional[str] = None,
) -> bytes:
    """Encode 1–2 frames into an MP4 and read avcC from the demuxed stream."""
    import av

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Write a tiny MP4
        out = av.open(tmp_path, mode="w")
        try:
            stream = out.add_stream(encoder, rate=int(round(fps)))
            stream.width = int(w)
            stream.height = int(h)
            if pix_fmt:
                stream.pix_fmt = pix_fmt

            # One synthetic frame is enough for the muxer to finalize headers
            x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
            r = np.broadcast_to(x, (h, w))
            g = np.broadcast_to(y, (h, w))
            b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
            rgb = np.dstack([r, g, b])
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            vf = vf.reformat(w, h, format=str(stream.pix_fmt or "yuv420p"))
            for pkt in stream.encode(vf):
                out.mux(pkt)
            for pkt in stream.encode():  # flush
                out.mux(pkt)
        finally:
            out.close()

        # Read back and fetch avcC from the demuxed codec context
        inc = av.open(tmp_path, mode="r")
        try:
            vstreams = [s for s in inc.streams if getattr(s, "type", "") == "video"]
            if not vstreams:
                raise RuntimeError("No video stream found in probe file")
            st = vstreams[0]
            extra = getattr(st.codec_context, "extradata", None)
            avcc = bytes(extra) if extra else b""
            if not (len(avcc) >= 5 and avcc[0] == 1):
                raise RuntimeError("avcC not found in stream extradata")
            # Validate parse
            parse_avcc(avcc)
            return avcc
        finally:
            inc.close()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def encode_decode_demo(
    encoder: str,
    w: int,
    h: int,
    fps: float,
    frames: int,
    pix_fmt: Optional[str],
    use_container_encoder: bool = True,
) -> None:
    import av
    from napari_cuda.client.streaming.decoders.vt import VTLiveDecoder

    if not _is_darwin():
        raise RuntimeError("VT decode demo requires macOS (VideoToolbox)")

    # 1) Probe avcC once via muxer
    avcc = probe_avcc_via_muxer(encoder, w, h, fps, pix_fmt)
    sps, pps, nsz = parse_avcc(avcc)
    print(f"Probed avcC: nal_len_size={nsz} sps={len(sps)} pps={len(pps)}")

    # 2) Initialize VT decoder from avcC
    vt = VTLiveDecoder(avcc, w, h)

    # 3) Create encoder (container-backed or raw) for steady-state packet production
    stream = None
    enc = None
    out = None
    if use_container_encoder:
        out = av.open(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name, mode="w")
        stream = out.add_stream(encoder, rate=int(round(fps)))
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = pix_fmt or ("nv12" if encoder == "h264_videotoolbox" else "yuv420p")
        # Low-latency hints
        try:
            if encoder == "h264_videotoolbox":
                stream.codec_context.options = {"realtime": "1"}
                try:
                    stream.codec_context.max_b_frames = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif encoder == "libx264":
                stream.codec_context.options = {
                    "tune": "zerolatency",
                    "preset": "veryfast",
                    "bf": "0",
                    "x264-params": "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",
                    "annexb": "0",
                }
        except Exception:
            pass
    else:
        enc = av.CodecContext.create(encoder, "w")
        enc.width = int(w)
        enc.height = int(h)
        enc.pix_fmt = pix_fmt or ("nv12" if encoder == "h264_videotoolbox" else "yuv420p")
        enc.time_base = Fraction(1, int(round(fps)))
        # Encourage immediate keyframes / low-latency where sensible
        try:
            if encoder == "h264_videotoolbox":
                enc.options = {"realtime": "1"}
                try:
                    enc.max_b_frames = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif encoder == "libx264":
                enc.options = {
                    "tune": "zerolatency",
                    "preset": "veryfast",
                    "bf": "0",
                    "x264-params": "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1",
                    "annexb": "0",
                }
        except Exception:
            pass
        enc.open()

    # 4) Generate frames, encode, normalize to AVCC and feed to VT
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    r = np.broadcast_to(x, (h, w))
    g = np.broadcast_to(y, (h, w))
    b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
    emitted = decoded = 0

    def pkt_bytes(pkt) -> bytes:
        try:
            return pkt.to_bytes()
        except Exception:
            try:
                return bytes(pkt)
            except Exception:
                return memoryview(pkt).tobytes()

    def repack_to_len(au_bytes: bytes, preferred: int) -> Tuple[bytes, list[int]]:
        # Return AVCC AU with desired length size and nal types
        if is_annexb(au_bytes):
            nals = split_annexb(au_bytes)
        else:
            nals = []
            src_len = None
            for nsz_try in (4, 3, 2, 1):
                nals = split_avcc_by_len(au_bytes, nsz_try)
                if nals:
                    src_len = nsz_try
                    break
            if not nals:
                return au_bytes, []
            if src_len != preferred:
                # Repack to preferred size
                ba = bytearray()
                for n in nals:
                    ba.extend(len(n).to_bytes(preferred, 'big'))
                    ba.extend(n)
                au_bytes = bytes(ba)
        nts = [(n[0] & 0x1F) for n in nals if n]
        return au_bytes, nts

    for i in range(max(1, int(frames))):
        rgb = np.dstack([r, g, b])
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        try:
            vf = vf.reformat(w, h, format=str(enc.pix_fmt))
        except Exception:
            pass
        # Force an IDR on the first frame for faster decoder startup
        try:
            if i == 0:
                vf.pict_type = "I"  # type: ignore[attr-defined]
        except Exception:
            pass
        if use_container_encoder:
            pkts = list(stream.encode(vf))  # type: ignore[union-attr]
        else:
            pkts = enc.encode(vf)  # type: ignore[union-attr]
        if not pkts:
            continue
        au_raw = b"".join(pkt_bytes(p) for p in pkts)
        # Normalize to preferred AVCC length size and collect types
        au, nts = repack_to_len(au_raw, nsz)
        if i < 3:
            has_idr = any(t == 5 for t in nts)
            print(f"AU{i} types={nts} idr={has_idr} len={len(au)}")
        # For keyframes, ensure SPS/PPS are present in-band to satisfy VT
        if any(t == 5 for t in nts):
            need_sps = all(t != 7 for t in nts)
            need_pps = all(t != 8 for t in nts)
            if need_sps or need_pps:
                prefix = bytearray()
                if need_sps and sps:
                    prefix.extend(len(sps[0]).to_bytes(nsz, 'big'))
                    prefix.extend(sps[0])
                if need_pps and pps:
                    prefix.extend(len(pps[0]).to_bytes(nsz, 'big'))
                    prefix.extend(pps[0])
                au = bytes(prefix) + au
        ts = float(i) / float(max(1.0, fps))
        ok = vt.decode(au, ts)
        emitted += 1
        # Try to pull any frame immediately available
        # Try for a brief moment to allow VT output
        t0 = time.time()
        while True:
            item = vt.get_frame_nowait()
            if item:
                cap, pts = item
                arr = None
                try:
                    from napari_cuda.client.vt_shim import VTShimDecoder as _VT
                    arr = _VT.map_to_rgb(vt._shim._vt, cap)  # type: ignore[attr-defined]
                except Exception:
                    pass
                shp = None if arr is None else tuple(arr.shape)
                print(f"decoded frame {decoded} pts={pts} shape={shp}")
                decoded += 1
                break
            if (time.time() - t0) > 0.02:  # ~20ms per frame budget
                break
            time.sleep(0.001)

    # Flush encoder and decoder
    try:
        if use_container_encoder:
            for p in stream.encode():  # type: ignore[union-attr]
                au_raw = pkt_bytes(p)
                au, _ = repack_to_len(au_raw, nsz)
                _ = vt.decode(au, None)
            if out:
                try:
                    out.close()
                except Exception:
                    pass
        else:
            for p in enc.encode(None):  # type: ignore[union-attr]
                au = pkt_bytes(p)
                if is_annexb(au):
                    au = annexb_to_avcc(au, out_len=nsz)
                _ = vt.decode(au, None)
    except Exception:
        pass
    try:
        vt.flush()
    except Exception:
        pass
    # Drain VT with a short timeout window
    drained = 0
    t_end = time.time() + 1.0
    while time.time() < t_end:
        item = vt.get_frame_nowait()
        if not item:
            time.sleep(0.005)
            continue
        drained += 1
    decoded += drained
    sub, out, qlen = vt.counts()
    print(f"done: emitted={emitted} decoded={decoded} vt_counts submit={sub} output={out} qlen={qlen}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VT encode/decode demo using muxer avcC probe")
    ap.add_argument("--encoder", default=("h264_videotoolbox" if _is_darwin() else "libx264"))
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--pixfmt", default=None)
    args = ap.parse_args()

    encode_decode_demo(args.encoder, int(args.w), int(args.h), float(args.fps), int(args.frames), args.pixfmt, True)


if __name__ == "__main__":
    main()
