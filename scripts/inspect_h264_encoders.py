#!/usr/bin/env python3
from __future__ import annotations

"""
Inspect H.264 encoders (libx264, h264_videotoolbox) and print bitstream details.

This tool encodes a few synthetic frames and reports:
- Encoder name, pixel format, extradata presence/size and whether it's avcC
- If avcC, parsed nal_length_size and SPS/PPS lengths
- For the first N packets: AnnexB/AVCC detection and NAL types per AU
- Whether SPS/PPS/IDR/AUD appear in-stream (repeat-headers behavior)

Usage examples:
- uv run python scripts/inspect_h264_encoders.py --encoder libx264 --w 1920 --h 1080 --fps 60 \
    --x264-params keyint=1:scenecut=0:repeat-headers=1 --annexb 1
- uv run python scripts/inspect_h264_encoders.py --encoder h264_videotoolbox --w 1920 --h 1080 --fps 60
"""

import argparse
import logging
from fractions import Fraction

import numpy as np

from napari_cuda.codec.avcc import (
    is_annexb,
    parse_avcc,
    split_annexb,
    split_avcc_by_len,
)


def nal_name_h264(b0: int) -> str:
    t = b0 & 0x1F
    names = {
        1: "nonIDR",
        5: "IDR",
        6: "SEI",
        7: "SPS",
        8: "PPS",
        9: "AUD",
    }
    return names.get(t, str(t))


def analyze_packets(pkts: list[bytes], avcc_len_size_hint: int | None = None) -> list[list[str]]:
    out: list[list[str]] = []
    for data in pkts:
        names: list[str] = []
        if is_annexb(data):
            for nal in split_annexb(data):
                if nal:
                    names.append(nal_name_h264(nal[0]))
        else:
            # Try multiple length sizes
            tried = [avcc_len_size_hint] if avcc_len_size_hint in (1, 2, 3, 4) else []
            tried += [n for n in (4, 3, 2, 1) if n not in tried]
            nals: list[bytes] = []
            for nsz in tried:
                nals = split_avcc_by_len(data, nsz)
                if nals:
                    break
            for nal in nals:
                if nal:
                    names.append(nal_name_h264(nal[0]))
        out.append(names)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect H.264 encoder outputs")
    ap.add_argument("--encoder", default="libx264", help="libx264 | h264_videotoolbox | h264")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--flush", action="store_true", help="Call enc.encode(None) once at end and include results")
    ap.add_argument("--pixfmt", default=None, help="Override pixel format (e.g., yuv420p, nv12)")
    ap.add_argument("--annexb", default=None, help="Set encoder option annexb=0/1")
    ap.add_argument("--x264-params", dest="x264_params", default=None, help="x264-params string")
    ap.add_argument("--realtime", default=None, help="Set encoder option realtime=1 for VT")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    log = logging.getLogger("inspect")

    import av

    enc = av.CodecContext.create(args.encoder, "w")
    enc.width = int(args.w)
    enc.height = int(args.h)
    if args.pixfmt:
        enc.pix_fmt = args.pixfmt
    else:
        if args.encoder == "h264_videotoolbox":
            enc.pix_fmt = "nv12"
        else:
            enc.pix_fmt = "yuv420p"
    enc.time_base = Fraction(1, int(round(args.fps)))

    opts = {}
    if args.x264_params is not None:
        opts["x264-params"] = args.x264_params
    if args.annexb is not None:
        opts["annexb"] = str(args.annexb)
    if args.realtime is not None:
        opts["realtime"] = str(args.realtime)
    if opts:
        enc.options = opts

    enc.open()

    # Note: Many encoders (esp. hardware) do not populate CodecContext.extradata
    # in this raw mode. Some provide avcC via packet side-data (NEW_EXTRADATA).
    ed = getattr(enc, "extradata", None)
    edb = bytes(ed) if ed else b""
    ed_len = len(edb)
    ed_head = edb[:8].hex() if ed_len > 0 else ""
    # Also print codec flags (to check GLOBAL_HEADER)
    try:
        from av.codec.context import Flags as CFlags  # type: ignore
        gh = bool(int(enc.flags) & int(CFlags.GLOBAL_HEADER))  # type: ignore[attr-defined]
    except Exception:
        gh = False
    log.info("encoder=%s pixfmt=%s global_header=%s extradata_len=%d head=%s",
             enc.name, enc.pix_fmt, gh, ed_len, ed_head)
    avcc_len_size = None
    if ed_len >= 5 and edb[0] == 1:
        try:
            sps_list, pps_list, nal_len_size = parse_avcc(edb)
            avcc_len_size = nal_len_size
            log.info("extradata=avcC nal_len_size=%d sps=%d pps=%d", nal_len_size, len(sps_list), len(pps_list))
        except Exception as e:
            log.info("extradata=avcC parse failed: %s", e)
    elif ed_len > 0:
        # Try to see if extradata contains AnnexB NALs
        ann = is_annexb(bytes(ed))
        log.info("extradata_annexb=%s", ann)

    # Generate frames
    w = int(args.w); h = int(args.h)
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    r = np.broadcast_to(x, (h, w))
    g = np.broadcast_to(y, (h, w))
    b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)

    frames_pkts: list[list[bytes]] = []
    # Some encoders (VT, certain wrappers) attach avcC as NEW_EXTRADATA on first packets.
    # Capture it if present so we can parse SPS/PPS and nal length size.
    pkt_side_extradata: bytes | None = None
    for i in range(max(1, int(args.frames))):
        rgb = np.dstack([r, g, b])
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        try:
            vf = vf.reformat(w, h, format=str(enc.pix_fmt))
        except Exception:
            pass
        out = enc.encode(vf)
        group: list[bytes] = []
        for p in out:
            try:
                group.append(p.to_bytes())
            except Exception:
                try:
                    group.append(bytes(p))
                except Exception:
                    group.append(memoryview(p).tobytes())
            # Extract NEW_EXTRADATA if present
            if pkt_side_extradata is None:
                try:
                    sd_list = getattr(p, "side_data", None)
                    if sd_list:
                        for sd in sd_list:
                            t = getattr(sd, "type", None)
                            tname = str(t).lower() if t is not None else ""
                            if "new_extradata" in tname:
                                try:
                                    pkt_side_extradata = sd.to_bytes()  # type: ignore[attr-defined]
                                except Exception:
                                    try:
                                        pkt_side_extradata = bytes(sd)  # type: ignore[arg-type]
                                    except Exception:
                                        pkt_side_extradata = memoryview(sd).tobytes()  # type: ignore[arg-type]
                                break
                except Exception:
                    pass
        frames_pkts.append(group)

    # If we found side-data extradata, prefer that for avcC parsing
    if pkt_side_extradata and len(pkt_side_extradata) >= 5 and pkt_side_extradata[0] == 1:
        try:
            sps_list, pps_list, nal_len_size = parse_avcc(pkt_side_extradata)
            avcc_len_size = nal_len_size
            log.info(
                "packet_side_extradata=avcC nal_len_size=%d sps=%d pps=%d",
                nal_len_size,
                len(sps_list),
                len(pps_list),
            )
        except Exception as e:
            log.info("packet_side_extradata parse failed: %s", e)

    # Analyze
    # Optionally flush delayed packets
    if args.flush:
        try:
            flush_pkts = enc.encode(None)
            group: list[bytes] = []
            for p in flush_pkts:
                try:
                    group.append(p.to_bytes())
                except Exception:
                    try:
                        group.append(bytes(p))
                    except Exception:
                        group.append(memoryview(p).tobytes())
            if group:
                frames_pkts.append(group)
        except Exception as e:
            logging.getLogger("inspect").info("flush encode(None) raised: %s", e)

    show_n = min(len(frames_pkts), max(1, int(args.frames)))
    for i in range(show_n):
        group = frames_pkts[i]
        names = analyze_packets([b"".join(group)], avcc_len_size_hint=avcc_len_size)[0]
        head = (group[0][:8].hex()) if group else ""
        log.info("frame=%d au_head=%s nals=%s", i, head, names)

    # Summary
    have_sps = False; have_pps = False; have_idr = False; have_aud = False
    for group in frames_pkts[:show_n]:
        names = analyze_packets([b"".join(group)], avcc_len_size_hint=avcc_len_size)[0]
        have_sps = have_sps or any(n == "SPS" for n in names)
        have_pps = have_pps or any(n == "PPS" for n in names)
        have_idr = have_idr or any(n == "IDR" for n in names)
        have_aud = have_aud or any(n == "AUD" for n in names)
    log.info("summary: sps=%s pps=%s idr=%s aud=%s", have_sps, have_pps, have_idr, have_aud)


if __name__ == "__main__":
    main()
