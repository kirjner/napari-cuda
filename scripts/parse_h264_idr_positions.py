#!/usr/bin/env python3
"""
Lightweight Annex B H.264 parser to list positions of IDR NAL units.

This is not a full AU parser, but good enough to spot IDR NALs (type=5)
and print their indices in the NAL stream. For precise keyframe timing,
use ffprobe; this script is a fallback when ffprobe isn't available.
"""
import sys
from pathlib import Path


def find_start_codes(data: bytes):
    i = 0
    n = len(data)
    while i < n - 3:
        if data[i] == 0 and data[i+1] == 0 and data[i+2] == 1:
            yield i, 3
            i += 3
        elif i < n - 4 and data[i] == 0 and data[i+1] == 0 and data[i+2] == 0 and data[i+3] == 1:
            yield i, 4
            i += 4
        else:
            i += 1

def parse_annexb(data: bytes):
    nal_starts = list(find_start_codes(data))
    nal_indices = []
    for idx, (pos, sc_len) in enumerate(nal_starts):
        hdr_pos = pos + sc_len
        if hdr_pos >= len(data):
            continue
        hdr = data[hdr_pos]
        nal_type = hdr & 0x1F
        nal_indices.append((idx, pos, sc_len, nal_type))
    return nal_indices

def parse_avcc(data: bytes):
    nal_indices = []
    i = 0
    idx = 0
    n = len(data)
    while i + 4 <= n:
        # 4-byte big-endian length
        ln = int.from_bytes(data[i:i+4], 'big')
        i += 4
        if ln <= 0 or i + ln > n:
            break
        hdr = data[i] if ln >= 1 else 0
        nal_type = hdr & 0x1F
        nal_indices.append((idx, i, 4, nal_type))
        i += ln
        idx += 1
    return nal_indices

def main():
    if len(sys.argv) < 2:
        print("Usage: parse_h264_idr_positions.py <input.h264>")
        sys.exit(1)
    p = Path(sys.argv[1])
    data = p.read_bytes()
    # Try Annex B first
    nal_indices = parse_annexb(data)
    if len(nal_indices) == 0:
        nal_indices = parse_avcc(data)

    # Print a small summary of IDR occurrences
    idr_positions = [i for (i, pos, sc, t) in nal_indices if t == 5]
    print(f"NALs parsed: {len(nal_indices)}")
    print(f"IDR NAL count: {len(idr_positions)}")
    print("First few IDR NAL indices:", idr_positions[:10])

    # Also show positions of SPS/PPS to confirm header cadence
    sps_positions = [i for (i, pos, sc, t) in nal_indices if t == 7]
    pps_positions = [i for (i, pos, sc, t) in nal_indices if t == 8]
    print("First few SPS indices:", sps_positions[:5])
    print("First few PPS indices:", pps_positions[:5])

if __name__ == '__main__':
    main()
