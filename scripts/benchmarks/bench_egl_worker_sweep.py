#!/usr/bin/env python
"""
Sweep benchmark for EGLRendererWorker across resolutions and save JSON.

Runs the headless EGL + VisPy + CUDA interop + NVENC path to measure server-side
KPIs that match the production pipeline.

Environment variables:
- BENCH_FRAMES=100          Number of frames per resolution
- BENCH_USE_VOLUME=0/1      Use Volume visual (MIP) instead of 2D Image
- OUT_JSON=egl_worker_results.json  Output file path (default shown)

Prints a concise summary and writes per-resolution stats to JSON.
"""

from __future__ import annotations

import os
import json
import statistics as stats
import numpy as np
from typing import Dict, Any, List, Tuple
import argparse
from pathlib import Path
from datetime import datetime

from napari_cuda.server.egl_worker import EGLRendererWorker


RESOLUTIONS: List[Tuple[int, int, str]] = [
    (1280, 720, "HD"),
    (1920, 1080, "Full HD"),
    (2560, 1440, "2K/QHD"),
    (3840, 2160, "4K/UHD"),
]

FRAMES = int(os.getenv("BENCH_FRAMES", "1000"))
USE_VOLUME = os.getenv("BENCH_USE_VOLUME", "0") not in {"0", "false", "False"}
# If OUT_JSON is set explicitly, use it as a file path; otherwise we'll build a dated filename in --out-dir
OUT_JSON = os.getenv("OUT_JSON")
# Volume parameters
ENV_VOL_DEPTH = int(os.getenv("VOL_DEPTH", "64"))
ENV_VOL_DTYPE = os.getenv("VOL_DTYPE", "float32")
ENV_VOL_STEP = os.getenv("VOL_STEP")


def mean_or_none(vals):
    if not vals:
        return None
    return float(stats.fmean(vals))

def p95_p99(vals):
    if not vals:
        return (None, None)
    v = np.asarray(vals, dtype=float)
    return (float(np.percentile(v, 95)), float(np.percentile(v, 99)))


def run_once(width: int, height: int, frames: int, use_volume: bool,
             vol_depth: int, vol_dtype: str, vol_step: float | None) -> Dict[str, Any]:
    worker = EGLRendererWorker(
        width=width,
        height=height,
        use_volume=use_volume,
        volume_depth=vol_depth,
        volume_dtype=vol_dtype,
        volume_relative_step=vol_step,
    )
    # Warmup
    for i in range(10):
        worker.render_frame(azimuth_deg=i * 3)
        worker.capture_and_encode()

    render_ms: List[float] = []
    blit_ns: List[int] = []
    map_ms: List[float] = []
    copy_ms: List[float] = []
    enc_ms: List[float] = []
    total_ms: List[float] = []
    sizes: List[int] = []

    for i in range(frames):
        worker.render_frame(azimuth_deg=(i * 3) % 360)
        t = worker.capture_and_encode()
        render_ms.append(t.render_ms)
        if t.blit_gpu_ns is not None:
            blit_ns.append(t.blit_gpu_ns)
        map_ms.append(t.map_ms)
        copy_ms.append(t.copy_ms)
        enc_ms.append(t.encode_ms)
        total_ms.append(t.total_ms)
        if t.packet_bytes:
            sizes.append(t.packet_bytes)

    worker.cleanup()

    # Summaries
    m_total = mean_or_none(total_ms)
    fps = (1000.0 / m_total) if m_total else None
    avg_pkt = float(np.mean(sizes)) if sizes else None
    bitrate_mbps = (fps * avg_pkt * 8 / 1e6) if (fps and avg_pkt) else None
    blit_ms = (float(np.mean(blit_ns) / 1e6) if blit_ns else None)

    # Percentiles and stddevs
    r_p95, r_p99 = p95_p99(render_ms)
    m_p95, m_p99 = p95_p99(map_ms)
    c_p95, c_p99 = p95_p99(copy_ms)
    e_p95, e_p99 = p95_p99(enc_ms)
    t_p95, t_p99 = p95_p99(total_ms)
    b_p95 = b_p99 = None
    if blit_ns:
        blit_ms_vals = [ns / 1e6 for ns in blit_ns]
        b_p95, b_p99 = p95_p99(blit_ms_vals)
    total_std = float(np.std(total_ms)) if total_ms else None

    result = {
        "width": width,
        "height": height,
        "frames": frames,
        "use_volume": use_volume,
        "render_ms": mean_or_none(render_ms),
        "render_p95_ms": r_p95,
        "render_p99_ms": r_p99,
        "capture_blit_gpu_ms": blit_ms,
        "capture_blit_p95_ms": b_p95,
        "capture_blit_p99_ms": b_p99,
        "map_ms": mean_or_none(map_ms),
        "map_p95_ms": m_p95,
        "map_p99_ms": m_p99,
        "copy_ms": mean_or_none(copy_ms),
        "copy_p95_ms": c_p95,
        "copy_p99_ms": c_p99,
        "encode_ms": mean_or_none(enc_ms),
        "encode_p95_ms": e_p95,
        "encode_p99_ms": e_p99,
        "total_ms": m_total,
        "total_p95_ms": t_p95,
        "total_p99_ms": t_p99,
        "total_std_ms": total_std,
        "fps": fps,
        "avg_packet_bytes": avg_pkt,
        "bitrate_mbps": bitrate_mbps,
    }
    if use_volume:
        result.update({"volume_depth": vol_depth, "volume_dtype": vol_dtype, "volume_relative_step": vol_step})
    return result


def main():
    parser = argparse.ArgumentParser(description="EGLRendererWorker resolution sweep")
    parser.add_argument("--frames", type=int, default=FRAMES, help="Frames per resolution (default from BENCH_FRAMES or 100)")
    parser.add_argument("--visual", choices=["image", "volume", "both"], default="both",
                        help="Which visual(s) to benchmark (default: both)")
    parser.add_argument("--volume", action="store_true", help="[Deprecated] Same as --visual volume")
    parser.add_argument("--out", default=OUT_JSON, help="Output path or directory. If ends with .json, used as file; otherwise treated as directory.")
    parser.add_argument("--out-dir", default=os.getenv("OUT_DIR", "benchmarks/egl_worker"), help="Directory to save results when --out is not a file path")
    parser.add_argument("--vol-depth", type=int, default=ENV_VOL_DEPTH, help="Volume depth (slices), default 64 or VOL_DEPTH env")
    parser.add_argument("--vol-dtype", choices=["float32", "float16", "uint8"], default=ENV_VOL_DTYPE, help="Volume dtype")
    parser.add_argument("--vol-step", type=float, default=(float(ENV_VOL_STEP) if ENV_VOL_STEP is not None else None), nargs="?", help="Volume relative step size (optional)")
    args = parser.parse_args()

    frames = int(args.frames)
    visual_arg = args.visual
    if args.volume:
        visual_arg = "volume"
    visuals = [visual_arg] if visual_arg in ("image", "volume") else ["image", "volume"]
    vol_depth = int(args.vol_depth)
    vol_dtype = str(args.vol_dtype)
    vol_step = float(args.vol_step) if args.vol_step is not None else None
    # Determine output path
    out_arg = args.out if args.out is not None else OUT_JSON
    out_dir = Path(args.out_dir)
    out_path: Path
    if out_arg and out_arg.lower().endswith('.json'):
        out_path = Path(out_arg)
        out_dir = out_path.parent
    else:
        # Build filename with args and timestamp
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        visual_tag = visual_arg
        vol_tag = ""
        if visual_tag in ("volume", "both"):
            step_tag = f"_s{vol_step}" if vol_step is not None else ""
            vol_tag = f"_vD{vol_depth}_vT{vol_dtype}{step_tag}"
        filename = f"egl_worker_{visual_tag}_{frames}f{vol_tag}_{ts}.json"
        out_dir = Path(out_arg) if out_arg and out_arg != OUT_JSON else out_dir
        out_path = out_dir / filename

    print("EGLRendererWorker sweep")
    extra = f", vol_depth={vol_depth}, vol_dtype={vol_dtype}, vol_step={vol_step}" if ('volume' in visuals) else ''
    print(f"Frames per test: {frames}; Visuals: {', '.join(visuals)}{extra}\n")

    results: Dict[str, Any] = {"frames": frames, "runs": {}}

    for vis in visuals:
        print(f"=== Visual: {vis} ===")
        use_volume = (vis == "volume")
        per_res: List[Dict[str, Any]] = []
        for w, h, name in RESOLUTIONS:
            print(f"Testing {w}x{h} ({name})...")
            summary = run_once(w, h, frames, use_volume, vol_depth, vol_dtype, vol_step)
            per_res.append(summary)
            ms = summary["total_ms"] or 0.0
            fps = summary["fps"] or 0.0
            print(f"  TOTAL: {ms:.3f} ms  ({fps:.0f} FPS)\n")
        results["runs"][vis] = per_res

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {out_path}")


if __name__ == "__main__":
    main()
