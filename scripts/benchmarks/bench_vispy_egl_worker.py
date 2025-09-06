#!/usr/bin/env python
"""
Benchmark VisPy rendering with EGLRendererWorker (no Qt), measuring:
- Render (CPU wall time)
- Capture blit (GPU ns via timer query)
- CUDA map (CPU)
- CUDA Memcpy2D (GPU events)
- NVENC Encode (CPU wall time)
- Total per-frame time

Environment:
- BENCH_WIDTH, BENCH_HEIGHT, BENCH_FRAMES
- BENCH_USE_VOLUME=1 to test Volume visual (MIP)

This exercises the exact path intended for headless server rendering.
"""

import os
import statistics as stats
import numpy as np

from napari_cuda.server.egl_worker import EGLRendererWorker

WIDTH = int(os.getenv('BENCH_WIDTH', '1920'))
HEIGHT = int(os.getenv('BENCH_HEIGHT', '1080'))
FRAMES = int(os.getenv('BENCH_FRAMES', '100'))
USE_VOLUME = os.getenv('BENCH_USE_VOLUME', '0') not in {'0', 'false', 'False'}


def mean_std(a):
    return (stats.fmean(a), (stats.pstdev(a) if len(a) > 1 else 0.0))


def main():
    print(f"EGLRendererWorker Benchmark: {WIDTH}x{HEIGHT}, frames={FRAMES}, volume={USE_VOLUME}")
    worker = EGLRendererWorker(width=WIDTH, height=HEIGHT, use_volume=USE_VOLUME)

    # Warmup
    for i in range(10):
        worker.render_frame(azimuth_deg=i * 3)
        worker.capture_and_encode()

    render_ms, blit_ns, map_ms, copy_ms, enc_ms, total_ms, sizes = ([] for _ in range(7))

    for i in range(FRAMES):
        # simple motion
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
        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}: total={t.total_ms:.2f} ms, render={t.render_ms:.2f} ms")

    # Report
    print('\n' + '=' * 60)
    print('EGLRendererWorker RESULTS')
    print(f'Resolution: {WIDTH}x{HEIGHT}, Frames: {FRAMES}, Visual: {"Volume" if USE_VOLUME else "Image"}')
    print('=' * 60)
    m_r, s_r = mean_std(render_ms)
    print(f'Render (CPU return): {m_r:.3f} ± {s_r:.3f} ms')
    if blit_ns:
        print(f'Capture blit (GPU):  {np.mean(blit_ns)/1e6:.3f} ms')
    m_m, s_m = mean_std(map_ms)
    print(f'Map GL texture:      {m_m:.3f} ± {s_m:.3f} ms')
    m_c, s_c = mean_std(copy_ms)
    print(f'Copy GL→CuPy:        {m_c:.3f} ± {s_c:.3f} ms')
    m_e, s_e = mean_std(enc_ms)
    print(f'NVENC Encode:        {m_e:.3f} ± {s_e:.3f} ms')
    m_t, s_t = mean_std(total_ms)
    print('-' * 60)
    print(f'TOTAL:               {m_t:.3f} ± {s_t:.3f} ms  ({(1000.0/m_t if m_t>0 else 0):.0f} FPS)')
    if sizes:
        fps = 1000.0/m_t if m_t>0 else 0
        bitrate_mbps = (np.mean(sizes) * 8 * fps) / 1e6
        print(f'Bitrate estimate:    {bitrate_mbps:.2f} Mbps (avg {np.mean(sizes)/1024:.1f} KB/frame)')

    worker.cleanup()


if __name__ == '__main__':
    main()
