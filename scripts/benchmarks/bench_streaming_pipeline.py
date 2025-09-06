#!/usr/bin/env python
"""
Production-ready benchmark for the napari-cuda streaming pipeline.

Supports two modes:
1. Simulated texture mode (default): Creates a test texture to measure encoding pipeline
2. Napari/VisPy mode: Full end-to-end with real napari rendering

Environment Variables:
- BENCH_USE_NAPARI=1  Enable napari/vispy rendering (default: 0)
- BENCH_WIDTH=1920    Frame width (default: 1920)
- BENCH_HEIGHT=1080   Frame height (default: 1080)  
- BENCH_FRAMES=100    Number of frames to benchmark (default: 100)
- BENCH_CPU_UPLOAD=1  Simulate CPU->GPU upload in texture mode (default: 1)

Measures:
- Napari render time (if enabled)
- Texture upload/generation (if CPU upload)
- GL->CUDA map
- CUDA 2D copy GL->CuPy
- CuPy->Torch zero-copy bridge
- NVENC Encode() call latency
"""

import os
import json
import time
import statistics as stats
from typing import Optional

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import cupy as cp
import torch
import PyNvVideoCodec as pnvc
from OpenGL import GL
import pycuda.driver as cuda
import pycuda.gl
from pycuda.gl import RegisteredImage, graphics_map_flags

WIDTH, HEIGHT = int(os.getenv('BENCH_WIDTH', '1920')), int(os.getenv('BENCH_HEIGHT', '1080'))
FRAMES = int(os.getenv('BENCH_FRAMES', '100'))
# If True, simulates CPU-side compute + CPU->GL texture update via glTexSubImage2D
# If False, assumes texture already populated on GPU and measures only encode path
CPU_UPLOAD = os.getenv('BENCH_CPU_UPLOAD', '1') not in {'0', 'false', 'False'}
# If True, use napari/vispy for real rendering instead of simulated texture
USE_NAPARI = os.getenv('BENCH_USE_NAPARI', '0') not in {'0', 'false', 'False'}
# Optional JSON output path for machine-readable results
OUT_JSON = os.getenv('BENCH_OUT_JSON')


def torch_from_cupy_zero_copy(arr: cp.ndarray) -> torch.Tensor:
    """
    Import a CuPy array into Torch without a copy.

    Uses CuPy's toDlpack exporter and Torch's from_dlpack importer.
    """
    dl = arr.toDlpack()
    return torch.utils.dlpack.from_dlpack(dl)


def cuda_event_ms(fn) -> float:
    """Time a callable GPU function with CUDA events and return elapsed milliseconds."""
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.time_till(end)


def gl_timer_query_ns(fn) -> Optional[int]:
    """Time GL function with GL timer query (returns nanoseconds, or None on error)"""
    import ctypes
    try:
        query_id = GL.glGenQueries(1)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, query_id)
        fn()
        GL.glEndQuery(GL.GL_TIME_ELAPSED)
        
        # Wait for result
        available = GL.GLint(0)
        max_wait = 100
        while not available.value and max_wait > 0:
            GL.glGetQueryObjectiv(query_id, GL.GL_QUERY_RESULT_AVAILABLE, available)
            max_wait -= 1
        
        if available.value:
            result = GL.GLuint64(0)
            GL.glGetQueryObjectui64v(query_id, GL.GL_QUERY_RESULT, result)
            GL.glDeleteQueries(1, [query_id])
            return int(result.value)
        else:
            GL.glDeleteQueries(1, [query_id])
            return None
    except Exception:
        return None

def gl_fence_cpu_wait_ms() -> float:
    """Block CPU until GPU reaches the fence; return CPU wait in ms.

    Uses GL sync objects to measure how long the CPU waits for the specific
    upload to complete, instead of a coarse glFinish().
    """
    import time
    try:
        sync = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        GL.glFlush()
        t0 = time.perf_counter()
        # Wait with a reasonable timeout; loop until signaled
        # 1e9 ns = 1s timeout units in nanoseconds for client wait
        while True:
            status = GL.glClientWaitSync(sync, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000)
            # GL_ALREADY_SIGNALED (0x911A) or GL_CONDITION_SATISFIED (0x911C)
            if status in (GL.GL_ALREADY_SIGNALED, GL.GL_CONDITION_SATISFIED):
                break
        dt_ms = (time.perf_counter() - t0) * 1000.0
        GL.glDeleteSync(sync)
        return dt_ms
    except Exception:
        # Fallback to glFinish timing if sync not supported
        import time
        t0 = time.perf_counter()
        GL.glFinish()
        return (time.perf_counter() - t0) * 1000.0


def setup_egl_context():
    """Setup EGL context for headless OpenGL."""
    import OpenGL.EGL as EGL
    import ctypes
    
    egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    if egl_display == EGL.EGL_NO_DISPLAY:
        raise RuntimeError("Failed to get EGL display")
    
    major = EGL.EGLint()
    minor = EGL.EGLint()
    if not EGL.eglInitialize(egl_display, major, minor):
        raise RuntimeError("Failed to initialize EGL")
    
    config_attribs = [
        EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
        EGL.EGL_RED_SIZE, 8,
        EGL.EGL_GREEN_SIZE, 8,
        EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_ALPHA_SIZE, 8,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
        EGL.EGL_NONE
    ]
    config_attribs_p = (EGL.EGLint * len(config_attribs))(*config_attribs)
    egl_config = EGL.EGLConfig()
    num_configs = EGL.EGLint()
    if not EGL.eglChooseConfig(egl_display, config_attribs_p, ctypes.byref(egl_config), 1, ctypes.byref(num_configs)):
        raise RuntimeError("Failed to choose EGL config")
    
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    egl_context = EGL.eglCreateContext(egl_display, egl_config, EGL.EGL_NO_CONTEXT, None)
    if egl_context == EGL.EGL_NO_CONTEXT:
        raise RuntimeError("Failed to create EGL context")
    
    # Create pbuffer surface
    pbuffer_attribs = [
        EGL.EGL_WIDTH, WIDTH,
        EGL.EGL_HEIGHT, HEIGHT,
        EGL.EGL_NONE
    ]
    pbuffer_attribs_p = (EGL.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    egl_surface = EGL.eglCreatePbufferSurface(egl_display, egl_config, pbuffer_attribs_p)
    if egl_surface == EGL.EGL_NO_SURFACE:
        raise RuntimeError("Failed to create EGL pbuffer surface")
    
    if not EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
        raise RuntimeError("Failed to make EGL context current")
    
    return egl_display


def setup_napari_scene(width, height):
    """Setup napari viewer with test scene."""
    import napari
    from qtpy.QtWidgets import QApplication
    
    # Ensure Qt app exists
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    # Create viewer
    viewer = napari.Viewer(show=False)
    viewer.window.resize(width, height)
    
    # Add test data
    volume_data = np.random.rand(50, 512, 512).astype(np.float32)
    viewer.add_image(volume_data, name='volume', colormap='viridis', rendering='mip')
    
    points_data = np.random.rand(200, 3) * [50, 512, 512]
    viewer.add_points(points_data, name='points', size=5, face_color='cyan')
    
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (30, 30, 0)
    viewer.camera.zoom = 0.8
    
    return viewer, app


def benchmark_simulated_texture():
    """Benchmark with simulated texture (original mode)."""
    # Setup EGL
    egl_display = setup_egl_context()
    
    # Create texture
    texture_id = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, WIDTH, HEIGHT, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    
    # CUDA setup
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.retain_primary_context()
    ctx.push()
    
    try:
        pycuda.gl.init()
        registered_texture = RegisteredImage(int(texture_id), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
        
        # Buffers
        dev_frame = cp.empty((HEIGHT, WIDTH, 4), dtype=cp.uint8)
        host = np.random.randint(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8) if CPU_UPLOAD else None
        
        # Encoder
        enc = pnvc.CreateEncoder(width=WIDTH, height=HEIGHT, fmt='ABGR', usecpuinputbuffer=False)

        # Warmup
        for _ in range(10):
            if CPU_UPLOAD:
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, host)
                gl_fence_cpu_wait_ms()

            mapping = registered_texture.map()
            cuda_array = mapping.array(0, 0)
            m = cuda.Memcpy2D()
            m.set_src_array(cuda_array)
            m.set_dst_device(int(dev_frame.data.ptr))
            m.width_in_bytes = WIDTH * 4
            m.height = HEIGHT
            m.dst_pitch = dev_frame.strides[0]
            m(aligned=True)
            mapping.unmap()
            enc.Encode(torch_from_cupy_zero_copy(dev_frame))

        # Timing arrays
        cpu_compute = []
        cpu_gl_upload_ms = []
        gpu_gl_upload_ns = []
        map_times = []
        copy_times = []
        convert_times = []
        encode_times = []
        total_times = []
        encoded_sizes = []

        print(f'Benchmarking {FRAMES} frames (simulated texture mode)...')

        for i in range(FRAMES):
            t0 = time.perf_counter()

            # Simulate compute: touch/modify host buffer to avoid driver shortcuts
            t_cs = time.perf_counter()
            if CPU_UPLOAD:
                # Toggle bits in-place to write the whole buffer
                host ^= 1
            t_ce = time.perf_counter()
            cpu_compute.append((t_ce - t_cs) * 1000)

            # Upload if needed
            gpu_upload_ns = None
            if CPU_UPLOAD:
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                def upload():
                    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, host)
                # GPU time with timer query
                gpu_upload_ns = gl_timer_query_ns(upload)
                # CPU wait via fence to ensure this upload completes
                cpu_wait_ms = gl_fence_cpu_wait_ms()
                cpu_gl_upload_ms.append(cpu_wait_ms)
                if gpu_upload_ns is not None:
                    gpu_gl_upload_ns.append(gpu_upload_ns)

            # Map texture
            t_ms = time.perf_counter()
            mapping = registered_texture.map()
            cuda_array = mapping.array(0, 0)
            t_me = time.perf_counter()
            map_times.append((t_me - t_ms) * 1000)

            # Copy to CuPy
            copy_ms = cuda_event_ms(lambda: (
                (lambda m: (m.set_src_array(cuda_array),
                           m.set_dst_device(int(dev_frame.data.ptr)),
                           setattr(m, 'width_in_bytes', WIDTH * 4),
                           setattr(m, 'height', HEIGHT),
                           setattr(m, 'dst_pitch', dev_frame.strides[0]),
                           m(aligned=True)))(cuda.Memcpy2D())
            ))
            copy_times.append(copy_ms)
            mapping.unmap()

            # Convert to torch
            t_cs2 = time.perf_counter()
            torch_frame = torch_from_cupy_zero_copy(dev_frame)
            t_ce2 = time.perf_counter()
            convert_times.append((t_ce2 - t_cs2) * 1000)

            # Encode
            t_es = time.perf_counter()
            pkt = enc.Encode(torch_frame)
            if pkt:
                encoded_sizes.append(len(pkt))
            t_ee = time.perf_counter()
            encode_times.append((t_ee - t_es) * 1000)

            total_times.append((time.perf_counter() - t0) * 1000)

            if (i + 1) % 20 == 0:
                print(f'  Frame {i+1}: Total={total_times[-1]:.2f}ms')

        final_pkt = enc.EndEncode()
        if final_pkt:
            encoded_sizes.append(len(final_pkt))

        # Cleanup
        registered_texture.unregister()
        GL.glDeleteTextures([texture_id])

        # Return timing data
        return {
            'compute': cpu_compute,
            'upload': cpu_gl_upload_ms,
            'upload_gpu_ns': gpu_gl_upload_ns,
            'map': map_times,
            'copy': copy_times,
            'convert': convert_times,
            'encode': encode_times,
            'total': total_times,
            'encoded_sizes': encoded_sizes
        }
        
    finally:
        ctx.pop()
        ctx.detach()
        import OpenGL.EGL as EGL
        EGL.eglTerminate(egl_display)


def ensure_gl_current(canvas):
    """Ensure the VisPy/Qt GL context is current on this thread."""
    backend = canvas._scene_canvas._backend
    if hasattr(backend, '_vispy_set_current'):
        backend._vispy_set_current()
        return True
    return False


def benchmark_napari_rendering():
    """Benchmark with real napari rendering."""
    from napari_cuda.server.vispy_intercept import GLFrameInterceptor
    
    # Setup napari
    viewer, app = setup_napari_scene(WIDTH, HEIGHT)
    canvas = viewer.window._qt_viewer.canvas
    
    # Force initial draw to create GL resources
    canvas.on_draw(None)
    app.processEvents()
    
    # CRITICAL: Make GL context current BEFORE any GL/CUDA operations
    print("\nMaking GL context current...")
    if not ensure_gl_current(canvas):
        print("ERROR: Could not make GL context current!")
        viewer.close()
        app.quit()
        return None
    print("GL context is current")
    
    # NOW attach interceptor (after context is current)
    interceptor = GLFrameInterceptor(canvas)
    interceptor.attach()
    
    # Check GL context details
    try:
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        version = GL.glGetString(GL.GL_VERSION)
        print(f"\nGL Context Info:")
        print(f"  Vendor: {vendor}")
        print(f"  Renderer: {renderer}")
        print(f"  Version: {version}")
        
        if b'llvmpipe' in renderer.lower() or b'software' in renderer.lower():
            print("WARNING: Software renderer detected! Performance will be poor.")
    except Exception as e:
        print(f"Could not get GL info: {e}")
    
    # CUDA setup - MUST be done while GL context is current!
    print("\nInitializing CUDA...")
    cuda.init()
    dev = cuda.Device(0)
    print(f"CUDA device: {dev.name()}")
    
    ctx = dev.retain_primary_context()
    ctx.push()
    
    try:
        # Ensure GL context is still current before GL-CUDA interop
        ensure_gl_current(canvas)
        
        print("Initializing CUDA-GL interop...")
        pycuda.gl.init()
        print("SUCCESS: CUDA-GL interop initialized!")
        
        # We'll register the texture after first capture
        registered_texture = None
        dev_frame = cp.empty((HEIGHT, WIDTH, 4), dtype=cp.uint8)
        enc = pnvc.CreateEncoder(width=WIDTH, height=HEIGHT, fmt='ABGR', usecpuinputbuffer=False)
        
        # Warmup
        for _ in range(10):
            canvas.on_draw(None)
            app.processEvents()
            
            tex_id = interceptor.texture_id
            if tex_id and not registered_texture:
                registered_texture = RegisteredImage(int(tex_id), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
            
            if registered_texture:
                mapping = registered_texture.map()
                cuda_array = mapping.array(0, 0)
                m = cuda.Memcpy2D()
                m.set_src_array(cuda_array)
                m.set_dst_device(int(dev_frame.data.ptr))
                m.width_in_bytes = WIDTH * 4
                m.height = HEIGHT
                m.dst_pitch = dev_frame.strides[0]
                m(aligned=True)
                mapping.unmap()
                enc.Encode(torch_from_cupy_zero_copy(dev_frame))
        
        # Timing arrays
        render_times = []
        capture_gpu_ns = []
        map_times = []
        copy_times = []
        convert_times = []
        encode_times = []
        total_times = []
        encoded_sizes = []
        
        print(f'Benchmarking {FRAMES} frames (napari/vispy rendering)...')
        
        for i in range(FRAMES):
            t0 = time.perf_counter()
            
            # Animate scene
            viewer.camera.angles = (30, (i * 2) % 360, 0)
            viewer.dims.current_step = (i % 50, 0, 0)
            
            # Render
            t_rs = time.perf_counter()
            canvas.on_draw(None)
            app.processEvents()
            t_re = time.perf_counter()
            render_times.append((t_re - t_rs) * 1000)
            
            # Get capture stats
            stats_cap = interceptor.last_stats
            if stats_cap and stats_cap.gpu_time_ns:
                capture_gpu_ns.append(stats_cap.gpu_time_ns)
            
            tex_id = interceptor.texture_id
            if not tex_id:
                continue
                
            if not registered_texture:
                registered_texture = RegisteredImage(int(tex_id), GL.GL_TEXTURE_2D, graphics_map_flags.READ_ONLY)
            
            # Map texture
            t_ms = time.perf_counter()
            mapping = registered_texture.map()
            cuda_array = mapping.array(0, 0)
            t_me = time.perf_counter()
            map_times.append((t_me - t_ms) * 1000)
            
            # Copy to CuPy
            copy_ms = cuda_event_ms(lambda: (
                (lambda m: (m.set_src_array(cuda_array),
                           m.set_dst_device(int(dev_frame.data.ptr)),
                           setattr(m, 'width_in_bytes', WIDTH * 4),
                           setattr(m, 'height', HEIGHT),
                           setattr(m, 'dst_pitch', dev_frame.strides[0]),
                           m(aligned=True)))(cuda.Memcpy2D())
            ))
            copy_times.append(copy_ms)
            mapping.unmap()
            
            # Convert to torch
            t_cs = time.perf_counter()
            torch_frame = torch_from_cupy_zero_copy(dev_frame)
            t_ce = time.perf_counter()
            convert_times.append((t_ce - t_cs) * 1000)
            
            # Encode
            t_es = time.perf_counter()
            pkt = enc.Encode(torch_frame)
            if pkt:
                encoded_sizes.append(len(pkt))
            t_ee = time.perf_counter()
            encode_times.append((t_ee - t_es) * 1000)
            
            total_times.append((time.perf_counter() - t0) * 1000)
            
            if (i + 1) % 20 == 0:
                print(f'  Frame {i+1}: Total={total_times[-1]:.2f}ms, Render={render_times[-1]:.2f}ms')
        
        final_pkt = enc.EndEncode()
        if final_pkt:
            encoded_sizes.append(len(final_pkt))
        
        # Cleanup
        interceptor.detach()
        if registered_texture:
            registered_texture.unregister()
        viewer.close()
        
        # Return timing data
        return {
            'render': render_times,
            'capture_gpu_ns': capture_gpu_ns,
            'map': map_times,
            'copy': copy_times,
            'convert': convert_times,
            'encode': encode_times,
            'total': total_times,
            'encoded_sizes': encoded_sizes
        }
        
    finally:
        ctx.pop()
        ctx.detach()
        app.quit()


def print_results(timing_data, mode_name):
    """Print benchmark results."""
    def mean_std(a):
        return (stats.fmean(a), (stats.pstdev(a) if len(a) > 1 else 0.0))
    
    print('\n' + '=' * 60)
    print(f'BENCHMARK RESULTS ({mode_name})')
    print(f'Resolution: {WIDTH}x{HEIGHT}, Frames: {FRAMES}')
    print('=' * 60)
    
    if 'render' in timing_data:
        m, s = mean_std(timing_data['render'])
        print(f'Napari Render:      {m:.3f} ± {s:.3f} ms')
        if timing_data.get('capture_gpu_ns'):
            gpu_ns = timing_data['capture_gpu_ns']
            print(f'Capture blit (GPU): {np.mean(gpu_ns)/1e6:.3f} ms')
    
    if 'compute' in timing_data:
        m, s = mean_std(timing_data['compute'])
        print(f'Compute:            {m:.3f} ± {s:.3f} ms')
    
    if 'upload' in timing_data and timing_data['upload']:
        m, s = mean_std(timing_data['upload'])
        print(f'GL upload (CPU wait): {m:.3f} ± {s:.3f} ms')
        if timing_data.get('upload_gpu_ns'):
            gpu_ns = timing_data['upload_gpu_ns']
            gpu_ms = np.mean(gpu_ns)/1e6
            print(f'  GPU time (query):   {gpu_ms:.3f} ms')
            # Bandwidth estimate
            bytes_per_frame = WIDTH * HEIGHT * 4
            if gpu_ms > 0:
                bw_gbps = (bytes_per_frame * 8) / (gpu_ms * 1e6)
                print(f'  Upload bandwidth:   {bw_gbps:.2f} Gbps')
    
    m_map, s_map = mean_std(timing_data['map'])
    m_copy, s_copy = mean_std(timing_data['copy'])
    m_conv, s_conv = mean_std(timing_data['convert'])
    m_enc, s_enc = mean_std(timing_data['encode'])
    m_total, s_total = mean_std(timing_data['total'])
    
    print(f'Map GL texture:     {m_map:.3f} ± {s_map:.3f} ms')
    print(f'Copy GL→CuPy:       {m_copy:.3f} ± {s_copy:.3f} ms')
    print(f'CuPy→Torch:         {m_conv:.3f} ± {s_conv:.3f} ms')
    print(f'NVENC Encode:       {m_enc:.3f} ± {s_enc:.3f} ms')
    print('-' * 60)
    print(f'TOTAL:              {m_total:.3f} ± {s_total:.3f} ms')
    fps = 1000.0 / m_total if m_total > 0 else 0.0
    print(f'FPS capability:     {fps:.0f} FPS')
    
    if 'render' in timing_data:
        m_render = mean_std(timing_data['render'])[0]
        print(f'Render percentage:  {100 * m_render / m_total:.1f}%')
    
    # Bitrate estimation from encoded packet sizes (if available)
    if timing_data.get('encoded_sizes'):
        avg_size = np.mean(timing_data['encoded_sizes'])
        bitrate_mbps = (avg_size * 8 * fps) / 1e6 if fps > 0 else 0.0
        print(f'Bitrate estimate:   {bitrate_mbps:.2f} Mbps (avg {avg_size/1024:.1f} KB/frame)')
    
    print('=' * 60)


def save_json(timing_data, mode_name):
    if not OUT_JSON:
        return
    summary = {
        'mode': mode_name,
        'width': WIDTH,
        'height': HEIGHT,
        'frames': FRAMES,
        'cpu_upload': CPU_UPLOAD if not USE_NAPARI else None,
    }
    # Compute summaries
    def mean_or_none(key):
        vals = timing_data.get(key)
        return float(stats.fmean(vals)) if vals else None
    # mean helpers
    upload_cpu_ms = (float(stats.fmean(timing_data['upload']))
                     if timing_data.get('upload') else None)
    upload_gpu_ms = (float(np.mean(timing_data['upload_gpu_ns'])/1e6)
                     if timing_data.get('upload_gpu_ns') else None)
    bytes_per_frame = WIDTH * HEIGHT * 4
    upload_bw_gbps = ((bytes_per_frame * 8) / (upload_gpu_ms * 1e6)
                      if upload_gpu_ms and upload_gpu_ms > 0 else None)

    summary.update({
        'map_ms': mean_or_none('map'),
        'copy_ms': mean_or_none('copy'),
        'convert_ms': mean_or_none('convert'),
        'encode_ms': mean_or_none('encode'),
        'total_ms': mean_or_none('total'),
        'render_ms': mean_or_none('render') if 'render' in timing_data else None,
        'capture_blit_gpu_ms': (float(np.mean(timing_data['capture_gpu_ns'])/1e6)
                                if timing_data.get('capture_gpu_ns') else None),
        'fps': (1000.0 / mean_or_none('total')) if mean_or_none('total') else None,
        'avg_packet_bytes': (float(np.mean(timing_data['encoded_sizes']))
                             if timing_data.get('encoded_sizes') else None),
        'upload_cpu_ms': upload_cpu_ms,
        'upload_gpu_ms': upload_gpu_ms,
        'upload_bandwidth_gbps': upload_bw_gbps,
    })
    if summary.get('fps') and summary.get('avg_packet_bytes'):
        summary['bitrate_mbps'] = summary['fps'] * summary['avg_packet_bytes'] * 8 / 1e6
    with open(OUT_JSON, 'w') as f:
        json.dump({'summary': summary, 'timings': timing_data}, f, indent=2)
    print(f"Saved JSON results to {OUT_JSON}")


def main():
    """Main benchmark entry point."""
    print(f"Streaming Pipeline Benchmark")
    print(f"Mode: {'Napari/VisPy Rendering' if USE_NAPARI else 'Simulated Texture'}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Frames: {FRAMES}")
    if not USE_NAPARI:
        print(f"CPU Upload: {CPU_UPLOAD}")
    print()
    
    if USE_NAPARI:
        timing_data = benchmark_napari_rendering()
        print_results(timing_data, "Napari/VisPy Rendering")
        save_json(timing_data, 'napari_vispy')
    else:
        timing_data = benchmark_simulated_texture()
        print_results(timing_data, f"Simulated Texture (CPU_UPLOAD={CPU_UPLOAD})")
        save_json(timing_data, 'simulated_texture')


if __name__ == '__main__':
    main()
