#!/usr/bin/env python
"""
Check if GPU is actually being used for napari rendering.
"""

import os
import sys
import numpy as np
import napari
import time


def check_opengl_info():
    """Check what OpenGL context we're actually using."""
    print("\n" + "="*60)
    print("OPENGL CONTEXT CHECK")
    print("="*60)
    
    viewer = napari.Viewer(show=False)
    
    # Access the vispy canvas
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    
    # Get OpenGL context info
    try:
        from vispy import gloo
        
        # Force context creation by making it current
        canvas.context.make_current()
        
        # Get GL info
        gl_info = {
            'GL_VENDOR': gloo.gl.glGetString(gloo.gl.GL_VENDOR),
            'GL_RENDERER': gloo.gl.glGetString(gloo.gl.GL_RENDERER),
            'GL_VERSION': gloo.gl.glGetString(gloo.gl.GL_VERSION),
        }
        
        for key, value in gl_info.items():
            if value:
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                print(f"{key:15}: {value}")
        
        # Check if we're using software rendering
        renderer = gl_info.get('GL_RENDERER', '')
        if isinstance(renderer, bytes):
            renderer = renderer.decode('utf-8')
        
        if 'llvmpipe' in renderer.lower():
            print("\n⚠️  WARNING: Using SOFTWARE RENDERING (llvmpipe)")
            print("   This is CPU-based Mesa software rasterizer!")
            
        elif 'software' in renderer.lower() or 'mesa' in renderer.lower():
            print("\n⚠️  WARNING: Likely using SOFTWARE RENDERING")
            print("   GPU acceleration may not be available!")
            
        elif 'apple' in renderer.lower() or 'nvidia' in renderer.lower() or 'amd' in renderer.lower() or 'intel' in renderer.lower():
            print("\n✅ Using HARDWARE GPU acceleration")
            
        else:
            print(f"\n❓ Unknown renderer: {renderer}")
                
    except Exception as e:
        print(f"Error getting OpenGL info: {e}")
    
    viewer.close()
    

def check_qt_platform():
    """Check Qt platform and rendering backend."""
    print("\n" + "="*60)
    print("QT PLATFORM CHECK")
    print("="*60)
    
    from qtpy import QtCore
    from qtpy.QtWidgets import QApplication
    
    app = QApplication.instance() or QApplication([])
    
    # Platform info
    print(f"Qt Version: {QtCore.__version__}")
    print(f"Qt Platform: {app.platformName()}")
    
    # Check for offscreen platform
    if app.platformName() == 'offscreen':
        print("\n⚠️  WARNING: Using OFFSCREEN platform")
        print("   This typically means SOFTWARE rendering!")
    elif app.platformName() == 'cocoa':
        print("\n✅ Using native macOS (Cocoa) platform")
    elif app.platformName() == 'xcb' or app.platformName() == 'wayland':
        print(f"\n✅ Using native Linux ({app.platformName()}) platform")
    elif app.platformName() == 'windows':
        print("\n✅ Using native Windows platform")
        
    # Check environment variables that affect rendering
    print("\n--- Environment Variables ---")
    env_vars = [
        'QT_QPA_PLATFORM',
        'QT_XCB_GL_INTEGRATION', 
        'QT_OPENGL',
        'QT_QUICK_BACKEND',
        'LIBGL_ALWAYS_SOFTWARE',
        'MESA_GL_VERSION_OVERRIDE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
            if var == 'LIBGL_ALWAYS_SOFTWARE' and value == '1':
                print("   ⚠️  FORCED SOFTWARE RENDERING!")
                

def benchmark_gpu_vs_cpu():
    """Compare rendering performance to detect GPU usage."""
    print("\n" + "="*60)
    print("GPU VS CPU BENCHMARK")
    print("="*60)
    
    viewer = napari.Viewer(show=False)
    
    # Test with increasingly complex scenes
    sizes = [
        (100, 256, 256, "Small"),
        (100, 512, 512, "Medium"),
        (100, 1024, 1024, "Large"),
    ]
    
    for depth, height, width, label in sizes:
        # Create data
        data = np.random.random((depth, height, width)).astype(np.float32)
        
        # Clear and add
        viewer.layers.clear()
        layer = viewer.add_image(data)
        
        # Force render
        viewer.window._qt_viewer.canvas.on_draw(None)
        
        # Benchmark rendering
        times = []
        for _ in range(10):
            start = time.perf_counter()
            viewer.window._qt_viewer.canvas.on_draw(None)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"{label:10} ({width}x{height}): {avg_time:6.2f} ms")
        
        # Heuristic: if rendering time scales badly with size, likely CPU
        if label == "Large" and avg_time > 50:
            print("   ⚠️  Slow rendering suggests CPU/software rendering")
    
    viewer.close()
    

def check_framebuffer_readback():
    """Test framebuffer readback performance (the real bottleneck)."""
    print("\n" + "="*60)
    print("FRAMEBUFFER READBACK PERFORMANCE")
    print("="*60)
    
    viewer = napari.Viewer(show=False)
    data = np.random.random((50, 1024, 1024)).astype(np.float32)
    viewer.add_image(data)
    
    # Test readback at different resolutions
    sizes = [(512, 512), (1024, 1024), (1920, 1080), (2048, 2048)]
    
    for width, height in sizes:
        viewer.window._qt_viewer.canvas.size = (width, height)
        
        # Warm up
        viewer.window._qt_viewer.canvas.on_draw(None)
        
        # Test direct GL readback
        times = []
        for _ in range(5):
            start = time.perf_counter()
            
            # This is what screenshot() does internally
            img = viewer.window._qt_viewer.canvas.native.grabFramebuffer()
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        bandwidth = (width * height * 4) / (avg_time / 1000) / 1024 / 1024  # MB/s
        
        print(f"{width}x{height}: {avg_time:6.2f} ms ({bandwidth:.0f} MB/s)")
        
        if bandwidth < 100:
            print("   ⚠️  VERY SLOW readback - likely software rendering")
        elif bandwidth < 1000:
            print("   ⚠️  Slow readback - possible PCIe bottleneck or old GPU")
    
    viewer.close()


def main():
    """Run all GPU checks."""
    print("\n🔬 GPU INVOLVEMENT CHECK FOR NAPARI")
    print("This will determine if GPU is actually being used...\n")
    
    check_qt_platform()
    check_opengl_info()
    benchmark_gpu_vs_cpu()
    check_framebuffer_readback()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    print("""
If you see:
- 'llvmpipe' or 'Mesa' → CPU software rendering (NO GPU!)
- Slow readback speeds → GPU→CPU transfer bottleneck
- Apple/NVIDIA/AMD/Intel → Hardware GPU is being used

To force GPU on Linux/WSL:
  export LIBGL_ALWAYS_SOFTWARE=0
  export QT_XCB_GL_INTEGRATION=xcb_glx

To force GPU on macOS:
  # Should work by default

To check GPU usage while running:
  macOS: Activity Monitor → Window → GPU History
  Linux: nvidia-smi or intel_gpu_top
  Windows: Task Manager → Performance → GPU
""")


if __name__ == "__main__":
    main()