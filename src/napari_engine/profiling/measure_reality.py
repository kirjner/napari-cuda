"""Profile what ACTUALLY takes time in napari rendering.
Following Casey Muratori's philosophy: measure first, assume nothing.
"""

import time
import numpy as np
import napari
from contextlib import contextmanager


@contextmanager
def timed_section(name):
    """Simple profiler - no fancy libraries, just facts."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{name:40} {elapsed_ms:8.2f}ms")


def profile_napari_reality():
    """What actually costs time in napari?"""
    
    print("=== NAPARI PERFORMANCE REALITY CHECK ===\n")
    
    # Setup
    with timed_section("Create headless viewer"):
        viewer = napari.Viewer(show=False)
    
    # Test data
    data_3d = np.random.random((100, 512, 512))
    data_huge = np.random.random((10, 2048, 2048))
    
    # ACTUAL BOTTLENECKS
    
    with timed_section("Add 3D image layer"):
        viewer.add_image(data_3d)
    
    with timed_section("First screenshot (cold)"):
        img1 = viewer.screenshot()
    
    with timed_section("Second screenshot (warm)"):
        img2 = viewer.screenshot()
    
    # Camera operations
    with timed_section("Camera zoom change"):
        viewer.camera.zoom = 2.0
    
    with timed_section("Screenshot after zoom"):
        img3 = viewer.screenshot()
    
    with timed_section("Camera pan (100 pixels)"):
        viewer.camera.center = (100, 100)
    
    with timed_section("Screenshot after pan"):
        img4 = viewer.screenshot()
    
    # The killer: dimension changes
    with timed_section("Change Z slice"):
        viewer.dims.current_step = (50, 0, 0)
    
    with timed_section("Screenshot after slice change"):
        img5 = viewer.screenshot()
    
    # Rapid operations (game engine test)
    print("\n--- RAPID FIRE TEST (60 FPS target = 16.67ms) ---")
    
    for i in range(10):
        with timed_section(f"Frame {i} (pan + screenshot)"):
            viewer.camera.center = (i * 10, i * 10)
            img = viewer.screenshot()
    
    # The real question for streaming
    print("\n--- STREAMING CRITICAL PATH ---")
    
    with timed_section("Screenshot only"):
        raw_img = viewer.screenshot()
    
    with timed_section("Convert to bytes"):
        img_bytes = raw_img.tobytes()
    
    with timed_section("Compress (PNG simulation)"):
        from PIL import Image
        import io
        img = Image.fromarray(raw_img)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        compressed = buffer.getvalue()
    
    with timed_section("Compress (JPEG simulation)"):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        compressed = buffer.getvalue()
    
    print(f"\n--- DATA SIZES ---")
    print(f"Raw image:        {len(img_bytes) / 1024 / 1024:.2f} MB")
    print(f"PNG compressed:   {len(buffer.getvalue()) / 1024 / 1024:.2f} MB")
    
    viewer.close()
    
    print("\n=== REALITY CHECK COMPLETE ===")
    print("If any operation > 16.67ms, it can't run at 60 FPS.")
    print("If any operation > 33.33ms, it can't run at 30 FPS.")


def profile_state_sync_cost():
    """What's the cost of state synchronization?"""
    
    print("\n=== STATE SYNC COST ANALYSIS ===\n")
    
    viewer = napari.Viewer(show=False)
    data = np.random.random((100, 512, 512))
    layer = viewer.add_image(data)
    
    # Measure state access costs
    with timed_section("Get all layer properties"):
        state = {
            'name': layer.name,
            'visible': layer.visible,
            'opacity': layer.opacity,
            'colormap': layer.colormap.name,
            'contrast_limits': layer.contrast_limits,
            'gamma': layer.gamma,
            'interpolation': layer.interpolation,
            'rendering': layer.rendering,
            'blending': layer.blending,
        }
    
    with timed_section("Serialize state (JSON simulation)"):
        import json
        json_state = json.dumps(state, default=str)
    
    with timed_section("Get camera state"):
        camera_state = {
            'center': viewer.camera.center,
            'zoom': viewer.camera.zoom,
            'angles': viewer.camera.angles,
        }
    
    with timed_section("Get dims state"):
        dims_state = {
            'current_step': viewer.dims.current_step,
            'ndisplay': viewer.dims.ndisplay,
            'order': viewer.dims.order,
        }
    
    print(f"\nState sizes:")
    print(f"Layer state JSON: {len(json_state)} bytes")
    
    viewer.close()


if __name__ == "__main__":
    profile_napari_reality()
    profile_state_sync_cost()