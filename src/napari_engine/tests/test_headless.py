"""Test headless napari rendering capability."""

import numpy as np
import napari


def test_headless_viewer_creation():
    """Test that a napari viewer can be created without showing the window."""
    # Create viewer without showing
    viewer = napari.Viewer(show=False)
    
    # Add some test data
    data = np.random.random((100, 100, 100))
    viewer.add_image(data, name="test_data")
    
    # Verify viewer was created
    assert viewer is not None
    assert viewer.window is not None
    assert len(viewer.layers) == 1
    
    # Close viewer to cleanup
    viewer.close()
    print("✓ Headless viewer creation successful")


def test_headless_screenshot():
    """Test that we can capture screenshots in headless mode."""
    viewer = napari.Viewer(show=False)
    
    # Add test data
    data = np.random.random((100, 100))
    viewer.add_image(data, colormap='viridis')
    
    # Try to capture a screenshot
    try:
        screenshot = viewer.screenshot(canvas_only=True, flash=False)
        assert screenshot is not None
        assert screenshot.shape[0] > 0  # Height
        assert screenshot.shape[1] > 0  # Width
        assert screenshot.shape[2] in [3, 4]  # RGB or RGBA
        print(f"✓ Screenshot captured: {screenshot.shape}")
    except Exception as e:
        print(f"✗ Screenshot failed: {e}")
        raise
    finally:
        viewer.close()


def test_offscreen_rendering():
    """Test direct canvas rendering without display."""
    viewer = napari.Viewer(show=False)
    
    # Add some 3D data
    data = np.random.random((50, 100, 100))
    viewer.add_image(data)
    
    # Access the Qt viewer and canvas
    qt_viewer = viewer.window._qt_viewer
    canvas = qt_viewer.canvas
    
    # Test that we can access the framebuffer
    try:
        # Force a render
        canvas.on_draw(None)
        
        # Try to grab the framebuffer
        fb_image = canvas.native.grabFramebuffer()
        assert fb_image is not None
        
        # Convert to numpy array
        from napari._qt.utils import QImg2array
        img_array = QImg2array(fb_image)
        
        assert img_array.shape[0] > 0
        assert img_array.shape[1] > 0
        print(f"✓ Offscreen rendering successful: {img_array.shape}")
    except Exception as e:
        print(f"✗ Offscreen rendering failed: {e}")
        raise
    finally:
        viewer.close()


def test_viewport_manipulation():
    """Test that we can manipulate camera and capture different views."""
    viewer = napari.Viewer(show=False)
    
    # Create a 3D dataset with recognizable pattern
    z, y, x = np.ogrid[:50, :100, :100]
    data = np.sin(0.1 * x) * np.cos(0.1 * y) * np.exp(-0.01 * z)
    viewer.add_image(data, name="3d_pattern")
    
    screenshots = []
    
    # Test different camera positions
    positions = [
        (50, 50),  # Center
        (25, 25),  # Top-left
        (75, 75),  # Bottom-right
    ]
    
    for pos in positions:
        viewer.camera.center = pos
        viewer.camera.zoom = 2.0
        
        # Capture screenshot
        screenshot = viewer.screenshot(canvas_only=True, flash=False)
        screenshots.append(screenshot)
        print(f"✓ Captured view at position {pos}: {screenshot.shape}")
    
    # Verify we got different views (images should be different)
    assert not np.array_equal(screenshots[0], screenshots[1])
    assert not np.array_equal(screenshots[1], screenshots[2])
    
    viewer.close()
    print("✓ Viewport manipulation successful")


if __name__ == "__main__":
    print("Testing napari headless rendering capabilities...\n")
    
    try:
        test_headless_viewer_creation()
        test_headless_screenshot()
        test_offscreen_rendering()
        test_viewport_manipulation()
        
        print("\n✅ All headless rendering tests passed!")
        print("\nConclusion: napari CAN render headless using show=False")
        print("The viewer.screenshot() method works without a visible window")
        print("We can manipulate the camera and capture different views")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()