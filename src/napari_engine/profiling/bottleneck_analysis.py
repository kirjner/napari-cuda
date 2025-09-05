#!/usr/bin/env python
"""
The real bottleneck analysis for napari-engine.
This will tell us if streaming frames is even viable.

Run this to get the ground truth about performance.
"""

import time
import numpy as np
import napari
from io import BytesIO
from contextlib import contextmanager
import json
from typing import Dict, List
import gc


@contextmanager
def timed(name: str) -> dict:
    """Time a section and store result."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result['time_ms'] = (time.perf_counter() - start) * 1000
        result['name'] = name


class BottleneckAnalyzer:
    """Analyze where time is actually spent in remote rendering pipeline."""
    
    def __init__(self):
        self.results = {}
        
    def profile_local_operations(self):
        """How fast is local napari really?"""
        print("\n" + "="*60)
        print("LOCAL NAPARI PERFORMANCE (baseline)")
        print("="*60)
        
        viewer = napari.Viewer(show=False)
        data = np.random.random((100, 1024, 1024)).astype(np.float32)
        layer = viewer.add_image(data, name="test_data")
        
        results = []
        
        # Warm up
        viewer.camera.zoom = 1.5
        viewer.screenshot()
        
        # Test different operations
        operations = [
            ("Camera zoom 2x", lambda: setattr(viewer.camera, 'zoom', 2.0)),
            ("Camera zoom 0.5x", lambda: setattr(viewer.camera, 'zoom', 0.5)),
            ("Camera pan +100px", lambda: setattr(viewer.camera, 'center', (600, 600))),
            ("Camera pan -100px", lambda: setattr(viewer.camera, 'center', (400, 400))),
            ("Change Z slice", lambda: setattr(viewer.dims, 'current_step', (50, 0, 0))),
            ("Toggle layer visibility", lambda: setattr(layer, 'visible', not layer.visible)),
            ("Change opacity", lambda: setattr(layer, 'opacity', 0.5)),
            ("Change colormap", lambda: setattr(layer, 'colormap', 'viridis')),
        ]
        
        for op_name, op_func in operations:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                op_func()
                # Force render
                viewer.window._qt_viewer.canvas.on_draw(None)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{op_name:30} {avg_time:6.2f} ± {std_time:4.2f} ms")
            results.append({
                'operation': op_name,
                'avg_ms': avg_time,
                'std_ms': std_time,
                'all_times': times
            })
        
        viewer.close()
        self.results['local_operations'] = results
        return results
    
    def profile_screenshot_pipeline(self):
        """Profile each step of the screenshot pipeline."""
        print("\n" + "="*60)
        print("SCREENSHOT PIPELINE BREAKDOWN")
        print("="*60)
        
        viewer = napari.Viewer(show=False)
        
        # Test different data sizes
        sizes = [
            (512, 512, "512x512 (0.25 MP)"),
            (1024, 1024, "1024x1024 (1 MP)"),
            (1920, 1080, "1920x1080 (2 MP, Full HD)"),
            (2048, 2048, "2048x2048 (4 MP)"),
        ]
        
        results = {}
        
        for width, height, label in sizes:
            print(f"\n--- {label} ---")
            
            # Create test data
            data = np.random.random((50, height, width)).astype(np.float32)
            viewer.layers.clear()
            viewer.add_image(data)
            
            # Set canvas size
            viewer.window._qt_viewer.canvas.size = (width, height)
            
            # Warm up
            for _ in range(3):
                viewer.screenshot()
                gc.collect()
            
            step_times = {
                'screenshot': [],
                'to_bytes': [],
                'compress_png': [],
                'compress_jpeg_95': [],
                'compress_jpeg_85': [],
                'compress_jpeg_75': [],
                'compress_webp': [],
            }
            
            for i in range(10):
                # Step 1: Screenshot (GPU → CPU)
                start = time.perf_counter()
                screenshot = viewer.screenshot(canvas_only=True, flash=False)
                step_times['screenshot'].append((time.perf_counter() - start) * 1000)
                
                # Step 2: Convert to bytes
                start = time.perf_counter()
                raw_bytes = screenshot.tobytes()
                step_times['to_bytes'].append((time.perf_counter() - start) * 1000)
                
                # Step 3a: PNG compression
                from PIL import Image
                img = Image.fromarray(screenshot)
                
                start = time.perf_counter()
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=False)
                png_size = len(buffer.getvalue())
                step_times['compress_png'].append((time.perf_counter() - start) * 1000)
                
                # Step 3b: JPEG compression (various qualities)
                # Convert RGBA to RGB for JPEG
                if img.mode == 'RGBA':
                    img_rgb = img.convert('RGB')
                else:
                    img_rgb = img
                    
                for quality, key in [(95, 'compress_jpeg_95'), 
                                    (85, 'compress_jpeg_85'), 
                                    (75, 'compress_jpeg_75')]:
                    start = time.perf_counter()
                    buffer = BytesIO()
                    img_rgb.save(buffer, format='JPEG', quality=quality)
                    jpeg_size = len(buffer.getvalue())
                    step_times[key].append((time.perf_counter() - start) * 1000)
                
                # Step 3c: WebP compression (if available)
                try:
                    start = time.perf_counter()
                    buffer = BytesIO()
                    img.save(buffer, format='WEBP', quality=85)
                    webp_size = len(buffer.getvalue())
                    step_times['compress_webp'].append((time.perf_counter() - start) * 1000)
                except:
                    step_times['compress_webp'].append(0)
            
            # Print results
            print(f"{'Operation':<20} {'Avg (ms)':>10} {'Std (ms)':>10} {'Size (MB)':>10}")
            print("-" * 52)
            
            for key, times in step_times.items():
                if times[0] > 0:  # Skip if not measured
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    
                    # Calculate size
                    if key == 'screenshot' or key == 'to_bytes':
                        size_mb = len(raw_bytes) / 1024 / 1024
                    elif key == 'compress_png':
                        size_mb = png_size / 1024 / 1024
                    elif 'jpeg' in key:
                        size_mb = jpeg_size / 1024 / 1024
                    else:
                        size_mb = 0
                    
                    print(f"{key:<20} {avg_time:10.2f} {std_time:10.2f} {size_mb:10.2f}")
                    
            results[label] = step_times
            
            # Calculate total pipeline time
            total_jpeg85 = (np.mean(step_times['screenshot']) + 
                          np.mean(step_times['compress_jpeg_85']))
            
            print(f"\nTotal pipeline (JPEG 85): {total_jpeg85:.2f} ms")
            print(f"Max achievable FPS: {1000/total_jpeg85:.1f}")
            
            if total_jpeg85 > 33.33:
                print("⚠️  TOO SLOW for 30 FPS!")
            elif total_jpeg85 > 16.67:
                print("⚠️  TOO SLOW for 60 FPS (but OK for 30 FPS)")
            else:
                print("✅ Fast enough for 60 FPS")
        
        viewer.close()
        self.results['screenshot_pipeline'] = results
        return results
    
    def profile_network_simulation(self):
        """Simulate network latency impact."""
        print("\n" + "="*60)
        print("NETWORK LATENCY SIMULATION")
        print("="*60)
        
        # Simulate different network conditions
        conditions = [
            ("LAN (1ms)", 0.001),
            ("Fast internet (10ms)", 0.010),
            ("Average internet (50ms)", 0.050),
            ("Poor internet (100ms)", 0.100),
            ("Cross-continent (200ms)", 0.200),
        ]
        
        viewer = napari.Viewer(show=False)
        data = np.random.random((50, 1024, 1024)).astype(np.float32)
        viewer.add_image(data)
        
        # Get baseline timings
        screenshot = viewer.screenshot()
        
        from PIL import Image
        img = Image.fromarray(screenshot)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        compressed = buffer.getvalue()
        frame_size = len(compressed) / 1024  # KB
        
        print(f"Frame size (JPEG 85): {frame_size:.1f} KB")
        print(f"\n{'Network':<20} {'RTT (ms)':>10} {'Total (ms)':>12} {'Max FPS':>10}")
        print("-" * 54)
        
        results = []
        for label, latency in conditions:
            # Simple model: RTT + transmission time
            # Assuming 10 Mbps = 1.25 MB/s = 1280 KB/s
            transmission_time = (frame_size / 1280) * 1000  # ms
            total_time = latency * 1000 + transmission_time
            max_fps = 1000 / total_time if total_time > 0 else float('inf')
            
            print(f"{label:<20} {latency*1000:10.1f} {total_time:12.2f} {max_fps:10.1f}")
            
            results.append({
                'condition': label,
                'rtt_ms': latency * 1000,
                'total_ms': total_time,
                'max_fps': max_fps
            })
        
        viewer.close()
        self.results['network_simulation'] = results
        return results
    
    def profile_state_sync_overhead(self):
        """How expensive is state synchronization?"""
        print("\n" + "="*60)
        print("STATE SYNCHRONIZATION OVERHEAD")
        print("="*60)
        
        viewer = napari.Viewer(show=False)
        
        # Create complex scene
        for i in range(10):
            data = np.random.random((10, 256, 256)).astype(np.float32)
            viewer.add_image(data, name=f"layer_{i}")
        
        import json
        
        # Measure state extraction
        times = []
        for _ in range(100):
            start = time.perf_counter()
            
            state = {
                'camera': {
                    'center': list(viewer.camera.center),
                    'zoom': float(viewer.camera.zoom),
                    'angles': list(viewer.camera.angles),
                },
                'dims': {
                    'current_step': list(viewer.dims.current_step),
                    'ndisplay': viewer.dims.ndisplay,
                },
                'layers': [
                    {
                        'name': layer.name,
                        'visible': layer.visible,
                        'opacity': float(layer.opacity),
                        'blending': layer.blending,
                    }
                    for layer in viewer.layers
                ]
            }
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        extract_time = np.mean(times)
        
        # Measure serialization
        times = []
        for _ in range(100):
            start = time.perf_counter()
            json_state = json.dumps(state)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        serialize_time = np.mean(times)
        state_size = len(json_state.encode()) / 1024  # KB
        
        print(f"State extraction:  {extract_time:.3f} ms")
        print(f"JSON serialization: {serialize_time:.3f} ms")
        print(f"State size: {state_size:.2f} KB")
        print(f"Total overhead: {extract_time + serialize_time:.3f} ms")
        
        viewer.close()
        
        self.results['state_sync'] = {
            'extract_ms': extract_time,
            'serialize_ms': serialize_time,
            'size_kb': state_size,
            'total_ms': extract_time + serialize_time
        }
        
        return self.results['state_sync']
    
    def generate_report(self):
        """Generate final analysis report."""
        print("\n" + "="*60)
        print("FINAL ANALYSIS")
        print("="*60)
        
        # Find the bottleneck
        if 'screenshot_pipeline' in self.results:
            # Get 1080p results
            hd_results = self.results['screenshot_pipeline'].get('1920x1080 (2 MP, Full HD)', {})
            if hd_results:
                screenshot_time = np.mean(hd_results.get('screenshot', [0]))
                compress_time = np.mean(hd_results.get('compress_jpeg_85', [0]))
                
                print(f"\nFor Full HD (1920x1080):")
                print(f"  Screenshot: {screenshot_time:.2f} ms")
                print(f"  JPEG compression: {compress_time:.2f} ms")
                print(f"  Total: {screenshot_time + compress_time:.2f} ms")
                
                total_time = screenshot_time + compress_time
                if total_time > 0:
                    max_fps = 1000 / total_time
                    print(f"  Max local FPS: {max_fps:.1f}")
                    
                    if max_fps < 30:
                        print("\n❌ VERDICT: Frame streaming NOT VIABLE at Full HD")
                        print("   Screenshot/compression alone exceeds 30 FPS budget")
                        print("   Consider: Streaming tiles or reducing resolution")
                    elif max_fps < 60:
                        print("\n⚠️  VERDICT: Frame streaming viable for 30 FPS only")
                        print("   Cannot achieve 60 FPS even locally")
                    else:
                        print("\n✅ VERDICT: Frame streaming technically viable")
                        print("   Local performance adequate, network is main constraint")
        
        # Save results to JSON
        with open('bottleneck_analysis.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("\nDetailed results saved to bottleneck_analysis.json")
    
    def run_all_tests(self):
        """Run complete bottleneck analysis."""
        print("\n🔬 NAPARI-ENGINE BOTTLENECK ANALYSIS")
        print("This will tell us if frame streaming is viable...\n")
        
        self.profile_local_operations()
        self.profile_screenshot_pipeline()
        self.profile_network_simulation()
        self.profile_state_sync_overhead()
        self.generate_report()


if __name__ == "__main__":
    analyzer = BottleneckAnalyzer()
    analyzer.run_all_tests()