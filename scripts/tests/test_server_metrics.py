#!/usr/bin/env python
"""
Test script to trigger frame captures on the server and collect metrics.

This connects as a client to trigger the actual GPU pipeline.
"""

import asyncio
import json
import time
import websockets
import requests
from typing import Dict, Any

async def trigger_frames(state_uri: str, pixel_uri: str, num_frames: int = 50):
    """
    Connect to server and trigger frame captures.
    
    Parameters
    ----------
    state_uri : str
        WebSocket URI for state channel (e.g., ws://localhost:8081)
    pixel_uri : str
        WebSocket URI for pixel channel (e.g., ws://localhost:8082)
    num_frames : int
        Number of frames to request
    """
    print(f"Connecting to server at {state_uri} and {pixel_uri}...")
    
    # Connect to both channels
    async with websockets.connect(state_uri) as state_ws, \
               websockets.connect(pixel_uri) as pixel_ws:
        
        print(f"Connected! Triggering {num_frames} frame captures...")
        
        frames_received = 0
        start_time = time.perf_counter()
        
        # Request frames by sending camera updates
        for i in range(num_frames):
            # Send a camera update to trigger rendering
            camera_update = {
                "type": "set_camera",
                "zoom": 1.5 + (i * 0.01),  # Slight zoom changes
                "center": [256, 256, 50]
            }
            await state_ws.send(json.dumps(camera_update))
            
            # Try to receive frame data (non-blocking)
            try:
                frame_data = await asyncio.wait_for(pixel_ws.recv(), timeout=0.1)
                frames_received += 1
                if frames_received % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    fps = frames_received / elapsed
                    print(f"  Received {frames_received}/{num_frames} frames - {fps:.1f} FPS")
            except asyncio.TimeoutError:
                pass
            
            # Small delay to not overwhelm
            await asyncio.sleep(0.01)
        
        # Final stats
        total_time = time.perf_counter() - start_time
        print(f"\nCompleted: {frames_received} frames in {total_time:.2f}s")
        print(f"Average FPS: {frames_received/total_time:.1f}")


def fetch_metrics_json(metrics_url: str) -> Dict[str, Any]:
    """Fetch JSON metrics snapshot."""
    try:
        r = requests.get(metrics_url, timeout=5)
        if r.status_code != 200:
            print(f"Failed to fetch metrics: HTTP {r.status_code}")
            return {}
        return r.json()
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}


def print_metrics_summary(delta_counters: Dict[str, float], snapshot: Dict[str, Any]):
    """
    Print a summary of collected metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dictionary
    """
    print("\n" + "=" * 70)
    print("Server Performance Metrics")
    print("=" * 70)
    
    # Frame metrics
    frames = int(delta_counters.get('napari_cuda_frames_total', 0))
    bytes_total = float(delta_counters.get('napari_cuda_bytes_total', 0.0))
    
    if frames > 0:
        print(f"\n📊 Frame Statistics:")
        print(f"  Total frames:    {int(frames)}")
        print(f"  Total data:      {bytes_total/1024/1024:.1f} MB")
        print(f"  Avg frame size:  {bytes_total/frames/1024:.1f} KB")
    
    # Pipeline timing metrics
    timing_metrics = [
        ('napari_cuda_map_ms', 'Texture mapping'),
        ('napari_cuda_copy_ms', 'Data copy'),
        ('napari_cuda_encode_ms', 'H.264 encoding'),
        ('napari_cuda_total_ms', 'Total pipeline')
    ]
    
    print(f"\n⏱️  Pipeline Stage Timing (per frame):")
    h = snapshot.get('histograms', {})
    for metric_key, label in timing_metrics:
        stat = h.get(metric_key)
        if stat:
            avg_time = float(stat.get('mean_ms', 0.0))
            print(f"  {label:20s}: {avg_time:>8.3f} ms")
    
    # Calculate theoretical max FPS
    if 'napari_cuda_total_ms' in h:
        avg_total = float(h['napari_cuda_total_ms'].get('mean_ms', 0.0))
        max_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
        print(f"\n🚀 Theoretical max FPS: {max_fps:.1f}")
        
        # Compare to CPU baseline (approximate)
        cpu_baseline_ms = 40  # Typical CPU screenshot time
        speedup = cpu_baseline_ms / avg_total
        print(f"✨ Estimated speedup vs CPU: {speedup:.1f}x")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test napari-cuda server metrics")
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--state-port', type=int, default=8081, help='State WebSocket port')
    parser.add_argument('--pixel-port', type=int, default=8082, help='Pixel WebSocket port')
    parser.add_argument('--metrics-port', type=int, default=8083, help='Metrics HTTP port')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to request')
    
    args = parser.parse_args()
    
    # Build URIs
    state_uri = f"ws://{args.host}:{args.state_port}"
    pixel_uri = f"ws://{args.host}:{args.pixel_port}"
    metrics_url = f"http://{args.host}:{args.metrics_port}/metrics.json"
    
    print("🎯 napari-cuda Server Performance Test")
    print(f"Server: {args.host}")
    print(f"Ports: state={args.state_port}, pixel={args.pixel_port}, metrics={args.metrics_port}")
    
    # Fetch initial metrics
    print("\nFetching initial metrics...")
    initial_snapshot = fetch_metrics_json(metrics_url)
    
    # Trigger frame captures
    await trigger_frames(state_uri, pixel_uri, args.frames)
    
    # Wait a moment for final frames to process
    await asyncio.sleep(2)
    
    # Fetch final metrics
    print("\nFetching final metrics...")
    final_snapshot = fetch_metrics_json(metrics_url)
    
    # Calculate deltas
    # Compute deltas for counters only
    delta_counters: Dict[str, float] = {}
    init_c = (initial_snapshot or {}).get('counters', {})
    final_c = (final_snapshot or {}).get('counters', {})
    for key, v in final_c.items():
        try:
            delta_counters[key] = float(v) - float(init_c.get(key, 0.0))
        except Exception:
            pass
    
    # Print summary
    print_metrics_summary(delta_counters, final_snapshot or {})


if __name__ == "__main__":
    asyncio.run(main())
