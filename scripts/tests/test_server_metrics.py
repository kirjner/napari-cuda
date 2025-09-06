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
from typing import Dict

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


def fetch_metrics(metrics_url: str) -> Dict[str, float]:
    """
    Fetch and parse Prometheus metrics.
    
    Parameters
    ----------
    metrics_url : str
        URL of metrics endpoint
        
    Returns
    -------
    Dict[str, float]
        Parsed metrics
    """
    try:
        response = requests.get(metrics_url, timeout=5)
        if response.status_code != 200:
            print(f"Failed to fetch metrics: HTTP {response.status_code}")
            return {}
        
        # Parse Prometheus format
        metrics = {}
        for line in response.text.split('\n'):
            if line.startswith('napari_cuda_'):
                # Handle histogram summaries
                if '_sum{' in line or '_sum ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].split('{')[0].replace('_sum', '')
                        try:
                            value = float(parts[-1])
                            metrics[key] = value
                        except ValueError:
                            pass
                # Handle counters and gauges
                elif '{' not in line and ' ' in line:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            metrics[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
        
        return metrics
        
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}


def print_metrics_summary(metrics: Dict[str, float]):
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
    frames = metrics.get('napari_cuda_frames_total', 0)
    bytes_total = metrics.get('napari_cuda_bytes_total', 0)
    
    if frames > 0:
        print(f"\n📊 Frame Statistics:")
        print(f"  Total frames:    {int(frames)}")
        print(f"  Total data:      {bytes_total/1024/1024:.1f} MB")
        print(f"  Avg frame size:  {bytes_total/frames/1024:.1f} KB")
    
    # Pipeline timing metrics
    timing_metrics = [
        ('napari_cuda_queue_wait_ms', 'Queue wait'),
        ('napari_cuda_cuda_lock_ms', 'CUDA lock'),
        ('napari_cuda_map_ms', 'Texture mapping'),
        ('napari_cuda_copy_ms', 'Data copy'),
        ('napari_cuda_encode_ms', 'H.264 encoding'),
        ('napari_cuda_send_ms', 'Network send'),
        ('napari_cuda_total_ms', 'Total pipeline')
    ]
    
    print(f"\n⏱️  Pipeline Stage Timing (per frame):")
    total_tracked = 0
    for metric_key, label in timing_metrics:
        if metric_key in metrics and frames > 0:
            avg_time = metrics[metric_key] / frames
            print(f"  {label:20s}: {avg_time:>8.3f} ms")
            if 'total' not in metric_key:
                total_tracked += avg_time
    
    # Calculate theoretical max FPS
    if 'napari_cuda_total_ms' in metrics and frames > 0:
        avg_total = metrics['napari_cuda_total_ms'] / frames
        max_fps = 1000 / avg_total
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
    metrics_url = f"http://{args.host}:{args.metrics_port}/metrics"
    
    print("🎯 napari-cuda Server Performance Test")
    print(f"Server: {args.host}")
    print(f"Ports: state={args.state_port}, pixel={args.pixel_port}, metrics={args.metrics_port}")
    
    # Fetch initial metrics
    print("\nFetching initial metrics...")
    initial_metrics = fetch_metrics(metrics_url)
    
    # Trigger frame captures
    await trigger_frames(state_uri, pixel_uri, args.frames)
    
    # Wait a moment for final frames to process
    await asyncio.sleep(2)
    
    # Fetch final metrics
    print("\nFetching final metrics...")
    final_metrics = fetch_metrics(metrics_url)
    
    # Calculate deltas
    delta_metrics = {}
    for key in final_metrics:
        if key in initial_metrics:
            delta_metrics[key] = final_metrics[key] - initial_metrics[key]
        else:
            delta_metrics[key] = final_metrics[key]
    
    # Print summary
    print_metrics_summary(delta_metrics)


if __name__ == "__main__":
    asyncio.run(main())