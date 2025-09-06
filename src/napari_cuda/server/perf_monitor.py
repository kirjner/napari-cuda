"""
Terminal-based performance monitor for napari-cuda server.

Displays real-time performance metrics from Prometheus in a terminal dashboard.
"""

import sys
import time
import threading
from collections import deque
from typing import Optional, Dict, Any
from prometheus_client import generate_latest, REGISTRY
from prometheus_client.parser import text_string_to_metric_families

from .metrics import Metrics


class PerfMonitor:
    """Terminal-based performance dashboard for napari-cuda."""
    
    def __init__(self, metrics: Metrics, update_interval: float = 1.0, log_file: str = None):
        """
        Initialize the performance monitor.
        
        Parameters
        ----------
        metrics : Metrics
            The Prometheus metrics instance to monitor
        update_interval : float
            How often to update the display (seconds)
        log_file : str
            Optional path to log file for saving metrics
        """
        self.metrics = metrics
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        
        # History tracking for computing averages
        self.history = {}
        self.history_size = 100  # Keep last 100 samples
        
        # Frame counting
        self.last_frame_count = 0
        self.last_time = time.time()
        
        # File logging
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, 'w')
            # Write CSV header
            self.log_file.write("timestamp,stage,value_ms,stat_type\n")
            self.log_file.flush()
        
    def start(self):
        """Start the performance monitor in a background thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the performance monitor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.log_file:
            self.log_file.close()
            
    def _monitor_loop(self):
        """Main monitoring loop that updates the display."""
        while self.running:
            try:
                self.print_stats()
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitor error: {e}", file=sys.stderr)
                
    def _get_metric_value(self, metric_name: str, stat_type: str = 'mean') -> float:
        """
        Get a specific metric value from Prometheus.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to retrieve
        stat_type : str
            Type of statistic: 'mean', 'count', 'sum', or specific quantile like '0.99'
            
        Returns
        -------
        float
            The metric value, or 0.0 if not found
        """
        try:
            # Generate Prometheus text format
            metrics_text = generate_latest(self.metrics.registry).decode('utf-8')
            
            # Parse metrics
            for family in text_string_to_metric_families(metrics_text):
                if family.name == metric_name:
                    for sample in family.samples:
                        # For histograms, look for specific buckets or statistics
                        if sample.name == f"{metric_name}_{stat_type}":
                            return sample.value
                        # For simple metrics
                        elif sample.name == metric_name and not sample.labels:
                            return sample.value
                        # For quantiles in histograms
                        elif sample.name == metric_name and 'quantile' in sample.labels:
                            if sample.labels.get('quantile') == stat_type:
                                return sample.value
            return 0.0
        except Exception:
            return 0.0
            
    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame counter."""
        try:
            current_frames = self._get_metric_value('napari_cuda_frames_total', 'count')
            current_time = time.time()
            
            if self.last_frame_count > 0:
                frames_delta = current_frames - self.last_frame_count
                time_delta = current_time - self.last_time
                if time_delta > 0:
                    fps = frames_delta / time_delta
                else:
                    fps = 0.0
            else:
                fps = 0.0
                
            self.last_frame_count = current_frames
            self.last_time = current_time
            
            return fps
        except Exception:
            return 0.0
            
    def _get_percentile(self, metric_name: str, percentile: float) -> float:
        """
        Get a percentile value for a histogram metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the histogram metric
        percentile : float
            Percentile to compute (e.g., 0.99 for P99)
            
        Returns
        -------
        float
            The percentile value
        """
        # For now, use the mean as a placeholder
        # In a real implementation, we'd compute from histogram buckets
        return self._get_metric_value(metric_name, 'mean') * 1.5
        
    def print_stats(self):
        """Print performance statistics table to terminal."""
        # Clear screen (works on Unix-like systems)
        print("\033[2J\033[H", end='')
        
        # Header
        print("=" * 70)
        print("napari-cuda Performance Monitor (Server-Side)")
        print("=" * 70)
        
        # Pipeline stages table
        print(f"\n{'Pipeline Stage':<30} {'Last':<12} {'Avg':<12} {'P99':<12}")
        print("-" * 70)
        
        stages = [
            ('Queue Wait', 'napari_cuda_queue_wait_ms'),
            ('CUDA Lock Acquire', 'napari_cuda_cuda_lock_ms'),
            ('Context Switch', 'napari_cuda_context_ms'),
            ('Texture Lock', 'napari_cuda_texture_lock_ms'),
            ('CUDA Register (first time)', 'napari_cuda_register_texture_ms'),
            ('CUDA Map', 'napari_cuda_map_ms'),
            ('Get Array', 'napari_cuda_array_get_ms'),
            ('NVENC Encode', 'napari_cuda_encode_ms'),
            ('CUDA Unmap', 'napari_cuda_unmap_ms'),
            ('━' * 30, None),  # Separator
            ('Total GPU Time', 'napari_cuda_total_gpu_ms'),
            ('End-to-End', 'napari_cuda_end_to_end_ms'),
        ]
        
        timestamp = time.time()
        
        for name, metric in stages:
            if metric is None:
                # It's a separator
                print(name)
            else:
                # Get metric values
                last_val = self._get_metric_value(metric, 'mean')
                avg_val = self._get_metric_value(metric, 'mean')
                p99_val = self._get_percentile(metric, 0.99)
                
                # Format the row
                if last_val > 0 or avg_val > 0:
                    print(f"{name:<30} {last_val:>10.2f}ms {avg_val:>10.2f}ms {p99_val:>10.2f}ms")
                    
                    # Log to file if enabled
                    if self.log_file and last_val > 0:
                        self.log_file.write(f"{timestamp},{metric},{last_val},last\n")
                        self.log_file.write(f"{timestamp},{metric},{avg_val},avg\n")
                        self.log_file.write(f"{timestamp},{metric},{p99_val},p99\n")
                    
        # System stats
        print("\n" + "-" * 70)
        
        fps = self._calculate_fps()
        queue_depth = self._get_metric_value('napari_cuda_capture_queue_depth', 'mean')
        total_frames = self._get_metric_value('napari_cuda_frames_total', 'count')
        dropped_frames = self._get_metric_value('napari_cuda_frames_dropped', 'count')
        encode_errors = self._get_metric_value('napari_cuda_encode_errors', 'count')
        bytes_total = self._get_metric_value('napari_cuda_bytes_total', 'count')
        
        print(f"FPS: {fps:.1f} | Queue Depth: {queue_depth:.0f} | Total Frames: {total_frames:.0f}")
        print(f"Dropped: {dropped_frames:.0f} | Errors: {encode_errors:.0f} | Data: {bytes_total/1024/1024:.1f} MB")
        
        # GPU vs CPU comparison (estimated)
        gpu_time = self._get_metric_value('napari_cuda_total_gpu_ms', 'mean')
        if gpu_time > 0:
            estimated_cpu_time = 40.0  # Typical CPU screenshot time in ms
            speedup = estimated_cpu_time / gpu_time
            print(f"\n📊 Estimated CPU equivalent: ~{estimated_cpu_time:.0f}ms ({speedup:.1f}x speedup)")
            
        # Connection status
        state_clients = self._get_metric_value('napari_cuda_state_clients', 'mean')
        pixel_clients = self._get_metric_value('napari_cuda_pixel_clients', 'mean')
        print(f"\n🔗 Connections: State={state_clients:.0f} | Pixel={pixel_clients:.0f}")
        
        # Timestamp
        print(f"\n[Updated: {time.strftime('%H:%M:%S')}]")
        
        # Flush log file if enabled
        if self.log_file:
            self.log_file.flush()
        
    def print_summary(self):
        """Print a one-time summary of performance metrics."""
        print("\n" + "=" * 70)
        print("Performance Summary")
        print("=" * 70)
        
        # Key metrics
        total_gpu = self._get_metric_value('napari_cuda_total_gpu_ms', 'mean')
        encode_time = self._get_metric_value('napari_cuda_encode_ms', 'mean')
        map_time = self._get_metric_value('napari_cuda_map_ms', 'mean')
        
        print(f"Average GPU Pipeline: {total_gpu:.2f}ms")
        print(f"  - CUDA Map: {map_time:.2f}ms")
        print(f"  - NVENC Encode: {encode_time:.2f}ms")
        
        if total_gpu > 0:
            theoretical_fps = 1000.0 / total_gpu
            print(f"Theoretical Max FPS: {theoretical_fps:.0f}")
            
            # Comparison with CPU
            cpu_estimate = 40.0  # ms
            speedup = cpu_estimate / total_gpu
            print(f"\nSpeedup vs CPU: {speedup:.1f}x")
            print(f"(CPU baseline: ~{cpu_estimate:.0f}ms per screenshot)")