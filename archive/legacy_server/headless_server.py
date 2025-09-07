"""
HeadlessServer - Main server orchestrator for napari-cuda.

Runs napari in headless mode on HPC, captures frames via CUDA,
and streams to connected clients.
"""

import asyncio
import json
import logging
import os
import queue
import numpy as np
from pathlib import Path

import napari
import qasync
import websockets
from qtpy.QtWidgets import QApplication
from napari._vispy.utils.visual import layer_to_visual
from napari._vispy.layers.image import VispyImageLayer
from napari.layers import Image

from .cuda_streaming_layer import CudaStreamingLayer
from .render_thread import CUDARenderThread
from .metrics import Metrics
from ..protocol.messages import StateMessage, FrameMessage

logger = logging.getLogger(__name__)


class HeadlessServer:
    """
    Main server class that coordinates:
    - Headless napari viewer
    - CUDA/OpenGL capture
    - WebSocket communication
    - State synchronization
    """
    
    def __init__(self, 
                 dataset_path=None,
                 host='localhost',
                 state_port=8081,
                 pixel_port=8082,
                 metrics_port=8083,
                 enable_cuda=True):
        """
        Initialize the headless server.
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to numpy array or image to load
        host : str
            Host address for WebSocket servers
        state_port : int
            Port for state synchronization stream
        pixel_port : int
            Port for pixel/video stream
        enable_cuda : bool
            Whether to enable CUDA acceleration
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.host = host
        self.state_port = state_port
        self.pixel_port = pixel_port
        self.enable_cuda = enable_cuda
        self.metrics_port = metrics_port
        
        self.viewer = None
        self.cuda_layer = None
        self.render_thread = None
        
        # WebSocket clients
        self.state_clients = set()
        self.pixel_clients = set()
        
        # Metrics
        self.metrics = Metrics()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"HeadlessServer initialized - CUDA: {enable_cuda}")
    
    def _register_streaming_layer_factory(self):
        """
        Register CudaStreamingLayer as the factory for Image layers.
        
        This is the "napari-native" way to replace layer visuals - by modifying
        the global layer_to_visual dictionary that napari uses to create visuals.
        """
        if not self.enable_cuda:
            logger.info("CUDA disabled - skipping streaming layer registration")
            return
        
        # Store original factory for fallback
        self._original_image_factory = layer_to_visual.get(Image, VispyImageLayer)
        
        def create_streaming_layer(layer, *args, **kwargs):
            """Factory function that creates our streaming layer."""
            # Check if render thread is ready
            if not self.render_thread:
                logger.warning(f"Render thread not ready for layer {layer.name}, using fallback")
                return self._original_image_factory(layer, *args, **kwargs)
            
            logger.debug(f"Creating CudaStreamingLayer for layer: {layer.name}")
            try:
                return CudaStreamingLayer(
                    layer, 
                    render_thread=self.render_thread,
                    pixel_stream=self,
                    *args, 
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to create CudaStreamingLayer: {e}")
                # Fallback to original
                return self._original_image_factory(layer, *args, **kwargs)
        
        # Replace the Image layer factory globally
        layer_to_visual[Image] = create_streaming_layer
        
        logger.info("Registered CudaStreamingLayer factory for Image layers")
    
    def start(self):
        """Start the server with all components."""
        logger.info("Starting HeadlessServer...")
        
        # Set headless environment
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        # Create Qt application if not already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create qasync event loop for Qt/asyncio integration
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # Create and setup napari viewer (this will create the render thread)
        self._setup_viewer()
        
        # Register streaming layer factory after render thread exists
        self._register_streaming_layer_factory()
        
        # Run unified Qt/asyncio event loop
        logger.info("Starting unified Qt/asyncio event loop...")
        with loop:
            # Schedule coroutines to run once loop starts
            loop.create_task(self._start_websocket_servers())
            loop.create_task(self._start_metrics_server())
            
            loop.run_forever()
        
        # Cleanup on exit
        self.stop()
    
    def _setup_viewer(self):
        """Set up the napari viewer without adding data yet."""
        logger.info("Setting up napari viewer...")
        
        # Create viewer without data
        self.viewer = napari.Viewer(show=False)
        
        # Setup CUDA streaming infrastructure if enabled
        if self.enable_cuda:
            self._setup_render_thread()
        
        # Now add test data (will use our custom layer factory)
        self._add_test_data()
        
        # Connect viewer events to send state updates
        self.viewer.camera.events.center.connect(self._on_camera_change)
        self.viewer.camera.events.zoom.connect(self._on_camera_change)
        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        
        logger.info("Viewer setup complete")
    
    def _setup_render_thread(self):
        """Initialize the CUDA render thread."""
        logger.info("Setting up CUDA render thread...")
        
        try:
            from .render_thread import CUDARenderThread
            canvas = self.viewer.window._qt_viewer.canvas
            gl_context = canvas.native.context()
            
            self.render_thread = CUDARenderThread(
                gl_context=gl_context,
                pixel_stream=self
            )
            self.render_thread.start()
            
            logger.info("CUDA render thread started")
            
        except Exception as e:
            logger.error(f"Failed to setup CUDA render thread: {e}")
            self.render_thread = None
    
    def _add_test_data(self):
        """Add test data to the viewer."""
        # Load dataset
        if self.dataset_path and self.dataset_path.exists():
            logger.info(f"Loading dataset from {self.dataset_path}")
            data = np.load(self.dataset_path)
        else:
            logger.info("Creating test data (no dataset provided)")
            # Create interesting test pattern
            x = np.linspace(-3, 3, 512)
            y = np.linspace(-3, 3, 512)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))
            data = np.stack([Z * np.roll(np.eye(3), i, axis=0).sum() 
                           for i in range(10)])
        
        # Add layer (will use our custom factory if registered)
        layer = self.viewer.add_image(
            data,
            name='remote_data',
            colormap='viridis'
        )
        
        logger.info(f"Added test data layer: {layer.name}")
    
    async def _start_websocket_servers(self):
        """Start both WebSocket servers for state and pixel streams."""
        logger.info(f"Starting WebSocket servers on {self.host}:{self.state_port}/{self.pixel_port}")
        
        # Start both servers
        state_server = await websockets.serve(
            self._handle_state_client,
            self.host,
            self.state_port
        )
        
        pixel_server = await websockets.serve(
            self._handle_pixel_client,
            self.host,
            self.pixel_port
        )
        
        logger.info("WebSocket servers started")
        
        # Keep servers running
        await asyncio.Future()

    async def _start_metrics_server(self):
        """Serve Prometheus metrics on /metrics via aiohttp."""
        from aiohttp import web
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        app = web.Application()

        async def handle_metrics(request):
            body = generate_latest(self.metrics.registry)
            return web.Response(body=body, content_type='text/plain')

        app.add_routes([web.get('/metrics', handle_metrics)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.metrics_port)
        await site.start()
        logger.info(f"Metrics endpoint started on {self.host}:{self.metrics_port}/metrics")
    
    async def _handle_state_client(self, websocket):
        """Handle state synchronization client."""
        logger.info(f"State client connected: {websocket.remote_address}")
        self.state_clients.add(websocket)
        self.metrics.set('napari_cuda_state_clients', float(len(self.state_clients)))
        self.metrics.inc('napari_cuda_state_connects')
        
        try:
            async for message in websocket:
                # Parse command from client
                data = json.loads(message)
                await self._process_state_command(data)
        except websockets.ConnectionClosed:
            logger.info("State client disconnected")
        finally:
            self.state_clients.discard(websocket)
            self.metrics.set('napari_cuda_state_clients', float(len(self.state_clients)))
    
    async def _handle_pixel_client(self, websocket):
        """Handle pixel stream client."""
        logger.info(f"Pixel client connected: {websocket.remote_address}")
        self.pixel_clients.add(websocket)
        self.metrics.set('napari_cuda_pixel_clients', float(len(self.pixel_clients)))
        self.metrics.inc('napari_cuda_pixel_connects')
        
        try:
            await websocket.wait_closed()
        except websockets.ConnectionClosed:
            logger.info("Pixel client disconnected")
        finally:
            self.pixel_clients.discard(websocket)
            self.metrics.set('napari_cuda_pixel_clients', float(len(self.pixel_clients)))
    
    async def _process_state_command(self, command):
        """Process state command from client."""
        cmd_type = command.get('type')
        logger.debug(f"Processing command: {cmd_type}")
        
        if cmd_type == 'set_camera':
            self.viewer.camera.center = command.get('center', self.viewer.camera.center)
            self.viewer.camera.zoom = command.get('zoom', self.viewer.camera.zoom)
            
        elif cmd_type == 'set_dims':
            steps = command.get('current_step')
            if steps:
                self.viewer.dims.current_step = steps
                
        elif cmd_type == 'ping':
            # Heartbeat
            await self.broadcast_state({'type': 'pong'})
    
    def _on_camera_change(self, event=None):
        """Send camera update to clients."""
        state = {
            'type': 'camera_update',
            'center': list(self.viewer.camera.center),
            'zoom': float(self.viewer.camera.zoom)
        }
        # Schedule coroutine on the qasync event loop
        # Using asyncio.create_task since qasync integrates with the standard asyncio API
        asyncio.create_task(self.broadcast_state(state))
    
    def _on_dims_change(self, event=None):
        """Send dimensions update to clients."""
        state = {
            'type': 'dims_update',
            'current_step': list(self.viewer.dims.current_step)
        }
        # Schedule coroutine on the qasync event loop
        # Using asyncio.create_task since qasync integrates with the standard asyncio API
        asyncio.create_task(self.broadcast_state(state))
    
    async def broadcast_state(self, state):
        """Broadcast state to all connected clients."""
        if self.state_clients:
            message = json.dumps(state)
            await asyncio.gather(*[
                client.send(message)
                for client in self.state_clients
            ], return_exceptions=True)
    
    async def broadcast_frame(self, frame_data):
        """Broadcast encoded frame to all pixel clients."""
        if self.pixel_clients:
            self.metrics.inc('napari_cuda_frames_total')
            self.metrics.inc('napari_cuda_bytes_total', len(frame_data))
            await asyncio.gather(*[
                client.send(frame_data)
                for client in self.pixel_clients
            ], return_exceptions=True)
    
    def send_frame(self, frame_data):
        """Called by CUDA thread to send frame."""
        # With qasync, we can schedule coroutines directly on the main loop
        asyncio.create_task(self.broadcast_frame(frame_data))
    
    def stop(self):
        """Stop the server and clean up."""
        logger.info("Stopping HeadlessServer...")
        
        # Restore original layer factory
        if hasattr(self, '_original_image_factory'):
            layer_to_visual[Image] = self._original_image_factory
            logger.info("Restored original Image layer factory")
        
        if self.render_thread:
            self.render_thread.stop()
            self.render_thread.wait()
        
        if self.cuda_layer:
            self.cuda_layer.close()
        
        # Stop qasync event loop
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            loop.stop()
        
        logger.info("HeadlessServer stopped")


def main():
    """Command-line entry point for napari-cuda server."""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment configuration
    load_dotenv('.env.hpc')
    
    parser = argparse.ArgumentParser(
        description='napari-cuda GPU-accelerated streaming server'
    )
    
    parser.add_argument(
        'dataset',
        nargs='?',
        default=None,
        help='Path to numpy array or image to load'
    )
    
    parser.add_argument(
        '--host',
        default=os.getenv('NAPARI_CUDA_HOST', '0.0.0.0'),
        help='Host address for WebSocket servers (default: 0.0.0.0 for HPC access)'
    )
    
    parser.add_argument(
        '--state-port',
        type=int,
        default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')),
        help='Port for state synchronization (default: 8081)'
    )
    
    parser.add_argument(
        '--pixel-port',
        type=int,
        default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')),
        help='Port for pixel stream (default: 8082)'
    )
    
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=int(os.getenv('NAPARI_CUDA_METRICS_PORT', '8083')),
        help='Port for Prometheus metrics (default: 8083)'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA acceleration (CPU fallback)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--perf',
        action='store_true',
        help='Enable real-time performance monitoring in terminal'
    )
    
    parser.add_argument(
        '--perf-interval',
        type=float,
        default=1.0,
        help='Performance monitor update interval in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = HeadlessServer(
        dataset_path=args.dataset,
        host=args.host,
        state_port=args.state_port,
        pixel_port=args.pixel_port,
        metrics_port=args.metrics_port,
        enable_cuda=not args.no_cuda
    )
    
    # Start performance monitor if requested
    perf_monitor = None
    if args.perf:
        from .perf_monitor import PerfMonitor
        
        # Create log file with timestamp
        import time
        log_file = f"napari_cuda_perf_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        perf_monitor = PerfMonitor(
            server.metrics, 
            update_interval=args.perf_interval,
            log_file=log_file
        )
        logger.info(f"Starting performance monitor (logging to {log_file})...")
        perf_monitor.start()
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C")
    finally:
        if perf_monitor:
            perf_monitor.stop()
        server.stop()


if __name__ == "__main__":
    main()
