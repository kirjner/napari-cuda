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
from threading import Thread

import napari
from qtpy.QtCore import QThread
import websockets

from .cuda_streaming_layer import CudaStreamingLayer
from .render_thread import CUDARenderThread
from ..protocol.messages import StateMessage, FrameMessage

logger = logging.getLogger(__name__)


class AsyncioThread(QThread):
    """Thread to run asyncio event loop alongside Qt."""
    
    def __init__(self):
        super().__init__()
        self.loop = None
        self.running = False
    
    def run(self):
        """Run asyncio event loop in this thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.running = True
        self.loop.run_forever()
    
    def stop(self):
        """Stop the event loop."""
        if self.loop and self.running:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.running = False


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
        
        self.viewer = None
        self.cuda_layer = None
        self.render_thread = None
        self.asyncio_thread = None
        
        # WebSocket clients
        self.state_clients = set()
        self.pixel_clients = set()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"HeadlessServer initialized - CUDA: {enable_cuda}")
    
    def start(self):
        """Start the server with all components."""
        logger.info("Starting HeadlessServer...")
        
        # Set headless environment
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        # Start asyncio thread for WebSocket servers
        self.asyncio_thread = AsyncioThread()
        self.asyncio_thread.start()
        
        # Wait for asyncio loop to be ready
        while self.asyncio_thread.loop is None:
            QThread.msleep(10)
        
        # Start WebSocket servers in asyncio thread
        asyncio.run_coroutine_threadsafe(
            self._start_websocket_servers(),
            self.asyncio_thread.loop
        )
        
        # Create and run napari in main thread
        with napari.gui_qt(visible=False):
            self._setup_viewer()
            logger.info("Starting napari event loop...")
            napari.run()
        
        # Cleanup on exit
        self.stop()
    
    def _setup_viewer(self):
        """Set up the napari viewer with test data."""
        logger.info("Setting up napari viewer...")
        
        # Create viewer
        self.viewer = napari.Viewer(show=False)
        
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
        
        # Add layer
        layer = self.viewer.add_image(
            data,
            name='remote_data',
            colormap='viridis'
        )
        
        if self.enable_cuda:
            self._setup_cuda_streaming(layer)
        
        # Connect viewer events to send state updates
        self.viewer.camera.events.center.connect(self._on_camera_change)
        self.viewer.camera.events.zoom.connect(self._on_camera_change)
        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        
        logger.info("Viewer setup complete")
    
    def _setup_cuda_streaming(self, layer):
        """Replace normal layer with CUDA streaming layer."""
        logger.info("Setting up CUDA streaming...")
        
        try:
            # Start render thread for CUDA operations
            from .render_thread import CUDARenderThread
            canvas = self.viewer.window._qt_viewer.canvas
            gl_context = canvas.native.context()
            
            self.render_thread = CUDARenderThread(
                gl_context=gl_context,
                pixel_stream=self
            )
            self.render_thread.start()
            
            # Replace vispy layer with streaming version
            self.cuda_layer = CudaStreamingLayer(
                layer,
                render_thread=self.render_thread,
                pixel_stream=self
            )
            
            # Inject into canvas
            canvas.layer_to_visual[layer] = self.cuda_layer
            
            logger.info("CUDA streaming layer installed")
            
        except Exception as e:
            logger.error(f"Failed to setup CUDA streaming: {e}")
            logger.info("Falling back to CPU mode")
    
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
    
    async def _handle_state_client(self, websocket, path):
        """Handle state synchronization client."""
        logger.info(f"State client connected: {websocket.remote_address}")
        self.state_clients.add(websocket)
        
        try:
            async for message in websocket:
                # Parse command from client
                data = json.loads(message)
                await self._process_state_command(data)
        except websockets.ConnectionClosed:
            logger.info("State client disconnected")
        finally:
            self.state_clients.discard(websocket)
    
    async def _handle_pixel_client(self, websocket, path):
        """Handle pixel stream client."""
        logger.info(f"Pixel client connected: {websocket.remote_address}")
        self.pixel_clients.add(websocket)
        
        try:
            await websocket.wait_closed()
        except websockets.ConnectionClosed:
            logger.info("Pixel client disconnected")
        finally:
            self.pixel_clients.discard(websocket)
    
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
        if self.asyncio_thread and self.asyncio_thread.loop:
            state = {
                'type': 'camera_update',
                'center': list(self.viewer.camera.center),
                'zoom': float(self.viewer.camera.zoom)
            }
            asyncio.run_coroutine_threadsafe(
                self.broadcast_state(state),
                self.asyncio_thread.loop
            )
    
    def _on_dims_change(self, event=None):
        """Send dimensions update to clients."""
        if self.asyncio_thread and self.asyncio_thread.loop:
            state = {
                'type': 'dims_update',
                'current_step': list(self.viewer.dims.current_step)
            }
            asyncio.run_coroutine_threadsafe(
                self.broadcast_state(state),
                self.asyncio_thread.loop
            )
    
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
            await asyncio.gather(*[
                client.send(frame_data)
                for client in self.pixel_clients
            ], return_exceptions=True)
    
    def send_frame(self, frame_data):
        """Called by CUDA thread to send frame."""
        if self.asyncio_thread and self.asyncio_thread.loop:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_frame(frame_data),
                self.asyncio_thread.loop
            )
    
    def stop(self):
        """Stop the server and clean up."""
        logger.info("Stopping HeadlessServer...")
        
        if self.asyncio_thread:
            self.asyncio_thread.stop()
            self.asyncio_thread.wait()
        
        if self.render_thread:
            self.render_thread.stop()
            self.render_thread.wait()
        
        if self.cuda_layer:
            self.cuda_layer.close()
        
        logger.info("HeadlessServer stopped")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create and run server
    server = HeadlessServer(dataset_path=dataset)
    server.start()