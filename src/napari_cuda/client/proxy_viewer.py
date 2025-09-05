"""
ProxyViewer - Client-side viewer that forwards commands to remote server.

Critical: This must prevent local Window creation while maintaining
the Viewer API for the QtViewer to interact with.
"""

import asyncio
import json
import logging
import websockets
from typing import Optional

import napari
from napari.viewer import Viewer
from napari.components import LayerList, Dims, Camera

logger = logging.getLogger(__name__)


class ProxyViewer(Viewer):
    """
    A viewer that forwards all commands to a remote server instead of 
    rendering locally.
    
    This is the critical client-side component that maintains the napari
    API while preventing local rendering.
    """
    
    def __init__(self, server_host='localhost', server_port=8081, **kwargs):
        """
        Initialize proxy viewer connected to remote server.
        
        Critical: We must carefully handle initialization to prevent
        creating a local Window.
        
        Parameters
        ----------
        server_host : str
            Remote server hostname
        server_port : int
            Remote server state sync port
        """
        self.server_host = server_host
        self.server_port = server_port
        self._state_websocket = None
        
        # CRITICAL: Two approaches to prevent Window creation:
        
        # Option A: Don't call super().__init__ at all
        # Instead manually initialize only what we need
        self._init_without_window()
        
        # Option B: Call super() with special flag (would require napari changes)
        # super().__init__(create_window=False, **kwargs)
        
        # Connect to server
        self._connect_to_server()
        
        logger.info(f"ProxyViewer initialized for {server_host}:{server_port}")
    
    def _init_without_window(self):
        """
        Initialize viewer components without creating a Window.
        
        This manually sets up the minimal components needed for the
        viewer to function as a proxy.
        """
        # Initialize base components that don't require rendering
        self.layers = LayerList()
        self.dims = Dims()
        self.camera = Camera()
        
        # Set window to None explicitly
        self._window = None
        self._overlays = {}
        
        # Initialize events (needed for Qt connections)
        from napari.utils.events import EmitterGroup
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            status=None,
            help=None,
            title=None,
            theme=None,
            reset=None,
        )
        
        # Connect local events to forward to server
        self.camera.events.center.connect(self._on_camera_change)
        self.camera.events.zoom.connect(self._on_camera_change)
        self.dims.events.current_step.connect(self._on_dims_change)
        
        # Disable layer modifications
        self.layers.events.disconnect()
    
    def _connect_to_server(self):
        """Establish WebSocket connection to server."""
        # Create asyncio task for connection
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Start connection in background
        asyncio.create_task(self._maintain_connection())
    
    async def _maintain_connection(self):
        """Maintain persistent connection to server."""
        url = f"ws://{self.server_host}:{self.server_port}"
        logger.info(f"Connecting to server at {url}")
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    self._state_websocket = websocket
                    logger.info("Connected to server")
                    
                    # Listen for state updates from server
                    async for message in websocket:
                        data = json.loads(message)
                        await self._handle_server_message(data)
                        
            except Exception as e:
                logger.error(f"Connection lost: {e}")
                self._state_websocket = None
                
                # Reconnect after delay
                await asyncio.sleep(5)
                logger.info("Attempting reconnection...")
    
    async def _handle_server_message(self, data):
        """Process state update from server."""
        msg_type = data.get('type')
        
        if msg_type == 'camera_update':
            # Update local camera without triggering events
            self.camera.events.center.block()
            self.camera.events.zoom.block()
            
            self.camera.center = data.get('center', self.camera.center)
            self.camera.zoom = data.get('zoom', self.camera.zoom)
            
            self.camera.events.center.unblock()
            self.camera.events.zoom.unblock()
            
        elif msg_type == 'dims_update':
            # Update local dims without triggering events
            self.dims.events.current_step.block()
            self.dims.current_step = data.get('current_step', self.dims.current_step)
            self.dims.events.current_step.unblock()
    
    def _on_camera_change(self, event=None):
        """Forward camera change to server."""
        if self._state_websocket:
            command = {
                'type': 'set_camera',
                'center': list(self.camera.center),
                'zoom': float(self.camera.zoom)
            }
            asyncio.create_task(self._send_command(command))
    
    def _on_dims_change(self, event=None):
        """Forward dimension change to server."""
        if self._state_websocket:
            command = {
                'type': 'set_dims',
                'current_step': list(self.dims.current_step)
            }
            asyncio.create_task(self._send_command(command))
    
    async def _send_command(self, command):
        """Send command to server."""
        if self._state_websocket:
            try:
                await self._state_websocket.send(json.dumps(command))
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
    
    # Override methods that would try to render locally
    def add_image(self, *args, **kwargs):
        """Block local image addition."""
        logger.warning("ProxyViewer: add_image blocked (server-side only)")
        return None
    
    def add_labels(self, *args, **kwargs):
        """Block local labels addition."""
        logger.warning("ProxyViewer: add_labels blocked (server-side only)")
        return None
    
    def add_points(self, *args, **kwargs):
        """Block local points addition."""
        logger.warning("ProxyViewer: add_points blocked (server-side only)")
        return None
    
    def add_shapes(self, *args, **kwargs):
        """Block local shapes addition."""
        logger.warning("ProxyViewer: add_shapes blocked (server-side only)")
        return None
    
    def screenshot(self, *args, **kwargs):
        """Screenshot happens server-side."""
        logger.info("Screenshot requested from server")
        # Could implement by requesting high-quality frame from server
        return None
    
    @property
    def window(self):
        """Return None as we have no local window."""
        return self._window
    
    def close(self):
        """Close connection to server."""
        if self._state_websocket:
            asyncio.create_task(self._state_websocket.close())
        logger.info("ProxyViewer closed")