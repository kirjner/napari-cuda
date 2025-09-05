#!/usr/bin/env python
"""
Minimum Viable Streaming Client.
Just display frames. Send camera commands.
"""

import asyncio
import struct
import json
import numpy as np
from io import BytesIO
from PIL import Image
import napari
from qtpy.QtCore import QTimer


class MVPClient:
    """Display streamed frames in local napari."""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        
        # Create a local viewer (will display streamed frames)
        self.viewer = napari.Viewer(title=f"napari-engine client → {host}:{port}")
        
        # Add a placeholder layer for displaying frames
        placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
        self.display_layer = self.viewer.add_image(
            placeholder, 
            name="remote_view",
            rgb=True
        )
        
        # Command writer (will be set on connection)
        self.cmd_writer = None
        
        # Hook camera events to send commands
        self.viewer.camera.events.center.connect(self._on_camera_change)
        self.viewer.camera.events.zoom.connect(self._on_camera_change)
        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        
    async def connect(self):
        """Connect to server."""
        # Frame stream connection
        self.frame_reader, self.frame_writer = await asyncio.open_connection(
            self.host, self.port
        )
        
        # Command stream connection  
        self.cmd_reader, self.cmd_writer = await asyncio.open_connection(
            self.host, self.port + 1
        )
        
        print(f"Connected to {self.host}:{self.port}")
        
    async def receive_frames(self):
        """Receive and display frames."""
        while True:
            try:
                # Read header
                header = await self.frame_reader.readexactly(8)
                frame_num, size = struct.unpack('!II', header)
                
                # Read frame data
                jpeg_bytes = await self.frame_reader.readexactly(size)
                
                # Decompress and display
                img = Image.open(BytesIO(jpeg_bytes))
                frame_array = np.array(img)
                
                # Update display layer
                self.display_layer.data = frame_array
                
            except asyncio.IncompleteReadError:
                print("Server disconnected")
                break
                
    def _on_camera_change(self, event):
        """Send camera update to server."""
        if self.cmd_writer:
            cmd = {
                'type': 'camera',
                'center': list(self.viewer.camera.center),
                'zoom': float(self.viewer.camera.zoom),
            }
            self._send_command(cmd)
    
    def _on_dims_change(self, event):
        """Send dims update to server."""
        if self.cmd_writer:
            cmd = {
                'type': 'dims',
                'step': list(self.viewer.dims.current_step),
            }
            self._send_command(cmd)
    
    def _send_command(self, cmd):
        """Send command to server."""
        if self.cmd_writer:
            cmd_bytes = json.dumps(cmd).encode()
            header = struct.pack('!I', len(cmd_bytes))
            self.cmd_writer.write(header + cmd_bytes)
            # Note: we're not awaiting drain() for lower latency
    
    async def run(self):
        """Connect and start receiving frames."""
        await self.connect()
        await self.receive_frames()


def main():
    """Run the MVP client."""
    import sys
    from qtpy.QtWidgets import QApplication
    
    # Parse args
    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    # Create Qt application
    app = QApplication.instance() or QApplication([])
    
    # Create client
    client = MVPClient(host, port)
    
    # Run asyncio in Qt event loop
    import qasync
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    with loop:
        loop.create_task(client.run())
        loop.run_forever()


if __name__ == "__main__":
    main()