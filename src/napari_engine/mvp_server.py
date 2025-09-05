#!/usr/bin/env python
"""
Minimum Viable Streaming Server.
Following Casey's approach: Just send frames. Ship it.
"""

import asyncio
import struct
import numpy as np
import napari
from io import BytesIO
from PIL import Image


class MVPServer:
    """Dead simple frame streaming. No state sync. Just pixels."""
    
    def __init__(self, data_path=None, port=8765):
        self.port = port
        self.viewer = napari.Viewer(show=False)
        
        # Load some data
        if data_path:
            # Load your zarr/data here
            pass
        else:
            # Test data
            self.viewer.add_image(
                np.random.random((50, 512, 512)),
                name="test_data"
            )
        
        self.clients = set()
        
    async def handle_client(self, reader, writer):
        """Handle one client connection."""
        addr = writer.get_extra_info('peername')
        print(f"Client connected: {addr}")
        self.clients.add(writer)
        
        try:
            # Send frames at 30 FPS
            while True:
                # Take screenshot
                frame = self.viewer.screenshot(canvas_only=True, flash=False)
                
                # Compress to JPEG (fastest reasonable compression)
                buffer = BytesIO()
                img = Image.fromarray(frame)
                img.save(buffer, format='JPEG', quality=85)
                jpeg_bytes = buffer.getvalue()
                
                # Send header + frame
                header = struct.pack('!II', 0, len(jpeg_bytes))  # frame_num=0 for now
                writer.write(header + jpeg_bytes)
                await writer.drain()
                
                # 30 FPS = 33.33ms per frame
                await asyncio.sleep(0.033)
                
        except (ConnectionError, asyncio.CancelledError):
            print(f"Client disconnected: {addr}")
        finally:
            self.clients.discard(writer)
            writer.close()
            await writer.wait_closed()
    
    async def handle_commands(self, reader, writer):
        """Handle command stream from client (camera, dims, etc)."""
        try:
            while True:
                # Read command size
                size_bytes = await reader.readexactly(4)
                size = struct.unpack('!I', size_bytes)[0]
                
                # Read command
                cmd_bytes = await reader.readexactly(size)
                
                # Parse and apply (pseudo-code)
                import json
                cmd = json.loads(cmd_bytes)
                
                if cmd['type'] == 'camera':
                    self.viewer.camera.center = cmd.get('center', self.viewer.camera.center)
                    self.viewer.camera.zoom = cmd.get('zoom', self.viewer.camera.zoom)
                elif cmd['type'] == 'dims':
                    self.viewer.dims.current_step = cmd.get('step', self.viewer.dims.current_step)
                    
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
    
    async def run(self):
        """Start the server."""
        # Frame streaming server
        frame_server = await asyncio.start_server(
            self.handle_client, '0.0.0.0', self.port
        )
        
        # Command server (port + 1)
        cmd_server = await asyncio.start_server(
            self.handle_commands, '0.0.0.0', self.port + 1
        )
        
        addr = frame_server.sockets[0].getsockname()
        print(f"Streaming server running on {addr[0]}:{addr[1]}")
        print(f"Command server on port {self.port + 1}")
        print("Waiting for clients...")
        
        async with frame_server, cmd_server:
            await asyncio.gather(
                frame_server.serve_forever(),
                cmd_server.serve_forever()
            )


def main():
    """Run the MVP server."""
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    server = MVPServer(data_path)
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()