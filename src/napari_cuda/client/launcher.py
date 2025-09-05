"""
Launcher for napari-cuda streaming client.

This creates the client application that connects to a remote napari-cuda server.
"""

import sys
import logging
import argparse

import napari
from napari._qt.qt_viewer import QtViewer
from napari._qt.qt_main_window import Window

from .proxy_viewer import ProxyViewer
from .streaming_canvas import StreamingCanvas

logger = logging.getLogger(__name__)


def launch_streaming_client(server_host='localhost', 
                          state_port=8081,
                          pixel_port=8082,
                          debug=False):
    """
    Launch napari client connected to remote server.
    
    Parameters
    ----------
    server_host : str
        Remote server hostname/IP
    state_port : int
        Port for state synchronization
    pixel_port : int
        Port for pixel/video stream
    debug : bool
        Enable debug logging
    """
    # Setup logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Launching streaming client for {server_host}")
    
    # CRITICAL: Handle ProxyViewer initialization carefully
    
    # Option A: Create ProxyViewer without Window
    proxy_viewer = ProxyViewer(
        server_host=server_host,
        server_port=state_port
    )
    
    # Create Qt components with custom canvas
    qt_viewer = QtViewer(
        proxy_viewer,
        canvas_class=lambda viewer, **kwargs: StreamingCanvas(
            viewer,
            server_host=server_host,
            server_port=pixel_port,
            **kwargs
        )
    )
    
    # Create window manually
    window = Window(qt_viewer, show=True)
    
    # Store reference (some parts of napari expect this)
    proxy_viewer._window = window
    
    # Option B: Surgical replacement approach
    # viewer = napari.Viewer(show=False)
    # viewer._window.qt_viewer.canvas = StreamingCanvas(...)
    # ... replace viewer internals with proxy ...
    
    logger.info("Client launched successfully")
    
    # Run Qt event loop
    napari.run()
    
    # Cleanup
    proxy_viewer.close()
    logger.info("Client closed")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='napari-cuda streaming client'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Remote server hostname/IP (default: localhost)'
    )
    
    parser.add_argument(
        '--state-port',
        type=int,
        default=8081,
        help='State synchronization port (default: 8081)'
    )
    
    parser.add_argument(
        '--pixel-port',
        type=int,
        default=8082,
        help='Pixel stream port (default: 8082)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # For SSH tunnel setup hint
    parser.add_argument(
        '--tunnel-hint',
        action='store_true',
        help='Show SSH tunnel setup command'
    )
    
    args = parser.parse_args()
    
    if args.tunnel_hint:
        print("SSH Tunnel Setup:")
        print(f"ssh -L {args.state_port}:localhost:{args.state_port} \\")
        print(f"    -L {args.pixel_port}:localhost:{args.pixel_port} \\")
        print(f"    user@{args.host}")
        print("\nThen run client with: --host localhost")
        sys.exit(0)
    
    # Launch client
    launch_streaming_client(
        server_host=args.host,
        state_port=args.state_port,
        pixel_port=args.pixel_port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()