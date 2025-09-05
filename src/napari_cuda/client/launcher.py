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
    
    # Create ProxyViewer (inherits from ViewerModel, no Window created)
    proxy_viewer = ProxyViewer(
        server_host=server_host,
        server_port=state_port
    )
    
    # Create Window manually with our ProxyViewer
    # The Window constructor expects a Viewer, but ViewerModel works too
    window = Window(proxy_viewer, show=False)
    
    # Store window reference in the ProxyViewer
    proxy_viewer.window = window
    
    # Replace the canvas with our streaming canvas
    qt_viewer = window._qt_viewer
    
    # Create streaming canvas
    streaming_canvas = StreamingCanvas(
        proxy_viewer,
        server_host=server_host,
        server_port=pixel_port,
        parent=qt_viewer
    )
    
    # Replace the default canvas with streaming canvas
    old_canvas = qt_viewer.canvas
    qt_viewer.canvas = streaming_canvas
    
    # Replace in the canvas splitter widget
    if hasattr(qt_viewer, 'canvas_splitter'):
        # Find the old canvas widget index
        for i in range(qt_viewer.canvas_splitter.count()):
            if qt_viewer.canvas_splitter.widget(i) == old_canvas.native:
                qt_viewer.canvas_splitter.replaceWidget(i, streaming_canvas.native)
                break
    
    # Clean up old canvas
    old_canvas.close()
    
    # Now show the window
    window.show()
    
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