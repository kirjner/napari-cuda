"""
Launcher for napari-cuda streaming client.

This creates the client application that connects to a remote napari-cuda server.
"""

import sys
import os
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
                          debug=False,
                          vt_smoke: bool = False):
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
    # Setup logging - check env var too
    if os.getenv('NAPARI_CUDA_DEBUG', '').lower() in ('1', 'true', 'yes'):
        debug = True
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Launching streaming client for {server_host}")
    
    # Create ProxyViewer (inherits from ViewerModel, no Window created)
    proxy_viewer = ProxyViewer(
        server_host=server_host,
        server_port=state_port,
        offline=bool(vt_smoke),
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
        vt_smoke=vt_smoke,
        key_map_handler=getattr(qt_viewer, '_key_map_handler', None),
        parent=qt_viewer
    )
    
    # Replace the default canvas with streaming canvas
    old_canvas = qt_viewer.canvas
    qt_viewer.canvas = streaming_canvas

    # If ProxyViewer connection was deferred, try again now that Qt loop exists
    if getattr(proxy_viewer, '_connection_pending', False):
        try:
            proxy_viewer._connect_to_server()
        except Exception as e:
            logger.debug("Deferred server connect failed (will retry later)", exc_info=True)
    
    # Replace canvas inside the QtViewer welcome overlay stack (current napari layout)
    try:
        ow = qt_viewer._welcome_widget  # QtWidgetOverlay(QStackedWidget)
        # If the old canvas.native is present in the overlay, replace it with the new one
        try:
            for i in range(ow.count()):
                if ow.widget(i) is old_canvas.native:
                    # Remove the old widget and insert our streaming canvas at index 0
                    ow.removeWidget(old_canvas.native)
                    break
        except Exception:
            logger.debug("launcher: failed to remove old canvas from overlay", exc_info=True)
        # Ensure our streaming canvas is the primary widget in the overlay
        try:
            # Avoid duplicate insert if already present
            present = any(ow.widget(i) is streaming_canvas.native for i in range(ow.count()))
            if not present:
                ow.insertWidget(0, streaming_canvas.native)
        except Exception:
            logger.debug("launcher: failed to insert streaming canvas into overlay", exc_info=True)
        # Hide the welcome screen so only the video canvas is visible
        try:
            qt_viewer.set_welcome_visible(False)
        except Exception:
            try:
                ow.set_welcome_visible(False)
            except Exception:
                logger.debug("launcher: failed to hide welcome overlay (overlay widget)", exc_info=True)
    except Exception:
        logger.debug("launcher: overlay manipulation failed", exc_info=True)
    
    # Clean up old canvas widget
    try:
        old_canvas.delete()
    except Exception:
        # Fallback: delete native widget if present
        try:
            old_canvas.native.deleteLater()
        except Exception:
            logger.debug("launcher: cleanup old canvas native failed", exc_info=True)
    
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
    # Unified smoke flag (offline; no server)
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run client-side smoke test (offline)'
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
    smoke = bool(args.smoke)
    launch_streaming_client(
        server_host=args.host,
        state_port=args.state_port,
        pixel_port=args.pixel_port,
        debug=args.debug,
        vt_smoke=smoke
    )


if __name__ == "__main__":
    main()
