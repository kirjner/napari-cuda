#!/usr/bin/env python3
"""
Validate local client environment for napari-cuda.

Tests WebSocket connectivity, video decoding capabilities, and napari imports.
"""

import sys
import asyncio
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('validate_local')


async def test_websocket(host='127.0.0.1', state_port=8081, pixel_port=8082, strict=False):
    """Test WebSocket connectivity to server ports."""
    try:
        import websockets
        
        ok = True
        # Try state port
        try:
            uri = f'ws://{host}:{state_port}'
            async with websockets.connect(uri, open_timeout=2) as ws:
                await ws.send('{"type": "ping"}')
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                log.info(f'✓ State port {state_port}: Connected and responsive')
        except Exception as e:
            log.warning(f'✗ State port {state_port}: {e}')
            ok = False
            
        # Try pixel port  
        try:
            uri = f'ws://{host}:{pixel_port}'
            async with websockets.connect(uri, open_timeout=2) as ws:
                log.info(f'✓ Pixel port {pixel_port}: Connected')
        except Exception as e:
            log.warning(f'✗ Pixel port {pixel_port}: {e}')
            ok = False
            
    except ImportError:
        log.error('✗ websockets not installed (uv sync --extra client)')
        return False
        
    return ok if strict else True


def test_video_decoder():
    """Test H.264 decoding capability."""
    try:
        import av
        
        # Test codec availability
        codec = av.codec.Codec('h264', 'r')
        log.info(f'✓ H.264 decoder available: {codec.name}')
        
        # Test creating decoder
        decoder = codec.create()
        log.info('✓ H.264 decoder context created')
        
        return True
        
    except ImportError:
        log.error('✗ PyAV not installed (uv sync --extra client)')
        return False
    except Exception as e:
        log.error(f'✗ H.264 decoder error: {e}')
        return False


def test_napari_client():
    """Test napari client imports."""
    try:
        import napari
        log.info(f'✓ napari version: {napari.__version__}')
        
        from napari_cuda.client.proxy_viewer import ProxyViewer
        log.info('✓ ProxyViewer importable')
        
        from napari_cuda.client.streaming_canvas import StreamingCanvas
        log.info('✓ StreamingCanvas importable')
        
        from napari_cuda.client.launcher import launch_streaming_client
        log.info('✓ Client launcher importable')
        
        return True
        
    except ImportError as e:
        log.error(f'✗ Import error: {e}')
        return False


def main():
    """Run all validation tests."""
    parser = argparse.ArgumentParser(description='Validate local client environment for napari-cuda')
    parser.add_argument('--host', default=os.getenv('NAPARI_CUDA_HOST', '127.0.0.1'))
    parser.add_argument('--state-port', type=int, default=int(os.getenv('NAPARI_CUDA_STATE_PORT', '8081')))
    parser.add_argument('--pixel-port', type=int, default=int(os.getenv('NAPARI_CUDA_PIXEL_PORT', '8082')))
    parser.add_argument('--skip-websocket', action='store_true', help='Skip WebSocket connectivity tests')
    parser.add_argument('--strict', action='store_true', help='Fail if WebSocket ports are unreachable')
    args = parser.parse_args()

    log.info('=== napari-cuda Client Validation ===\n')
    
    results = []
    
    # Test imports
    log.info('1. Testing napari client modules...')
    results.append(test_napari_client())
    print()
    
    # Test video decoder
    log.info('2. Testing H.264 video decoder...')
    results.append(test_video_decoder())
    print()
    
    # Test WebSocket connectivity
    if args.skip_websocket:
        log.info('3. Skipping WebSocket connectivity (requested)')
        results.append(True)
    else:
        log.info(f'3. Testing WebSocket connectivity to {args.host}:{args.state_port}/{args.pixel_port}...')
        results.append(asyncio.run(test_websocket(args.host, args.state_port, args.pixel_port, strict=args.strict)))
    print()
    
    # Summary
    if all(results):
        log.info('=== All tests passed! ===')
        log.info('You can now run: uv run napari-cuda-client')
        return 0
    else:
        log.error('=== Some tests failed ===')
        log.error('Fix the issues above before running the client')
        return 1


if __name__ == '__main__':
    sys.exit(main())
