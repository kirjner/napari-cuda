"""
CUDARenderThread - Manages CUDA/OpenGL contexts and performs GPU operations.

This thread owns the OpenGL and CUDA contexts and ensures all GPU operations
happen on the correct thread.
"""

import logging
import queue
import threading
import contextlib
from qtpy.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

# Try to import CUDA - will fail on local development
try:
    import pycuda.driver as cuda
    import pycuda.gl
    from pycuda.gl import RegisteredImage, graphics_map_flags
    import OpenGL.GL as GL
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    logger.warning("CUDA libraries not available")

# Thread-local storage for CUDA contexts
_thread_local = threading.local()

@contextlib.contextmanager
def cuda_context_guard(context):
    """
    Context manager for safe CUDA context operations.
    
    Ensures that CUDA operations happen with the correct context
    and handles context cleanup properly.
    """
    if not HAS_CUDA or context is None:
        yield
        return
    
    # Get current context
    old_context = None
    try:
        old_context = cuda.Context.get_current()
    except cuda.LogicError:
        # No context is current
        pass
    
    try:
        # Push our context if it's different
        if old_context != context:
            context.push()
        
        yield
        
    finally:
        # Restore previous context
        try:
            if old_context != context and old_context is not None:
                context.pop()
        except cuda.LogicError:
            # Context was already popped or invalid
            pass


class CUDARenderThread(QThread):
    """
    Dedicated thread for CUDA/OpenGL operations.
    
    This thread:
    - Owns the OpenGL context
    - Owns the CUDA context  
    - Registers OpenGL textures with CUDA
    - Performs NVENC encoding
    - Sends encoded frames to pixel stream
    """
    
    # Signal emitted when frame is ready
    frame_ready = pyqtSignal(bytes)
    
    def __init__(self, gl_context=None, pixel_stream=None):
        """
        Initialize render thread.
        
        Parameters
        ----------
        gl_context : QOpenGLContext
            OpenGL context to make current on this thread
        pixel_stream : HeadlessServer
            Server to send encoded frames to
        """
        super().__init__()
        
        self.gl_context = gl_context
        self.pixel_stream = pixel_stream
        self.cuda_context = None
        self.nvenc_encoder = None
        
        # Registered textures cache (thread-safe access needed)
        self.registered_textures = {}
        self._texture_lock = threading.RLock()
        
        # Queue for capture requests from main thread
        self.capture_queue = queue.Queue()
        
        # Control flag
        self.running = True
        
        # CUDA context lock for thread safety
        self._cuda_lock = threading.RLock()
        
        logger.info("CUDARenderThread initialized")
    
    def run(self):
        """Main thread loop - process capture requests."""
        logger.info("CUDARenderThread starting...")
        
        if not HAS_CUDA:
            logger.error("CUDA not available, thread exiting")
            return
        
        try:
            # Initialize contexts on this thread
            self._setup_contexts()
            
            # Initialize NVENC encoder
            self._setup_nvenc()
            
            # Process capture requests
            while self.running:
                try:
                    # Wait for capture request (timeout for checking running flag)
                    request = self.capture_queue.get(timeout=0.1)
                    
                    # Capture and encode
                    encoded_frame = self._capture_and_encode(request)
                    
                    if encoded_frame and self.pixel_stream:
                        # Send to pixel stream
                        self.pixel_stream.send_frame(encoded_frame)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize render thread: {e}")
        finally:
            self._cleanup()
    
    def _setup_contexts(self):
        """Initialize OpenGL and CUDA contexts on this thread."""
        logger.info("Setting up OpenGL/CUDA contexts...")
        
        # Make OpenGL context current on this thread
        if self.gl_context:
            self.gl_context.makeCurrent()
            logger.info("OpenGL context made current")
        
        # Initialize CUDA
        cuda.init()
        device_count = cuda.Device.count()
        logger.info(f"Found {device_count} CUDA devices")
        
        if device_count == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Create CUDA context on first device
        device = cuda.Device(0)
        self.cuda_context = device.make_context()
        
        logger.info(f"CUDA context created on device: {device.name()}")
        
        # Enable OpenGL interop
        import pycuda.gl
        pycuda.gl.init()
        
        logger.info("CUDA-OpenGL interop initialized")
    
    def _setup_nvenc(self):
        """Initialize NVENC encoder."""
        try:
            from ..cuda.nvenc_wrapper import NVENCEncoder
            
            self.nvenc_encoder = NVENCEncoder(
                width=1920,
                height=1080,
                fps=60,
                bitrate=10000000  # 10 Mbps
            )
            
            logger.info("NVENC encoder initialized")
            
        except ImportError:
            logger.warning("NVENC wrapper not available, using fallback encoding")
            self.nvenc_encoder = None
        except Exception as e:
            logger.error(f"Failed to initialize NVENC: {e}")
            self.nvenc_encoder = None
    
    def _capture_and_encode(self, request):
        """
        Capture OpenGL texture and encode to H.264.
        
        Parameters
        ----------
        request : dict
            Contains 'texture_id', 'frame_num', 'timestamp'
            
        Returns
        -------
        bytes
            Encoded H.264 frame
        """
        texture_id = request['texture_id']
        frame_num = request.get('frame_num', 0)
        
        logger.debug(f"Capturing texture {texture_id} (frame {frame_num})")
        
        # Use thread safety for all CUDA operations
        with self._cuda_lock:
            with cuda_context_guard(self.cuda_context):
                # Register texture with CUDA if not already done
                with self._texture_lock:
                    if texture_id not in self.registered_textures:
                        try:
                            self.registered_textures[texture_id] = RegisteredImage(
                                int(texture_id),
                                GL.GL_TEXTURE_2D,
                                graphics_map_flags.READ_ONLY
                            )
                            logger.debug(f"Registered texture {texture_id} with CUDA")
                        except Exception as e:
                            logger.error(f"Failed to register texture: {e}")
                            return None
                    
                    # Get registered image
                    reg_image = self.registered_textures[texture_id]
                
                # Map texture to CUDA (outside texture lock but inside CUDA lock)
                try:
                    reg_image.map()
                    cuda_array = reg_image.get_mapped_array(0, 0)
                    
                    # Encode with NVENC
                    if self.nvenc_encoder:
                        encoded = self.nvenc_encoder.encode_surface(cuda_array)
                    else:
                        # Fallback: Just send a placeholder
                        encoded = b'FRAME_%d' % frame_num
                    
                    reg_image.unmap()
                    
                    return encoded
                    
                except Exception as e:
                    logger.error(f"Failed to capture/encode frame: {e}")
                    try:
                        reg_image.unmap()
                    except:
                        pass
                    return None
    
    def queue_capture(self, texture_id, metadata=None):
        """
        Queue a texture for capture and encoding.
        
        Called from main thread.
        """
        request = {
            'texture_id': texture_id,
            'frame_num': metadata.get('frame_num', 0) if metadata else 0,
            'timestamp': metadata.get('timestamp') if metadata else None
        }
        
        try:
            self.capture_queue.put_nowait(request)
        except queue.Full:
            logger.warning("Capture queue full, dropping frame")
    
    def stop(self):
        """Stop the render thread."""
        logger.info("Stopping CUDARenderThread...")
        self.running = False
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up CUDARenderThread...")
        
        # Unregister all textures
        for tex_id, reg_image in self.registered_textures.items():
            try:
                reg_image.unregister()
            except:
                pass
        
        self.registered_textures.clear()
        
        # Clean up NVENC
        if self.nvenc_encoder:
            try:
                self.nvenc_encoder.close()
            except:
                pass
        
        # Clean up CUDA context
        if self.cuda_context:
            try:
                self.cuda_context.pop()
                self.cuda_context.detach()
            except:
                pass
        
        logger.info("CUDARenderThread cleaned up")