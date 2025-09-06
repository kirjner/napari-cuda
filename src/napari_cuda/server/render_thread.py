"""
CUDARenderThread - Manages CUDA/OpenGL contexts and performs GPU operations.

This thread owns the OpenGL and CUDA contexts and ensures all GPU operations
happen on the correct thread.
"""

import logging
import queue
import threading
import contextlib
from qtpy.QtCore import QThread, Signal as pyqtSignal

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
        
        # EGL objects (if created)
        self.egl_display = None
        self.egl_context = None
        self.egl_surface = None
        
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
                    try:
                        if hasattr(self.pixel_stream, 'metrics'):
                            self.pixel_stream.metrics.set('napari_cuda_capture_queue_depth', float(self.capture_queue.qsize()))
                    except Exception:
                        pass
                    
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
        
        # Initialize CUDA first
        cuda.init()
        device_count = cuda.Device.count()
        logger.info(f"Found {device_count} CUDA devices")
        
        if device_count == 0:
            raise RuntimeError("No CUDA devices found")
        
        # Create CUDA context on first device
        device = cuda.Device(0)
        self.cuda_context = device.make_context()
        
        logger.info(f"CUDA context created on device: {device.name()}")
        
        # Try to use provided GL context, or create EGL context for headless
        if self.gl_context:
            try:
                self.gl_context.makeCurrent()
                logger.info("Qt OpenGL context made current")
            except Exception as e:
                logger.warning(f"Could not make Qt context current: {e}")
                self.gl_context = None
        
        if not self.gl_context:
            # Create headless EGL context
            logger.info("Creating headless EGL context...")
            try:
                import os
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                
                from OpenGL import EGL
                from OpenGL import GL
                
                # Get EGL display
                egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
                if egl_display == EGL.EGL_NO_DISPLAY:
                    raise RuntimeError("Failed to get EGL display")
                
                # Initialize EGL
                major = EGL.EGLint()
                minor = EGL.EGLint()
                if not EGL.eglInitialize(egl_display, major, minor):
                    raise RuntimeError("Failed to initialize EGL")
                
                logger.info(f"EGL initialized: {major.value}.{minor.value}")
                
                # Choose config
                config_attribs = [
                    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                    EGL.EGL_RED_SIZE, 8,
                    EGL.EGL_GREEN_SIZE, 8,
                    EGL.EGL_BLUE_SIZE, 8,
                    EGL.EGL_ALPHA_SIZE, 8,
                    EGL.EGL_NONE
                ]
                
                num_configs = EGL.EGLint()
                configs = (EGL.EGLConfig * 1)()
                if not EGL.eglChooseConfig(egl_display, config_attribs, configs, 1, num_configs):
                    raise RuntimeError("Failed to choose EGL config")
                
                # Create context
                EGL.eglBindAPI(EGL.EGL_OPENGL_API)
                context_attribs = [EGL.EGL_NONE]
                egl_context = EGL.eglCreateContext(egl_display, configs[0], EGL.EGL_NO_CONTEXT, context_attribs)
                if egl_context == EGL.EGL_NO_CONTEXT:
                    raise RuntimeError("Failed to create EGL context")
                
                # Create pbuffer surface
                pbuffer_attribs = [
                    EGL.EGL_WIDTH, 1920,
                    EGL.EGL_HEIGHT, 1080,
                    EGL.EGL_NONE
                ]
                egl_surface = EGL.eglCreatePbufferSurface(egl_display, configs[0], pbuffer_attribs)
                if egl_surface == EGL.EGL_NO_SURFACE:
                    raise RuntimeError("Failed to create EGL surface")
                
                # Make current
                if not EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
                    raise RuntimeError("Failed to make EGL context current")
                
                # Store EGL objects for cleanup
                self.egl_display = egl_display
                self.egl_context = egl_context
                self.egl_surface = egl_surface
                
                logger.info("EGL context created and made current")
                
            except Exception as e:
                logger.error(f"Failed to create EGL context: {e}")
                raise
        
        # Enable OpenGL interop - should work now with valid GL context
        try:
            import pycuda.gl
            pycuda.gl.init()
            logger.info("CUDA-OpenGL interop initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA-GL interop: {e}")
            raise
    
    def _setup_nvenc(self):
        """Initialize NVENC encoder."""
        # Try PyNvVideoCodec first (NVIDIA's official Python bindings)
        try:
            from ..cuda.pynv_encoder import PyNvEncoder
            
            if PyNvEncoder.is_available():
                self.nvenc_encoder = PyNvEncoder(
                    width=1920,
                    height=1080,
                    fps=60,
                    bitrate=10000000  # 10 Mbps
                )
                logger.info("PyNvVideoCodec NVENC encoder initialized")
                return
        except ImportError:
            logger.debug("PyNvVideoCodec not available")
        except Exception as e:
            logger.warning(f"Failed to initialize PyNvVideoCodec: {e}")
        
        # Fallback: Try legacy NVENC wrapper if it exists
        try:
            from ..cuda.nvenc_wrapper import NVENCEncoder
            
            self.nvenc_encoder = NVENCEncoder(
                width=1920,
                height=1080,
                fps=60,
                bitrate=10000000  # 10 Mbps
            )
            
            logger.info("Legacy NVENC encoder initialized")
            
        except ImportError:
            logger.warning("No NVENC encoder available, using fallback encoding")
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
        import time
        
        texture_id = request['texture_id']
        frame_num = request.get('frame_num', 0)
        ts_request = request.get('ts_ns')
        
        # Track when request was dequeued
        t_dequeue = time.perf_counter_ns()
        
        logger.debug(f"Capturing texture {texture_id} (frame {frame_num})")
        
        # Use thread safety for all CUDA operations
        with self._cuda_lock:
            t_cuda_lock = time.perf_counter_ns()
            
            with cuda_context_guard(self.cuda_context):
                t_context_current = time.perf_counter_ns()
                
                # Register texture with CUDA if not already done
                with self._texture_lock:
                    t_texture_lock = time.perf_counter_ns()
                    
                    if texture_id not in self.registered_textures:
                        try:
                            t_register_start = time.perf_counter_ns()
                            self.registered_textures[texture_id] = RegisteredImage(
                                int(texture_id),
                                GL.GL_TEXTURE_2D,
                                graphics_map_flags.READ_ONLY
                            )
                            t_register_end = time.perf_counter_ns()
                            logger.debug(f"Registered texture {texture_id} with CUDA")
                            
                            # Track registration time
                            if hasattr(self.pixel_stream, 'metrics'):
                                ms = 1e-6
                                self.pixel_stream.metrics.observe_ms(
                                    'napari_cuda_register_texture_ms', 
                                    (t_register_end - t_register_start) * ms
                                )
                        except Exception as e:
                            logger.error(f"Failed to register texture: {e}")
                            return None
                    
                    # Get registered image
                    reg_image = self.registered_textures[texture_id]
                    t_texture_unlock = time.perf_counter_ns()
                
                # Map texture to CUDA (outside texture lock but inside CUDA lock)
                try:
                    t_map_start = time.perf_counter_ns()
                    reg_image.map()
                    t_map_done = time.perf_counter_ns()
                    
                    cuda_array = reg_image.get_mapped_array(0, 0)
                    t_array_get = time.perf_counter_ns()
                    
                    # Encode with NVENC
                    if self.nvenc_encoder:
                        t_encode_start = time.perf_counter_ns()
                        # Check if it's our PyNvEncoder
                        if hasattr(self.nvenc_encoder, 'encode_cuda_array'):
                            encoded = self.nvenc_encoder.encode_cuda_array(cuda_array)
                        else:
                            # Legacy encoder
                            encoded = self.nvenc_encoder.encode_surface(cuda_array)
                        t_encode_end = time.perf_counter_ns()
                    else:
                        # MVP fallback: copy to host and JPEG-encode
                        try:
                            import cupy as cp  # requires CUDA-capable environment
                            import cv2

                            # Query texture size from OpenGL
                            t_gl_query_start = time.perf_counter_ns()
                            GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture_id))
                            width = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_WIDTH))
                            height = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_HEIGHT))
                            t_gl_query_end = time.perf_counter_ns()

                            # Allocate device buffer (assume GL_RGBA8 -> 4 channels)
                            t_alloc_start = time.perf_counter_ns()
                            dev_frame = cp.empty((height, width, 4), dtype=cp.uint8)
                            t_alloc_end = time.perf_counter_ns()

                            # Copy CUDA array -> device buffer
                            t_memcpy_start = time.perf_counter_ns()
                            m = cuda.Memcpy2D()
                            m.set_src_array(cuda_array)
                            m.set_dst_device(int(dev_frame.data.ptr))
                            m.width_in_bytes = width * 4
                            m.height = height
                            m()
                            t_memcpy_end = time.perf_counter_ns()

                            # Move to host and encode JPEG (best-effort color conversion)
                            t_encode_start = time.perf_counter_ns()
                            host_frame = cp.asnumpy(dev_frame)
                            t_host_copy = time.perf_counter_ns()
                            
                            try:
                                bgr = cv2.cvtColor(host_frame, cv2.COLOR_RGBA2BGR)
                            except Exception:
                                bgr = host_frame[..., :3]
                            ok, jpeg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            encoded = jpeg.tobytes() if ok else None
                            t_encode_end = time.perf_counter_ns()
                            
                            # Track fallback metrics
                            if hasattr(self.pixel_stream, 'metrics'):
                                ms = 1e-6
                                self.pixel_stream.metrics.observe_ms('napari_cuda_gl_query_ms', (t_gl_query_end - t_gl_query_start) * ms)
                                self.pixel_stream.metrics.observe_ms('napari_cuda_alloc_ms', (t_alloc_end - t_alloc_start) * ms)
                                self.pixel_stream.metrics.observe_ms('napari_cuda_memcpy_ms', (t_memcpy_end - t_memcpy_start) * ms)
                                self.pixel_stream.metrics.observe_ms('napari_cuda_host_copy_ms', (t_host_copy - t_encode_start) * ms)
                        except Exception as e:
                            logger.error(f"JPEG fallback failed: {e}")
                            # Last-resort placeholder to keep pipeline alive
                            encoded = b'FRAME_%d' % frame_num
                            t_encode_end = time.perf_counter_ns()
                    
                    t_unmap_start = time.perf_counter_ns()
                    reg_image.unmap()
                    t_unmap_end = time.perf_counter_ns()

                    # Comprehensive metrics
                    try:
                        if hasattr(self.pixel_stream, 'metrics'):
                            ms = 1e-6
                            
                            # Queue wait time (if request timestamp available)
                            if ts_request is not None:
                                self.pixel_stream.metrics.observe_ms('napari_cuda_queue_wait_ms', (t_dequeue - ts_request) * ms)
                            
                            # Lock acquisition times
                            self.pixel_stream.metrics.observe_ms('napari_cuda_cuda_lock_ms', (t_cuda_lock - t_dequeue) * ms)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_context_ms', (t_context_current - t_cuda_lock) * ms)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_texture_lock_ms', (t_texture_lock - t_context_current) * ms)
                            
                            # Core GPU operations
                            self.pixel_stream.metrics.observe_ms('napari_cuda_map_ms', (t_map_done - t_map_start) * ms)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_array_get_ms', (t_array_get - t_map_done) * ms)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_encode_ms', (t_encode_end - t_encode_start) * ms)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_unmap_ms', (t_unmap_end - t_unmap_start) * ms)
                            
                            # Total GPU time (map to unmap)
                            self.pixel_stream.metrics.observe_ms('napari_cuda_total_gpu_ms', (t_unmap_end - t_map_start) * ms)
                            
                            # End-to-end time
                            if ts_request is not None:
                                self.pixel_stream.metrics.observe_ms('napari_cuda_end_to_end_ms', (t_unmap_end - ts_request) * ms)
                    except Exception:
                        pass
                    
                    return encoded
                    
                except Exception as e:
                    logger.error(f"Failed to capture/encode frame: {e}")
                    try:
                        if hasattr(self.pixel_stream, 'metrics'):
                            self.pixel_stream.metrics.inc('napari_cuda_encode_errors')
                    except Exception:
                        pass
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
            'ts_ns': metadata.get('ts_ns') if metadata else None
        }
        
        try:
            self.capture_queue.put_nowait(request)
            if hasattr(self.pixel_stream, 'metrics'):
                self.pixel_stream.metrics.set('napari_cuda_capture_queue_depth', float(self.capture_queue.qsize()))
        except queue.Full:
            logger.warning("Capture queue full, dropping frame")
            if hasattr(self.pixel_stream, 'metrics'):
                self.pixel_stream.metrics.inc('napari_cuda_frames_dropped')
    
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
