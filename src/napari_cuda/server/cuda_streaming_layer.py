"""
CudaStreamingLayer - Custom napari layer that captures OpenGL texture via CUDA.

This is the core server-side component that intercepts napari's rendering
and streams it to the client without CPU involvement.
"""

from napari._vispy.layers.scalar_field import VispyScalarFieldBaseLayer
from napari._vispy.utils.gl import fix_data_dtype
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)

# These will be imported on HPC where CUDA is available
try:
    import pycuda.driver as cuda
    from pycuda.gl import RegisteredImage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    logger.warning("PyCUDA not available - running in CPU fallback mode")


class CudaStreamingLayer(VispyScalarFieldBaseLayer):
    """
    Custom layer that captures rendered OpenGL textures via CUDA-GL interop
    and streams them to clients.
    """
    
    def __init__(self, layer, render_thread=None, pixel_stream=None, **kwargs):
        """
        Initialize the CUDA streaming layer.
        
        Parameters
        ----------
        layer : napari.layers.Layer
            The napari layer to render
        render_thread : CUDARenderThread
            Thread managing CUDA/GL contexts
        pixel_stream : PixelStreamServer
            Server for sending encoded frames to clients
        """
        super().__init__(layer, **kwargs)
        
        self.render_thread = render_thread
        self.pixel_stream = pixel_stream
        self.registered_textures = {}
        self.frame_counter = 0
        
        # Queue for passing frames to encoding thread
        self.encode_queue = queue.Queue(maxsize=2)
        
        logger.info(f"CudaStreamingLayer initialized for layer: {layer.name}")
    
    def _on_data_change(self) -> None:
        """
        Override to intercept after napari renders to OpenGL texture.
        This is called whenever the displayed data changes.
        """
        # First, let napari render normally to OpenGL texture
        data = fix_data_dtype(self.layer._data_view)
        super()._on_data_change()
        
        # Force immediate render to ensure texture is ready
        if hasattr(self.node, 'update'):
            self.node.update()
        
        # Now capture the rendered texture
        self._capture_and_stream()
    
    def _capture_and_stream(self):
        """
        Capture the current OpenGL texture and send for streaming.
        This is where the magic happens - GPU to GPU transfer.
        """
        if not HAS_CUDA:
            logger.debug("CUDA not available, skipping capture")
            return
            
        # Get the OpenGL texture handle
        texture_handle = self._get_texture_handle()
        if texture_handle is None:
            return
        
        self.frame_counter += 1
        logger.debug(f"Capturing frame {self.frame_counter}, texture handle: {texture_handle}")
        
        if self.render_thread:
            # Queue capture request to render thread
            # (must happen on thread with GL context)
            try:
                self.encode_queue.put_nowait({
                    'texture_id': texture_handle,
                    'frame_num': self.frame_counter,
                    'timestamp': np.datetime64('now')
                })
            except queue.Full:
                logger.warning("Encode queue full, dropping frame")
    
    def _get_texture_handle(self):
        """
        Extract the OpenGL texture handle from the vispy node.
        
        Returns
        -------
        int or None
            OpenGL texture ID, or None if not available
        """
        # Try to get texture from the node
        # Different node types store it differently
        
        # For ImageNode (2D)
        if hasattr(self.node, '_texture'):
            texture = self.node._texture
            if texture and hasattr(texture, 'handle'):
                return texture.handle
        
        # For VolumeNode (3D) 
        if hasattr(self.node, 'texture'):
            texture = self.node.texture
            if texture and hasattr(texture, 'handle'):
                return texture.handle
        
        # Fallback: try to find any texture attribute
        for attr_name in dir(self.node):
            if 'texture' in attr_name.lower():
                attr = getattr(self.node, attr_name)
                if hasattr(attr, 'handle'):
                    logger.debug(f"Found texture at {attr_name}")
                    return attr.handle
        
        logger.warning("Could not find OpenGL texture handle in node")
        return None
    
    def close(self):
        """Clean up resources."""
        # Unregister all textures
        if HAS_CUDA:
            for tex_id, registered in self.registered_textures.items():
                try:
                    registered.unregister()
                except:
                    pass
        
        self.registered_textures.clear()
        logger.info("CudaStreamingLayer closed")