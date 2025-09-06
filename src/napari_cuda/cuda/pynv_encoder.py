"""
PyNvVideoCodec-based NVENC encoder for hardware-accelerated H.264 encoding.

This encoder uses NVIDIA's PyNvVideoCodec to encode frames directly on the GPU
using NVENC hardware acceleration available on the L4 and other NVIDIA GPUs.
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import PyNvVideoCodec
try:
    import PyNvVideoCodec as pnvc
    HAS_PYNVCODEC = True
except ImportError:
    HAS_PYNVCODEC = False
    logger.warning("PyNvVideoCodec not available - NVENC encoding disabled")

# Try to import CUDA libraries
try:
    import pycuda.driver as cuda
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    logger.warning("CUDA libraries not available")


class PyNvEncoder:
    """
    Hardware-accelerated H.264 encoder using PyNvVideoCodec and NVENC.
    
    This encoder accepts CUDA arrays directly from the OpenGL texture
    and encodes them to H.264 using NVENC hardware acceleration.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080, 
                 fps: int = 60, bitrate: int = 10_000_000):
        """
        Initialize the PyNvVideoCodec encoder.
        
        Parameters
        ----------
        width : int
            Frame width in pixels
        height : int
            Frame height in pixels
        fps : int
            Target frames per second
        bitrate : int
            Target bitrate in bits per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.encoder = None
        self.frame_count = 0
        
        if not HAS_PYNVCODEC:
            raise RuntimeError("PyNvVideoCodec not available")
        
        if not HAS_CUDA:
            raise RuntimeError("CUDA libraries not available")
        
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the PyNvVideoCodec encoder."""
        try:
            # Create encoder with ARGB format (matches OpenGL texture)
            # We'll convert from RGBA to ARGB if needed
            self.encoder = pnvc.CreateEncoder(
                width=self.width,
                height=self.height,
                fmt="ARGB",  # Use ARGB for direct OpenGL compatibility
                usecpuinputbuffer=False  # Use GPU buffer for zero-copy
            )
            
            logger.info(f"PyNvVideoCodec encoder initialized: {self.width}x{self.height}@{self.fps}fps")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyNvVideoCodec encoder: {e}")
            raise
    
    def encode_cuda_array(self, cuda_array) -> Optional[bytes]:
        """
        Encode a CUDA array to H.264.
        
        Parameters
        ----------
        cuda_array : pycuda.driver.Array
            CUDA array from OpenGL texture (RGBA format)
            
        Returns
        -------
        bytes or None
            Encoded H.264 frame data or None on failure
        """
        if not self.encoder:
            logger.error("Encoder not initialized")
            return None
        
        try:
            # The CUDA array from OpenGL is typically RGBA
            # We need to convert to format encoder expects
            
            # For now, we'll use a simplified approach
            # In production, you'd want proper color conversion
            
            # Allocate device memory for frame if using raw CUDA
            # For CuPy arrays, we can pass directly
            
            # Encode the frame
            packets = self.encoder.Encode(cuda_array)
            
            if packets and len(packets) > 0:
                # Get the first packet (there should typically be one per frame)
                packet_data = bytes(packets[0])
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    logger.debug(f"Encoded {self.frame_count} frames")
                
                return packet_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None
    
    def encode_texture(self, texture_data: np.ndarray) -> Optional[bytes]:
        """
        Encode an OpenGL texture (as numpy array) to H.264.
        
        This is a fallback method when direct CUDA array encoding isn't available.
        
        Parameters
        ----------
        texture_data : np.ndarray
            RGBA texture data from OpenGL
            
        Returns
        -------
        bytes or None
            Encoded H.264 frame data or None on failure
        """
        if not self.encoder:
            logger.error("Encoder not initialized")
            return None
        
        try:
            # Convert RGBA to ARGB
            if texture_data.shape[-1] == 4:  # RGBA
                # Swap R and A channels: RGBA -> ARGB
                argb_data = np.empty_like(texture_data)
                argb_data[..., 0] = texture_data[..., 3]  # A
                argb_data[..., 1] = texture_data[..., 0]  # R
                argb_data[..., 2] = texture_data[..., 1]  # G
                argb_data[..., 3] = texture_data[..., 2]  # B
                texture_data = argb_data
            
            # Encode the frame
            packets = self.encoder.Encode(texture_data)
            
            if packets and len(packets) > 0:
                packet_data = bytes(packets[0])
                self.frame_count += 1
                return packet_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to encode texture: {e}")
            return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush any remaining frames from the encoder.
        
        Returns
        -------
        bytes or None
            Remaining encoded data or None
        """
        if not self.encoder:
            return None
        
        try:
            # PyNvVideoCodec uses EndEncode to flush
            remaining = self.encoder.EndEncode()
            if remaining and len(remaining) > 0:
                # Concatenate all remaining packets
                data = b''.join(bytes(packet) for packet in remaining)
                return data
            return None
            
        except Exception as e:
            logger.error(f"Failed to flush encoder: {e}")
            return None
    
    def close(self):
        """Clean up encoder resources."""
        if self.encoder:
            try:
                self.encoder.EndEncode()
                logger.info(f"PyNvEncoder closed after encoding {self.frame_count} frames")
            except:
                pass
            self.encoder = None
    
    def __del__(self):
        """Ensure encoder is cleaned up."""
        self.close()
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if PyNvVideoCodec NVENC encoding is available.
        
        Returns
        -------
        bool
            True if NVENC encoding is available
        """
        if not HAS_PYNVCODEC or not HAS_CUDA:
            return False
        
        try:
            # Try to create a small test encoder
            test_enc = pnvc.CreateEncoder(
                width=640,
                height=480,
                fmt="NV12",
                usecpuinputbuffer=True
            )
            # If we get here, it worked
            del test_enc
            return True
        except Exception:
            return False