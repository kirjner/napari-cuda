#!/usr/bin/env python
"""
Step 3: CUDA texture mapping - the zero-copy path.
"""
import os
import time

import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import OpenGL.EGL as EGL
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gl
from OpenGL import GL
from pycuda.gl import RegisteredImage, graphics_map_flags

print("Step 3: CUDA Texture Mapping")
print("=" * 60)

# Setup OpenGL
egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
EGL.eglInitialize(egl_display, None, None)

config_attribs = np.array([
    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
    EGL.EGL_RED_SIZE, 8,
    EGL.EGL_GREEN_SIZE, 8,
    EGL.EGL_BLUE_SIZE, 8,
    EGL.EGL_ALPHA_SIZE, 8,
    EGL.EGL_NONE
], dtype=np.int32)

configs = (EGL.EGLConfig * 1)()
num_configs = EGL.EGLint()
EGL.eglChooseConfig(egl_display, config_attribs, configs, 1, num_configs)

EGL.eglBindAPI(EGL.EGL_OPENGL_API)
egl_context = EGL.eglCreateContext(egl_display, configs[0], EGL.EGL_NO_CONTEXT, None)

width, height = 1920, 1080
pbuffer_attribs = np.array([
    EGL.EGL_WIDTH, width,
    EGL.EGL_HEIGHT, height,
    EGL.EGL_NONE
], dtype=np.int32)
egl_surface = EGL.eglCreatePbufferSurface(egl_display, configs[0], pbuffer_attribs)
EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)

print(f"✓ OpenGL ready: {width}x{height}")

# Create texture
texture_id = GL.glGenTextures(1)
GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
test_data = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, test_data)
GL.glFinish()

print(f"✓ Texture created: ID {texture_id}")

# Initialize CUDA-GL interop
pycuda.gl.init()

# Register texture with CUDA
registered_texture = RegisteredImage(
    int(texture_id),
    GL.GL_TEXTURE_2D,
    graphics_map_flags.READ_ONLY
)

print("✓ Texture registered with CUDA")

# MEASURE CUDA mapping
print("\n🚀 Measuring CUDA texture mapping (GPU path):")
print("-" * 40)

times = []

for i in range(30):
    # Update texture (simulating render)
    if i % 5 == 0:
        new_data = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height,
                          GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, new_data)
        GL.glFinish()

    cuda.Context.synchronize()
    t0 = time.perf_counter()

    # THIS IS WHAT WE'RE MEASURING
    mapping = registered_texture.map()

    # Get the CUDA array
    cuda_array = mapping.array(0, 0)

    # This is where NVENC would read from cuda_array
    # The data is already on GPU - zero copy!

    mapping.unmap()
    cuda.Context.synchronize()

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    times.append(elapsed_ms)

    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1}: {elapsed_ms:.3f}ms")

# Results
mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)

print("\n" + "=" * 60)
print("CUDA Texture Mapping Performance:")
print(f"  Mean: {mean_time:.3f} ± {std_time:.3f} ms")
print(f"  Min:  {min_time:.3f} ms")
print(f"  Max:  {max_time:.3f} ms")
print(f"  FPS:  {1000/mean_time:.0f}")
print("  Data: Zero-copy (texture already on GPU)")

# Cleanup
registered_texture.unregister()
GL.glDeleteTextures(1, [texture_id])
EGL.eglMakeCurrent(egl_display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
EGL.eglDestroyContext(egl_display, egl_context)
EGL.eglDestroySurface(egl_display, egl_surface)
EGL.eglTerminate(egl_display)

print("\n✅ Step 3 complete - we have GPU baseline")
