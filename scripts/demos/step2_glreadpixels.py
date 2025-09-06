#!/usr/bin/env python
"""
Step 2: Measure ACTUAL glReadPixels performance.
This is what napari does for screenshots.
"""
import os
import time
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from OpenGL import GL
import OpenGL.EGL as EGL

print("Step 2: glReadPixels Timing")
print("=" * 60)

# Setup OpenGL (from step 1, we know this works)
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

print(f"✓ OpenGL context ready: {width}x{height}")

# Create texture and FBO
texture_id = GL.glGenTextures(1)
GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
test_data = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, test_data)

fbo = GL.glGenFramebuffers(1)
GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                          GL.GL_TEXTURE_2D, texture_id, 0)

print("✓ Framebuffer ready with texture")

# MEASURE glReadPixels
print("\n📷 Measuring glReadPixels (CPU path):")
print("-" * 40)

buffer = np.empty((height, width, 4), dtype=np.uint8)
times = []

for i in range(30):
    # Render something new each frame
    GL.glClearColor(np.random.rand(), np.random.rand(), np.random.rand(), 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glFinish()  # Ensure render is complete
    
    t0 = time.perf_counter()
    
    # THIS IS WHAT WE'RE MEASURING
    GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buffer)
    GL.glFinish()  # Ensure transfer is complete
    
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    times.append(elapsed_ms)
    
    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1}: {elapsed_ms:.2f}ms")

# Results
mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)

print("\n" + "=" * 60)
print("glReadPixels Performance:")
print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
print(f"  Min:  {min_time:.2f} ms")
print(f"  Max:  {max_time:.2f} ms")
print(f"  FPS:  {1000/mean_time:.1f}")
print(f"  Data: {buffer.nbytes/1024/1024:.1f} MB per frame")

# Cleanup (fixed!)
GL.glDeleteFramebuffers(1, [fbo])
GL.glDeleteTextures(1, [texture_id])
EGL.eglMakeCurrent(egl_display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
EGL.eglDestroyContext(egl_display, egl_context)
EGL.eglDestroySurface(egl_display, egl_surface)
EGL.eglTerminate(egl_display)

print("\n✅ Step 2 complete - we have CPU baseline")