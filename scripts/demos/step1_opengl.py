#!/usr/bin/env python
"""
Step 1: Verify OpenGL context and texture creation works.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from OpenGL import GL
import OpenGL.EGL as EGL
import numpy as np

print("Step 1: OpenGL Setup")
print("=" * 60)

# 1. Initialize EGL
egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
major, minor = EGL.EGLint(), EGL.EGLint()
result = EGL.eglInitialize(egl_display, major, minor)
print(f"✓ EGL initialized: {major.value}.{minor.value}, result: {result}")

# 2. Choose config
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
result = EGL.eglChooseConfig(egl_display, config_attribs, configs, 1, num_configs)
print(f"✓ Config chosen: {result}, num_configs: {num_configs.value}")

# 3. Create context
EGL.eglBindAPI(EGL.EGL_OPENGL_API)
egl_context = EGL.eglCreateContext(egl_display, configs[0], EGL.EGL_NO_CONTEXT, None)
print(f"✓ Context created: {egl_context}")

# 4. Create surface
pbuffer_attribs = np.array([
    EGL.EGL_WIDTH, 1920,
    EGL.EGL_HEIGHT, 1080,
    EGL.EGL_NONE
], dtype=np.int32)
egl_surface = EGL.eglCreatePbufferSurface(egl_display, configs[0], pbuffer_attribs)
print(f"✓ Surface created: {egl_surface}")

# 5. Make current
result = EGL.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)
print(f"✓ Made current: {result}")

# 6. Create texture
texture_id = GL.glGenTextures(1)
GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
print(f"✓ Texture created: ID {texture_id}")

# 7. Upload data
width, height = 1920, 1080
test_data = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, test_data)
GL.glFinish()
print(f"✓ Data uploaded: {width}x{height} = {test_data.nbytes/1024/1024:.1f}MB")

# 8. Create FBO
fbo = GL.glGenFramebuffers(1)
GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                          GL.GL_TEXTURE_2D, texture_id, 0)
status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
print(f"✓ FBO created: ID {fbo}, status: {status == GL.GL_FRAMEBUFFER_COMPLETE}")

print("\n✅ OpenGL setup complete!")

# Cleanup
GL.glDeleteFramebuffers([fbo])
GL.glDeleteTextures([texture_id])
EGL.eglMakeCurrent(egl_display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
EGL.eglDestroyContext(egl_display, egl_context)
EGL.eglDestroySurface(egl_display, egl_surface)
EGL.eglTerminate(egl_display)