#pragma once

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vt_session vt_session_t;
typedef struct gl_cache gl_cache_t;

// Create a VT decoding session from avcC + dimensions.
// pixfmt: fourcc, e.g. 'NV12' or 'BGRA'. Returns NULL on failure.
vt_session_t* vt_create(const uint8_t* avcc, size_t avcc_len, int width, int height, uint32_t pixfmt);

// Destroy a session and free resources.
void vt_destroy(vt_session_t* s);

// Submit one AVCC access unit for decode. Returns 0 on success, non-zero on error.
int vt_decode(vt_session_t* s, const uint8_t* avcc_au, size_t len, double pts_seconds);

// Wait for asynchronous frames to be delivered (best-effort). Returns 0 on success.
int vt_flush(vt_session_t* s);

// Retrieve a decoded frame. On success returns 1 and stores a retained CVPixelBufferRef in out_buf
// and its pts in out_pts. Caller must eventually release via vt_release_frame().
int vt_get_frame(vt_session_t* s, double timeout_s, void** out_buf, double* out_pts);

// Release a CVPixelBufferRef obtained from vt_get_frame().
void vt_release_frame(void* buf);
void vt_retain_frame(void* buf);

// Diagnostics counters
void vt_counts(vt_session_t* s, uint32_t* submits, uint32_t* outputs, uint32_t* qlen);
// Extended stats: adds drops (queue overflow), retains and releases accounted per-session
void vt_stats(vt_session_t* s,
              uint32_t* submits,
              uint32_t* outputs,
              uint32_t* qlen,
              uint32_t* drops,
              uint32_t* retains,
              uint32_t* releases);

// --- OpenGL / zero-copy helpers (macOS only) ---
// Initialize a CVOpenGLTextureCache for the current OpenGL context. Returns NULL on failure.
gl_cache_t* gl_cache_init_for_current_context(void);
// Destroy a GL cache created above.
void gl_cache_destroy(gl_cache_t* cache);
// Flush the GL texture cache to release any internally cached textures.
void gl_cache_flush(gl_cache_t* cache);
// Retrieve GL cache counters: number of created and released CVOpenGLTextureRefs.
void gl_cache_counts(gl_cache_t* cache, uint32_t* creates, uint32_t* releases);
// Allocate a BGRA CVPixelBuffer compatible with OpenGL. If use_iosurface is nonzero,
// the buffer will be IOSurface-backed for optimal interop.
void* alloc_pixelbuffer_bgra(int width, int height, int use_iosurface);
// Return the CoreVideo pixel format type (OSType) for a CVPixelBufferRef capsule.
unsigned int pixel_format(void* cvpixelbuffer);
// Create a GL texture from a CVPixelBuffer using the cache. Returns a retained CVOpenGLTextureRef via out_tex,
// and fills out_name, out_target, out_w, out_h. Returns 0 on failure.
int gl_tex_from_cvpixelbuffer(gl_cache_t* cache, void* cvpixelbuffer, void** out_tex, unsigned int* out_name, unsigned int* out_target, int* out_w, int* out_h);
// Release a CVOpenGLTextureRef obtained from gl_tex_from_cvpixelbuffer.
void gl_release_tex(void* cvgltex);

#ifdef __cplusplus
}
#endif
