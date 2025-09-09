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

// Diagnostics counters
void vt_counts(vt_session_t* s, uint32_t* submits, uint32_t* outputs, uint32_t* qlen);

#ifdef __cplusplus
}
#endif

