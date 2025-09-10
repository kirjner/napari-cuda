#ifdef __APPLE__

#import <CoreFoundation/CoreFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#import <CoreVideo/CVOpenGLTexture.h>
#import <CoreVideo/CVOpenGLTextureCache.h>
#import <pthread.h>

#include "vt_shim.h"

typedef struct {
    CVPixelBufferRef buf; // retained
    double pts;
} vt_item_t;

struct vt_session {
    VTDecompressionSessionRef session;
    CMVideoFormatDescriptionRef fmt;
    uint32_t requested_pixfmt;
    // ring queue
    vt_item_t *q;
    int qcap;
    int qhead;
    int qtail;
    int qcount;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    // counters
    uint32_t submits;
    uint32_t outputs;
};

// GL cache wrapper
struct gl_cache {
    CVOpenGLTextureCacheRef cache;
};

static void vt_output_cb(void *refcon,
                         void *sourceFrameRefCon,
                         OSStatus status,
                         VTDecodeInfoFlags infoFlags,
                         CVImageBufferRef imageBuffer,
                         CMTime pts,
                         CMTime duration) {
    vt_session_t *s = (vt_session_t*)refcon;
    if (!s) return;
    if (status != noErr || imageBuffer == NULL) {
        return;
    }
    // Retain for cross-thread handoff
    CFRetain(imageBuffer);
    double pts_s = NAN;
    if (CMTIME_IS_NUMERIC(pts)) {
        pts_s = (double)pts.value / (double)pts.timescale;
    }
    pthread_mutex_lock(&s->mu);
    // drop oldest if full
    if (s->qcount == s->qcap) {
        vt_item_t *old = &s->q[s->qhead];
        if (old->buf) CFRelease(old->buf);
        s->qhead = (s->qhead + 1) % s->qcap;
        s->qcount--;
    }
    vt_item_t *slot = &s->q[s->qtail];
    slot->buf = (CVPixelBufferRef)imageBuffer;
    slot->pts = pts_s;
    s->qtail = (s->qtail + 1) % s->qcap;
    s->qcount++;
    s->outputs++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static int parse_avcc(const uint8_t* avcc, size_t n, const uint8_t*** sets, size_t** sizes, int* num_sets, int* nal_len_size) {
    if (!avcc || n < 7) return -1;
    size_t i = 0;
    i += 4; // version+profile+compat+level
    uint8_t lsm1 = avcc[i++] & 0x3;
    *nal_len_size = (int)lsm1 + 1;
    int num_sps = avcc[i++] & 0x1F;
    int total = num_sps;
    size_t idx = i;
    for (int k=0;k<num_sps;k++){
        if (idx + 2 > n) return -1;
        uint16_t ln = (avcc[idx]<<8) | avcc[idx+1]; idx+=2;
        if (idx + ln > n) return -1;
        idx += ln;
    }
    if (idx >= n) return -1;
    int num_pps = avcc[idx++];
    total += num_pps;
    const uint8_t** ps = (const uint8_t**)calloc(total, sizeof(void*));
    size_t* sz = (size_t*)calloc(total, sizeof(size_t));
    int pos = 0;
    i = (size_t)( (avcc + 5) - avcc ); // recompute? safer to re-walk
    i = 5;
    int spsn = avcc[5] & 0x1F;
    i++;
    for (int k=0;k<spsn;k++){
        uint16_t ln = (avcc[i]<<8) | avcc[i+1]; i+=2;
        ps[pos] = avcc + i; sz[pos] = ln; pos++; i+=ln;
    }
    int ppsn = avcc[i++];
    for (int k=0;k<ppsn;k++){
        uint16_t ln = (avcc[i]<<8) | avcc[i+1]; i+=2;
        ps[pos] = avcc + i; sz[pos] = ln; pos++; i+=ln;
    }
    *sets = ps; *sizes = sz; *num_sets = total; return 0;
}

vt_session_t* vt_create(const uint8_t* avcc, size_t avcc_len, int width, int height, uint32_t pixfmt) {
    vt_session_t* s = (vt_session_t*)calloc(1, sizeof(vt_session_t));
    if (!s) return NULL;
    s->qcap = 16; s->q = (vt_item_t*)calloc(s->qcap, sizeof(vt_item_t));
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    s->requested_pixfmt = pixfmt;

    // Declare resources up-front for safe cleanup
    CMFormatDescriptionRef fmt = NULL;
    CFMutableDictionaryRef attrs = NULL;
    CFNumberRef pfnum = NULL;
    VTDecompressionSessionRef sess = NULL;
    CFDictionaryRef ioSurf = NULL;

    // Create format description from avcC SPS/PPS
    OSStatus st = noErr;
    int nal_len = 4, num_sets = 0; const uint8_t** sets = NULL; size_t* sizes = NULL;
    if (parse_avcc(avcc, avcc_len, &sets, &sizes, &num_sets, &nal_len) != 0) {
        goto fail;
    }
    st = CMVideoFormatDescriptionCreateFromH264ParameterSets(kCFAllocatorDefault,
                                                                      num_sets,
                                                                      sets,
                                                                      sizes,
                                                                      nal_len,
                                                                      &fmt);
    free((void*)sets); free((void*)sizes);
    if (st != noErr) goto fail;
    s->fmt = fmt;

    // Pixel buffer attrs
    attrs = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (!attrs) goto fail;
    pfnum = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixfmt);
    if (!pfnum) goto fail;
    CFDictionarySetValue(attrs, kCVPixelBufferPixelFormatTypeKey, pfnum);
    CFRelease(pfnum); pfnum = NULL;
    ioSurf = CFDictionaryCreate(kCFAllocatorDefault, NULL, NULL, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (ioSurf) { CFDictionarySetValue(attrs, kCVPixelBufferIOSurfacePropertiesKey, ioSurf); CFRelease(ioSurf); ioSurf = NULL; }

    // Callback record
    VTDecompressionOutputCallbackRecord rec;
    rec.decompressionOutputCallback = vt_output_cb;
    rec.decompressionOutputRefCon = s;

    st = VTDecompressionSessionCreate(kCFAllocatorDefault, fmt, NULL, attrs, &rec, &sess);
    if (attrs) { CFRelease(attrs); attrs = NULL; }
    if (st != noErr) goto fail;
    s->session = sess;

    // Real-time hint
    VTSessionSetProperty(sess, kVTDecompressionPropertyKey_RealTime, kCFBooleanTrue);
    return s;
fail:
    if (pfnum) CFRelease(pfnum);
    if (attrs) CFRelease(attrs);
    if (sess) { VTDecompressionSessionInvalidate(sess); CFRelease(sess); }
    if (fmt) CFRelease(fmt);
    if (s) {
        if (s->q) free(s->q);
        pthread_mutex_destroy(&s->mu);
        pthread_cond_destroy(&s->cv);
        free(s);
    }
    return NULL;
}

void vt_destroy(vt_session_t* s) {
    if (!s) return;
    if (s->session) {
        VTDecompressionSessionInvalidate(s->session);
        CFRelease(s->session);
    }
    if (s->fmt) CFRelease(s->fmt);
    pthread_mutex_lock(&s->mu);
    for (int i=0;i<s->qcap;i++) {
        if (s->q[i].buf) CFRelease(s->q[i].buf);
    }
    pthread_mutex_unlock(&s->mu);
    free(s->q);
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    free(s);
}

int vt_decode(vt_session_t* s, const uint8_t* avcc_au, size_t len, double pts_seconds) {
    if (!s || !s->session || !avcc_au || len == 0) return -1;
    OSStatus st;
    CMBlockBufferRef bb = NULL;
    st = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, NULL, len, kCFAllocatorDefault, NULL, 0, len, 0, &bb);
    if (st != noErr) return -2;
    st = CMBlockBufferReplaceDataBytes(avcc_au, bb, 0, len);
    if (st != noErr) { CFRelease(bb); return -3; }

    CMSampleBufferRef sb = NULL;
    CMSampleTimingInfo ti;
    if (pts_seconds > 0) {
        CMTime pts = CMTimeMakeWithSeconds(pts_seconds, 1000000);
        ti.presentationTimeStamp = pts;
    } else {
        ti.presentationTimeStamp = kCMTimeInvalid;
    }
    ti.duration = kCMTimeInvalid;
    ti.decodeTimeStamp = kCMTimeInvalid;
    size_t sz = len;
    st = CMSampleBufferCreateReady(kCFAllocatorDefault, bb, s->fmt, 1, 1, &ti, 1, &sz, &sb);
    CFRelease(bb);
    if (st != noErr) return -4;

    // Attachments
    CFArrayRef atts = CMSampleBufferGetSampleAttachmentsArray(sb, true);
    if (atts && CFArrayGetCount(atts) > 0) {
        CFMutableDictionaryRef d = (CFMutableDictionaryRef)CFArrayGetValueAtIndex(atts, 0);
        CFDictionarySetValue(d, kCMSampleAttachmentKey_DisplayImmediately, kCFBooleanTrue);
    }
    VTDecodeInfoFlags info = 0;
    st = VTDecompressionSessionDecodeFrame(s->session, sb, kVTDecodeFrame_EnableAsynchronousDecompression | kVTDecodeFrame_1xRealTimePlayback, NULL, &info);
    CFRelease(sb);
    if (st != noErr) return -5;
    __sync_fetch_and_add(&s->submits, 1);
    return 0;
}

int vt_flush(vt_session_t* s) {
    if (!s || !s->session) return -1;
    VTDecompressionSessionWaitForAsynchronousFrames(s->session);
    return 0;
}

int vt_get_frame(vt_session_t* s, double timeout_s, void** out_buf, double* out_pts) {
    if (!s || !out_buf || !out_pts) return -1;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long ns = ts.tv_nsec + (long)(timeout_s * 1e9);
    ts.tv_sec += ns / 1000000000L;
    ts.tv_nsec = ns % 1000000000L;

    pthread_mutex_lock(&s->mu);
    int rc = 0;
    while (s->qcount == 0) {
        if (timeout_s <= 0) { rc = 0; break; }
        if (pthread_cond_timedwait(&s->cv, &s->mu, &ts) == ETIMEDOUT) { rc = 0; break; }
    }
    if (s->qcount > 0) {
        vt_item_t* it = &s->q[s->qhead];
        *out_buf = it->buf; *out_pts = it->pts;
        // transfer ownership: clear slot without releasing
        it->buf = NULL;
        s->qhead = (s->qhead + 1) % s->qcap;
        s->qcount--;
        rc = 1;
    }
    pthread_mutex_unlock(&s->mu);
    return rc;
}

void vt_release_frame(void* buf) {
    if (buf) CFRelease((CFTypeRef)buf);
}

void vt_counts(vt_session_t* s, uint32_t* submits, uint32_t* outputs, uint32_t* qlen) {
    if (!s) return;
    if (submits) *submits = s->submits;
    if (outputs) *outputs = s->outputs;
    if (qlen) {
        pthread_mutex_lock(&s->mu);
        *qlen = (uint32_t)s->qcount;
        pthread_mutex_unlock(&s->mu);
    }
}

// ---- OpenGL helpers ----
gl_cache_t* gl_cache_init_for_current_context(void) {
    CGLContextObj ctx = CGLGetCurrentContext();
    if (!ctx) return NULL;
    CGLPixelFormatObj pf = CGLGetPixelFormat(ctx);
    if (!pf) return NULL;
    CVOpenGLTextureCacheRef cache = NULL;
    CVReturn rc = CVOpenGLTextureCacheCreate(kCFAllocatorDefault, NULL, ctx, pf, NULL, &cache);
    if (rc != kCVReturnSuccess || !cache) return NULL;
    gl_cache_t* g = (gl_cache_t*)calloc(1, sizeof(gl_cache_t));
    if (!g) { if (cache) CFRelease(cache); return NULL; }
    g->cache = cache;
    return g;
}

void gl_cache_destroy(gl_cache_t* cache) {
    if (!cache) return;
    if (cache->cache) CFRelease(cache->cache);
    free(cache);
}

void* alloc_pixelbuffer_bgra(int width, int height, int use_iosurface) {
    CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                            &kCFTypeDictionaryKeyCallBacks,
                                                            &kCFTypeDictionaryValueCallBacks);
    if (!attrs) return NULL;
    uint32_t pf = kCVPixelFormatType_32BGRA;
    CFNumberRef pfnum = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pf);
    CFDictionarySetValue(attrs, kCVPixelBufferPixelFormatTypeKey, pfnum);
    CFRelease(pfnum);
    CFNumberRef w = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &width);
    CFNumberRef h = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &height);
    CFDictionarySetValue(attrs, kCVPixelBufferWidthKey, w);
    CFDictionarySetValue(attrs, kCVPixelBufferHeightKey, h);
    CFRelease(w); CFRelease(h);
    CFDictionarySetValue(attrs, kCVPixelBufferOpenGLCompatibilityKey, kCFBooleanTrue);
    if (use_iosurface) {
        CFMutableDictionaryRef empty = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                                 &kCFTypeDictionaryKeyCallBacks,
                                                                 &kCFTypeDictionaryValueCallBacks);
        CFDictionarySetValue(attrs, kCVPixelBufferIOSurfacePropertiesKey, empty);
        CFRelease(empty);
    }
    CVPixelBufferRef pb = NULL;
    CVReturn rc = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pb);
    CFRelease(attrs);
    if (rc != kCVReturnSuccess) return NULL;
    return pb; // retained
}

unsigned int pixel_format(void* cvpixelbuffer) {
    if (!cvpixelbuffer) return 0;
    CVPixelBufferRef pb = (CVPixelBufferRef)cvpixelbuffer;
    return (unsigned int)CVPixelBufferGetPixelFormatType(pb);
}

int gl_tex_from_cvpixelbuffer(gl_cache_t* cache, void* cvpixelbuffer, void** out_tex,
                              unsigned int* out_name, unsigned int* out_target, int* out_w, int* out_h) {
    if (!cache || !cache->cache || !cvpixelbuffer || !out_tex || !out_name || !out_target || !out_w || !out_h)
        return 0;
    CVPixelBufferRef pb = (CVPixelBufferRef)cvpixelbuffer;
    CVOpenGLTextureRef tex = NULL;
    // Create texture with default attributes (may be GL_TEXTURE_RECTANGLE)
    CVReturn rc = CVOpenGLTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache->cache, pb, NULL, &tex);
    if (rc != kCVReturnSuccess || !tex) return 0;
    GLenum target = CVOpenGLTextureGetTarget(tex);
    GLuint name = CVOpenGLTextureGetName(tex);
    size_t w = CVPixelBufferGetWidth(pb);
    size_t h = CVPixelBufferGetHeight(pb);
    *out_tex = tex; // retained
    *out_name = (unsigned int)name;
    *out_target = (unsigned int)target;
    *out_w = (int)w; *out_h = (int)h;
    return 1;
}

void gl_release_tex(void* cvgltex) {
    if (cvgltex) CFRelease((CFTypeRef)cvgltex);
}

#else

#include "vt_shim.h"
typedef struct vt_session { int _; } vt_session_t;
vt_session_t* vt_create(const uint8_t*, size_t, int, int, uint32_t){return NULL;}
void vt_destroy(vt_session_t*){}
int vt_decode(vt_session_t*, const uint8_t*, size_t, double){return -1;}
int vt_flush(vt_session_t*){return -1;}
int vt_get_frame(vt_session_t*, double, void**, double*){return 0;}
void vt_release_frame(void*){}
void vt_counts(vt_session_t*, uint32_t*, uint32_t*, uint32_t*){}

#endif
