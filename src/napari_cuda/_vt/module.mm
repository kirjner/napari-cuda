#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "vt_shim.h"
#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#include <CoreFoundation/CoreFoundation.h>
static PyObject* m_map_to_rgb_impl(PyObject* self, PyObject* args);
#else
static PyObject* m_map_to_rgb_stub(PyObject* self, PyObject* args){ Py_RETURN_NONE; }
#endif

typedef struct {
    PyObject_HEAD
    vt_session_t* sess;
} PyVTSession;

static void PyVTSession_dealloc(PyVTSession* self){
    if (self->sess) vt_destroy(self->sess);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject PyVTSessionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_vt.Session",
    .tp_basicsize = sizeof(PyVTSession),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)PyVTSession_dealloc,
};

static PyObject* m_create(PyObject* self, PyObject* args){
    const uint8_t* avcc; Py_ssize_t avcc_len; int w,h; unsigned int pf;
    if (!PyArg_ParseTuple(args, "y#iiI", &avcc, &avcc_len, &w, &h, &pf)) return NULL;
    vt_session_t* s = vt_create(avcc, (size_t)avcc_len, w, h, pf);
    if (!s){ PyErr_SetString(PyExc_RuntimeError, "vt_create failed"); return NULL; }
    PyVTSession* obj = PyObject_New(PyVTSession, &PyVTSessionType);
    obj->sess = s; return (PyObject*)obj;
}

static PyObject* m_decode(PyObject* self, PyObject* args){
    PyVTSession* obj; const uint8_t* au; Py_ssize_t ln; double pts;
    if (!PyArg_ParseTuple(args, "Oy#d", (PyObject**)&obj, &au, &ln, &pts)) return NULL;
    int rc = vt_decode(obj->sess, au, (size_t)ln, pts);
    return PyLong_FromLong(rc);
}

static PyObject* m_flush(PyObject* self, PyObject* args){
    PyVTSession* obj; if (!PyArg_ParseTuple(args, "O", (PyObject**)&obj)) return NULL;
    int rc = vt_flush(obj->sess);
    return PyLong_FromLong(rc);
}

static PyObject* m_get_frame(PyObject* self, PyObject* args){
    PyVTSession* obj; double timeout;
    if (!PyArg_ParseTuple(args, "Od", (PyObject**)&obj, &timeout)) return NULL;
    void* buf = NULL; double pts = 0.0;
    int got = vt_get_frame(obj->sess, timeout, &buf, &pts);
    if (!got) Py_RETURN_NONE;
    // Return (capsule(buf), pts)
    PyObject* cap = PyCapsule_New(buf, "CVPixelBufferRef", NULL);
    PyObject* tup = Py_BuildValue("(Od)", cap, pts);
    Py_DECREF(cap);
    return tup;
}

static PyObject* m_release_frame(PyObject* self, PyObject* args){
    PyObject* cap; if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    void* buf = PyCapsule_GetPointer(cap, "CVPixelBufferRef");
    if (buf) vt_release_frame(buf);
    Py_RETURN_NONE;
}

static PyObject* m_counts(PyObject* self, PyObject* args){
    PyVTSession* obj; if (!PyArg_ParseTuple(args, "O", (PyObject**)&obj)) return NULL;
    uint32_t a=0,b=0,c=0; vt_counts(obj->sess, &a, &b, &c);
    return Py_BuildValue("(III)", a,b,c);
}

static PyMethodDef Methods[] = {
    {"create", m_create, METH_VARARGS, "Create VT session"},
    {"decode", m_decode, METH_VARARGS, "Submit AVCC AU"},
    {"flush", m_flush, METH_VARARGS, "Flush async frames"},
    {"get_frame", m_get_frame, METH_VARARGS, "Get decoded frame (capsule, pts) or None"},
    {"release_frame", m_release_frame, METH_VARARGS, "Release CVPixelBufferRef"},
    {"counts", m_counts, METH_VARARGS, "Get (submits, outputs, qlen)"},
    // map_to_rgb(capsule) -> (bytes, width, height)
#ifdef __APPLE__
    {"map_to_rgb", m_map_to_rgb_impl, METH_VARARGS, "Map CVPixelBufferRef to RGB bytes (w,h)"},
#else
    {"map_to_rgb", m_map_to_rgb_stub, METH_VARARGS, "Map CVPixelBufferRef to RGB bytes (w,h)"},
#endif
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_vt",
    "VT shim module",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__vt(void){
    if (PyType_Ready(&PyVTSessionType) < 0) return NULL;
    PyObject* m = PyModule_Create(&module);
    if (!m) return NULL;
    Py_INCREF(&PyVTSessionType);
    PyModule_AddObject(m, "SessionType", (PyObject*)&PyVTSessionType);
    return m;
}

#ifdef __APPLE__
static PyObject* m_map_to_rgb_impl(PyObject* self, PyObject* args){
    PyObject* cap; if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    void* buf = PyCapsule_GetPointer(cap, "CVPixelBufferRef");
    if (!buf) { PyErr_SetString(PyExc_ValueError, "invalid capsule"); return NULL; }
    CVPixelBufferRef pb = (CVPixelBufferRef)buf;
    CVPixelBufferLockBaseAddress(pb, 0);
    size_t w = (size_t)CVPixelBufferGetWidth(pb);
    size_t h = (size_t)CVPixelBufferGetHeight(pb);
    OSType pf = CVPixelBufferGetPixelFormatType(pb);
    PyObject* out = NULL;
    if (pf == kCVPixelFormatType_32BGRA) {
        size_t bpr = (size_t)CVPixelBufferGetBytesPerRow(pb);
        const unsigned char* base = (const unsigned char*)CVPixelBufferGetBaseAddress(pb);
        out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(w*h*3));
        if (out) {
            unsigned char* dst = (unsigned char*)PyBytes_AS_STRING(out);
            for (size_t y=0; y<h; ++y){
                const unsigned char* row = base + y*bpr;
                for (size_t x=0; x<w; ++x){
                    const unsigned char* p = row + 4*x; // BGRA
                    *dst++ = p[2]; // R
                    *dst++ = p[1]; // G
                    *dst++ = p[0]; // B
                }
            }
        }
    } else if (pf == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || pf == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        size_t y_bpr = (size_t)CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
        size_t uv_bpr = (size_t)CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
        const unsigned char* yb = (const unsigned char*)CVPixelBufferGetBaseAddressOfPlane(pb, 0);
        const unsigned char* uvb = (const unsigned char*)CVPixelBufferGetBaseAddressOfPlane(pb, 1);
        int full = (pf == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(w*h*3));
        if (out) {
            unsigned char* dst = (unsigned char*)PyBytes_AS_STRING(out);
            for (size_t j=0; j<h; ++j){
                const unsigned char* yrow = yb + j*y_bpr;
                const unsigned char* uvrow = uvb + (j/2)*uv_bpr;
                for (size_t i=0; i<w; i+=2){
                    int Y0 = yrow[i];
                    int Y1 = (i+1<w) ? yrow[i+1] : Y0;
                    // NV12 has interleaved U,V at even byte indices
                    int U = uvrow[i & ~1] - 128;
                    int V = uvrow[(i & ~1)+1] - 128;
                    int r0, g0, b0, r1, g1, b1;
                    if (full){
                        // Full range (BT.709-ish integer approx): Y in [0..255], U/V offset already removed
                        // Scale Y by 256 for fixed-point math
                        int C0 = Y0;
                        int C1 = Y1;
                        r0 = (256*C0 + 358*V + 128) >> 8;
                        g0 = (256*C0 -  88*U - 183*V + 128) >> 8;
                        b0 = (256*C0 + 454*U + 128) >> 8;
                        r1 = (256*C1 + 358*V + 128) >> 8;
                        g1 = (256*C1 -  88*U - 183*V + 128) >> 8;
                        b1 = (256*C1 + 454*U + 128) >> 8;
                    } else {
                        // Video range (16..235): C = Y - 16; use ITU-R BT.601/709 integer approx
                        int C0 = Y0 - 16; if (C0 < 0) C0 = 0;
                        int C1 = Y1 - 16; if (C1 < 0) C1 = 0;
                        r0 = (298*C0 + 409*V + 128) >> 8;
                        g0 = (298*C0 - 100*U - 208*V + 128) >> 8;
                        b0 = (298*C0 + 516*U + 128) >> 8;
                        r1 = (298*C1 + 409*V + 128) >> 8;
                        g1 = (298*C1 - 100*U - 208*V + 128) >> 8;
                        b1 = (298*C1 + 516*U + 128) >> 8;
                    }
                    if (r0<0) r0=0; if (r0>255) r0=255;
                    if (g0<0) g0=0; if (g0>255) g0=255;
                    if (b0<0) b0=0; if (b0>255) b0=255;
                    if (r1<0) r1=0; if (r1>255) r1=255;
                    if (g1<0) g1=0; if (g1>255) g1=255;
                    if (b1<0) b1=0; if (b1>255) b1=255;
                    *dst++ = (unsigned char)r0; *dst++ = (unsigned char)g0; *dst++ = (unsigned char)b0;
                    if (i+1<w){ *dst++ = (unsigned char)r1; *dst++ = (unsigned char)g1; *dst++ = (unsigned char)b1; }
                }
            }
        }
    } else {
        CVPixelBufferUnlockBaseAddress(pb, 0);
        PyErr_SetString(PyExc_RuntimeError, "Unsupported pixel format");
        return NULL;
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    if (!out) return NULL;
    PyObject* ret = Py_BuildValue("(y#kk)", PyBytes_AS_STRING(out), PyBytes_GET_SIZE(out), (unsigned long long)w, (unsigned long long)h);
    Py_DECREF(out);
    return ret;
}
#endif
