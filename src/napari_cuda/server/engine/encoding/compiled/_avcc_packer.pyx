# cython: language_level=3
"""
Fast AVCC packer implemented in Cython.

Functions:
    pack_to_avcc_fast(packets, cache) -> (bytes|None, bool is_key)

Behaviors:
- Accepts a bytes-like or a sequence of bytes-like chunks from the encoder.
- If a chunk starts with Annex B start code (0x000001 or 0x00000001), it is
  converted to AVCC (4-byte big-endian NAL length prefix).
- Otherwise, the chunk is treated as AVCC and copied through.
- Keyframe detection: H.264 IDR (type 5) or H.265 IDR/CRA (types 19/20/21).
- Caches VPS/SPS/PPS into the provided `cache` object if attributes exist.

Note: This is deliberately minimal and avoids SPS/PPS injection. The encoder
should be configured to repeat SPS/PPS on keyframes (repeatspspps=1).
"""

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from libc.string cimport memcmp, memcpy
from libc.stdint cimport uint8_t, uint32_t
cimport cython

@cython.cfunc
cdef inline bint _is_annexb(const uint8_t* data, Py_ssize_t n) noexcept:
    if n >= 3 and data[0] == 0 and data[1] == 0 and data[2] == 1:
        return True
    if n >= 4 and data[0] == 0 and data[1] == 0 and data[2] == 0 and data[3] == 1:
        return True
    return False

@cython.cfunc
cdef inline void _write_be32(uint8_t* dst, uint32_t v) noexcept:
    dst[0] = <uint8_t>((v >> 24) & 0xFF)
    dst[1] = <uint8_t>((v >> 16) & 0xFF)
    dst[2] = <uint8_t>((v >> 8) & 0xFF)
    dst[3] = <uint8_t>(v & 0xFF)

@cython.cfunc
cdef inline void _update_cache_and_key(const uint8_t* nal, Py_ssize_t n, object cache, bint *is_key) noexcept:
    if n <= 0:
        return
    cdef uint8_t b0 = nal[0]
    cdef int n264 = b0 & 0x1F
    cdef int n265 = (b0 >> 1) & 0x3F
    try:
        # H.264
        if n264 == 7:
            cache.sps = PyBytes_FromStringAndSize(<const char*>nal, n)
        elif n264 == 8:
            cache.pps = PyBytes_FromStringAndSize(<const char*>nal, n)
        elif n264 == 5:
            is_key[0] = True
        # H.265
        if n265 == 32:
            cache.vps = PyBytes_FromStringAndSize(<const char*>nal, n)
        elif n265 == 33:
            cache.sps = PyBytes_FromStringAndSize(<const char*>nal, n)
        elif n265 == 34:
            cache.pps = PyBytes_FromStringAndSize(<const char*>nal, n)
        elif n265 == 19 or n265 == 20 or n265 == 21:
            is_key[0] = True
    except Exception:
        # Cache object may not have these attributes; ignore failures.
        pass

@cython.cfunc
cdef bytes _annexb_to_avcc(const uint8_t* data, Py_ssize_t n, object cache, bint *is_key):
    # First pass: find start codes, estimate total output size
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t start
    cdef Py_ssize_t next_start
    cdef Py_ssize_t count = 0
    cdef uint8_t *p
    cdef Py_ssize_t total = 0
    cdef Py_ssize_t k
    cdef Py_ssize_t prev = 0
    cdef uint8_t* out = NULL
    cdef Py_ssize_t out_len = 0
    cdef uint8_t* out2 = NULL
    cdef uint8_t* w = NULL
    cdef Py_ssize_t ln = 0
    cdef Py_ssize_t end = 0
    cdef object py = None

    p = <uint8_t*>data

    # Collect NAL boundaries (start positions) into a temporary Python list
    starts = []
    while i <= n - 3:
        if i <= n - 4 and p[i] == 0 and p[i+1] == 0 and p[i+2] == 0 and p[i+3] == 1:
            starts.append(i + 4)
            i += 4
        elif p[i] == 0 and p[i+1] == 0 and p[i+2] == 1:
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    starts.append(n)

    # If we didn't find any proper start codes, return original as a single NAL
    if len(starts) <= 1:
        # Wrap as 4-byte length + payload
        out_len = n + 4
        out = <uint8_t*>PyMem_Malloc(out_len)
        if out == NULL:
            raise MemoryError()
        _write_be32(out, <uint32_t>n)
        memcpy(out + 4, data, n)
        try:
            _update_cache_and_key(data, n, cache, is_key)
            return PyBytes_FromStringAndSize(<const char*>out, out_len)
        finally:
            PyMem_Free(out)

    # Compute total output size based on true NAL boundaries (exclude next startcode)
    total = 0
    cdef Py_ssize_t m = len(starts)
    for k in range(m - 1):
        start = starts[k]
        next_start = starts[k + 1]
        if k + 1 < m - 0:  # have next start
            # Determine the preceding start-code length (3 or 4) before next payload start
            if next_start >= 4 and p[next_start - 4] == 0 and p[next_start - 3] == 0 and p[next_start - 2] == 0 and p[next_start - 1] == 1:
                end = next_start - 4
            elif next_start >= 3 and p[next_start - 3] == 0 and p[next_start - 2] == 0 and p[next_start - 1] == 1:
                end = next_start - 3
            else:
                end = next_start
        else:
            end = n
        if end > start:
            total += 4 + (end - start)

    if total <= 0:
        return PyBytes_FromStringAndSize(NULL, 0)

    # Allocate final Python bytes and fill directly to avoid extra copy
    py = PyBytes_FromStringAndSize(NULL, total)
    if py is None:
        raise MemoryError()
    w = <uint8_t*>PyBytes_AS_STRING(py)

    # Second pass: write length-prefixed NALs
    for k in range(m - 1):
        start = starts[k]
        next_start = starts[k + 1]
        if k + 1 < m - 0:
            if next_start >= 4 and p[next_start - 4] == 0 and p[next_start - 3] == 0 and p[next_start - 2] == 0 and p[next_start - 1] == 1:
                end = next_start - 4
            elif next_start >= 3 and p[next_start - 3] == 0 and p[next_start - 2] == 0 and p[next_start - 1] == 1:
                end = next_start - 3
            else:
                end = next_start
        else:
            end = n
        if end <= start:
            continue
        ln = end - start
        _write_be32(w, <uint32_t>ln)
        memcpy(w + 4, p + start, ln)
        _update_cache_and_key(p + start, ln, cache, is_key)
        w += 4 + ln

    return <bytes>py


@cython.cfunc
cdef void _avcc_scan(const uint8_t* data, Py_ssize_t n, object cache, bint *is_key):
    # Validate length prefixes and scan for keyframe and cache VPS/SPS/PPS.
    cdef Py_ssize_t i = 0
    cdef uint32_t ln
    cdef const uint8_t* p = data
    while i + 4 <= n:
        ln = (<uint32_t>p[i] << 24) | (<uint32_t>p[i+1] << 16) | (<uint32_t>p[i+2] << 8) | (<uint32_t>p[i+3])
        i += 4
        if ln == 0 or i + ln > n:
            break
        _update_cache_and_key(p + i, ln, cache, is_key)
        i += ln


cpdef tuple pack_to_avcc_fast(object packets, object cache):
    """Fast path: returns (payload_bytes|None, is_keyframe)."""
    cdef bint is_key = False
    cdef bytes buf
    cdef const uint8_t* d
    cdef Py_ssize_t n
    if packets is None:
        return None, False
    # Normalize packets to a sequence of bytes objects
    chunks = []
    if isinstance(packets, (bytes, bytearray, memoryview)):
        chunks.append(bytes(packets))
    elif isinstance(packets, (list, tuple)):
        for p in packets:
            if p is None:
                continue
            chunks.append(bytes(p))
    else:
        try:
            chunks.append(bytes(packets))
        except Exception:
            return None, False

    # Fast path: single chunk
    if len(chunks) == 1:
        buf = chunks[0]
        d = <const uint8_t*>buf
        n = len(buf)
        if _is_annexb(d, n):
            out = _annexb_to_avcc(d, n, cache, &is_key)
            return out, bool(is_key)
        else:
            _avcc_scan(d, n, cache, &is_key)
            return buf, bool(is_key)

    # Multi-chunk: process each and concatenate
    outs = []
    for buf in chunks:
        d = <const uint8_t*>buf
        n = len(buf)
        if _is_annexb(d, n):
            outs.append(_annexb_to_avcc(d, n, cache, &is_key))
        else:
            _avcc_scan(d, n, cache, &is_key)
            outs.append(buf)
    return b''.join(outs), bool(is_key)
