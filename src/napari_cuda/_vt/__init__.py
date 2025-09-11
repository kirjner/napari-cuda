"""
macOS VideoToolbox shim package.

This package exposes the compiled extension `_vt` under a flat API so callers
can `from napari_cuda import _vt as vt` and access functions like `vt.create`.
"""

try:
    from ._vt import (  # type: ignore
        create,
        decode,
        flush,
        get_frame,
        release_frame,
        counts,
        map_to_rgb,
        retain_frame,
        # Zero-copy / GL helpers
        gl_cache_init_for_current_context,
        gl_cache_destroy,
        alloc_pixelbuffer_bgra,
        pixel_format,
        gl_tex_from_cvpixelbuffer,
        gl_release_tex,
        pb_lock_base,
        pb_unlock_base,
        # Types
        SessionType,
    )
except Exception as e:  # pragma: no cover
    # Allow import on non-macOS or when the extension is not built
    # Accessing attributes will raise AttributeError in that case
    pass
