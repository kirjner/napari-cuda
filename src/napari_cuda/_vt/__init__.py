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
        retain_frame,
        counts,
        stats,  # extended counters
        map_to_rgb,
        # Zero-copy / GL helpers
        gl_cache_init_for_current_context,
        gl_cache_destroy,
        gl_cache_flush,
        gl_cache_counts,
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

# Optional convenience to discover the compiled extension path when present
def compiled_path() -> str | None:  # pragma: no cover
    try:
        from importlib import import_module
        mod = import_module('napari_cuda._vt._vt')  # type: ignore
        return getattr(mod, '__file__', None)  # type: ignore
    except Exception:
        return None

__all__ = [
    'create','decode','flush','get_frame','release_frame','retain_frame',
    'counts','stats','map_to_rgb',
    'gl_cache_init_for_current_context','gl_cache_destroy','gl_cache_flush','gl_cache_counts',
    'alloc_pixelbuffer_bgra','pixel_format','gl_tex_from_cvpixelbuffer','gl_release_tex',
    'pb_lock_base','pb_unlock_base','SessionType','compiled_path',
]
