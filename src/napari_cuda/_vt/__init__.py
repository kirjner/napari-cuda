"""
macOS VideoToolbox shim package.

This package exposes the compiled extension `_vt` under a flat API so callers
can `from napari_cuda import _vt as vt` and access functions like `vt.create`.
"""

try:
    from ._vt import (  # type: ignore
        # Types
        SessionType,
        alloc_pixelbuffer_bgra,
        counts,
        create,
        decode,
        flush,
        get_frame,
        gl_cache_counts,
        gl_cache_destroy,
        gl_cache_flush,
        # Zero-copy / GL helpers
        gl_cache_init_for_current_context,
        gl_release_tex,
        gl_tex_from_cvpixelbuffer,
        map_to_rgb,
        pb_lock_base,
        pb_unlock_base,
        pixel_format,
        release_frame,
        retain_frame,
        stats,  # extended counters
    )
except Exception:  # pragma: no cover
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
    'SessionType',
    'alloc_pixelbuffer_bgra',
    'compiled_path',
    'counts',
    'create',
    'decode',
    'flush',
    'get_frame',
    'gl_cache_counts',
    'gl_cache_destroy',
    'gl_cache_flush',
    'gl_cache_init_for_current_context',
    'gl_release_tex',
    'gl_tex_from_cvpixelbuffer',
    'map_to_rgb',
    'pb_lock_base',
    'pb_unlock_base',
    'pixel_format',
    'release_frame',
    'retain_frame',
    'stats',
]
