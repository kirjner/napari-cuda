"""
macOS VideoToolbox shim package.

This package exposes the compiled extension `_vt` under a flat API so callers
can `from napari_cuda import _vt as vt` and access functions like `vt.create`.
"""

try:
    from ._vt import (
        create,
        decode,
        flush,
        get_frame,
        release_frame,
        counts,
        map_to_rgb,
        SessionType,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    # Allow import on non-macOS or when the extension is not built
    # Accessing attributes will raise AttributeError in that case
    pass

