"""Qt helper to force layer-list thumbnail repaint.

This module provides a small, feature‑flagged injector that emits
QAbstractItemModel.dataChanged for the row corresponding to a napari layer in
the Qt layer list view. It bypasses the timing of napari's thumbnail event to
ensure immediate repaint when a new thumbnail is applied on the model side.

Enable by setting environment variable:
    NAPARI_CUDA_DIRECT_THUMBNAIL=1

The functions are no-ops if the Qt viewer widgets are unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - best effort import
    QtCore = None  # type: ignore

logger = logging.getLogger(__name__)


_FLAG_CACHE: Optional[bool] = None


def _flag_enabled() -> bool:
    """Check env flag once and cache the result."""
    global _FLAG_CACHE
    if _FLAG_CACHE is None:
        flag = (os.getenv("NAPARI_CUDA_DIRECT_THUMBNAIL") or "").lower()
        _FLAG_CACHE = flag in {"1", "true", "yes", "on", "dbg", "debug"}
    return bool(_FLAG_CACHE)


def emit_layer_thumbnail_data_changed(viewer: Any, layer: Any) -> bool:
    """Emit dataChanged for the model row of `layer` in the Qt layer list.

    Returns True if a row was located and a signal was emitted. Returns False
    if Qt is unavailable, the Qt viewer widgets are not yet constructed, or
    the layer row could not be resolved.
    """
    if not _flag_enabled():
        return False
    if QtCore is None:  # pragma: no cover - GUI not available
        return False
    try:
        window = getattr(viewer, "window", None)
        qt_viewer = getattr(window, "_qt_viewer", None)
        if qt_viewer is None:
            return False
        ll = getattr(qt_viewer, "layers", None)
        if ll is None:
            return False
        model = getattr(ll, "model", lambda: None)()
        if model is None:
            return False
        # Resolve source model if a proxy is used
        source = getattr(model, "sourceModel", lambda: None)() or model
        row_count = getattr(source, "rowCount")()
        src_index = None
        for row in range(int(row_count)):
            idx = source.index(row, 0)
            item = None
            try:
                # LayerListModel exposes getItem(index)
                item = source.getItem(idx)  # type: ignore[attr-defined]
            except Exception:
                pass
            if item is layer:
                src_index = idx
                break
        if src_index is None:
            return False
        # Prefer emitting on the source model; fall back to proxy
        try:
            source.dataChanged.emit(src_index, src_index, [])
        except Exception:
            try:
                proxy_index = model.mapFromSource(src_index)
                model.dataChanged.emit(proxy_index, proxy_index, [])
            except Exception:
                return False
        # Nudge the viewport in case the view needs an update
        try:
            vp = getattr(ll, "viewport", lambda: None)()
            if vp is not None and hasattr(vp, "update"):
                vp.update()
        except Exception:
            pass
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ThumbnailInjector: dataChanged emitted for layer row")
        return True
    except Exception:
        logger.debug("ThumbnailInjector: emit failed", exc_info=True)
        return False


def flush_all_layers(viewer: Any) -> int:
    """Emit dataChanged for all current layers.

    Returns the number of rows for which a signal was emitted.
    """
    if not _flag_enabled():
        return 0
    count = 0
    try:
        layers = getattr(viewer, "layers", None)
        if layers is None:
            return 0
        for layer in list(layers):
            try:
                if emit_layer_thumbnail_data_changed(viewer, layer):
                    count += 1
            except Exception:
                logger.debug("ThumbnailInjector: flush for one layer failed", exc_info=True)
        return count
    except Exception:
        logger.debug("ThumbnailInjector: flush failed", exc_info=True)
        return 0


__all__ = [
    "emit_layer_thumbnail_data_changed",
    "flush_all_layers",
]

