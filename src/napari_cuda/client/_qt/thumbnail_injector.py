"""Qt helper to force layer-list thumbnail repaint.

This module emits QAbstractItemModel.dataChanged for the row corresponding to
the napari layer in the Qt layer list view. It bypasses the timing of napari's
thumbnail event to ensure immediate repaint when a new thumbnail is applied on
the model side.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from qtpy import QtCore  # type: ignore
except Exception:  # pragma: no cover - Qt must be available in the client app
    QtCore = None  # type: ignore

logger = logging.getLogger(__name__)


def emit_layer_thumbnail_data_changed(viewer: Any, layer: Any) -> bool:
    """Emit dataChanged for the model row of `layer` in the Qt layer list.

    Returns True if a row was located and a signal was emitted. Returns False
    if Qt is unavailable, the Qt viewer widgets are not yet constructed, or
    the layer row could not be resolved.
    """
    assert QtCore is not None, "Qt must be available in the client app"

    window = viewer.window
    qt_viewer = window._qt_viewer
    ll = qt_viewer.layers
    model = ll.model()
    source = model.sourceModel() or model

    row_count = source.rowCount()
    src_index = None
    for row in range(int(row_count)):
        idx = source.index(row, 0)
        # LayerListModel exposes getItem(index)
        item = source.getItem(idx)  # type: ignore[attr-defined]
        if item is layer:
            src_index = idx
            break
    if src_index is None:
        return False

    # Emit on the proxy model to guarantee view update
    proxy_index = model.mapFromSource(src_index)
    model.dataChanged.emit(proxy_index, proxy_index, [])

    # Nudge the viewport in case the view needs an update
    ll.viewport().update()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("ThumbnailInjector: dataChanged emitted for layer row")
    return True


def flush_all_layers(viewer: Any) -> int:
    """Emit dataChanged for all current layers.

    Returns the number of rows for which a signal was emitted.
    """
    count = 0
    layers = viewer.layers
    for layer in list(layers):
        if emit_layer_thumbnail_data_changed(viewer, layer):
            count += 1
    return count


__all__ = [
    "emit_layer_thumbnail_data_changed",
    "flush_all_layers",
]
