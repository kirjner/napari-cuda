from __future__ import annotations

from typing import Any

from napari_cuda.server.data import SliceROI


class SliceDataApplier:
    """Apply slab updates to the napari layer with deterministic semantics."""

    def __init__(
        self,
        *,
        layer: Any,
    ) -> None:
        self._layer = layer

    def apply(
        self,
        *,
        slab,
        roi: SliceROI,
        scale: tuple[float, float],
    ) -> None:
        sy, sx = scale
        translate = (
            float(roi.y_start) * float(max(1e-12, sy)),
            float(roi.x_start) * float(max(1e-12, sx)),
        )
        import logging
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "roi.apply: roi=(y:%d..%d x:%d..%d) scale=(%.6f,%.6f) translate=(%.3f,%.3f)",
                int(roi.y_start), int(roi.y_stop), int(roi.x_start), int(roi.x_stop), float(sy), float(sx), translate[0], translate[1]
            )
        if not hasattr(self._layer, "translate"):
            raise AttributeError("napari layer must expose a 'translate' attribute")
        self._layer.translate = translate  # type: ignore[assignment]
        if not hasattr(self._layer, "data"):
            raise AttributeError("napari layer must expose a 'data' attribute")
        self._layer.data = slab  # type: ignore[assignment]
