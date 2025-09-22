from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

from napari_cuda.server.scene_types import SliceROI


@dataclass(frozen=True)
class SliceUpdateDecision:
    refresh: bool
    new_last_roi: Optional[tuple[int, SliceROI]]


class SliceUpdatePlanner:
    """Determine whether a slice needs to be refreshed based on ROI drift."""

    def __init__(self, edge_threshold: int) -> None:
        self._edge_threshold = int(edge_threshold)

    def evaluate(
        self,
        *,
        level: int,
        roi: SliceROI,
        last_roi: Optional[tuple[int, SliceROI]],
    ) -> SliceUpdateDecision:
        threshold = self._edge_threshold
        if last_roi is None or int(last_roi[0]) != int(level):
            return SliceUpdateDecision(refresh=True, new_last_roi=(int(level), roi))
        prev = last_roi[1]
        if (
            abs(int(roi.y_start) - int(prev.y_start)) >= threshold
            or abs(int(roi.y_stop) - int(prev.y_stop)) >= threshold
            or abs(int(roi.x_start) - int(prev.x_start)) >= threshold
            or abs(int(roi.x_stop) - int(prev.x_stop)) >= threshold
        ):
            return SliceUpdateDecision(refresh=True, new_last_roi=(int(level), roi))
        return SliceUpdateDecision(refresh=False, new_last_roi=last_roi)


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
        scale: Tuple[float, float],
    ) -> None:
        sy, sx = scale
        translate = (
            float(roi.y_start) * float(max(1e-12, sy)),
            float(roi.x_start) * float(max(1e-12, sx)),
        )
        if not hasattr(self._layer, "translate"):
            raise AttributeError("napari layer must expose a 'translate' attribute")
        self._layer.translate = translate  # type: ignore[assignment]
        if not hasattr(self._layer, "data"):
            raise AttributeError("napari layer must expose a 'data' attribute")
        self._layer.data = slab  # type: ignore[assignment]
