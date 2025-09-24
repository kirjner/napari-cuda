"""Scene state application helper for the EGL worker.

Consolidates applying pending scene updates (dims/Z slice in 2D and volume
parameters in 3D) so the render loop can delegate without duplicating logic.
Procedural and minimal: no extra indirection beyond an explicit context and
return value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence, Tuple
import math
import logging

import numpy as np

from napari_cuda.server.scene_types import SliceROI
from napari_cuda.server.roi_applier import SliceDataApplier
from napari_cuda.server.scene_state import ServerSceneState
from napari_cuda.server.state_machine import SceneStateQueue


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneStateApplyContext:
    # Mode & view
    use_volume: bool
    viewer: Any  # napari ViewerModel or None
    camera: Any  # vispy camera or None
    visual: Any  # vispy visual (Image/Volume) or None
    layer: Any   # napari layer (for 2D slab) or None

    # Source and level selection
    scene_source: Any  # ZarrSceneSource or None
    active_ms_level: int
    z_index: Optional[int]

    # ROI placement and flags
    last_roi: Optional[tuple[int, SliceROI]]
    preserve_view_on_switch: bool
    sticky_contrast: bool
    idr_on_z: bool

    # Current data size in pixels (W, H)
    data_wh: Tuple[int, int]

    # Synchronization for source mutators
    state_lock: Any  # threading.Lock

    # Collaborator functions
    ensure_scene_source: Callable[[], Any]
    plane_scale_for_level: Callable[[Any, int], Tuple[float, float]]
    load_slice: Callable[[Any, int, int], Any]
    notify_scene_refresh: Callable[[], None]
    mark_render_tick_needed: Callable[[], None]
    request_encoder_idr: Optional[Callable[[], None]] = None


@dataclass(frozen=True)
class SceneStateApplyResult:
    z_index: Optional[int] = None
    data_wh: Optional[Tuple[int, int]] = None
    last_step: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class SceneDrainResult:
    z_index: Optional[int] = None
    data_wh: Optional[Tuple[int, int]] = None
    last_step: Optional[Tuple[int, ...]] = None
    render_marked: bool = False
    policy_refresh_needed: bool = False


class SceneStateApplier:
    @staticmethod
    def apply_dims_and_slice(
        ctx: SceneStateApplyContext,
        *,
        current_step: Optional[Sequence[int]]
    ) -> SceneStateApplyResult:
        if ctx.use_volume or current_step is None:
            return SceneStateApplyResult()

        steps = tuple(int(x) for x in current_step)

        viewer = ctx.viewer
        assert viewer is not None, "viewer must be initialised in 2D mode"
        viewer.dims.current_step = steps  # type: ignore[attr-defined]

        # Compute z from source axes when available
        z_new: Optional[int] = None
        source = ctx.scene_source or ctx.ensure_scene_source()
        axes = getattr(source, "axes", [])
        if isinstance(axes, (list, tuple)) and "z" in axes:
            zi = axes.index("z")
            if zi < len(steps):
                z_new = int(steps[zi])

        if z_new is None or (ctx.z_index is not None and int(z_new) == int(ctx.z_index)):
            ctx.mark_render_tick_needed()
            return SceneStateApplyResult(last_step=steps)

        # Update the source step preserving other indices
        axes = source.axes
        zi = axes.index("z") if "z" in axes else 0
        base = list(getattr(source, "current_step", None) or steps)
        lvl_shape = source.level_shape(ctx.active_ms_level)
        if len(base) < len(lvl_shape):
            base = base + [0] * (len(lvl_shape) - len(base))
        base[zi] = int(z_new)
        with ctx.state_lock:
            _ = source.set_current_level(ctx.active_ms_level, step=tuple(int(x) for x in base))

        # Load slab and update layer/visual
        slab = ctx.load_slice(source, ctx.active_ms_level, int(z_new))

        roi_for_layer = None
        if ctx.last_roi is not None and int(ctx.last_roi[0]) == int(ctx.active_ms_level):
            roi_for_layer = ctx.last_roi[1]

        sy, sx = SceneStateApplier.apply_slice_to_layer(
            ctx,
            source=source,
            slab=slab,
            roi=roi_for_layer,
            update_contrast=not ctx.sticky_contrast,
        )

        visual = ctx.visual
        assert visual is not None, "visual must be initialised in 2D mode"
        visual.set_data(slab)  # type: ignore[attr-defined]

        # Update camera range unless preserving the view
        cam = ctx.camera
        if not ctx.preserve_view_on_switch:
            assert cam is not None, "camera must be available to update range"
            h, w = int(slab.shape[0]), int(slab.shape[1])
            world_w = max(1.0, float(w) * float(max(1e-12, sx)))
            world_h = max(1.0, float(h) * float(max(1e-12, sy)))
            cam.set_range(x=(0.0, world_w), y=(0.0, world_h))

        if ctx.idr_on_z and ctx.request_encoder_idr is not None:
            ctx.request_encoder_idr()

        ctx.notify_scene_refresh()
        ctx.mark_render_tick_needed()

        h, w = int(slab.shape[0]), int(slab.shape[1])
        return SceneStateApplyResult(z_index=int(z_new), data_wh=(w, h), last_step=steps)

    @staticmethod
    def apply_slice_to_layer(
        ctx: SceneStateApplyContext,
        *,
        source: Any,
        slab: Any,
        roi: Optional[SliceROI],
        update_contrast: bool,
        ) -> Tuple[float, float]:
        layer = ctx.layer
        assert layer is not None, "napari layer must be initialised in 2D mode"

        sy, sx = ctx.plane_scale_for_level(source, int(ctx.active_ms_level))
        roi_to_apply = roi or SliceROI(0, int(slab.shape[0]), 0, int(slab.shape[1]))
        SliceDataApplier(layer=layer).apply(slab=slab, roi=roi_to_apply, scale=(sy, sx))

        layer.visible = True  # type: ignore[assignment]
        layer.opacity = 1.0  # type: ignore[assignment]
        layer.blending = "opaque"  # type: ignore[assignment]

        if update_contrast:
            smin = float(np.nanmin(slab)) if hasattr(np, "nanmin") else float(np.min(slab))
            smax = float(np.nanmax(slab)) if hasattr(np, "nanmax") else float(np.max(slab))
            if not math.isfinite(smin) or not math.isfinite(smax) or smax <= smin:
                layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
            else:
                if 0.0 <= smin <= 1.0 and 0.0 <= smax <= 1.1:
                    layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
                else:
                    layer.contrast_limits = [smin, smax]  # type: ignore[assignment]

        return sy, sx

    @staticmethod
    def apply_volume_layer(
        ctx: SceneStateApplyContext,
        *,
        volume: Any,
        contrast: Tuple[float, float],
    ) -> Tuple[Tuple[int, int], int]:
        layer = ctx.layer
        if layer is not None:
            layer.data = volume
            layer.contrast_limits = [float(contrast[0]), float(contrast[1])]  # type: ignore[assignment]

        depth = int(volume.shape[0])
        height = int(volume.shape[1])
        width = int(volume.shape[2]) if volume.ndim >= 3 else int(volume.shape[-1])
        return (width, height), depth

    @staticmethod
    def apply_volume_params(
        ctx: SceneStateApplyContext,
        *,
        mode: Optional[str] = None,
        colormap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        opacity: Optional[float] = None,
        sample_step: Optional[float] = None,
    ) -> None:
        if not ctx.use_volume:
            return
        vis = ctx.visual
        if vis is None:
            return
        if mode:
            mm = str(mode).lower()
            if mm in ("mip", "translucent", "iso"):
                vis.method = mm  # type: ignore[attr-defined]
        if colormap:
            name = str(colormap).lower()
            if name == "gray":
                name = "grays"
            vis.cmap = name  # type: ignore[attr-defined]
        if isinstance(clim, tuple) and len(clim) >= 2:
            lo = float(clim[0]); hi = float(clim[1])
            if hi < lo:
                lo, hi = hi, lo
            vis.clim = (lo, hi)  # type: ignore[attr-defined]
        if opacity is not None:
            vis.opacity = float(max(0.0, min(1.0, float(opacity))))  # type: ignore[attr-defined]
        if sample_step is not None:
            vis.relative_step_size = float(max(0.1, min(4.0, float(sample_step))))  # type: ignore[attr-defined]

    @staticmethod
    def drain_updates(
        ctx: SceneStateApplyContext,
        *,
        state: ServerSceneState,
        queue: SceneStateQueue,
    ) -> SceneDrainResult:
        """Apply pending scene state updates and report side effects."""

        render_marked = False
        original_mark = ctx.mark_render_tick_needed

        def _mark_render() -> None:
            nonlocal render_marked
            render_marked = True
            original_mark()

        ctx_for_apply = replace(ctx, mark_render_tick_needed=_mark_render)

        z_index = ctx.z_index
        data_wh = ctx.data_wh
        last_step: Optional[Tuple[int, ...]] = None

        if not ctx.use_volume and state.current_step is not None:
            res = SceneStateApplier.apply_dims_and_slice(
                ctx_for_apply,
                current_step=state.current_step,
            )
            if res.z_index is not None:
                z_index = int(res.z_index)
            if res.data_wh is not None:
                data_wh = (int(res.data_wh[0]), int(res.data_wh[1]))
            if res.last_step is not None:
                last_step = tuple(int(x) for x in res.last_step)

        if ctx.use_volume:
            SceneStateApplier.apply_volume_params(
                ctx_for_apply,
                mode=state.volume_mode,
                colormap=state.volume_colormap,
                clim=state.volume_clim,
                opacity=state.volume_opacity,
                sample_step=state.volume_sample_step,
            )

        cam = ctx.camera
        if cam is None:
            queue.update_state_signature(state)
            return SceneDrainResult(
                z_index=int(z_index) if z_index is not None else None,
                data_wh=(int(data_wh[0]), int(data_wh[1])) if data_wh is not None else None,
                last_step=last_step,
                render_marked=render_marked,
                policy_refresh_needed=True,
            )

        if state.center is not None:
            assert hasattr(cam, "center"), "Camera missing center property"
            cam.center = state.center  # type: ignore[attr-defined]
        if state.zoom is not None:
            assert hasattr(cam, "zoom"), "Camera missing zoom property"
            cam.zoom = state.zoom  # type: ignore[attr-defined]
        if state.angles is not None:
            assert hasattr(cam, "angles"), "Camera missing angles property"
            cam.angles = state.angles  # type: ignore[attr-defined]

        policy_refresh = queue.update_state_signature(state)

        return SceneDrainResult(
            z_index=int(z_index) if z_index is not None else None,
            data_wh=(int(data_wh[0]), int(data_wh[1])) if data_wh is not None else None,
            last_step=last_step,
            render_marked=render_marked,
            policy_refresh_needed=policy_refresh,
        )


__all__ = [
    "SceneStateApplier",
    "SceneStateApplyContext",
    "SceneStateApplyResult",
    "SceneDrainResult",
]
