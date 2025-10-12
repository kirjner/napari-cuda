"""Scene state application helper for the EGL worker.

Consolidates applying pending scene updates (dims/Z slice in 2D and volume
parameters in 3D) so the render loop can delegate without duplicating logic.
Procedural and minimal: no extra indirection beyond an explicit context and
return value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import math
import logging

import numpy as np

from napari.layers.base._base_constants import Blending as NapariBlending
from napari.layers.image._image_constants import ImageRendering as NapariImageRendering

from napari_cuda.server.runtime.scene_types import SliceROI
from napari_cuda.server.data.roi_applier import SliceDataApplier
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.render_update_queue import RenderUpdateQueue

from napari._vispy.layers.image import _napari_cmap_to_vispy
from napari.utils.colormaps.colormap_utils import ensure_colormap


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
    mark_render_tick_needed: Callable[[], None]
    request_encoder_idr: Optional[Callable[[], None]] = None
    volume_scale: Optional[Tuple[float, float, float]] = None


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

        src_step = None
        try:
            if ctx.scene_source is not None:
                src_step = tuple(int(x) for x in (ctx.scene_source.current_step or ()))
        except Exception:
            src_step = None

        viewer_before = None
        try:
            viewer_before = tuple(int(x) for x in (ctx.viewer.dims.current_step or ()))  # type: ignore[attr-defined]
        except Exception:
            viewer_before = None

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
            return SceneStateApplyResult()

        # Update the source step preserving other indices
        axes = source.axes
        zi = axes.index("z") if "z" in axes else 0
        base = list(getattr(source, "current_step", None) or steps)
        lvl_shape = source.level_shape(ctx.active_ms_level)
        if len(base) < len(lvl_shape):
            base = base + [0] * (len(lvl_shape) - len(base))
        base[zi] = int(z_new)
        with ctx.state_lock:
            _ = source.set_current_slice(tuple(int(x) for x in base), ctx.active_ms_level)

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
        if not layer.blending:
            layer.blending = NapariBlending.OPAQUE.value  # type: ignore[assignment]

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
            lo = float(contrast[0])
            hi = float(contrast[1])
            if hi <= lo:
                hi = lo + 1.0
            layer.translate = tuple(0.0 for _ in range(int(volume.ndim)))  # type: ignore[assignment]
            data_min = float(np.nanmin(volume)) if hasattr(np, 'nanmin') else float(np.min(volume))
            data_max = float(np.nanmax(volume)) if hasattr(np, 'nanmax') else float(np.max(volume))
            normalized = (-0.05 <= data_min <= 1.05) and (-0.05 <= data_max <= 1.05)
            logger.debug(
                "apply_volume_layer stats: min=%.6f max=%.6f contrast=(%.6f, %.6f) normalized=%s",
                data_min,
                data_max,
                lo,
                hi,
                normalized,
            )
            if normalized:
                layer.contrast_limits = [0.0, 1.0]  # type: ignore[assignment]
            else:
                layer.contrast_limits = [lo, hi]  # type: ignore[assignment]
            if ctx.volume_scale is not None:
                scale_vals: Tuple[float, ...] = tuple(float(s) for s in ctx.volume_scale)
            elif ctx.scene_source is not None:
                lvl_scale = ctx.scene_source.level_scale(int(ctx.active_ms_level))
                scale_vals = tuple(float(s) for s in lvl_scale)
            else:
                scale_vals = tuple(1.0 for _ in range(int(volume.ndim)))
            if len(scale_vals) < int(volume.ndim):
                pad = int(volume.ndim) - len(scale_vals)
                scale_vals = tuple(1.0 for _ in range(pad)) + scale_vals
            scale_vals = tuple(scale_vals[-int(volume.ndim):])
            layer.scale = scale_vals  # type: ignore[assignment]

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
            token = str(mode).strip().lower()
            mm = NapariImageRendering(token).value
            vis.method = mm  # type: ignore[attr-defined]
        if colormap is not None:
            cmap = ensure_colormap(colormap)
            vis.cmap = _napari_cmap_to_vispy(cmap)  # type: ignore[attr-defined]
        if isinstance(clim, tuple) and len(clim) >= 2:
            lo = float(clim[0]); hi = float(clim[1])
            if hi < lo:
                lo, hi = hi, lo
            vis.clim = (lo, hi)  # type: ignore[attr-defined]
        if opacity is not None:
            vis.opacity = float(max(0.0, min(1.0, float(opacity))))  # type: ignore[attr-defined]
        if sample_step is not None:
            if hasattr(vis, "relative_step_size"):
                vis.relative_step_size = float(max(0.1, min(4.0, float(sample_step))))  # type: ignore[attr-defined]
            else:
                logger.debug(
                    "volume params: visual %s missing relative_step_size; skipping sample_step update",
                    type(vis).__name__,
                )

    @staticmethod
    def apply_layer_updates(
        ctx: SceneStateApplyContext,
        updates: Mapping[str, Mapping[str, Any]],
    ) -> None:
        layer = ctx.layer
        if layer is None or not updates:
            return

        def _apply(prop: str, value: Any) -> None:
            if prop == "visible":
                layer.visible = bool(value)  # type: ignore[assignment]
            elif prop == "opacity":
                layer.opacity = float(max(0.0, min(1.0, float(value))))  # type: ignore[assignment]
            elif prop == "blending":
                layer.blending = str(value)  # type: ignore[assignment]
            elif prop == "interpolation":
                layer.interpolation = str(value)  # type: ignore[assignment]
            elif prop == "gamma":
                gamma = float(value)
                if not gamma > 0.0:
                    raise ValueError("gamma must be positive")
                layer.gamma = gamma  # type: ignore[assignment]
                if ctx.visual is not None:
                    ctx.visual.gamma = gamma  # type: ignore[assignment]
            elif prop == "contrast_limits":
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    raise ValueError("contrast_limits must be a length-2 sequence")
                lo = float(value[0])
                hi = float(value[1])
                if hi < lo:
                    lo, hi = hi, lo
                layer.contrast_limits = [lo, hi]  # type: ignore[assignment]
                if ctx.visual is not None:
                    ctx.visual.clim = (lo, hi)  # type: ignore[attr-defined]
            elif prop == "colormap":
                cmap = ensure_colormap(value)
                layer.colormap = cmap  # type: ignore[assignment]
                if ctx.visual is not None:
                    logger.debug(
                        "apply_layer_updates: updating visual colormap to %s", cmap.name
                    )
                    ctx.visual.cmap = _napari_cmap_to_vispy(cmap)
            elif prop == "depiction":
                layer.depiction = str(value)  # type: ignore[assignment]
            elif prop == "rendering":
                layer.rendering = str(value)  # type: ignore[assignment]
            elif prop == "attenuation":
                layer.attenuation = float(value)  # type: ignore[assignment]
            elif prop == "iso_threshold":
                layer.iso_threshold = float(value)  # type: ignore[assignment]
            else:
                raise KeyError(f"Unsupported layer property '{prop}'")

        for props in updates.values():
            for key, val in props.items():
                _apply(key, val)
        ctx.mark_render_tick_needed()

    @staticmethod
    def drain_updates(
        ctx: SceneStateApplyContext,
        *,
        state: RenderLedgerSnapshot,
        mailbox: Optional[RenderUpdateQueue] = None,
        queue: Optional[RenderUpdateQueue] = None,
    ) -> SceneDrainResult:
        """Apply pending scene state updates and report side effects."""

        render_marked = False
        original_mark = ctx.mark_render_tick_needed

        def _mark_render() -> None:
            nonlocal render_marked
            render_marked = True
            original_mark()

        ctx_for_apply = replace(ctx, mark_render_tick_needed=_mark_render)

        render_mailbox = mailbox or queue
        if render_mailbox is None:
            raise ValueError("RenderUpdateQueue instance required")

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

        layer_updates = state.layer_updates or {}
        if layer_updates:
            SceneStateApplier.apply_layer_updates(ctx_for_apply, layer_updates)

        cam = ctx.camera
        if cam is None:
            render_mailbox.update_state_signature(state)
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

        policy_refresh = render_mailbox.update_state_signature(state)

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
