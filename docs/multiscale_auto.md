# Multiscale Pyramid: Auto Level Selection (Design)

This document confirms current multiscale behavior in our forked napari code, and proposes a precise plan to implement automatic multiscale level switching in the CUDA headless server/worker.

## Confirmed Behavior (napari fork in this repo)

Auto level selection for multiscale in 2D already exists in the napari layer stack and is computed per draw. Key references:

- `src/napari/layers/utils/layer_utils.py`
  - `compute_multiscale_level(requested_shape, shape_threshold, downsample_factors)`
  - `compute_multiscale_level_and_corners(corner_pixels, shape_threshold, downsample_factors)`
- `src/napari/layers/base/base.py` in the draw/update path (`_update_draw` equivalent):
  - For 2D + multiscale: computes the axis-aligned data bounding box from camera/canvas corners, then:
    - Calls `compute_multiscale_level_and_corners(...)`
    - If the resulting level or corners change, updates `self._data_level` and `self.corner_pixels` and refreshes the view.
  - For 3D + multiscale: always selects the lowest resolution (`self._data_level = len(self.level_shapes) - 1`).

Effectively:
- 2D: pick the coarsest (highest index) level such that the on‑screen field of view still has ≥ 1 data pixel per screen pixel (with proper corner/crop alignment). This avoids excessive oversampling.
- 3D: default to lowest resolution level.

## Goal

Bring automatic multiscale level switching to the EGL headless server/worker so the encoder streams the appropriate level as the view zooms/pans, and update the dims meta so clients’ HUDs reflect it.

## High‑Level Design

- Decision location: EGL render worker
  - Has direct access to camera (PanZoomCamera or TurntableCamera), canvas size, and current dataset “levels”.
  - Can compute the data field of view in data coordinates per frame, just like napari layers do.
- State reflection: server
  - Server remains the source of truth for `ms_state` presented in `dims.update.meta`.
  - The worker notifies the server when it switches levels, and the server re‑broadcasts updated meta.

## Algorithms

### 2D (PanZoomCamera)

Mirror napari’s logic:
- Compute four canvas corner points in world coordinates, transform to data coordinates.
- Build the axis‑aligned integer data bbox ([min,max] per axis), compute `requested_shape` = max − min.
- Call `compute_multiscale_level(requested_shape, shape_threshold, downsample_factors)` where:
  - `shape_threshold` ≈ canvas size in pixels for the displayed axes.
  - `downsample_factors` comes from NGFF multiscales (per-axis factors), ordered to match the displayed axes.
- If the recommended level differs from current:
  - Respect hysteresis: only switch if sustained for dwell_ms and outside thresholds (below).
  - Switch via `request_multiscale_level(level, path)` and trigger a clean keyframe (reuse server `_ensure_keyframe()`).

### 3D (TurntableCamera)

Two options:
- Minimal parity with our napari fork: keep lowest resolution (coarsest) level in 3D, matching `base.py` behavior.
- Improved (optional) heuristic:
  - Estimate world‑per‑pixel at the view plane from camera distance/FOV:
    - `y_world_ppx = 2*distance*tan(fov/2) / screen_h`
  - Choose level whose XY downsample best matches ~1 data pixel per screen pixel using the same hysteresis/dwell scheme.

Default for first cut: keep lowest resolution in 3D for stability; add the heuristic under an opt‑in env flag.

## Hysteresis and Dwell (to avoid flipping)

- Tunables (env):
  - `NAPARI_CUDA_MS_AUTO_HYST_UP=2.0` (switch coarser when oversampled by >×2)
  - `NAPARI_CUDA_MS_AUTO_HYST_DOWN=0.6` (switch finer when undersampled by <×0.6)
  - `NAPARI_CUDA_MS_AUTO_DWELL_MS=300` (required sustained breach before switching)
  - `NAPARI_CUDA_MS_AUTO_COOLDOWN_MS=500` (no reswitch until cooldown elapses)
  - `NAPARI_CUDA_MS_AUTO_DEBUG=0|1` (reason logs)

## Worker Changes

- Fields
  - `self._ms_levels: list[dict]` (already present upstream via server meta; include `downsample` and `path`).
  - `self._ms_current_level: int` (mirror of server state/current dataset index).
  - Auto control vars: dwell timers, cooldown, thresholds.
- Methods
  - `_maybe_auto_switch_level_2d()`: compute corners from `canvas` + view transforms, then recommended level via `compute_multiscale_level(...)`. Manage dwell/cooldown and call `request_multiscale_level(...)` if needed.
  - Optional `_maybe_auto_switch_level_3d()` when enabled.
  - On applying a switch in `_apply_multiscale_switch(...)`, notify the server (callback) to update `ms_state.current_level` and rebroadcast dims meta.

## Server Changes

- Add a worker→server callback (set once when starting worker) for `on_ms_level_changed(level)` that:
  - Sets `self._ms_state['current_level'] = level; self._ms_state['policy'] = 'auto'`
  - Calls `await self._rebroadcast_meta(last_client_id=None)`
  - Triggers `_ensure_keyframe()` to make the change immediate.
- Expose env flags for auto enable/tuning: `NAPARI_CUDA_MS_AUTO=1`, thresholds above.

## Data Requirements

- NGFF multiscales parsed at server start (already implemented): `levels[{path, downsample[z,y,x]}]`.
- For 2D: `downsample_factors[:, displayed_axes]` ordered to {y,x} for the active dataset.
- For 3D (optional): XY downsample per level; Z ignored for the view sampling decision.

## Metrics & Logging

- Metrics:
  - `napari_cuda_ms_auto_switches` (counter)
  - `napari_cuda_ms_switch_dwell_ms` (histogram)
- Logs (when `NAPARI_CUDA_MS_AUTO_DEBUG=1`):
  - `ms.auto: rec=L2 cur=L1 reason=oversub ratio=2.4 dwell=320ms`.

## Testing / Validation (manual)

- 2D Zarr: zoom in/out across thresholds; verify level switches with dwell and no thrash.
- 3D Zarr: if heuristic off, stays at lowest level; if on, test a few distances and confirm sensible switches.
- Encoder: verify `_ensure_keyframe()` triggers and watchdog cancels after the keyframe.
- HUD: `ms lvl:a/b` updates as levels change.

## Phased Implementation Plan

1) Worker: 2D auto level selection
- Implement `_maybe_auto_switch_level_2d()` called once per frame before encode.
- Use napari’s `compute_multiscale_level(...)` (copy/adapt small helper if needed) with proper axis order.
- Add dwell/cooldown gates and call `request_multiscale_level()` if needed.

2) Server sync + IDR
- Add worker→server level changed callback; server updates meta and calls `_ensure_keyframe()`.

3) Envs + metrics + debug logs
- Wire env flags, counters, and concise rationale logs.

4) Optional 3D auto (behind flag)
- Implement distance/FOV heuristic; keep default disabled for the first release to match napari’s behavior.

5) Prefetch & budget (later)
- Preload neighbor levels (±1) with dask when cheap; enforce VRAM budget.

## Open Questions

- 3D default policy: stick to lowest level (parity) vs. heuristic by default?
- CLIM/colormap continuity: we preserve current behavior by reusing cached CLIM per dataset; OK for now.
- Axis conventions: ensure NGFF axes mapping (z,y,x) → displayed dims is handled consistently for downsample vectors.

