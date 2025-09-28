Napari Level Selector (Compute-Only) — Plan

Purpose
- Make napari the authority for 2D multiscale level choice without changing our render/ROI/encode path.
- Keep our stabilizer (directional thresholds + cooldown) and budget clamps.
- Gate the change for easy A/B and safe rollback.

Scope (Small/Medium)
- Compute-only: call napari’s level-selection math; do not add a live multiscale layer, do not change our ROI/rendering.
- 2D only. 3D selection stays ours (future sprint) — napari forces coarsest level in 3D.

Design Overview
- Inputs
  - Viewport in data coords (YX) at full resolution.
  - Per-level downsample factors derived from `ZarrSceneSource.level_scale(level)` relative to level 0.
  - Canvas pixel size as `shape_threshold`.
- Compute
  - Use napari’s `compute_multiscale_level(_and_corners)` to get desired `data_level`.
  - Reference: `src/napari/layers/utils/layer_utils.py` (compute_multiscale_level_and_corners and compute_multiscale_level).
- Apply
  - Feed desired level through our stabilizer (hysteresis thresholds + cooldown), enforce budgets, then apply using existing path:
    - Proportional Z remap (prev depth → new depth)
    - Set `viewer.dims.range` before `viewer.dims.current_step`
    - ROI fetch and placement via `layer.scale` + `layer.translate`

Implementation Steps
1) Helper: `lod.compute_napari_level(viewer, source) -> int`
   - Map current viewport corners (canvas/world) to data coordinates at full-res using the vispy transform and level-0 scale on Y/X.
   - Build `downsample_factors[level] = level_scale(level)/level_scale(0)` on displayed axes.
   - Use current canvas size as `shape_threshold`.
   - Call napari’s compute and return desired level.

2) Selection gate in worker
- In `render_worker._evaluate_level_policy`, when `NAPARI_CUDA_USE_NAPARI_LEVEL=1`:
     - `desired = lod.compute_napari_level(self._viewer, source)`
     - `selected = lod.stabilize_level(desired, current, hysteresis)`
     - Apply cooldown; enforce budgets; apply level via existing code.
     - Log one concise line: `ms.switch … reason=napari-compute desired=X applied=Y`.
   - Else (default): keep current oversampling heuristic.

3) Telemetry (optional but recommended)
   - Track agreement (napari desired vs applied after clamp) over last 100 decisions.
   - Expose counters in `/metrics.json` and emit a single INFO if disagree ≥ threshold (e.g., ≥10/100).

Files to Touch
- `src/napari_cuda/server/lod.py`: add `compute_napari_level`; keep selector helpers (`LevelPolicy*`, `select_level`) co-located to avoid duplicate modules.
- `src/napari_cuda/server/render_worker.py`: wire gate in `_evaluate_level_policy` (no change to apply path).

Gating / Rollout
- Enable: `export NAPARI_CUDA_USE_NAPARI_LEVEL=1` (default off).
- On compute errors, fall back to heuristic for that decision and log once.

Validation Plan
- A/B live on real data (gate on/off):
  - No oscillation near boundaries (cooldown holds; thresholds honored).
  - Comparable or improved switch cadence.
  - Z slider and pixels remain in sync (unchanged apply path).
  - Agreement ≥ 95% over a session.
- Logs: look for `ms.switch: … reason=napari-compute …` and low disagreement rate.

Future Work (not in this sprint)
- Remove heuristic path entirely once napari selector is stable (C6i). Keep stabilizer/cooldown/budgets.
- Optional guard-band in render ROI to reduce edge sensitivity (chunk-aligned outward expansion; env-tunable).
- 3D selection: add `compute_volume_level` that uses YX density + volume budgets (napari remains 2D-only advisor).

Risks & Mitigations
- Transform mismatch: use the same vispy view/scene transform path we rely on for ROI; test with anisotropic scales.
- Axis labeling/ordering: derive displayed axes from `source.axes`; map only Y/X for compute.
- Hidden IO risk: avoided by compute-only wrapper (no multiscale layer added).

Decision to Proceed
- Default keep-off gate minimizes risk. If no clear win on your dataset, leave off and defer removal of heuristic.
