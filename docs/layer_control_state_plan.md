# Layer Control State Plan

This document captures the ongoing work to make layer intents (opacity, visibility, blending, interpolation, etc.) share a single source of truth across the headless server. The initial implementation (LayerControlState + mirrored controls payload) landed in commit `b2caf6e7`; remaining steps below track the final cleanup.

## Motivations

- **Stability:** Layer opacity currently reverts to legacy defaults when the worker refreshes the napari adapter layer. This is caused by scattered state (extras vs render hints) and defensive resets inside `SceneStateApplier`.
- **Consistency:** Volume and planar pipelines expose the same surface to the user yet write to different payload slots, leading to subtle client regressions (e.g., the bridge seeing 1.0 in `render.opacity`).
- **Reuse:** Future intent surfaces (MCP tools, dashboards) need a well-defined structure rather than ad-hoc dicts.

## Proposed structure

1. **LayerControlState dataclass** ✅
   - Lives in `server/scene/data.py` and captures canonical properties (`visible`, `opacity`, `blending`, `interpolation`, `gamma`, `colormap`, `contrast_limits`, etc.).
   - `ServerSceneData` stores `layer_controls: dict[layer_id, LayerControlState]`.

2. **Intent helpers** ✅
   - `control/state_reducers.py` mutates `layer_controls` (instead of raw dicts) and records deltas in `ServerSceneData.pending_layer_updates` for the worker ingest path.

3. **Worker application** ✅
   - `runtime/scene_state_applier.py` pulls values from the control state when mutating the napari adapter layer (planar) or volume visual (3D), eliminating hard-coded resets.

4. **Snapshot emission** ✅
   - Layer snapshot blocks (the payload of `notify.scene`) now include a dedicated `controls` map plus a lean `source` section; transport metadata (source path, volume flags, policy metrics) lives under the scene-level `metadata` block.
   - Renderer-only toggles (shade/sample-step) stay under the `render` mapping so UI controls are not duplicated.

5. **ViewerSceneManager** ✅
   - `update_from_sources` receives the control state and merges it before querying the napari adapter, guaranteeing the emitted `notify.scene` snapshot reflects the canonical values even if the worker has not yet mutated the layer object.

6. **Testing**
   - ✅ Regression coverage added for intent helpers and spec builders; manual smoke confirms opacity persists.
   - ⬜ Add mixed-mode test (2D → 3D) to ensure the control state flows into both planar and volume payloads once the client surface lands.

## Client considerations

- Clients should consume the `controls` map exclusively; the legacy `extras` bag has been deleted and `render` only carries renderer-specific hints.
- Remote layer mirrors must apply acknowledgements from `controls` (opacity, visibility, rendering, etc.) and ignore the deprecated paths.
- The bridge can treat `controls` as authoritative and drop the fallback normalisation logic once the corresponding client patch lands.
- Colormap intents deliver `name` strings that must match napari’s registered palette names; the server normalises case but no longer honours the legacy `colormap` field.

## Next steps

1. Implement `LayerControlState` and migrate intent helpers to it. ✅
2. Update `SceneStateApplier` / `ViewerSceneManager` to consume the canonical bag. ✅
3. Rename `extras` → `controls` in `LayerSpec` and strip mirrored fields. ✅
4. Roll client changes so the bridge and registry rely solely on the new map. ✅
