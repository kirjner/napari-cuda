# Layer Construction in the CUDA Server (Design)

This document outlines how we will introduce a minimal yet extensible
"layer" concept to the EGL headless server/worker so we can evolve from a
single scalar field to a small, well‑structured stack of layers compatible
with multiscale, 2D/3D, and our existing control protocol.

## Goals

- Support constructing image/volume layers from OME‑Zarr (and synthetic) sources.
- Keep per‑layer state (visibility, opacity, colormap/rendering, transforms).
- Own multiscale state per layer (levels, current_level, policy) and report via dims meta.
- Maintain good performance: one render pass per frame; no extra readbacks.
- Preserve existing protocol; add optional layer management intents incrementally.

## Terminology

- Layer: a logical drawable entity (Image2D or Volume3D) with its own properties.
- Visual: the actual VisPy node in the worker (`ImageVisual` or `Volume`).
- Source: backing data (Zarr array/group; synthetic generator).

## Architecture

- Server (authoritative state + protocol)
  - Maintains `layers: Dict[str, LayerSpec]` keyed by `layer_id`.
  - Each `LayerSpec` captures: `type`, `source`, `zarr_path`, `axes`, `levels`,
    `current_level`, `policy`, `visible`, `opacity`, `colormap` or `render`.
  - Dims meta aggregates a summary across layers and includes multiscale per layer.
  - For now, a single `layer_id='main'` preserves current behavior; more layers can be added later.

- Worker (render + IO)
  - Holds `LayerRuntime` records mirroring `LayerSpec`.
  - Owns one VisPy visual per layer and composes them in deterministic z‑order.
  - Performs per‑layer multiscale selection/prefetch and applies changes on the render thread.

- Composition
  - Initial phase: single layer (no explicit composition changes).
  - Next: ordered list of layers rendered back‑to‑front using VisPy blending.

## LayerSpec (server model)

Minimal fields (extensible):
- id: str (stable key; e.g., 'main')
- kind: 'image2d' | 'volume3d' (matches worker visual type)
- source:
  - zarr_root: str
  - dataset_path: str (e.g., 'level_03' or inferred)
  - axes: str (e.g., 'zyx')
- multiscale: { levels: [{path, downsample[z,y,x]}], current_level: int, policy: 'auto'|'fixed' }
- style (image): { colormap: str, clim: [lo,hi], opacity: float }
- style (volume): { mode: 'mip'|'translucent'|'iso', opacity: float, sample_step: float, colormap?: str, clim?: [lo,hi] }
- flags: { visible: bool }

## LayerRuntime (worker model)

- id, kind, zarr handles, dask arrays (lazy), visuals (Image or Volume), cached CLIM.
- multiscale cache and current dataset selection.
- Methods:
  - `load_zarr(level_path)`, `apply_multiscale_switch(level_idx, path)`
  - `apply_style(style_dict)`, `set_visible(bool)`, `set_opacity(float)`

## Protocol (incremental)

We will add optional intents. Existing single‑layer flows continue to work.

- Add layer (optional for future multi‑layer)
  - `layer.intent.add { id, kind, zarr_root, axes, dataset?:path }`
- Remove layer (future)
  - `layer.intent.remove { id }`
- Update layer style (reuses current volume intents for the main layer; generalized later)
  - `layer.intent.set_style { id, style:{...} }`
- Multiscale (generalized)
  - `multiscale.intent.set_policy { id, policy }`
  - `multiscale.intent.set_level { id, level }`

For this phase, the server will map all current `volume.intent.*` and `multiscale.intent.*`
commands to `id='main'`.

## Dims Meta Extensions

- Add a `layers` array to `dims.update.meta`, each entry summarizing per‑layer
  multiscale and style. Example (single layer shown):

```
meta: {
  ndisplay: 3,
  volume: true,
  layers: [
    {
      id: 'main', kind: 'volume3d',
      multiscale: { levels:[...], current_level: 2, policy: 'fixed' },
      render: { mode:'mip', colormap:'grays', clim:[0.02,0.98], opacity:0.6 }
    }
  ]
}
```

Clients can ignore `layers` if not needed; HUD may show a compact summary.

## Multiscale Integration

- Each layer owns its multiscale levels and policy; worker auto‑selection runs per layer.
- Server aggregates `ms_state` for the default layer into top‑level meta for backward compatibility.

## Phased Rollout

1) Single‑layer refactor (no behavior change)
- Introduce `LayerSpec`/`LayerRuntime` with id='main'.
- Move current zarr/visual creation under layer construction helpers.
- Keep existing intents; map to id='main'.
- Keep dims meta top‑level fields; optionally add `layers[0]` mirror.

2) Multiscale auto (2D) per layer
- Implement per‑layer auto logic and server sync (see `docs/multiscale_auto.md`).
- Trigger keyframe on switch.

3) Optional: multi‑layer
- Add add/remove layer intents; maintain z‑order and visibility.
- Compose visuals using blending; re‑broadcast meta with `layers` entries.

4) Ergonomics & metrics
- Track per‑layer switches, memory, timing.
- Add concise logs on layer changes.

## UI Integration (Client/Dashboard)

The EGL headless server does not host the Qt Layer list UI. We will surface
equivalent controls via the client and (optionally) the Dash dashboard:

- Client (Qt) overlay controls
  - Minimal layer HUD: list of layers with visibility toggle and active marker.
  - Shortcuts (when multi‑layer enabled):
    - Next/prev layer selection, toggle visibility, and quick opacity up/down.
  - Wire to existing intents (`layer.intent.set_style`, `multiscale.intent.*`).

- Dashboard (Dash) controls
  - Add a “Layers” panel with a table:
    - id, kind, visible (toggle), opacity (slider), colormap/render mode (dropdown),
      ms level (slider) and policy (auto/fixed toggle).
  - Interact by issuing the same state intents to the server.

- Backward compatibility
  - For single‑layer (current default), keep the existing HUD; show a compact
    summary of layer state without requiring extra UI.

## Implementation Notes

- Keep server code DRY: centralize zarr multiscale parsing into a helper used by layer construction.
- Ensure all mutations (style, level, visibility) coalesce into the worker render thread via pending state.
- Guard GPU memory: volume loads must respect VRAM budget; fallback to coarser level with clear logs.
