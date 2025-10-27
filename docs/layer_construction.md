# Layer Construction Notes

The client no longer hydrates legacy `LayerSpec` dataclasses. Instead, the
control channel delivers authoritative layer snapshots via `notify.scene`
frames. The flow is:

1. **Server** (`snapshot_scene` helpers)
   - Builds a `SceneSnapshot` whose `layers` tuple contains JSON-safe blocks for
     each remote layer. These blocks include shape, axis labels, multiscale
     metadata, controls, and a compact `source` section when the worker exposes
     dataset identifiers.

2. **Client registry** (`RemoteLayerRegistry`)
   - Caches the raw layer blocks from the snapshot.
   - Instantiates `RemoteImageLayer` directly with
     `RemoteImageLayer(layer_id=..., block=block)`.
   - Applies deltas by mutating the cached block and calling
     `RemoteImageLayer.update_from_block()`.

3. **Remote layer** (`RemoteImageLayer`)
   - Keeps a copy of the last block in `_remote_block`.
   - Reuses existing napari infrastructure by configuring itself from the block
     (axis labels, contrast limits, multiscale levels, source data, etc.) and by
     constructing lightweight `RemoteArray`/`RemoteMultiscale` shims for napari’s
     data protocol.
   - Control updates (opacity, visibility, gamma, etc.) come straight from the
     `controls` mapping embedded in the block.

This removes the need for the historical `LayerSpec`/`SceneSpec` dataclasses and
avoids double-serialization or speculative fallbacks.
