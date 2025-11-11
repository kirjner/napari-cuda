Reducers & Transactions (Cut-Down)

Dims/View/Level
- `reduce_dims_update(axis, prop, value|step_delta)`
  - Index (legacy "step"): clamp against `level_shapes[current_level][axis]`; write
    `dims_spec` and the ledger `current_index` scope (legacy key `current_step`).
  - Margins: set `margin_left/right_*` for axis (world + steps); no camera changes.
- `reduce_view_update(ndisplay, order?, displayed?)`
  - Toggle ndisplay; set order/displayed; snapshot minimal plane/volume state as needed.
- `reduce_level_update(level, index_tuple?)`
  - Set `current_level`; optionally remap/clamp the index tuple (legacy step) and
    update `level_shapes` if shape provided.

Camera
- `reduce_camera_update(center|zoom|angles|distance|fov|rect)`
  - Write `camera_plane/*` or `camera_volume/*` on applied pose (worker origin only); ack with applied values.
  - No dims/ActiveView writes.

Layer
- `reduce_layer_property(layer_id, prop, value)` writes to `('layer', id, prop)`; notify.layers consumes these.

Transactions (scoped)
- `apply_dims_step_transaction(ledger, step, dims_spec_payload, origin, ts, op_seq, op_kind)`
- `apply_view_toggle_transaction(ledger, op_seq, target_ndisplay, order_value, displayed_value, dims_spec_payload, origin, ts)`
- `apply_level_switch_transaction(ledger, op_seq, level, step? [index tuple], dims_spec_payload, origin, ts)`
- `apply_plane_restore_transaction(...)`, `apply_volume_restore_transaction(...)`
- `apply_camera_update_transaction(ledger, updates[(scope,target,key,value,metadata?)...], origin, ts, op_seq, op_kind)`

Lean Rules
- Reducers construct canonical payloads; txns only write keys.
- No intent blobs; ActiveView is the only cross-cutting key (mode+level).
- No try/except in reducers; prevalidate and raise; handlers translate to protocol errors.
