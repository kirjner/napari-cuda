Per-Function Contracts (Server & Worker)

Conventions
- Preconditions: must hold on entry; handlers validate at the boundary and reject if not met.
- Postconditions: state changes and outputs guaranteed on success.
- Side effects: ledger writes, notify broadcasts, render enqueues.
- Error handling: no try/except in reducers/txns; handlers catch and map to protocol errors with logging.
- Schema alignment: as we roll out the `view / axes / index / lod / camera`
  ledger design, reducers must dual-write the new scopes while they continue to
  serve legacy `dims_spec` + `current_step` consumers. This doc flags those
  transitional writes explicitly.

1) Control: State Update Handlers (state_update_handlers)

- handle_dims_update(ctx)
  - Preconditions: target axis provided; key in {'index','step','margin_left','margin_right'}
    (where 'step' is the legacy alias for index); values typed (int for
    index; float for margins).
  - Postconditions: calls reduce_dims_update; acks with applied value and version; logs summary at INFO/DEBUG.
  - Side effects: ledger writes (dims_spec/current_index — legacy scope
    `current_step`), mirrors may broadcast notify.dims.
  - Errors: reject with state.invalid on type/unsupported key; state.error on reducer failure.

- handle_view_ndisplay(ctx)
  - Preconditions: ctx.value coercible to int; 2D vs 3D resolved to {2,3+}.
  - Postconditions: calls reduce_view_update; acks with new ndisplay and version; enqueues render update with snapshot + plane/volume state; schedules keyframe.
  - Side effects: ledger writes (dims_spec); runtime enqueue; notify from mirrors.
  - Errors: reject on invalid type/value or reducer failure.

- handle_camera_pan/orbit/zoom/reset(ctx)
  - Preconditions: required deltas provided and typed (floats); target camera is supported.
  - Postconditions: acks with the delta and sequence; enqueues CameraDeltaCommand to worker queue.
  - Side effects: none on ledger; notify.camera broadcast of deltas for clients.
  - Errors: reject on invalid payload.

- handle_camera_set(ctx)
  - Preconditions: pose components provided (subset of center/zoom/angles/rect).
  - Postconditions: calls reduce_camera_update; acks with applied components and version; may broadcast notify.camera(state).
  - Side effects: ledger writes to camera_plane/* or camera_volume/*.
  - Errors: reject on invalid payload or reducer failure.

- handle_layer_property(ctx)
  - Preconditions: layer id exists; prop/value valid per layer type.
  - Postconditions: calls reduce_layer_property; acks; notify.layers delta broadcast.
  - Side effects: ledger writes under ('layer', id, prop).
  - Errors: reject on invalid input.

2) Control: Reducers (state_reducers)

- reduce_dims_update(ledger, axis, prop, value|step_delta, intent_id?, timestamp?, origin)
  - Preconditions: axis resolvable; types per prop; index tuple (legacy "step")
    clamped to level shape.
  - Postconditions: versioned writes to dims_spec + current_index (legacy
    key `current_step`) and per-axis index; plane state mirrored in 2D; ActiveView updated only when level changes.
  - Side effects: none beyond ledger writes.

- reduce_view_update(ledger, ndisplay, order?, displayed?, intent_id?, timestamp?, origin)
  - Preconditions: ndisplay ∈ {2,3+}; order/displayed form consistent subsets.
  - Postconditions: versioned dims_spec with updated ndisplay/order/displayed; plane/volume states snapshot for restore; ActiveView written.

- reduce_level_update(ledger, level, index_tuple?, level_shape?, shape?, applied?, intent_id?, timestamp?, origin, mode?, plane_state?, volume_state?, preserve_step=False)
  - Preconditions: level resolvable; index tuple (legacy step) up to ndim; shapes valid if provided.
  - Postconditions: dims_spec current_level/current_index (legacy `current_step`) updated; level_shapes patch optional; ActiveView written with provided/derived mode.

- reduce_plane_restore / reduce_volume_restore
  - Preconditions: pose components typed; level valid.
  - Postconditions: respective camera_* scoped writes; (plane) dims_spec level/index updated; ActiveView written.

- reduce_camera_update(ledger, center|zoom|angles|distance|fov|rect, ...)
  - Preconditions: at least one component provided.
  - Postconditions: versioned camera_plane/* or camera_volume/* writes; returns ack dict and version; no dims/ActiveView mutation.

- reduce_layer_property(ledger, layer_id, prop, value, ...)
  - Preconditions: layer id exists; value typed per prop.
 - Postconditions: versioned ('layer', id, prop) write.

3) Control: Transactions (control/transactions)

- apply_dims_step_transaction(...)
  - Preconditions: serialized dims_spec payload valid; index tuple (legacy step) normalized.
  - Postconditions: writes dims_spec/current_index (legacy scope `current_step`) and returns versioned entries.

- apply_view_toggle_transaction(...), apply_level_switch_transaction(...)
  - Postconditions: write dims_spec and any related keys; return versioned entries.

- apply_plane_restore_transaction(...), apply_volume_restore_transaction(...)
  - Postconditions: write camera_* scoped entries atomically.

  - apply_camera_update_transaction(ledger, updates, origin, ts, op_seq, op_kind)
  - Preconditions: updates contain (scope,target,key,value[,metadata]) tuples; 4–6 tuple entries accepted.
  - Postconditions: writes all camera_* entries; dedupe same value+metadata.

4) Control: ActiveView

- apply_active_view_transaction(ledger, mode, level, origin, timestamp)
  - Preconditions: mode in {'plane','volume'}; level is int >= 0.
  - Postconditions: writes ("viewport","active","state") = {mode, level}; no other keys mutated.

5) Scene Builders (scene/builders.py)

- snapshot_render_state(ledger) -> RenderLedgerSnapshot
  - Preconditions: ledger contains ('dims','main','dims_spec').
  - Postconditions: returns snapshot with dims spec, camera poses (optional), layer visuals, active view (optional), version gates.
  - Side effects: none.

- build_ledger_snapshot(ledger) (internal)
  - Preconditions: ledger snapshot dict provided.
  - Postconditions: assembles RenderLedgerSnapshot; no recomputation beyond direct transforms.

- pull_render_snapshot(server)
  - Preconditions: server state lock acquired internally.
  - Postconditions: returns current RenderLedgerSnapshot; no mutation.

6) Worker Call Stack (final form)

- Worker tick (render loop):
  1. Observe `scene.main.op_seq` bump via lightweight “poke” (no RenderUpdate mailbox).
  2. Pull `{view, axes, index, lod, camera, layers}` scopes directly from the ledger.
  3. Apply blocks selectively:
     - `apply_dims_block(view, axes, index)` → set viewer dims, z-index, ROI hints.
     - `apply_camera_block(camera)` → set plane/volume cameras, emit pose if requested.
     - `apply_layers_block(layers)` → patch visuals only for changed props.
  4. Emit intents back through reducers:
     - `worker._emit_current_camera_pose` → `reduce_camera_update`.
     - `WorkerIntentMailbox.enqueue_level_switch` → `reduce_level_update`.
  5. Record per-block signatures locally to skip re-apply when unchanged.
- No RenderUpdate mailbox, planner, or viewport caches remain—the ledger/tick pointer is the only coordination mechanism.

7) Notify Broadcasters (control/notify)

- broadcast_dims_state(server, payload, intent_id?, timestamp?)
  - Preconditions: NotifyDimsPayload contains embedded dims_spec and consistent top-level fields.
  - Postconditions: sends notify.dims to all state clients with feature enabled; INFO logs summary when enabled.

- broadcast_camera_update(server, mode, delta?, state?, intent_id, origin, timestamp?)
  - Preconditions: at least one of delta or state present.
  - Postconditions: sends notify.camera to clients; no ledger changes.

- broadcast_layers_delta(server, ...)
  - Postconditions: sends notify.layers delta for changed sections.

- broadcast_level(server, payload, intent_id?, timestamp?)
  - Preconditions: NotifyLevelPayload contains current_level; payload is sourced exclusively from ActiveView.
  - Postconditions: sends resumable notify.level to clients; no dims/camera/ledger changes.

- start_worker(server, loop, state)
  - Preconditions: no live worker thread.
  - Postconditions: spawns EGL renderer thread; wires callbacks:
    - worker tick observes `scene.main.op_seq`, pulls new ledger scopes, and applies them block-by-block.
    - level_intent_cb → WorkerIntentMailbox.enqueue_level_switch → server `_handle_worker_level_intents` (loopback through reducers).
    - camera_pose_cb → server `_apply_worker_camera_pose` (reduce_camera_update).
  - Side effects: attaches ledger to worker; initializes scene; starts pixel encoding callbacks.

- stop_worker(state)
  - Postconditions: signals stop; joins thread; clears references.

- server._handle_worker_level_intents()
  - Preconditions: WorkerIntentMailbox has a LevelSwitchIntent.
  - Postconditions: calls reduce_level_update(...) to commit level/shape/state; worker notices the `scene.main.op_seq` bump and rehydrates from the ledger on the next render tick.
  - Side effects: ledger writes; pixel channel enqueue; mirrors may broadcast notify.

- server._apply_worker_camera_pose(pose)
  - Preconditions: pose contains applied camera components.
  - Postconditions: calls reduce_camera_update(...) with origin='worker.state.camera'; broadcasts notify.camera(state).
  - Side effects: ledger writes; no dims changes.

- WorkerIntentMailbox.enqueue/pop_level_switch, enqueue/pop_thumbnail_capture
  - Postconditions: latest-wins storage for intents; pop clears the stored value; thread-safe.

9) Ledger (server/ledger.py)

- ServerStateLedger.record_confirmed(scope, target, key, value, origin, timestamp?, metadata?, version?, dedupe=True)
  - Preconditions: mapping types for metadata; version None or int; lock held internally.
  - Postconditions: version bump (or dedupe); logs summary at INFO; notifies per-key and global subscribers.

- ServerStateLedger.batch_record_confirmed(entries, origin, timestamp?, dedupe=True)
  - Preconditions: entries length 4–6 as tuples; metadata optional.
  - Postconditions: writes multiple entries atomically; INFO logs with value summarization.

Notes
- Notify (state channel) is emitted as a consequence of ledger commits. Render (pixel channel) pulls snapshots from the ledger and never emits notify.
