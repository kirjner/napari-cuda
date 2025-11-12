Architecture Design Doc (ADD): napari-cuda Scene State and Runtime

Status: Draft (baseline pre-viewport-intent)
Owners: Server Runtime Team, Client Runtime Team
Last updated: 2025-11-09

1. Purpose and Scope
- Define the canonical architecture for scene state: what we store, what we send, and how we apply it at runtime.
- Replace grown patterns with explicit contracts and shared helpers.
- Provide a safe foundation for Layer Parity (multi-layer, per-axis/per-layer metadata) without coupling or regressions.

Non-goals
- Not a UI spec. HUD/UX follows from data contracts defined here.
- Not a protocol rewrite. We reuse existing message types, extending payloads where needed.

2. Principles (Substance + Style)
- Single source of truth per concern: ledger for committed state, in-memory channels for deltas.
- Modular blocks: dims, camera pose, layers, ROI handled independently end-to-end.
- Explicit contracts: typed schemas + shared serializers/deserializers for every block.
- Deterministic apply: never replay unrelated blocks (camera update must not reapply dims).
- Offensive coding practices (must):
  - No try/except in hot paths; catch only at subsystem boundaries, always log with logger.exception.
  - No dynamic attribute access; accessors/types explicit.
  - No None checks where invariants guarantee presence; use assertions to enforce invariants.
  - Defensive guards only at I/O boundaries (websocket, FFmpeg/PyAV, VT shims) with proper finally lifecycles.
  - Absolute imports only (relative imports banned by config).
  - Public APIs fully typed; 4-space indentation; UTF-8; ruff-enforced lint/format.

3. Data Model (Canonical Blocks)
3.1 DimsSpec (authoritative dims)
*Naming note*: the view/axes/lod redesign renames our multi-dimensional cursor
field to `current_index`. Legacy code and ledger scopes still emit
`current_step`; until the implementation lands, this doc references
`current_index` (legacy `current_step`) to make the transition explicit.
- Fields per axis: index, label, role, displayed, order_position,
  current_index (legacy `current_step`), per_level_steps, per_level_world
  (start, stop, step), margins (left/right in steps and world units).
- Global fields: version, ndim, ndisplay, order, displayed, current_level,
  current_index (legacy `current_step`), level_shapes, plane_mode, labels,
  levels (opaque metadata).
- Schema lives in code (src/napari_cuda/shared/dims_spec.py). The ADD requires that any new dims attribute be added here first and flow end-to-end.

Dataclasses (reference implementation):
```
from dataclasses import dataclass
from typing import Mapping, Tuple

@dataclass(frozen=True)
class AxisExtent:
    start: float
    stop: float
    step: float

@dataclass(frozen=True)
class DimsSpecAxis:
    index: int
    label: str
    role: str
    displayed: bool
    order_position: int
    current_index: int  # legacy field name `current_step`
    margin_left_steps: float
    margin_right_steps: float
    margin_left_world: float
    margin_right_world: float
    per_level_steps: Tuple[int, ...]
    per_level_world: Tuple[AxisExtent, ...]

@dataclass(frozen=True)
class DimsSpec:
    version: int
    ndim: int
    ndisplay: int
    order: Tuple[int, ...]
    displayed: Tuple[int, ...]
    current_level: int
    current_index: Tuple[int, ...]  # legacy payload key `current_step`
    level_shapes: Tuple[Tuple[int, ...], ...]
    plane_mode: bool
    axes: Tuple[DimsSpecAxis, ...]
    levels: Tuple[Mapping[str, object], ...]
    labels: Tuple[str, ...] | None
```

Serializers (must exist and be reused):
```
def dims_spec_to_payload(spec: DimsSpec) -> dict[str, object]: ...
def dims_spec_from_payload(data: Mapping[str, object]) -> DimsSpec: ...
```

3.2 View/Axes/Index/Lod Blocks (target schema)
- Purpose: replace the monolithic DimsSpec for runtime consumption while
  maintaining compatibility with legacy payloads during the migration window.
- Status (2025-11-10): reducers now dual-write these scopes under the
  `NAPARI_CUDA_ENABLE_VIEW_AXES_INDEX` flag (`scene/blocks/__init__.py`), so the
  ledger always has typed `ViewBlock`, `AxesBlock`, `IndexBlock`, `LodBlock`, and
  `CameraBlock` records whenever the flag is set. Control‑channel tests
  (`test_state_channel_updates.py`) run with the flag on/off to guarantee notify
  payload parity while we stage the consumer flip.
- Scopes (ledger keys names TBD until code lands):
  - `view.main.state`: `{mode: 'plane'|'volume', displayed_axes: tuple[int, ...], ndim: int}`
  - `axes.main.state`: tuple of axis records with `{axis_id, label, role, world_extent, margins}`
  - `index.main.cursor`: canonical multi-dimensional index tuple (what used to be `current_step`)
  - `lod.main.state`: `{level: int, roi: tuple[int, ...] | None, policy: str | None}`
  - `camera.main.state`: plane/volume pose metadata (mirrors `camera_plane`/`camera_volume` but unified)
- Target dataclasses (illustrative):
```
@dataclass(frozen=True)
class ViewBlock:
    mode: Literal["plane", "volume"]
    displayed_axes: Tuple[int, ...]
    ndim: int

@dataclass(frozen=True)
class AxisBlock:
    axis_id: int
    label: str
    role: str
    world_extent: AxisExtent
    margin_left_world: float
    margin_right_world: float
    displayed: bool

@dataclass(frozen=True)
class AxesBlock:
    axes: Tuple[AxisBlock, ...]

@dataclass(frozen=True)
class IndexBlock:
    value: Tuple[int, ...]

@dataclass(frozen=True)
class LodBlock:
    level: int
    roi: Tuple[int, ...] | None
    policy: str | None
```
- Implementation plan:
  1. Dual-write these payloads whenever reducers currently mutate `dims_spec`.
  2. Worker call stack simplifies to:
     - Control reducer writes new scopes + bumps `scene.main.op_seq`.
     - Server pokes worker (tick) → worker reads `{view, axes, index, lod, camera, layer}` scopes directly from the ledger.
     - Worker applies each block in place (dims, camera, layers) based on per-block signatures; no planner/mailbox/cache.
     - Worker emits camera poses and level intents back through reducers (via `reduce_camera_update` / worker intent mailbox).
  3. Update snapshot builders and notify serializers to read the new scopes first,
     falling back to `dims_spec` for legacy consumers only.
  4. Once worker/apply/mirrors rely solely on the new scopes, remove the legacy
     `current_step` mirrors and shrink DimsSpec to archival/compat use only.

3.3 PlanePose / VolumePose (camera)
- Plane: rect (L,B,W,H), center (Y,X), zoom.
- Volume: center (Z,Y,X), angles (az, el, roll), distance, fov.

Dataclasses (server/scene/viewport.py):
```
@dataclass
class PlanePose:
    rect: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    zoom: float | None

@dataclass
class VolumePose:
    center: tuple[float, float, float] | None
    angles: tuple[float, float, float] | None
    distance: float | None
    fov: float | None
```

3.3 LayerVisuals
- Typed per layer type; the builders must enumerate keys per type in one place (see 6. Snapshot Builders).

3.4 ActiveView (new)
- Ledger key viewport/active:state with {mode: plane|volume, level: int}. Single authoritative indicator of what’s live.
```
@dataclass(frozen=True)
class ActiveViewState:
    mode: str  # 'plane' | 'volume'
    level: int
```

4. Ledger Model (Authoritative, Committed Only)
- Dims: ("dims","main","dims_spec") holds the canonical DimsSpec payload;
  ("dims","main","current_index") mirrors the multi-dimensional index
  (legacy scope `"current_step"` until the rename lands); per-axis index entries
  are optional.
- Camera: ("camera_plane","main", {center,zoom,rect}), ("camera_volume","main", {center,angles,distance,fov}).
- Plane/Volume State: minimal state needed for restore; request/applied mirrors are deprecated once ActiveView exists.
- ActiveView: ("viewport","active","state") published by reducers whenever mode/level changes. [Implemented]
- Versions: per key; transactions must return versioned entries.

5. Reducers and Transactions (Function Signatures)
Module: `src/napari_cuda/server/control/state_reducers.py`
```
def reduce_dims_update(
    ledger: ServerStateLedger,
    *,
    axis: object,
    prop: str,  # 'index' | 'margin_left' | 'margin_right' | ...
    value: object | None = None,
    step_delta: int | None = None,
    intent_id: str | None = None,
    timestamp: float | None = None,
    origin: str = 'control.dims',
) -> ServerLedgerUpdate: ...

def reduce_view_update(
    ledger: ServerStateLedger,
    *,
    ndisplay: int | None = None,
    order: tuple[int, ...] | None = None,
    displayed: tuple[int, ...] | None = None,
    intent_id: str | None = None,
    timestamp: float | None = None,
    origin: str = 'control.view',
) -> ServerLedgerUpdate: ...

def reduce_level_update(
    ledger: ServerStateLedger,
    *,
    level: int | None = None,
    step: tuple[int, ...] | None = None,
    level_shape: tuple[int, ...] | None = None,
    shape: tuple[int, ...] | None = None,
    applied: Mapping[str, object] | object | None = None,
    intent_id: str | None = None,
    timestamp: float | None = None,
    origin: str = 'control.multiscale',
    mode: RenderMode | str | None = None,
    plane_state: PlaneViewportCache | Mapping[str, object] | None = None,
    volume_state: VolumeViewportCache | Mapping[str, object] | None = None,
    preserve_step: bool = False,
) -> ServerLedgerUpdate: ...

def reduce_plane_restore(
    ledger: ServerStateLedger,
    *,
    level: int,
    step: tuple[int, ...],
    center: tuple[float, float],
    zoom: float,
    rect: tuple[float, float, float, float],
    intent_id: str | None = None,
    timestamp: float | None = None,
    origin: str = 'control.view',
) -> dict[PropertyKey, LedgerEntry]: ...

def reduce_volume_restore(
    ledger: ServerStateLedger,
    *,
    level: int,
    center: tuple[float, float, float] | None = None,
    angles: tuple[float, float, float] | None = None,
    distance: float | None = None,
    fov: float | None = None,
    intent_id: str | None = None,
    timestamp: float | None = None,
    origin: str = 'control.view',
) -> dict[PropertyKey, LedgerEntry]: ...

def reduce_camera_update(
    ledger: ServerStateLedger,
    *,
    center: tuple[float, ...] | None = None,
    zoom: float | None = None,
    angles: tuple[float, ...] | None = None,
    distance: float | None = None,
    fov: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
    timestamp: float | None = None,
    origin: str = 'control.camera',
    metadata: Mapping[str, object] = MappingProxyType({}),
    intent_id: str | None = None,
) -> tuple[dict[str, object], int]: ...

def record_active_view(
    ledger: ServerStateLedger,
    *,
    mode: str,  # 'plane' | 'volume'
    level: int,
    origin: str,
    timestamp: float | None = None,
) -> None: ...
```

Transactions (module: `src/napari_cuda/server/control/transactions/*`)
```
def apply_dims_step_transaction(
    *,
    ledger: ServerStateLedger,
    step: tuple[int, ...],
    dims_spec_payload: Mapping[str, object],
    origin: str,
    timestamp: float,
    op_seq: int,
    op_kind: str,
) -> dict[PropertyKey, LedgerEntry]: ...

def apply_view_toggle_transaction(...): ...
def apply_level_switch_transaction(...): ...
def apply_plane_restore_transaction(...): ...
def apply_volume_restore_transaction(...): ...
def apply_camera_update_transaction(...): ...
```

6. Snapshot Builders (Shared Serializers)
Module: `src/napari_cuda/server/scene/builders.py`
```
@dataclass(frozen=True)
class RenderDimsBlock:
    spec: DimsSpec

@dataclass(frozen=True)
class RenderCameraBlock:
    plane: PlanePose | None
    volume: VolumePose | None

@dataclass(frozen=True)
class RenderLayerBlock:
    values: dict[str, LayerVisualState]

@dataclass(frozen=True)
class RenderSceneSnapshot:
    dims: RenderDimsBlock
    camera: RenderCameraBlock
    layers: RenderLayerBlock
    active: ActiveViewState | None

def snapshot_render_state(ledger: ServerStateLedger) -> RenderSceneSnapshot: ...

def build_notify_dims_payload(spec: DimsSpec) -> Mapping[str, object]: ...
def build_notify_camera_payload(plane: PlanePose | None, volume: VolumePose | None) -> Mapping[str, object]: ...
def build_notify_level(active: ActiveViewState) -> Mapping[str, object]: ...
```

6.1 Dual Channels: State vs Pixel (Authoritative Flows)

- State Channel (notify.*)
  - Trigger: Ledger writes committed by reducers/transactions.
  - Path: handlers → reducers/txns → ledger → notify builders → websocket broadcast.
  - Purpose: synchronize metadata/state with clients (dims, camera poses, active view, layers). No rendering side effects.
  - Timing: best‑effort immediate after ledger commit; mirrors/broadcasters may coalesce.

- Pixel Channel (render loop)
  - Trigger: worker observes `scene.main.op_seq` bump.
  - Path: ledger → fetch scene blocks (`{view, axes, index, lod, camera, layers}`) → per-block signature diff → apply helpers → render/apply policies.
  - Purpose: render the scene (ROI, levels, camera) and stream pixels. No notify messages are produced here.

Separation Guarantees
- Notify is driven solely by ledger commits; render is driven solely by snapshots pulled from the ledger.
- No apply/plan step is required for notify; no notify is emitted by the render loop.

6.2 Worker Intent Mailbox (Loopback to Reducers)
- Purpose: transport worker-originated intents that must be reflected in the ledger before render continues, preserving the single authoritative write path.
- Intents:
  - `LevelSwitchIntent`: worker multiscale/level policy decisions. Server calls `reduce_level_update(...)` with the selected level (and optional shape/state), then pulls a fresh `RenderLedgerSnapshot` and enqueues it for render. This guarantees state consistency (ledger → notify, and render consumes the same committed state).
  - `ThumbnailCapture`: worker-captured thumbnails. Control loop may dedupe and persist via the thumbnail transaction.

7. Runtime Apply (Deterministic)
Modules: `src/napari_cuda/server/runtime/render_loop/applying/*`
```
def apply_dims_block(snapshot_iface: RenderApplyInterface, spec: DimsSpec) -> None:
    # Must set: axis_labels, ndim, order, displayed, ndisplay, current_index (legacy current_step), margins.
    ...

def apply_camera_block(
    snapshot_iface: RenderApplyInterface,
    plane: PlanePose | None,
    volume: VolumePose | None,
) -> None: ...

def apply_layers_block(
    snapshot_iface: RenderApplyInterface,
    layers: Mapping[str, LayerVisualState],
) -> None: ...
```

9. Notify Path and Client Consumption
Module: `src/napari_cuda/protocol/messages.py`
```
@dataclass
class NotifyDimsPayload:
    step: tuple[int, ...]
    current_index: tuple[int, ...] | None  # legacy payload field `current_step`
    levels: tuple[Mapping[str, object], ...]
    current_level: int
    mode: str  # 'plane' | 'volume'
    ndisplay: int
    level_shapes: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...] | None
    dims_spec: Mapping[str, object]  # full DimsSpec payload including margins
```
Client emitters (`client/control/emitters/napari_dims_intent_emitter.py`) must ingest the payload and update the local viewer; margins are compared and emitted only when changed.

10. Active View (Mode + Level)
Canonical publication path:
- Reducers set viewport/active:state whenever mode/level changes (ViewUpdate, LevelUpdate, Plane/Volume Restores, Worker level confirms).
- Server broadcasts notify.level using ActiveView and logs mode+level.
- Client HUD uses notify.level for volume; dims.current_level for plane.

Acceptance criteria: mode and level consistent in ledger, logs, and HUD in the same control tick.

11. Coding Style and Enforcement (Offensive Practices)
- Assertions for invariants inside our code paths (present collaborators, non-null post-init). No None-guards where invariants hold.
- No try/except in hot paths; at subsystem boundaries (websocket I/O, codec/Vt shims) catch specific exceptions, always log, and release resources in finally.
- No dynamic attribute access; favor typed dataclasses and explicit accessors.
- Absolute imports; ruff-enforced formatting; docstrings with double quotes; typed public APIs.
- Logging:
  - INFO: high-level lifecycle (active view changes, applied dims/camera summaries), never dump large arrays.
  - DEBUG: hot-path details with exc_info=True when useful; never flood.

12. Testing Strategy (Function-Level and E2E)
- Unit: dims_spec (de)serialization; camera/layer serializers; signature builders.
- E2E (per block):
  - dims: reduce_dims_update → txn → snapshot → notify → apply_dims_block → assert viewer.dims (order/displayed/margins) and client parity.
  - camera: worker pose → reduce_camera_update → snapshot → notify.camera → apply_camera_block → assert no dims reapply.
  - view: reduce_view_update toggles ndisplay; verify ActiveView + notify.level + HUD.
  - layers: layer visual change updates only layer block; verify signatures don’t trigger dims/camera apply.
- Performance: assert camera-only deltas do not call apply_dims_block.

13. Migration Plan (Phased)
Phase A: Canonical serializers for all blocks; update snapshot builders and notify to use them.
Phase B: Modularize render mailbox signatures + block-specific apply; add tests eliminating dims replay on camera updates.
Phase C: Introduce ActiveView ledger key + broadcasts + HUD/logs integration.
Phase D: Extend notify.dims payload to include all DimsSpec fields (margins now) and implement apply of margins in worker.
Phase E: Planner reads from applied blocks (no ledger reads on hot-path); deprecate request/applied mirrors.
Phase F: Layer parity extensions (multi-layer attributes) built by extending block schemas only.

14. Risks and Mitigations
- Drift between ledger and notify: eliminated by shared serializers.
- Snapshot apply thrash: eliminated by block signatures.
- Regression in ROI/camera planning: mitigated by tests and staged planner shift from ledger reads to applied blocks.

15. File/Module Ownership Map
- shared/dims_spec.py: schema + (de)serialization.
- server/control/state_reducers.py: reducers; record_active_view helper.
- server/control/transactions/*: domain txns, strictly scoped.
- server/scene/builders.py: snapshot builders; notify builders using shared serializers.
- server/runtime/worker/lifecycle.py + server/utils/signatures.py: op_seq watcher + block signatures.
- server/runtime/render_loop/applying/*: block-specific apply functions.
- client/control/emitters/*: consume notify.*; compare against viewer; emit state.update.
- client/rendering/presenter_facade.py: HUD current level from notify.level in volume mode.

16. Glossary
- Ledger: authoritative, versioned state store (committed values only).
- Snapshot: in-memory structure for a render tick built from ledger values.
- Block: independent portion of scene state (dims, camera, layers, active_view).
- ActiveView: canonical mode+level indicator for what’s live.

Reducer Contracts (Pre/Postconditions & Side-effects)

- reduce_dims_update
  - Preconditions:
    - `axis` resolves to an integer index in [0, ndim) of the current DimsSpec via labels/order (reject if not resolvable).
    - If `prop == 'index'`: `value` or `step_delta` provided; computed target
      index is clamped into [0, level_shape[idx]).
    - If `prop in {'margin_left','margin_right'}`: `value` is a finite float ≥ 0.
  - Postconditions:
    - Writes versioned ('dims','main','dims_spec') with updated current_index
      (legacy `current_step`) and, for margin updates, axis margin_* in world and
      steps (steps = world/step_size; step_size from per_level_world[level_idx].step > 0).
    - Writes versioned ('dims','main','current_index') (legacy key
      `'current_step'`) and optional per-axis index.
    - If ndisplay < 3: updates viewport/plane/state (targets/applied) and clears pose-dirty; does not move camera.
    - If level changed: records ActiveView {'mode':'plane','level':current_level}.
  - Invariants:
    - order contains each axis index exactly once; displayed equals the last
      ndisplay of order; len(current_index) == len(axes).
  - Logging: DEBUG axis label, prop, requested index, new spec version, origin.

- reduce_view_update
  - Preconditions: ndisplay ∈ {2,3+}; order/displayed (if provided) are consistent permutations/subsets.
  - Postconditions:
    - Writes versioned dims_spec with ndisplay/order/displayed/plane_mode.
    - Snapshots plane/volume state for restore and records ActiveView {'mode':'volume','level':coarsest} or {'mode':'plane','level':spec.current_level}.
  - Invariants: displayed length == min(len(order), ndisplay).
  - Logging: DEBUG with ndisplay/order/displayed/versions/origin.

- reduce_level_update
  - Preconditions: level resolves; if an index tuple (legacy step) is provided,
    len(index) <= ndim.
  - Postconditions:
    - Updates level_shapes if shape provided; clamps/remaps the index tuple as
      needed; writes versioned dims_spec with current_level/current_index (legacy
      `current_step`).
    - Records ActiveView with provided/derived mode and level.
  - Logging: INFO/DEBUG summarizing level and mode.

- reduce_plane_restore / reduce_volume_restore
  - Postconditions: apply pose to state; write camera_* scoped keys via txn;
    (plane) update dims_spec level/index; record ActiveView.
  - Guarantees: reducers do not emit camera deltas; broadcaster handles notify.

- reduce_camera_update
  - Preconditions: at least one of center/zoom/angles/distance/fov/rect; validated by mode in worker path.
  - Postconditions: writes versioned camera_* scoped keys via txn; returns ack; records plane/volume pose cache only; does not mutate dims or ActiveView.
  - Logging: INFO summarizing applied camera components.

Deduplication & Versioning
- Txns increment versions per key; reducers never set versions directly.
- When value+metadata == previous, txn may dedupe and preserve version; callers must accept unchanged outcomes.

Error Handling
- Reducers raise ValueError/TypeError on invalid inputs (no try/except here). I/O handlers convert to protocol errors and log exceptions.

Builder Contracts
- Snapshot builders read only scoped ledger keys; no side effects.
- Return fully populated blocks or None (optional camera blocks) matching dataclasses.
- Do not recompute derived values already stored (e.g., ActiveView) to avoid divergence.

Signature Content (Stable, Minimal)
- dims_content_signature(spec): tuple(current_index (legacy current_step)
  normalized length, current_level, ndisplay, order tuple, displayed tuple,
  axis labels tuple, level_shapes, per-axis margins (left/right world) rounded
  to tolerance).
- camera_pose_signature(plane, volume): tuple(plane.center, plane.zoom, plane.rect, volume.center, volume.angles, volume.distance, volume.fov) with float rounding.
- layers_signature: sorted tuple of (layer_id, canonical visual items).
- active_view_signature: (mode, level) or None.

Concurrency
- Mailbox methods are internally locked; signature updates and scene enqueues are atomic per tick.

Runtime Apply (Deterministic) — Pseudocode Details
- apply_dims_block(snapshot_iface, spec):
  1) labels → dims.axis_labels
  2) ndim from max(spec.ndim, len(level_shapes[current_level]), viewer.ndim)
  3) order/displayed/ndisplay with assertions; displayed must equal tail(order, ndisplay)
  4) current_index (legacy current_step) normalized to ndim (pad/truncate)
  5) margins → dims.margin_left/right (world units)
  6) snapshot_iface.set_current_level_index(current_level)
- apply_camera_block(snapshot_iface, plane, volume): set plane/volume cameras; never touches dims.
- apply_layers_block(snapshot_iface, layers): apply visuals only; never touches dims/camera.

Notify Contracts
- notify.dims: dims_spec is the single source of truth; duplicated top-level fields must equal embedded spec.
- notify.camera: carries only poses.
- notify.level: carries only ActiveView (mode+level). Clients must not infer mode/level from other messages.

Consistency Gates
- When reducers write ActiveView, assert: if mode=='plane' then level==dims_spec.current_level; if mode=='volume' then level equals worker’s agreed level.
- Server broadcaster sources notify.level exclusively from ActiveView.

Performance Budgets (Guidance)
- Camera delta path: mailbox update + camera apply < 1 ms p50 on reference workstation; no dims apply.
- Dims apply: O(ndim); no dynamic allocation beyond tuple normalization.
- Snapshot build: scoped key reads + trivial assembly; no recomputation of derived values.
