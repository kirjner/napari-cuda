# Control State Refactor Plan (Post-Bridge Cleanup)

## 1. Current Inventory
- **Shared context + plumbing**
  - `ControlStateContext`, `on_state_connected/disconnected`, `_emit_state_update`, `_update_runtime_from_ack_outcome`, runtime key/rate-gating helpers.
  - Call sites: all emitters (dims/layer/camera), `ClientStreamLoop` (`_initialize_mirrors_and_emitters`, `_ingest_ack_state`, volume/multiscale commands), runtime tests.
- **Dims-facing helpers**
  - `_axis_to_index`, `_axis_target_label`, `_is_axis_playing`, dims intent entry points, HUD snapshot, dims ack handlers (now largely housed in `NapariDimsMirror`/`NapariDimsIntentEmitter`).
  - Call sites: `NapariDimsIntentEmitter`, `NapariDimsMirror`, `_PresenterStub` tests, `ClientStreamLoop` key handling + HUD snapshot.
- **Camera helpers (legacy)**
  - `camera_zoom/pan/orbit/reset/set`, `handle_notify_camera`, `_normalize_camera_delta_value`.
  - Call sites: `NapariCameraIntentEmitter` only uses `_normalize_camera_delta_value`; other camera helpers are now unused outside legacy tests.
- **Volume + multiscale helpers**
  - `volume_set_*`, `multiscale_set_policy/level`, `hud_snapshot`, `current_ndisplay`.
  - Call sites: `ClientStreamLoop` public API (`volume_set_*`, `multiscale_set_*`, `view_set_ndisplay`), hud snapshot for presenter overlay.
- **Input helpers**
  - `handle_key_event`, `_axis_index_from_target` (indirect), used by `ClientStreamLoop._on_key_event` and dims emitter.
- **Tests**
  - `control/_tests/test_state_update_actions.py` exercises most functions directly; runtime/dims tests rely on behaviour indirectly.

## 2. Proposed Split Targets
1. **`control_state.py` (new)**
   - Host `ControlStateContext`, `_emit_state_update`, `_update_runtime_from_ack_outcome`, rate-gating helpers, and ack bookkeeping.
   - Provide a tiny façade (`dispatch_state_update`, `update_runtime_from_ack`) consumed by all emitters/mirrors/runtime without dragging domain-specific logic.
2. **Dims module tightening**
   - Move `apply_scene_policies`, hud builders, `_axis_to_index/_axis_target_label/_is_axis_playing`, and dims ack handlers into `NapariDimsMirror`/`NapariDimsIntentEmitter`.
   - Update dims tests to cover the new methods directly; retire the equivalent cases from `test_state_update_actions`.
3. **Camera cleanup**
   - Drop the unused `camera_*` helpers and `handle_notify_camera`; `NapariCameraIntentEmitter` already owns emission/ack behaviour.
   - Keep `_normalize_camera_delta_value` alongside the emitter (or new utility file shared with the mirror).
4. **Volume & multiscale**
   - Relocate `volume_set_*` into a small `VolumeIntentEmitter` (either new class or methods on dims emitter) so `ClientStreamLoop` calls that instead of `control_actions`.
   - Keep `multiscale_set_*` with the new multiscale mirror/emitter pair; ensure presenter updates stay co-located.
5. **Input/key handling**
   - Move `handle_key_event` into a dedicated runtime helper (e.g. `client/runtime/client_loop/input_handlers.py`) and feed it the dims/camera emitters directly.
6. **Tests refactor**
   - Replace `test_state_update_actions` with focused suites per subsystem (dims/camera/volume) relying on the new home modules.
   - Ensure integration tests (`test_stream_runtime.py`) only touch the new emitters/mirrors.

## 3. Migration Steps
1. **Seed `control_state.py`**
   - Copy `ControlStateContext`, `_emit_state_update`, `_update_runtime_from_ack_outcome`, `_runtime_key`, `_rate_gate_settings`, and shared ack utilities.
   - Update emitters/mirrors/runtime to import from `control_state` instead of `state_update_actions`.
   - Adjust tests to reference the new module.
2. **Dims extraction**
   - Inline `apply_scene_policies`, dims metadata builders, and ack handlers into `NapariDimsMirror`/`NapariDimsIntentEmitter`.
   - Update dims tests + runtime tests accordingly.
3. **Camera trimming**
   - Delete unused `camera_*` helpers and `handle_notify_camera`; keep `_normalize_camera_delta_value` in the emitter/mirror as needed.
   - Remove redundant cases from legacy tests.
4. **Volume/multiscale intents**
   - Introduce a slim volume intent helper (either methods on dims emitter or a new emitter) and migrate `ClientStreamLoop.volume_*` calls.
   - Move `multiscale_set_policy/level` into the multiscale mirror/emitter, ensuring ledger updates stay in sync.
5. **Input helpers**
   - Move `handle_key_event` into a runtime-local helper that works with emitters directly, eliminating the last runtime dependency.
6. **Test consolidation + module deletion**
   - Once all call sites swap over, delete `state_update_actions.py` along with `test_state_update_actions.py`.
  - Run control and server/data suites to confirm behaviour matches the previous layer-emitter rewrite.

## 4. Open Questions
- Do we want a dedicated `VolumeIntentEmitter`, or should volume stay bundled with dims for now?
- Should `_normalize_camera_delta_value` live in a generic utilities module shared between mirror/emitter, or stay private to the emitter?
- Are there external consumers (e.g., server tooling) importing `ControlStateContext` that need shims during the transition?

## 5. Dependencies & Sequencing
- Complete the control_state extraction first to avoid churn across emitters.
- Follow the layer-emitter migration playbook: migrate logic per subsystem, update tests, then delete the old module.
- Coordinate with any in-flight plans touching volume/multiscale so the new emitters/mirrors remain the single source of truth.
