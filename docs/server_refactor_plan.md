# Server De-Godification & Defensive-Guard Reduction Plan

## Current Snapshot
- `egl_worker.py`: 2.4k LOC, still hosts camera/state/ROI/policy/encode logic. Longest blocks remain `__init__` (252 lines), `_apply_pending_state` (239), `_viewport_roi_for_level` (186), `_apply_level_internal` (182), `_maybe_select_level` (154).
- `try:` count: 129 in worker. Many wrap logging/sample instrumentation. `getattr` (85) and `isinstance` (17) still pepper hot paths.
- Env/env_bool sprinkled across init (encoder, ROI, debug). No centralized logging/flag policy yet.
- Server (`egl_headless_server.py`) untouched in this pass; still >2k LOC.

## Guiding Principles
1. **No "just in case" guards in hot paths**: Logging, attributes we control, and invariants should assert, not silently fall back.
2. **Single responsibilities**: Worker should delegate to thin helpers for ROI, policy, encode, state application, etc.
3. **Explicit boundaries**: Only I/O (GL/CUDA/NVENC/websockets) earns try/except; all else should fail loudly.
4. **Testable helpers**: Move math/policy/ROI code into pure modules with docstring specs.

## Phase Plan

### Phase A — NVENC Boundary
- **Extract Encoder helper** (`rendering/encoder.py`): init, encode, reset, force IDR, logging. Replace `_init_encoder`, `_encode_frame`, `reset_encoder`, `force_idr` (current top offenders with nested env reads and try/except blocks).
- **Outcome**: Worker interacts with `Encoder` via a minimal protocol (`setup(Context)`, `encode(frame)`, `reset()`, `force_idr()`), with no os.getenv inside worker.

### Phase B — State/Camera Core
- **State machine**: Pull `apply_state`, `_apply_pending_state`, `process_camera_commands` into `state_machine.py`. Convert to pure dataclass-based updates with assertive handling.
- **Camera ops**: Already extracted (module). Remove remaining `getattr` checks where invariants ensure types (e.g., `self.view.camera` once scene init completes).

### Phase C — ROI & LOD Minimization
- **ROI helper**: Move `_viewport_roi_for_level`, related chunk alignment, oversampling to `lod.py` or new `roi.py`. Worker should request `roi = compute_roi(view_context)` and let helper raise on failure.
- **LOD selection**: Replace `_maybe_select_level` with composition of (1) input intent (zoom queue), (2) pure `lod.select_level(current, overs_map, thresholds)`. No env reads inside worker; thresholds come from config.

### Phase D — Logging & Debug Policy
- Establish `logging_policy.py` or config entries controlling debug flags. Remove per-call env parsing (`NAPARI_CUDA_DEBUG_*`). Convert to bools resolved at init via `ServerCtx`.
- Strip logging try/except; ensure format strings are safe or pre-format values.

### Phase E — Server Decomposition
- Mirror worker work: extract `PixelBroadcaster`, `StateServer`, `SceneSpecBuilder`. Apply same hostility to try/except & getattr.

### Targets & Metrics
- Reduce worker `try:` count <40 (boundary-only) & `getattr` <30 in final state.
- Limit longest worker methods <120 lines (hard cap 80 ideal).
- Shift env read paths entirely into `ServerCtx`/new config modules.

### Supporting Work
- **Docs**: Capture module responsibilities & invariants in `docs/server_refactor_plan.md` and inline docstrings.
- **Tests**: Add unit tests around new helpers (encoder, ROI/LOD, state machine) for regression control. Prefer pure functions for easy testing.
- **Lint rules**: Introduce ruff rules to flag `except Exception` & `logger.*` inside try/except.

---

This doc will serve as the reset point before the next iteration, focusing strictly on de-godification and eliminating defensive clutter.
