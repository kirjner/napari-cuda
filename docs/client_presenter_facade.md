# Client Presenter Facade Plan

## Motivation
- Historically `src/napari_cuda/client/streaming_canvas.py` owned Qt bootstrap, draw-hook wiring, FPS HUD timers, and ad-hoc presenter plumbing. This duplicated logic already present in `ClientStreamLoop` and made refactors risky.
- `ClientStreamLoop` is the authoritative home for the presenter (`FixedLatencyPresenter`), renderer, intent dispatch, and dims mirroring. We now expose that functionality through a façade so Qt-specific code can stay thin and testable.
- A dedicated façade gives the upcoming `LayerStateBridge` a stable hook while keeping the Qt canvas focused on window deferral and keymap setup.

## Target Module
- New module: `src/napari_cuda/client/streaming/presenter_facade.py` (lives alongside `presenter.py`, `renderer.py`, etc.).
- Exported class: `PresenterFacade`.
- Dependencies: limited to `QtCore`, `QtWidgets`, `FixedLatencyPresenter`, `SourceMux`, `GLRenderer`, and lightweight helpers (`ClientConfig`, metrics hooks). Avoid importing `StreamingCanvas` to keep layering clean.

## Public Surface
### `PresenterFacade.start_presenting(*)`
- Parameters (first draft):
  - `scene_canvas`: the Vispy scene canvas (`StreamingCanvas._scene_canvas`).
  - `loop`: `ClientStreamLoop` instance (for draw, metrics, state callbacks).
  - `presenter`: `FixedLatencyPresenter` retained by the loop.
  - `renderer`: `GLRenderer` retained by the loop.
  - Optional knobs: `display_loop`, `metrics`, `hud_config`, `wake_proxy`.
- Responsibilities:
  - Connects `scene_canvas.events.draw` to `loop.draw` (ensuring dims play hook stays first, draw handler last).
  - Starts or reuses display-loop (Qt timer vs Vispy timer) and ensures it can be toggled by env flag exactly once.
  - Installs FPS/latency HUD label and timer, sourcing stats via `presenter.stats()`.
  - Registers a lightweight intent hook placeholder (`self._intent_dispatcher`) so the forthcoming layer intent bridge can bind without modifying the canvas again.
  - Emits the "first dims ready" callback so window deferral logic (`StreamingCanvas.defer_window_show`) can remain outside the façade.

### `PresenterFacade.apply_dims_update(data: dict)`
- Called from `ClientStreamLoop._mirror_dims_to_viewer` (or its successor once the state bag lands).
- Responsibilities:
  - Mirror dims metadata/step to the attached viewer via a stored weakref (currently `ClientStreamLoop.attach_viewer_mirror`).
  - Trigger presenter HUD updates if range/ndisplay changes affect overlay text.
  - Call into an optional intent dispatcher with `intent='dims'` events so the layer bridge can coalesce changes.
  - Maintain internal caches required for `PresenterFacade` to answer lightweight queries (e.g., `current_ndisplay()` for UI toggles).

### `PresenterFacade.shutdown()`
- Disconnect draw handlers and timers, stop display loop, clear HUD label, detach intent dispatcher, and release any weakrefs.
- Called from `ClientStreamLoop.stop()` and from `StreamingCanvas.closeEvent` to guarantee deterministic teardown.

### Hook for Layer Intents
- `PresenterFacade.set_intent_dispatcher(callable | None)` registers a callable that accepts `(intent_type, payload)`; default is `None`.
- During bridge work, `LayerStateBridge` will register here so Qt layer widgets can emit through the façade without poking at the loop or canvas.

## Implementation Notes
- `PresenterFacade` owns the draw-event wiring, optional display-loop lifecycle, and HUD timing updates. It holds only references to loop collaborators; no new threads are spawned.
- `StreamingCanvas` now creates `ClientStreamLoop`, invokes the façade’s `bind_canvas` helper, and delegates presenter/HUD responsibilities entirely to the façade.
- Window deferral (`StreamingCanvas.defer_window_show`) remains local to the canvas; the façade focuses on presenter-facing work.

## Migration Steps
1. **Introduce façade module**
   - Move HUD helpers, draw-event wiring, and display-loop startup into `PresenterFacade` while keeping bindings identical.
   - Add façade construction inside `ClientStreamLoop.start()` (after presenter/renderer are ready) and store it as `self._presenter_facade`.
   - Provide `ClientStreamLoop.present_facade` property so `StreamingCanvas` can call `defer_window_show` and register the first-dims callback without reaching into presenter internals.
2. **Update `StreamingCanvas`** *(done)*
   - The canvas now delegates draw wiring, HUD, and display-loop setup to `PresenterFacade` and no longer owns presenter, decoder, or queue state.
   - Legacy smoke/offline paths were removed; all presentation flows through `ClientStreamLoop.draw` via the façade hook.
3. **Expose dims/application events** *(done)*
   - `ClientStreamLoop` forwards dims payloads to `PresenterFacade.apply_dims_update`, which caches them for future intent bridge work.
4. **Teardown wiring**
   - On loop shutdown, invoke `self._presenter_facade.shutdown()` before renderer teardown.
   - Ensure `StreamingCanvas.closeEvent` (or equivalent) calls through to the loop, which fans out to the façade.

## Regression Tests (pre-hoist before deleting old code)
- **Qt Draw Wiring**: Qt test harness ensuring `scene_canvas.events.draw` calls `ClientStreamLoop.draw` exactly once per repaint and retains `enable_dims_play` ordering. (Existing `_tests/test_streaming_canvas_draw.py` can be adapted.)
- **HUD Stats Source**: Unit test that injects a fake presenter with deterministic `stats()` output and verifies the HUD label text updates after timer tick.
- **Display Loop Opt-in**: Test toggling `NAPARI_CUDA_USE_DISPLAY_LOOP` asserts façade starts/stops `DisplayLoop` and does not leak when shutdown is called twice.
- **Dims Update Mirror**: Test that `apply_dims_update` forwards payload to the viewer mirror, triggers `on_first_dims_ready`, and accepts repeated payloads without duplicating logs.
- **Intent Hook Smoke**: Test registering a stub dispatcher, calling `apply_dims_update`, and asserting the dispatcher sees a `('dims', payload)` tuple.

## Open Questions / Follow-ups
- With the `ClientLoopState` dataclass in place, migrate façade construction to accept the state bag and collaborators explicitly (reduces `self` reach).
- `PresenterFacade` exposes `set_intent_dispatcher`; the upcoming `LayerStateBridge` must document how it plugs into this hook.

## Current Status
- `PresenterFacade` manages draw wiring, HUD updates, and optional display loop creation. It caches viewer weakrefs and dims payloads for the forthcoming layer-intent bridge.
- `StreamingCanvas` is now a thin Qt shim responsible for vsync setup, keymap handling, and deferred window display; all presenter logic is delegated.
- Smoke/offline paths were deleted alongside the hoist to keep the façade surface focused on real streaming flows.
