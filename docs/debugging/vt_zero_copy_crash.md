# VT Zero-Copy Crash Notes

This note captures the current hypotheses and hardening ideas around the
sporadic segmentation faults observed while presenting VT zero-copy frames in
`GLRenderer._draw_vt_texture`.

## Observed Symptoms

- Segfault happens shortly after `GLRenderer: VT zero-copy draw engaged` logs.
- `VT dbg` counter shows `ret` and `rel` deltas of roughly `+120/+180` per
  second; `outstd` trends negative even when the pipeline keeps up.
- Crashes reproduce even with `NAPARI_CUDA_VT_GL_SAFE=1` (current safe mode
  performs a blocking `glFinish()` before calling `gl_release_tex`).
- 2025-09-25 spike: segfault now reproduces immediately after the stream loop
  logs `VT gate lifted on keyframe (...); presenter=VT`. The top of the Python
  traceback is `GLRenderer._draw_vt_texture`, invoked via Vispy's `paintGL`
  handler, matching the zero-copy path.

### Stack (abridged)

```
GLRenderer._draw_vt_texture -> GLRenderer.draw -> ClientStreamLoop.draw
StreamingCanvas._draw_video_frame -> vispy.app.backends._qt.QGLWidget.paintGL
napari._qt.qt_main_window.Window.event -> napari._qt.qt_event_loop.run
napari_cuda.client.launcher.launch_streaming_client -> main
```

## Accounting Fixes (Done)

- `vt_stats()` now increments `retains` in the VideoToolbox callback and uses
  `vt_release_frame()` whenever the ring buffer or session drops a frame. This
  keeps `releases - retains` close to the number of frames actually presented.

## Remaining Crash Hypotheses

- **Texture released while GPU still uses it.** We now gate releases through a
  fence-backed queue when `NAPARI_CUDA_VT_GL_SAFE=1` (see Renderer section
  below). The queue issues `glFenceSync` per frame, retires entries with
  `glClientWaitSync`, and only then calls `gl_release_tex()`. When the queue is
  disabled we still fall back to immediate release, so the original issue can
  be reproduced for comparison.

- **Context pollution / missing cleanup.** Attribute arrays and bound textures
  stay enabled on exceptions; falling through different code paths could leave
  stale GL state and crash the driver. Wrapping the draw body in `try/finally`
  blocks that restore state would mitigate this.

- **Invalid texture cache capsule.** We create `self._vt_cache` once; if Qt
  destroys/recreates the GL context (e.g., window move to new GPU), the cache
  becomes invalid but we keep using it. Detecting context changes and
  rebuilding the cache would avoid use-after-free inside Apple's code.

- **Double-release on VT fallback.** When zero-copy draw fails we release the
  payload before handing it back to the presenter. If the presenter still holds
  a reference we might drop the buffer early. Audit fallback paths to ensure
  releases happen exactly once per retain.

- **Wrong thread / missing current context.** If `draw()` runs without the Qt
  context current (e.g., wake scheduled from a worker), any GL call can crash.
  Confirm all wake scheduling routes through the main thread and guard the draw
  entry point with `QOpenGLContext.currentContext()` checks.

## 2025-09-25 Spike Tasks

- Capture an `lldb` native backtrace with symbols for the Apple VT stack while
  the segfault reproduces. This will confirm whether the crash happens inside
  `IOSurface` release or within our GL shim.
- Verify VT retain/release balance during the gate lift: ensure the initial
  keyframe retains the buffer exactly twice (cache + presenter). Audit the base
  `vt.release_frame` in the presenter to rule out double-release at the end of
  the warmup window.
- Re-run with `NAPARI_CUDA_VT_GL_SAFE=1` and the new logging to confirm whether
  the safe fence queue drains before the crash.

## Suggested Hardening Tasks

1. Wrap attribute enables, VBO binds, and texture binds inside context manager
   helpers that restore state in `finally` blocks.
2. Recreate the VT texture cache when the Vispy canvas reports a context
   change (`on_initialize`, `on_resize`, or via a context UUID check).
3. Add presenter-side assertions to ensure every retained VT frame has exactly
   two outstanding references (cache + renderer) while the frame is live.

## Next Steps

- Rebuild the extension so the new retain/release counters ship.
- Exercise the fence + deferred release logic under the safe flag and confirm
  whether segfaults stop. Compare crash frequency with the flag disabled to
  validate the queue is providing cover.
- If crashes persist, capture a native backtrace with `lldb` while logging the
  release queue to identify which texture ID explodes.
