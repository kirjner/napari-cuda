"""
Thumbnail wiring probe for napari (offscreen-safe).

Purpose
-------
Empirically determine when the Qt UI subscribes to `layer.events.thumbnail`
relative to `viewer.layers.events.inserted` and to dims events, and which
invocation style is safe to trigger a thumbnail update deterministically —
including driving updates from a non-GUI thread via a signal proxy.

Usage
-----
- 2D probe (offscreen):
  QT_QPA_PLATFORM=offscreen uv run python tools/thumbnail_probe.py
- 3D + dims toggles (offscreen):
  QT_QPA_PLATFORM=offscreen uv run python tools/thumbnail_probe.py --ndim 3

What it does
------------
- Creates a napari Viewer and Window (not shown).
- Connects to `viewer.layers.events.inserted` and dims events
  (`viewer.dims.events.ndisplay`, `viewer.dims.events.current_step`).
- Drives repaint deterministically by emitting Qt model `dataChanged` for the
  layer row; does NOT rely on `layer.events.thumbnail` timing.
- Schedules a sequence of dims updates (2D↔3D and index changes).
- Starts a background Python thread that posts GUI work (dataChanged emits
  and dims changes) via a Qt signal proxy to validate cross-thread posting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from typing import Optional, Tuple

import numpy as np

from qtpy import QtCore, QtWidgets

import napari
from napari._qt.qt_main_window import Window
from napari.layers import Image


def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def callbacks_count(layer: Image) -> int:
    # Kept for optional diagnostics; not used in the direct-inject path
    try:
        emitter = getattr(layer.events, 'thumbnail', None)
        return len(getattr(emitter, 'callbacks', []) or []) if emitter is not None else 0
    except Exception:
        return 0


class GuiProxy(QtCore.QObject):
    """Queue callables onto the GUI thread from any thread."""

    call = QtCore.Signal(object)

    def __init__(self, parent=None):  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.call.connect(self._on_call)

    @QtCore.Slot(object)
    def _on_call(self, fn):  # type: ignore[no-untyped-def]
        if callable(fn):
            fn()


def qimage_from_array(arr: np.ndarray) -> Optional[object]:  # returns QImage; object to avoid hard typing
    """Best-effort ndarray -> QImage in common formats.

    Accepts HxW (grayscale), HxWx3 (RGB), HxWx4 (RGBA) arrays. Values can be
    uint8 [0..255] or float in [0..1]. Returns a QImage or None if unsupported.
    """
    from qtpy import QtGui  # local import to avoid hard dependency at import time

    try:
        a = np.asarray(arr)
        if a.size == 0:
            return None
        if a.dtype.kind == 'f':
            a = np.clip(a, 0.0, 1.0)
            a = (a * 255.0 + 0.5).astype(np.uint8)
        if a.ndim == 2:
            h, w = a.shape
            fmt = QtGui.QImage.Format_Grayscale8
            bytes_per_line = w
        elif a.ndim == 3 and a.shape[2] == 3:
            h, w, _ = a.shape
            fmt = QtGui.QImage.Format_RGB888
            bytes_per_line = 3 * w
        elif a.ndim == 3 and a.shape[2] == 4:
            h, w, _ = a.shape
            fmt = QtGui.QImage.Format_RGBA8888
            bytes_per_line = 4 * w
        else:
            return None
        # Ensure contiguous for QImage
        a = np.ascontiguousarray(a)
        qimg = QtGui.QImage(a.data, w, h, bytes_per_line, fmt)
        # Keep a reference to the array data to avoid GC while QImage exists
        qimg._np_ref = a  # type: ignore[attr-defined]
        return qimg
    except Exception:
        return None


def maybe_direct_inject(viewer: napari.Viewer, layer: Image) -> bool:
    """Force the layer-list view to repaint the thumbnail for `layer`.

    This bypasses napari's thumbnail EventEmitter timing by directly emitting
    a dataChanged signal on the Qt model (ThumbnailRole) for the target row.
    Returns True if a target row was found and a signal was emitted.
    """
    try:
        window = getattr(viewer, 'window', None)
        qt_viewer = getattr(window, '_qt_viewer', None)
        if qt_viewer is None:
            return False
        ll = getattr(qt_viewer, 'layers', None)
        if ll is None:
            return False
        model = getattr(ll, 'model', lambda: None)()
        if model is None:
            return False
        # Get underlying source model if using a proxy
        source = getattr(model, 'sourceModel', lambda: None)() or model
        # Find the source index for our layer
        row_found = -1
        for row in range(getattr(source, 'rowCount')()):
            idx = source.index(row, 0)
            try:
                item = source.getItem(idx)  # type: ignore[attr-defined]
            except Exception:
                item = None
            if item is layer:
                row_found = row
                src_index = idx
                break
        if row_found < 0:
            log("direct-inject: model row for layer not found")
            return False
        # Emit dataChanged on the source model with no roles; proxy/view will update
        try:
            source.dataChanged.emit(src_index, src_index, [])
        except Exception:
            # Fallback: emit on the proxy model if needed
            try:
                proxy_index = model.mapFromSource(src_index)
                model.dataChanged.emit(proxy_index, proxy_index, [])
            except Exception as exc:
                log(f"direct-inject: failed to emit dataChanged: {exc!r}")
                return False
        # Optionally nudge viewport
        vp = getattr(ll, 'viewport', lambda: None)()
        if vp is not None and hasattr(vp, 'update'):
            vp.update()
        log("direct-inject: emitted dataChanged(no roles)")
        return True
    except Exception as exc:
        log(f"direct-inject: failed: {exc!r}")
        return False


# (No synthetic thumbnail generator; we rely on napari to compute thumbnails
# from the real image data via layer._update_thumbnail())


def schedule_dims_sequence(viewer: napari.Viewer, layer: Image, ndim: int = 2) -> None:
    """Schedule a short sequence of dims state changes for the probe.

    - Toggle ndisplay 2D↔3D if ndim == 3
    - Step through a few z indices on axis 0
    """
    def set_ndisplay(val: int) -> None:
        try:
            viewer.dims.ndisplay = int(val)
        except Exception as exc:
            log(f"set_ndisplay failed: {exc!r}")

    def set_step(z: int) -> None:
        try:
            cs = list(getattr(viewer.dims, 'current_step'))
            cs[0] = int(z)
            viewer.dims.current_step = tuple(cs)
        except Exception:
            try:
                viewer.dims.set_point(0, int(z))
            except Exception as exc:
                log(f"set_point failed: {exc!r}")

    t = 20
    if ndim == 3:
        QtCore.QTimer.singleShot(t, lambda: set_ndisplay(3))
        QtCore.QTimer.singleShot(t + 40, lambda: set_ndisplay(2))
        t += 80
    for dz in (1, 4, 8, 16, 24):
        QtCore.QTimer.singleShot(t, lambda z=dz: set_step(z))
        t += 30


def main() -> int:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    parser = argparse.ArgumentParser(description='Napari thumbnail wiring probe')
    parser.add_argument('--ndim', type=int, default=2, choices=(2, 3), help='Layer dimensionality for the probe (2 or 3)')
    parser.add_argument('--bg', action='store_true', help='Start a background thread to post GUI work via proxy')
    parser.add_argument('--direct', action='store_true', help='Attempt direct Qt injection of thumbnail pixmap')
    parser.add_argument('--verify', action='store_true', help='Verify repaint deterministically (delegate counter + viewport diff)')
    parser.add_argument('--capture-delay-ms', type=int, default=20, help='Delay before capturing thumbnail ROI (ms) to allow paint')
    parser.add_argument('--capture-cycles', type=int, default=3, help='How many paint cycles to wait before capture')
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = napari.Viewer()
    window = Window(viewer, show=False)
    qt_viewer = window._qt_viewer

    # Optional: instrument delegate paints and prepare viewport digest helper
    paint_counter = [0]
    event_seq = [0]

    def _next_key(prefix: str) -> str:
        event_seq[0] += 1
        return f"{prefix}#{event_seq[0]}"

    def _install_delegate_probe():
        try:
            ll = getattr(qt_viewer, 'layers', None)
            if ll is None:
                return False
            delegate = ll.itemDelegate()
            orig = getattr(delegate, '_paint_thumbnail', None)
            if orig is None:
                return False
            import types

            def _wrapped(self, painter, option, index):  # type: ignore[no-redef]
                paint_counter[0] += 1
                return orig(painter, option, index)

            setattr(delegate, '_paint_thumbnail', types.MethodType(_wrapped, delegate))
            log('verify: delegate probe installed')
            return True
        except Exception as exc:
            log(f'verify: delegate probe install failed: {exc!r}')
            return False

    def _viewport_digest() -> str:
        try:
            from qtpy import QtGui
            ll = getattr(qt_viewer, 'layers', None)
            if ll is None or ll.viewport() is None:
                return ''
            pm = ll.viewport().grab()
            img = pm.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
            ptr = img.bits()
            ptr.setsize(img.byteCount())  # type: ignore[attr-defined]
            data = bytes(ptr)
            import hashlib

            return hashlib.md5(data).hexdigest()
        except Exception as exc:
            log(f'verify: digest failed: {exc!r}')
            return ''

    def _thumb_rect_and_index(layer: Image):
        try:
            from qtpy import QtCore as _QtCore
            ll = getattr(qt_viewer, 'layers', None)
            if ll is None:
                return None, None
            model = getattr(ll, 'model', lambda: None)()
            if model is None:
                return None, None
            source = getattr(model, 'sourceModel', lambda: None)() or model
            src_index = None
            for row in range(getattr(source, 'rowCount')()):
                idx = source.index(row, 0)
                try:
                    item = source.getItem(idx)  # type: ignore[attr-defined]
                except Exception:
                    item = None
                if item is layer:
                    src_index = idx
                    break
            if src_index is None:
                return None, None
            proxy_index = model.mapFromSource(src_index) if hasattr(model, 'mapFromSource') else src_index
            item_rect = ll.visualRect(proxy_index)
            if not item_rect.isValid():
                return None, None
            # Mirror LayerDelegate logic for thumb_rect
            # size hint
            try:
                h = ll.sizeHintForRow(proxy_index.row())
                if not h:
                    h = proxy_index.data(_QtCore.Qt.ItemDataRole.SizeHintRole).height()  # type: ignore[union-attr]
            except Exception:
                h = 34
            thumb_rect = _QtCore.QRect(item_rect)
            thumb_rect.translate(-2, 2)
            thumb_rect.setWidth(max(1, int(h) - 4))
            thumb_rect.setHeight(max(1, int(h) - 4))
            return thumb_rect, proxy_index
        except Exception as exc:
            log(f'verify: rect/index failed: {exc!r}')
            return None, None

    # Track previous thumbnail-region hash for change detection
    last_thumb_hash = ['']

    def _thumb_region_hash_and_center_rgba(layer: Image):
        try:
            from qtpy import QtGui as _QtGui
            rect, _ = _thumb_rect_and_index(layer)
            ll = getattr(qt_viewer, 'layers', None)
            if rect is None or ll is None or ll.viewport() is None:
                return '', None
            pm = ll.viewport().grab()
            dpr = float(getattr(pm, 'devicePixelRatio', lambda: 1.0)())
            img = pm.toImage().convertToFormat(_QtGui.QImage.Format_RGBA8888)
            x = int(rect.x() * dpr)
            y = int(rect.y() * dpr)
            w = max(1, int(rect.width() * dpr))
            h = max(1, int(rect.height() * dpr))
            roi = img.copy(x, y, w, h)
            ptr = roi.bits()
            ptr.setsize(roi.byteCount())  # type: ignore[attr-defined]
            data = bytes(ptr)
            import hashlib
            digest = hashlib.md5(data).hexdigest()
            cx = x + w // 2
            cy = y + h // 2
            color = img.pixelColor(cx, cy)
            return digest, (color.red(), color.green(), color.blue(), color.alpha())
        except Exception:
            return '', None

    def _capture_thumb(label: str, layer: Image) -> None:
        d, rgba = _thumb_region_hash_and_center_rgba(layer)
        if not d:
            log(f'verify: {label}: no hash (thumb region unavailable)')
            return
        same = (d == last_thumb_hash[0])
        last_thumb_hash[0] = d
        if rgba is not None:
            log(f'verify: {label}: thumb same={same} hash={d} center={rgba}')
        else:
            log(f'verify: {label}: thumb same={same} hash={d}')

    def _schedule_capture(qtcore, delay_ms: int, cycles: int, fn) -> None:
        # Chain N one-shot timers to allow multiple paint cycles
        if cycles <= 1:
            qtcore.QTimer.singleShot(delay_ms, fn)
        else:
            qtcore.QTimer.singleShot(delay_ms, lambda: _schedule_capture(qtcore, delay_ms, cycles - 1, fn))

    if args.verify:
        _install_delegate_probe()

    def on_inserted(event=None):
        layer = getattr(event, 'value', None)
        if layer is None:
            idx = getattr(event, 'index', None)
            try:
                layer = viewer.layers[int(idx)] if idx is not None else None
            except Exception:
                layer = None
        if not isinstance(layer, Image):
            return
        log(f"inserted: layer={getattr(layer, 'name', 'unnamed')} thread={QtCore.QThread.currentThread()}")

        inserted_layer['obj'] = layer
        # After insertion, schedule dims toggles for the probe
        schedule_dims_sequence(viewer, layer, ndim=args.ndim)

        # Optionally attempt direct injection now (pixmap)
        if args.direct:
            # Compute thumbnail from current data using napari's pipeline
            try:
                layer._update_thumbnail()
            except Exception:
                pass
            ok = maybe_direct_inject(viewer, layer)
            log(f"direct-inject(at-insert): {ok}")
            if args.verify:
                _schedule_capture(QtCore, args.capture_delay_ms, args.capture_cycles, lambda: _capture_thumb('after-insert', layer))

        # Optional verification: ensure repaint deterministically
        if args.verify:
            before_paints = paint_counter[0]
            before_digest = _viewport_digest()
            # install a distinct thumbnail without emitting napari events
            try:
                shape = tuple(getattr(layer, 'thumbnail').shape)
                if not shape or shape[-1] != 4:
                    raise RuntimeError('unexpected thumbnail shape')
                # Fill with a bright green RGBA color (fully opaque)
                distinct = np.zeros(shape, dtype=np.uint8)
                distinct[..., 0] = 0
                distinct[..., 1] = 255
                distinct[..., 2] = 0
                distinct[..., 3] = 255
                setattr(layer, '_thumbnail', distinct)  # no napari event
            except Exception as exc:
                log(f'verify: prepare distinct thumbnail failed: {exc!r}')
            # Now emit dataChanged to force repaint
            ok2 = maybe_direct_inject(viewer, layer)
            # Service events briefly
            try:
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            except Exception:
                pass
            after_paints = paint_counter[0]
            after_digest = _viewport_digest()
            log(f'verify: paints delta={after_paints - before_paints} (before={before_paints}, after={after_paints})')
            log(f'verify: viewport changed={after_digest != before_digest}')
            # Sample the center pixel of the thumbnail rect to confirm color
            try:
                rect, pidx = _thumb_rect_and_index(layer)
                ll = getattr(qt_viewer, 'layers', None)
                if rect is not None and ll is not None:
                    pm = ll.viewport().grab()
                    dpr = float(getattr(pm, 'devicePixelRatio', lambda: 1.0)())
                    from qtpy import QtGui as _QtGui
                    img = pm.toImage().convertToFormat(_QtGui.QImage.Format_RGBA8888)
                    cx = int((rect.center().x()) * dpr)
                    cy = int((rect.center().y()) * dpr)
                    color = img.pixelColor(cx, cy)
                    log(f'verify: thumb center rgba=({color.red()},{color.green()},{color.blue()},{color.alpha()})')
            except Exception as exc:
                log(f'verify: sampling failed: {exc!r}')

    viewer.layers.events.inserted.connect(on_inserted)

    # Dims event handlers
    def on_ndisplay(event=None):
        layer = inserted_layer['obj']
        if not isinstance(layer, Image):
            return
        val = getattr(viewer.dims, 'ndisplay', None)
        log(f"dims.ndisplay -> {val} thread={QtCore.QThread.currentThread()}")
        # Mimic server: produce a new thumbnail for this dims toggle
        if args.direct:
            try:
                layer._update_thumbnail()
            except Exception:
                pass
            ok = maybe_direct_inject(viewer, layer)
            log(f"dims: direct-inject -> {ok}")
            if args.verify:
                _schedule_capture(QtCore, args.capture_delay_ms, args.capture_cycles, lambda: _capture_thumb(f'after-ndisplay-{val}', layer))

    def on_step(event=None):
        layer = inserted_layer['obj']
        if not isinstance(layer, Image):
            return
        step = getattr(viewer.dims, 'current_step', None)
        log(f"dims.step -> {step} thread={QtCore.QThread.currentThread()}")
        # Mimic server: produce a new thumbnail keyed by the step
        if args.direct:
            try:
                layer._update_thumbnail()
            except Exception:
                pass
            ok = maybe_direct_inject(viewer, layer)
            log(f"step: direct-inject -> {ok}")
            if args.verify:
                _schedule_capture(QtCore, args.capture_delay_ms, args.capture_cycles, lambda: _capture_thumb(f'after-step-{step}', layer))

    viewer.dims.events.ndisplay.connect(on_ndisplay)
    viewer.dims.events.current_step.connect(on_step)

    # Add a layer and force a thumbnail path
    inserted_layer: dict[str, Optional[Image]] = {'obj': None}
    if args.ndim == 3:
        data = np.random.random((32, 64, 64)).astype(np.float32)
    else:
        data = np.random.random((64, 64)).astype(np.float32)
    layer = viewer.add_image(data, name='probe')

    # Optionally start a background thread to post GUI work
    if args.bg:
        proxy = GuiProxy()

        def bg_worker():
            # Post deterministic repaints and dims edits via GUI proxy
            for i in range(5):
                def _bg_emit(label=f'bg-{i}'):
                    try:
                        # advance z by one (wrap) to simulate new frame
                        cs = list(getattr(viewer.dims, 'current_step'))
                        cs[0] = int((cs[0] + 1) % max(1, layer.data.shape[0]))
                        viewer.dims.current_step = tuple(cs)
                        layer._update_thumbnail()
                    except Exception:
                        pass
                    maybe_direct_inject(viewer, layer)
                    if args.verify:
                        _schedule_capture(QtCore, args.capture_delay_ms, args.capture_cycles, lambda: _capture_thumb(label, layer))
                proxy.call.emit(_bg_emit)
                time.sleep(0.02)
            # Dims edits must also occur on GUI thread; post them
            if args.ndim == 3:
                proxy.call.emit(lambda: setattr(viewer.dims, 'ndisplay', 3))
                time.sleep(0.01)
                proxy.call.emit(lambda: setattr(viewer.dims, 'ndisplay', 2))
            # Step changes
            def _set_step(z: int):
                try:
                    cs = list(getattr(viewer.dims, 'current_step'))
                    cs[0] = int(z)
                    viewer.dims.current_step = tuple(cs)
                except Exception:
                    try:
                        viewer.dims.set_point(0, int(z))
                    except Exception:
                        pass
            for z in (1, 2, 3, 4, 5):
                proxy.call.emit(lambda z=z: _set_step(z))
                time.sleep(0.005)

        threading.Thread(target=bg_worker, daemon=True).start()

    # Let the event loop run a bit to service queued calls and dims sequence
    def finish():
        log("finishing…")
        app.quit()

    QtCore.QTimer.singleShot(750, finish)
    app.exec_()

    # Probe if UI created a Qt widget for the layer (optional diagnostics)
    try:
        qlayers = getattr(qt_viewer, 'layers', None)
        info = str(type(qlayers))
        log(f"qt layers widget type: {info}")
    except Exception:
        pass

    log("done")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
