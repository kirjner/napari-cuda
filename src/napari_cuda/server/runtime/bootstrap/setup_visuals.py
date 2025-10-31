"""Visual registration helpers for the viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from napari_cuda.server.runtime.worker.egl import EGLRendererWorker


class _VisualHandle:
    """Explicit handle for a VisPy visual node managed by the worker."""

    def __init__(self, node: Any, order: int) -> None:
        self.node = node
        self.order = order
        self._attached = False

    def attach(self, view: Any) -> None:
        scene_parent = view.scene
        if self.node.parent is not scene_parent:
            view.add(self.node)
        self.node.order = self.order
        self.node.visible = True
        self._attached = True

    def detach(self) -> None:
        self.node.visible = False
        self.node.parent = None
        self._attached = False

    def is_attached(self) -> bool:
        return self._attached


def _register_plane_visual(worker: EGLRendererWorker, node: Any) -> None:
    worker._plane_visual_handle = _VisualHandle(node, order=10_000)


def _register_volume_visual(worker: EGLRendererWorker, node: Any) -> None:
    worker._volume_visual_handle = _VisualHandle(node, order=10_010)


def _ensure_plane_visual(worker: EGLRendererWorker) -> Any:
    view = worker.view
    assert view is not None, "VisPy view must exist before activating the plane visual"
    handle = worker._plane_visual_handle
    assert handle is not None, "plane visual not registered"
    if worker._volume_visual_handle is not None:
        worker._volume_visual_handle.detach()  # type: ignore[attr-defined]
    handle.attach(view)
    return handle.node


def _ensure_volume_visual(worker: EGLRendererWorker) -> Any:
    view = worker.view
    assert view is not None, "VisPy view must exist before activating the volume visual"
    handle = worker._volume_visual_handle
    assert handle is not None, "volume visual not registered"
    if worker._plane_visual_handle is not None:
        worker._plane_visual_handle.detach()  # type: ignore[attr-defined]
    handle.attach(view)
    return handle.node


__all__ = [
    "_VisualHandle",
    "_ensure_plane_visual",
    "_ensure_volume_visual",
    "_register_plane_visual",
    "_register_volume_visual",
]
