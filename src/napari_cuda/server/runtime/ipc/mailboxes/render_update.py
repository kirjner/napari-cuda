"""Render-thread mailbox for coalescing scene updates before each frame."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Optional

from napari_cuda.server.runtime.snapshots import RenderLedgerSnapshot
from napari_cuda.server.runtime.viewport import PlaneState, RenderMode, VolumeState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderUpdate:
    """Latest-wins state drained by the render worker."""

    scene_state: Optional[RenderLedgerSnapshot]
    mode: Optional[RenderMode] = None
    plane_state: Optional[PlaneState] = None
    volume_state: Optional[VolumeState] = None
    op_seq: Optional[int] = None


@dataclass(frozen=True)
class RenderZoomHint:
    """Latest zoom hint emitted by the control loop."""

    ratio: float
    timestamp: float


class RenderUpdateMailbox:
    """Thread-safe mailbox that coalesces render updates for the worker."""

    def __init__(self, *, time_fn: Callable[[], float] = time.perf_counter) -> None:
        self._time_fn = time_fn
        self._scene_state: Optional[RenderLedgerSnapshot] = None
        self._scene_op_seq: Optional[int] = None
        self._mode: Optional[RenderMode] = None
        self._plane_state: Optional[PlaneState] = None
        self._volume_state: Optional[VolumeState] = None
        self._zoom_hint: Optional[RenderZoomHint] = None
        self._camera_ops: list = []
        self._last_signature: Optional[tuple] = None
        self._lock = threading.Lock()

    def set_scene_state(self, state: RenderLedgerSnapshot) -> None:
        op_seq = int(state.op_seq) if state.op_seq is not None else 0
        with self._lock:
            if self._scene_op_seq is not None:
                current_op = int(self._scene_op_seq)
                if op_seq < current_op:
                    return
                if op_seq == current_op:
                    signature = self._build_signature(state)
                    if signature == self._last_signature:
                        return
            self._scene_state = state
            self._scene_op_seq = op_seq

    def set_viewport_state(
        self,
        *,
        mode: Optional[RenderMode],
        plane_state: Optional[PlaneState],
        volume_state: Optional[VolumeState],
    ) -> None:
        with self._lock:
            self._mode = mode
            self._plane_state = deepcopy(plane_state) if plane_state is not None else None
            self._volume_state = deepcopy(volume_state) if volume_state is not None else None

    def drain(self) -> RenderUpdate:
        with self._lock:
            op_seq = self._scene_op_seq
            drained = RenderUpdate(
                scene_state=self._scene_state,
                mode=self._mode,
                plane_state=self._plane_state,
                volume_state=self._volume_state,
                op_seq=op_seq,
            )
            self._scene_state = None
            self._mode = None
            self._plane_state = None
            self._volume_state = None
            self._scene_op_seq = op_seq
        return drained

    def record_zoom_hint(self, ratio: float, *, timestamp: Optional[float] = None) -> None:
        if ratio <= 0.0:
            raise ValueError("zoom ratio must be positive")
        ts = self._time_fn() if timestamp is None else float(timestamp)
        with self._lock:
            self._zoom_hint = RenderZoomHint(float(ratio), ts)

    def consume_zoom_hint(self, max_age: float) -> Optional[RenderZoomHint]:
        now = self._time_fn()
        with self._lock:
            zoom = self._zoom_hint
            if zoom is None:
                return None
            age = now - float(zoom.timestamp)
            if age > float(max_age):
                self._zoom_hint = None
                return None
            self._zoom_hint = None
            return zoom

    # ---- Camera ops ---------------------------------------------------------
    def append_camera_ops(self, ops) -> None:
        """Append a batch of camera ops for the render loop to consume.

        Type of ``ops`` is intentionally untyped here to avoid a hard import
        on the camera command class; we treat them as opaque objects and hand
        them to the existing camera controller on the render thread.
        """
        if not ops:
            return
        with self._lock:
            self._camera_ops.extend(list(ops))

    def drain_camera_ops(self):
        with self._lock:
            if not self._camera_ops:
                return []
            drained = list(self._camera_ops)
            self._camera_ops.clear()
            return drained

    def update_state_signature(self, state: RenderLedgerSnapshot) -> bool:
        signature = self._build_signature(state)
        with self._lock:
            if signature == self._last_signature:
                return False
            self._last_signature = signature
            return True

    @staticmethod
    def _build_signature(state: RenderLedgerSnapshot) -> tuple:
        def _canonical(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return tuple(_canonical(v) for v in value)
            if isinstance(value, dict):
                return tuple(sorted((str(k), _canonical(v)) for k, v in value.items()))
            return value

        dims_token: tuple
        if state.dims_version is not None:
            dims_token = ("dv", int(state.dims_version))
        else:
            dims_token = (
                "dvals",
                _canonical(state.current_step),
                _canonical(state.order),
                _canonical(state.displayed),
            )

        view_token: tuple
        if state.view_version is not None:
            view_token = ("vv", int(state.view_version))
        else:
            view_token = (
                "vvals",
                int(state.ndisplay) if state.ndisplay is not None else None,
                _canonical(state.displayed),
                _canonical(state.order),
            )

        multiscale_token: tuple
        if state.multiscale_level_version is not None:
            multiscale_token = ("lv", int(state.multiscale_level_version))
        else:
            multiscale_token = (
                "lvals",
                int(state.current_level) if state.current_level is not None else None,
                _canonical(state.level_shapes),
            )

        if state.camera_versions:
            camera_token = (
                "cv",
                tuple(sorted((str(k), int(v)) for k, v in state.camera_versions.items())),
            )
        else:
            camera_token = (
                "cvals",
                (
                    _canonical(state.plane_center),
                    _canonical(state.plane_zoom),
                    _canonical(state.plane_rect),
                    _canonical(state.volume_center),
                    _canonical(state.volume_angles),
                    _canonical(state.volume_distance),
                    _canonical(state.volume_fov),
                ),
            )

        layer_token = None
        if state.layer_versions:
            version_items: list[tuple[str, str, int]] = []
            for layer_id, props in state.layer_versions.items():
                if not props:
                    continue
                for prop, version in props.items():
                    version_items.append((str(layer_id), str(prop), int(version)))
            if version_items:
                layer_token = ("lver", tuple(sorted(version_items)))
        elif state.layer_values:
            layer_items = []
            for layer_id, props in state.layer_values.items():
                if not props:
                    continue
                normalized = tuple(
                    sorted((str(key), _canonical(val)) for key, val in props.items())
                )
                layer_items.append((str(layer_id), normalized))
            if layer_items:
                layer_token = ("lvals", tuple(sorted(layer_items)))

        volume_token = (
            _canonical(state.volume_mode),
            _canonical(state.volume_colormap),
            _canonical(state.volume_clim),
            _canonical(state.volume_opacity),
            _canonical(state.volume_sample_step),
        )

        return (
            dims_token,
            view_token,
            multiscale_token,
            camera_token,
            layer_token,
            volume_token,
        )


__all__ = [
    "RenderUpdate",
    "RenderUpdateMailbox",
    "RenderZoomHint",
]
