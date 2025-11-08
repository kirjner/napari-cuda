"""Serialized viewport intent contract shared between reducers and worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from napari_cuda.server.scene.viewport import (
    PlaneState,
    RenderMode,
    VolumeState,
)
from napari_cuda.server.state_ledger import ServerStateLedger
from napari_cuda.shared.dims_spec import (
    DimsSpec,
    dims_spec_from_payload,
    dims_spec_to_payload,
)


def _as_tuple(value: Sequence[Any] | None, *, item_type: type = int) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(item_type(v) for v in value)


@dataclass(frozen=True)
class PlanePoseIntent:
    """Plane camera pose + cached ROI metadata."""

    rect: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    zoom: float | None = None
    roi_signature: tuple[int, int, int, int] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.rect is not None:
            payload["rect"] = [float(v) for v in self.rect]
        if self.center is not None:
            payload["center"] = [float(v) for v in self.center]
        if self.zoom is not None:
            payload["zoom"] = float(self.zoom)
        if self.roi_signature is not None:
            payload["roi_signature"] = [int(v) for v in self.roi_signature]
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, Any] | None) -> PlanePoseIntent | None:
        if payload is None:
            return None
        rect_payload = payload["rect"]
        center_payload = payload["center"]
        sig_payload = payload["roi_signature"]
        zoom_payload = payload["zoom"]
        return PlanePoseIntent(
            rect=tuple(float(v) for v in rect_payload),
            center=tuple(float(v) for v in center_payload),
            zoom=float(zoom_payload),
            roi_signature=tuple(int(v) for v in sig_payload),
        )


@dataclass(frozen=True)
class VolumePoseIntent:
    """Volume camera pose + level metadata."""

    center: tuple[float, float, float] | None = None
    angles: tuple[float, float, float] | None = None
    distance: float | None = None
    fov: float | None = None
    level: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.center is not None:
            payload["center"] = [float(v) for v in self.center]
        if self.angles is not None:
            payload["angles"] = [float(v) for v in self.angles]
        if self.distance is not None:
            payload["distance"] = float(self.distance)
        if self.fov is not None:
            payload["fov"] = float(self.fov)
        if self.level is not None:
            payload["level"] = int(self.level)
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, Any] | None) -> VolumePoseIntent | None:
        if payload is None:
            return None
        center_payload = payload["center"]
        angles_payload = payload["angles"]
        distance_payload = payload["distance"]
        fov_payload = payload["fov"]
        level_payload = payload["level"]
        return VolumePoseIntent(
            center=tuple(float(v) for v in center_payload),
            angles=tuple(float(v) for v in angles_payload),
            distance=float(distance_payload),
            fov=float(fov_payload),
            level=int(level_payload),
        )


@dataclass(frozen=True)
class ViewportIntent:
    """Canonical viewport state that reducers publish and worker consumes."""

    seq: int
    mode: RenderMode
    dims_spec: DimsSpec
    plane_pose: PlanePoseIntent | None = None
    volume_pose: VolumePoseIntent | None = None
    roi_signature: tuple[int, int, int, int] | None = None
    dims_version: int | None = None
    multiscale_version: int | None = None
    plane_state_version: int | None = None
    volume_state_version: int | None = None
    op_seq: int | None = None
    op_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ndisplay(self) -> int:
        return int(self.dims_spec.ndisplay)

    @property
    def axis_order(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.dims_spec.order)

    @property
    def displayed_axes(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.dims_spec.displayed)

    @property
    def current_level(self) -> int:
        return int(self.dims_spec.current_level)

    @property
    def current_step(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.dims_spec.current_step)

    @property
    def level_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(dim) for dim in shape) for shape in self.dims_spec.level_shapes)

    @property
    def levels(self) -> tuple[Mapping[str, Any], ...]:
        return self.dims_spec.levels

    @property
    def labels(self) -> tuple[str, ...] | None:
        return self.dims_spec.labels

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a ledger-friendly mapping."""
        ds_payload = dims_spec_to_payload(self.dims_spec)
        assert ds_payload is not None, "dims_spec serialization failed"
        payload: dict[str, Any] = {
            "seq": int(self.seq),
            "mode": self.mode.name,
            "dims_spec": ds_payload,
        }
        if self.plane_pose is not None:
            payload["plane_pose"] = self.plane_pose.to_payload()
        if self.volume_pose is not None:
            payload["volume_pose"] = self.volume_pose.to_payload()
        if self.roi_signature is not None:
            payload["roi_signature"] = [int(v) for v in self.roi_signature]
        if self.dims_version is not None:
            payload["dims_version"] = int(self.dims_version)
        if self.multiscale_version is not None:
            payload["multiscale_version"] = int(self.multiscale_version)
        if self.plane_state_version is not None:
            payload["plane_state_version"] = int(self.plane_state_version)
        if self.volume_state_version is not None:
            payload["volume_state_version"] = int(self.volume_state_version)
        if self.op_seq is not None:
            payload["op_seq"] = int(self.op_seq)
        if self.op_kind is not None:
            payload["op_kind"] = str(self.op_kind)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> ViewportIntent:
        """Deserialize from ledger payload."""
        mode_value = payload.get("mode", "plane")
        try:
            mode = RenderMode[mode_value.upper()]
        except KeyError:
            mode = RenderMode.PLANE
        dims_spec_payload = payload["dims_spec"]
        dims_spec = dims_spec_from_payload(dims_spec_payload)
        assert dims_spec is not None, "viewport intent dims_spec payload missing"
        plane_pose = PlanePoseIntent.from_payload(payload.get("plane_pose"))  # type: ignore[arg-type]
        volume_pose = VolumePoseIntent.from_payload(payload.get("volume_pose"))  # type: ignore[arg-type]
        roi_signature_payload = payload.get("roi_signature")
        metadata_payload = payload.get("metadata")
        roi_signature = tuple(int(v) for v in roi_signature_payload) if roi_signature_payload is not None else None
        metadata = dict(metadata_payload) if metadata_payload is not None else {}
        return ViewportIntent(
            seq=int(payload["seq"]),
            mode=mode,
            dims_spec=dims_spec,
            plane_pose=plane_pose,
            volume_pose=volume_pose,
            roi_signature=roi_signature,
            dims_version=(
                int(payload["dims_version"]) if "dims_version" in payload else None
            ),
            multiscale_version=(
                int(payload["multiscale_version"])
                if "multiscale_version" in payload
                else None
            ),
            plane_state_version=(
                int(payload["plane_state_version"])
                if "plane_state_version" in payload
                else None
            ),
            volume_state_version=(
                int(payload["volume_state_version"])
                if "volume_state_version" in payload
                else None
            ),
            op_seq=int(payload["op_seq"]) if "op_seq" in payload else None,
            op_kind=str(payload["op_kind"]) if "op_kind" in payload else None,
            metadata=metadata,
        )


__all__ = [
    "PlanePoseIntent",
    "VolumePoseIntent",
    "ViewportIntent",
]


def plane_pose_from_state(state: PlaneState | None) -> PlanePoseIntent | None:
    if state is None:
        return None
    rect = tuple(float(v) for v in state.pose.rect) if state.pose.rect is not None else None
    center = tuple(float(v) for v in state.pose.center) if state.pose.center is not None else None
    zoom = float(state.pose.zoom) if state.pose.zoom is not None else None
    roi_sig = (
        tuple(int(v) for v in state.applied_roi_signature)
        if state.applied_roi_signature is not None
        else None
    )
    return PlanePoseIntent(rect=rect, center=center, zoom=zoom, roi_signature=roi_sig)


def volume_pose_from_state(state: VolumeState | None) -> VolumePoseIntent | None:
    if state is None:
        return None
    center = tuple(float(v) for v in state.pose.center) if state.pose.center is not None else None
    angles = tuple(float(v) for v in state.pose.angles) if state.pose.angles is not None else None
    distance = float(state.pose.distance) if state.pose.distance is not None else None
    fov = float(state.pose.fov) if state.pose.fov is not None else None
    return VolumePoseIntent(
        center=center,
        angles=angles,
        distance=distance,
        fov=fov,
        level=int(state.level) if state.level is not None else None,
    )


def store_viewport_intent(
    ledger: ServerStateLedger,
    intent: ViewportIntent,
    *,
    origin: str,
    timestamp: float,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = intent.to_payload()
    if metadata:
        ledger.batch_record_confirmed(
            [("scene", "main", "viewport_intent", payload, dict(metadata))],
            origin=origin,
            timestamp=timestamp,
        )
    else:
        ledger.batch_record_confirmed(
            [("scene", "main", "viewport_intent", payload)],
            origin=origin,
            timestamp=timestamp,
        )
