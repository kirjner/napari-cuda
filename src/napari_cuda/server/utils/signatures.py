"""Canonical, content-based signature helpers shared across server subsystems.

This module defines content signatures that represent "what the user sees" for
scenes, layers, and dims/pose state. They are op_seq-free and suitable for
deduping outward emissions or coalescing updates by content equality.

Note: legacy RenderSignature has been removed in favor of content signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple
import hashlib

from napari_cuda.server.scene import LayerVisualState, RenderLedgerSnapshot

# Layer properties that must be excluded from visual version signatures because
# they are derived emissions or noisy for content-dedupe (would cause feedback).
_EXCLUDED_LAYER_VERSION_KEYS: tuple[str, ...] = (
    "thumbnail",
    "metadata",
)

# Pixel-affecting visual keys (base layer properties)
_PIXEL_VISUAL_KEYS: tuple[str, ...] = (
    "visible",
    "opacity",
    "blending",
    "interpolation",
    "colormap",
    "gamma",
    "contrast_limits",
    "depiction",
    "rendering",
    "attenuation",
    "iso_threshold",
)


@dataclass(frozen=True)
class CameraSignature:
    """Normalized lens of camera-related version counters."""

    versions: Tuple[Tuple[str, int], ...]

    @staticmethod
    def _normalize_items(items: Iterable[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted((str(key), int(value)) for key, value in items))

    @classmethod
    def from_snapshot(cls, snapshot: RenderLedgerSnapshot) -> "CameraSignature":
        camera_versions = snapshot.camera_versions or {}
        return cls(cls._normalize_items(camera_versions.items()))

    @classmethod
    def from_mapping(cls, mapping: Optional[dict[str, int]]) -> "CameraSignature":
        if not mapping:
            return cls(())
        return cls(cls._normalize_items(mapping.items()))


@dataclass(frozen=True)
class DimsSignature:
    """Canonical dims signature used by worker/applying helpers."""

    ndisplay: Optional[int]
    order: Tuple[int, ...]
    displayed: Tuple[int, ...]
    current_step: Tuple[int, ...]
    current_level: Optional[int]
    axis_labels: Tuple[str, ...]

    @classmethod
    def from_snapshot(cls, snapshot: RenderLedgerSnapshot) -> "DimsSignature":
        ndisplay = int(snapshot.ndisplay) if snapshot.ndisplay is not None else None
        order = tuple(int(v) for v in (snapshot.order or ()))
        displayed = tuple(int(v) for v in (snapshot.displayed or ()))
        current_step = tuple(int(v) for v in (snapshot.current_step or ()))
        current_level = int(snapshot.current_level) if snapshot.current_level is not None else None
        axis_labels = tuple(str(v) for v in (snapshot.axis_labels or ()))
        return cls(
            ndisplay=ndisplay,
            order=order,
            displayed=displayed,
            current_step=current_step,
            current_level=current_level,
            axis_labels=axis_labels,
        )

# -----------------------------------------------------------------------------
# Content-based signatures for layers and scenes (op_seq-free)


def _qf(value: Optional[float], *, places: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), places)


def _canon(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(_canon(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((str(k), _canon(v)) for k, v in val.items()))
    return val


def _dims_token(snapshot: RenderLedgerSnapshot) -> tuple:
    """Canonical dims token tuple for content gating."""
    return (
        ("ndisplay", None if snapshot.ndisplay is None else int(snapshot.ndisplay)),
        ("order", tuple(int(v) for v in (snapshot.order or ()))),
        ("displayed", tuple(int(v) for v in (snapshot.displayed or ()))),
        ("current_step", tuple(int(v) for v in (snapshot.current_step or ()))),
        (
            "current_level",
            None if snapshot.current_level is None else int(snapshot.current_level),
        ),
        ("axis_labels", tuple(str(v) for v in (snapshot.axis_labels or ()))),
    )


def _camera_token(snapshot: RenderLedgerSnapshot) -> tuple:
    """Stable camera token based on counters, not raw floats."""
    cv = snapshot.camera_versions or {}
    return tuple(sorted((str(k), int(v)) for k, v in cv.items()))


def _filter_visual_versions(
    versions: Mapping[str, int] | None,
    *,
    ndisplay: Optional[int],
) -> tuple[tuple[str, int], ...]:
    if not versions:
        return ()
    nd = int(ndisplay) if ndisplay is not None else 2
    items: list[tuple[str, int]] = []
    for k, v in versions.items():
        sk = str(k)
        if sk in _EXCLUDED_LAYER_VERSION_KEYS:
            continue
        if sk in _PIXEL_VISUAL_KEYS:
            items.append((sk, int(v)))
            continue
        # Include volume.* extras only for 3D mode
        if nd >= 3 and sk.startswith("volume."):
            items.append((sk, int(v)))
    return tuple(sorted(items))


def layer_token(
    snapshot: RenderLedgerSnapshot,
    layer_id: str,
    *,
    dataset_id: Optional[str] = None,
    include_camera: bool = False,
) -> tuple:
    """Inputs-only content token for a single layer, for gating expensive work.

    Never includes outputs, timestamps, or non-visual metadata.
    """
    lv = (snapshot.layer_values or {}).get(str(layer_id))
    visuals = _filter_visual_versions(getattr(lv, "versions", {}), ndisplay=snapshot.ndisplay)
    token: list[tuple[Any, Any]] = [
        ("layer", str(layer_id)),
        ("dataset", dataset_id),
        ("dims", _dims_token(snapshot)),
        ("visuals", visuals),
    ]
    if include_camera:
        token.append(("camera", _camera_token(snapshot)))
    return tuple(token)


def scene_token(
    snapshot: RenderLedgerSnapshot,
    *,
    dataset_id: Optional[str] = None,
    include_camera: bool = False,
) -> tuple:
    """Inputs-only content token for the entire scene.

    Useful for gating global work; not required for thumbnails.
    """
    layers_visuals: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    if snapshot.layer_values:
        for layer_id, state in sorted(snapshot.layer_values.items()):
            if isinstance(state, LayerVisualState):
                items = _filter_visual_versions(state.versions, ndisplay=snapshot.ndisplay)
                layers_visuals.append((str(layer_id), items))
    token: list[tuple[Any, Any]] = [
        ("dataset", dataset_id),
        ("dims", _dims_token(snapshot)),
        ("layers_visuals", tuple(layers_visuals)),
        (
            "volume_settings",
            (
                _canon(snapshot.volume_mode),
                _canon(snapshot.volume_colormap),
                _canon(snapshot.volume_clim),
                _canon(snapshot.volume_opacity),
                _canon(snapshot.volume_sample_step),
            ),
        ),
    ]
    if include_camera:
        token.append(("camera", _camera_token(snapshot)))
    return tuple(token)


@dataclass(frozen=True)
class PlanePoseSignature:
    center: Optional[tuple[float, float]]
    zoom: Optional[float]
    rect: Optional[tuple[float, float, float, float]]

    @classmethod
    def from_snapshot(cls, snapshot: RenderLedgerSnapshot) -> "PlanePoseSignature":
        center = None
        if snapshot.plane_center is not None and len(snapshot.plane_center) >= 2:
            center = (_qf(snapshot.plane_center[0]), _qf(snapshot.plane_center[1]))  # type: ignore[index]
        zoom = _qf(snapshot.plane_zoom)
        rect = None
        if snapshot.plane_rect is not None and len(snapshot.plane_rect) >= 4:
            rect = tuple(_qf(float(v)) for v in snapshot.plane_rect[:4])  # type: ignore[assignment]
        return cls(center=center, zoom=zoom, rect=rect)


@dataclass(frozen=True)
class VolumePoseSignature:
    center: Optional[tuple[float, float, float]]
    angles: Optional[tuple[float, float, float]]
    distance: Optional[float]
    fov: Optional[float]

    @classmethod
    def from_snapshot(cls, snapshot: RenderLedgerSnapshot) -> "VolumePoseSignature":
        center = None
        if snapshot.volume_center is not None and len(snapshot.volume_center) >= 3:
            center = tuple(_qf(float(v)) for v in snapshot.volume_center[:3])  # type: ignore[assignment]
        angles = None
        if snapshot.volume_angles is not None and len(snapshot.volume_angles) >= 2:
            ang2 = _qf(snapshot.volume_angles[2]) if len(snapshot.volume_angles) >= 3 else _qf(0.0)
            angles = (
                _qf(snapshot.volume_angles[0]),
                _qf(snapshot.volume_angles[1]),
                ang2,
            )  # type: ignore[index]
        distance = _qf(snapshot.volume_distance)
        fov = _qf(snapshot.volume_fov)
        return cls(center=center, angles=angles, distance=distance, fov=fov)


@dataclass(frozen=True)
class LayerVisualSignature:
    versions: Tuple[Tuple[str, int], ...]

    @classmethod
    def from_layer_state(cls, state: LayerVisualState) -> "LayerVisualSignature":
        if not state.versions:
            return cls(())
        items = tuple(
            sorted(
                (str(k), int(v))
                for k, v in state.versions.items()
                if str(k) not in _EXCLUDED_LAYER_VERSION_KEYS
            )
        )
        return cls(items)


@dataclass(frozen=True)
class DatasetSignature:
    dataset_id: Optional[str]


@dataclass(frozen=True)
class LayerContentSignature:
    layer_id: str
    dataset: DatasetSignature
    dims: DimsSignature
    pose_plane: Optional[PlanePoseSignature]
    pose_volume: Optional[VolumePoseSignature]
    visuals: LayerVisualSignature


@dataclass(frozen=True)
class SceneContentSignature:
    dataset: DatasetSignature
    dims: DimsSignature
    pose_plane: Optional[PlanePoseSignature]
    pose_volume: Optional[VolumePoseSignature]
    layers_visuals: Tuple[Tuple[str, LayerVisualSignature], ...]
    volume_settings: Tuple[Any, ...]


def build_layer_content_signature(
    snapshot: RenderLedgerSnapshot,
    layer_state: LayerVisualState,
    *,
    dataset_id: Optional[str] = None,
) -> LayerContentSignature:
    dims = DimsSignature.from_snapshot(snapshot)
    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    pose_plane = PlanePoseSignature.from_snapshot(snapshot) if nd < 3 else None
    pose_volume = VolumePoseSignature.from_snapshot(snapshot) if nd >= 3 else None
    visuals = LayerVisualSignature.from_layer_state(layer_state)
    ds = DatasetSignature(dataset_id=dataset_id)
    return LayerContentSignature(
        layer_id=str(layer_state.layer_id),
        dataset=ds,
        dims=dims,
        pose_plane=pose_plane,
        pose_volume=pose_volume,
        visuals=visuals,
    )


def build_scene_content_signature(
    snapshot: RenderLedgerSnapshot,
    *,
    dataset_id: Optional[str] = None,
) -> SceneContentSignature:
    dims = DimsSignature.from_snapshot(snapshot)
    nd = int(snapshot.ndisplay) if snapshot.ndisplay is not None else 2
    pose_plane = PlanePoseSignature.from_snapshot(snapshot) if nd < 3 else None
    pose_volume = VolumePoseSignature.from_snapshot(snapshot) if nd >= 3 else None

    layers_visuals: list[tuple[str, LayerVisualSignature]] = []
    if snapshot.layer_values:
        for layer_id, state in sorted(snapshot.layer_values.items()):
            if isinstance(state, LayerVisualState):
                layers_visuals.append((str(layer_id), LayerVisualSignature.from_layer_state(state)))

    volume_settings = (
        _canon(snapshot.volume_mode),
        _canon(snapshot.volume_colormap),
        _canon(snapshot.volume_clim),
        _canon(snapshot.volume_opacity),
        _canon(snapshot.volume_sample_step),
    )
    ds = DatasetSignature(dataset_id=dataset_id)
    return SceneContentSignature(
        dataset=ds,
        dims=dims,
        pose_plane=pose_plane,
        pose_volume=pose_volume,
        layers_visuals=tuple(layers_visuals),
        volume_settings=volume_settings,
    )


def scene_content_signature_tuple(
    snapshot: RenderLedgerSnapshot,
    *,
    dataset_id: Optional[str] = None,
) -> Tuple[Any, ...]:
    sig = build_scene_content_signature(snapshot, dataset_id=dataset_id)
    plane = sig.pose_plane
    volume = sig.pose_volume
    return (
        ("dataset", sig.dataset.dataset_id),
        ("dims", sig.dims),
        (
            "plane",
            None if plane is None else (plane.center, plane.zoom, plane.rect),
        ),
        (
            "volume",
            None
            if volume is None
            else (volume.center, volume.angles, volume.distance, volume.fov),
        ),
        ("layers_visuals", tuple((lid, lv.versions) for lid, lv in sig.layers_visuals)),
        ("volume_settings", sig.volume_settings),
    )


def dims_payload_signature(payload: Any) -> Tuple[Any, ...]:
    """Canonical dims payload signature (for notify.dims)."""
    levels_sig = tuple(tuple(sorted(level.items())) for level in payload.levels)
    return (
        payload.current_step,
        payload.current_level,
        payload.ndisplay,
        payload.mode,
        payload.displayed,
        payload.axis_labels,
        payload.order,
        payload.labels,
        levels_sig,
        payload.level_shapes,
        payload.downgraded,
    )


def layer_payload_signature(state: LayerVisualState) -> Tuple[Any, ...]:
    """Canonical signature of an outbound LayerVisualState subset by values.

    The subset is typically produced via LayerVisualState.subset(props), which
    limits the keys present. We sign only the visible values, not versions.
    """
    keys = tuple(sorted(state.keys()))
    items: list[tuple[str, Any]] = []
    for key in keys:
        items.append((str(key), _canon(state.get(key))))
    return tuple(items)


def signature_hash_tuple(token: Tuple[Any, ...]) -> str:
    """Stable hex hash for a signature tuple."""
    data = repr(token).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "CameraSignature",
    "DimsSignature",
    # Content signatures
    "PlanePoseSignature",
    "VolumePoseSignature",
    "LayerVisualSignature",
    "DatasetSignature",
    "LayerContentSignature",
    "SceneContentSignature",
    "build_layer_content_signature",
    "build_scene_content_signature",
    "scene_content_signature_tuple",
    "dims_payload_signature",
    "layer_payload_signature",
    "signature_hash_tuple",
]
