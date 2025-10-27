from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
from napari._vispy.layers.image import _napari_cmap_to_vispy
from napari.utils.colormaps.colormap_utils import ensure_colormap

from napari_cuda.server.runtime.scene_state_applier import (
    SceneDrainResult,
    SceneStateApplier,
    SceneStateApplyContext,
)
from napari_cuda.server.runtime.render_ledger_snapshot import RenderLedgerSnapshot
from napari_cuda.server.runtime.render_update_mailbox import RenderUpdateMailbox
from napari_cuda.server.runtime.scene_types import SliceROI


class _StubViewer:
    def __init__(self) -> None:
        self.dims = SimpleNamespace(current_step=(0, 0, 0))


class _StubLayer:
    def __init__(self) -> None:
        self.data = None
        self.translate = (0.0, 0.0)
        self.visible = False
        self.opacity = 0.0
        self.blending = ""
        self.contrast_limits = [0.0, 1.0]
        self.gamma = 1.0


class _StubVisual:
    def __init__(self) -> None:
        self.data = None

    def set_data(self, slab) -> None:  # pragma: no cover - simple setter
        self.data = slab


class _StubCamera:
    def __init__(self) -> None:
        self.ranges = []
        self.center = (0.0, 0.0, 0.0)
        self.zoom = 1.0
        self.angles = (0.0, 0.0, 0.0)

    def set_range(self, *, x, y) -> None:
        self.ranges.append((tuple(x), tuple(y)))


class _StubSceneSource:
    def __init__(self, axes: tuple[str, ...], shape: tuple[int, ...]) -> None:
        self.axes = axes
        self._shape = shape
        self._current_step = tuple(0 for _ in shape)

    @property
    def current_step(self) -> tuple[int, ...]:
        return self._current_step

    def level_shape(self, _: int) -> tuple[int, ...]:
        return self._shape

    def level_scale(self, _: int) -> tuple[float, ...]:
        if len(self._shape) >= 3:
            return (1.0, 0.5, 0.25)
        return (0.5, 0.25)

    def set_current_slice(self, step: tuple[int, ...], level: int) -> tuple[int, ...]:
        self._current_step = tuple(int(x) for x in step)
        return self._current_step


def _plane_scale_for_level(_source: _StubSceneSource, _level: int) -> tuple[float, float]:
    return 0.5, 0.25


def _load_slice(_source: _StubSceneSource, _level: int, z_index: int):
    base = z_index * 10
    return np.array([[base + 0, base + 1], [base + 2, base + 3]], dtype=float)


def test_apply_dims_and_slice_updates_viewer_and_source_metadata() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    plane_visual = _StubVisual()
    camera = _StubCamera()
    source = _StubSceneSource(("z", "y", "x"), (3, 4, 5))

    mark_calls: list[None] = []
    idr_requests: list[None] = []

    ctx = SceneStateApplyContext(
        use_volume=False,
        viewer=viewer,
        camera=camera,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=0,
        last_roi=(0, SliceROI(2, 4, 6, 8)),
        preserve_view_on_switch=False,
        sticky_contrast=False,
        idr_on_z=True,
        data_wh=(256, 256),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: mark_calls.append(None),
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: SimpleNamespace(),
        request_encoder_idr=lambda: idr_requests.append(None),
    )

    result = SceneStateApplier.apply_dims_and_slice(ctx, current_step=(1, 1, 1))

    assert viewer.dims.current_step == (1, 1, 1)
    assert layer.data is None
    assert layer.translate == (0.0, 0.0)
    assert layer.visible is False
    assert layer.opacity == 0.0
    assert layer.blending == ""
    assert layer.contrast_limits == [0.0, 1.0]
    assert camera.ranges == []
    assert len(mark_calls) == 1
    assert len(idr_requests) == 1
    assert result.z_index == 1
    assert result.last_step == (1, 1, 1)
    assert source.current_step == (1, 0, 0)


def test_apply_dims_and_slice_does_not_reset_camera_when_preserving_view() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    plane_visual = _StubVisual()
    camera = _StubCamera()
    source = _StubSceneSource(("z", "y", "x"), (3, 4, 5))

    ctx = SceneStateApplyContext(
        use_volume=False,
        viewer=viewer,
        camera=camera,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=True,
        idr_on_z=False,
        data_wh=(256, 256),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: SimpleNamespace(),
    )

    result = SceneStateApplier.apply_dims_and_slice(ctx, current_step=(2, 3, 4))

    assert camera.ranges == []
    assert layer.data is None
    assert result.z_index == 2
    assert result.last_step == (2, 3, 4)


def test_apply_dims_and_slice_when_z_unchanged_marks_render_only() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    plane_visual = _StubVisual()
    camera = _StubCamera()
    source = _StubSceneSource(("z", "y", "x"), (3, 4, 5))

    mark_calls: list[None] = []

    ctx = SceneStateApplyContext(
        use_volume=False,
        viewer=viewer,
        camera=camera,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=1,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=True,
        idr_on_z=False,
        data_wh=(256, 256),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: mark_calls.append(None),
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: SimpleNamespace(),
        request_encoder_idr=lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    result = SceneStateApplier.apply_dims_and_slice(ctx, current_step=(1, 2, 3))

    assert mark_calls == [None]
    assert result.z_index == 1
    assert result.last_step == (1, 2, 3)


def test_apply_volume_params_sets_visual_fields() -> None:
    viewer = _StubViewer()
    visual = SimpleNamespace(method="", cmap=None, clim=None, opacity=0.0, relative_step_size=1.0)
    plane_visual = SimpleNamespace()

    ctx = SceneStateApplyContext(
        use_volume=True,
        viewer=viewer,
        camera=None,
        layer=None,
        scene_source=None,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=True,
        idr_on_z=False,
        data_wh=(0, 0),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: None,
        plane_scale_for_level=lambda *_: (1.0, 1.0),
        load_slice=lambda *_: None,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: visual,
        request_encoder_idr=None,
    )

    SceneStateApplier.apply_volume_params(
        ctx,
        mode="MIP",
        colormap="gray",
        clim=(2.0, 1.0),
        opacity=0.7,
        sample_step=0.2,
    )

    assert visual.method == "mip"
    expected_gray = _napari_cmap_to_vispy(ensure_colormap("gray"))
    assert hasattr(visual.cmap, "colors")
    assert np.allclose(np.asarray(visual.cmap.colors), np.asarray(expected_gray.colors))
    assert visual.clim == (1.0, 2.0)
    assert visual.opacity == 0.7
    assert visual.relative_step_size == 0.2


def test_apply_volume_params_accepts_named_colormap() -> None:
    viewer = _StubViewer()
    visual = SimpleNamespace(method="", cmap=None, clim=None, opacity=0.0, relative_step_size=1.0)
    plane_visual = SimpleNamespace()

    ctx = SceneStateApplyContext(
        use_volume=True,
        viewer=viewer,
        camera=None,
        layer=None,
        scene_source=None,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=True,
        idr_on_z=False,
        data_wh=(0, 0),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: None,
        plane_scale_for_level=lambda *_: (1.0, 1.0),
        load_slice=lambda *_: None,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: visual,
        request_encoder_idr=None,
    )

    SceneStateApplier.apply_volume_params(ctx, colormap="green")

    expected_green = _napari_cmap_to_vispy(ensure_colormap("green"))
    assert hasattr(visual.cmap, "colors")
    assert np.allclose(np.asarray(visual.cmap.colors), np.asarray(expected_green.colors))


def test_apply_layer_updates_sets_gamma_on_visual() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    visual = SimpleNamespace(gamma=1.0)
    plane_visual = SimpleNamespace(gamma=1.0)

    ctx = SceneStateApplyContext(
        use_volume=True,
        viewer=viewer,
        camera=None,
        layer=layer,
        scene_source=None,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=True,
        idr_on_z=False,
        data_wh=(0, 0),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: None,
        plane_scale_for_level=lambda *_: (1.0, 1.0),
        load_slice=lambda *_: None,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: visual,
        request_encoder_idr=None,
    )

    SceneStateApplier.apply_layer_updates(
        ctx,
        updates={
            "layer-0": {
                "gamma": 0.5,
            }
        },
    )

    assert layer.gamma == 0.5
    assert visual.gamma == 0.5


def test_apply_volume_layer_resets_translate() -> None:
    viewer = _StubViewer()
    camera = _StubCamera()
    source = _StubSceneSource(("z", "y", "x"), (3, 4, 5))
    layer = SimpleNamespace(translate=(5.0, 6.0, 7.0), contrast_limits=[0.0, 1.0], data=None, scale=(1.0, 1.0, 1.0))
    plane_visual = SimpleNamespace()
    volume_visual = SimpleNamespace()

    ctx = SceneStateApplyContext(
        use_volume=True,
        viewer=viewer,
        camera=camera,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=False,
        idr_on_z=False,
        data_wh=(256, 256),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: volume_visual,
    )

    data = np.ones((3, 4, 5), dtype=float)
    SceneStateApplier.apply_volume_layer(ctx, volume=data, contrast=(0.0, 1.0))

    assert layer.translate == (0.0, 0.0, 0.0)


def test_drain_updates_records_render_and_policy_without_camera() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    plane_visual = _StubVisual()
    source = _StubSceneSource(("z", "y", "x"), (4, 4, 4))

    marks: list[None] = []

    ctx = SceneStateApplyContext(
        use_volume=False,
        viewer=viewer,
        camera=None,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=0,
        last_roi=None,
        preserve_view_on_switch=True,
        sticky_contrast=False,
        idr_on_z=False,
        data_wh=(128, 128),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: marks.append(None),
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: SimpleNamespace(),
        request_encoder_idr=None,
    )

    queue = RenderUpdateMailbox()
    state = RenderLedgerSnapshot(current_step=(1, 0, 0))

    result = SceneStateApplier.drain_updates(ctx, state=state, mailbox=queue)

    assert isinstance(result, SceneDrainResult)
    assert result.render_marked is True
    assert result.policy_refresh_needed is True
    assert marks == [None]


def test_drain_updates_applies_camera_fields_and_signature() -> None:
    viewer = _StubViewer()
    layer = _StubLayer()
    plane_visual = _StubVisual()
    camera = _StubCamera()
    source = _StubSceneSource(("z", "y", "x"), (4, 4, 4))

    ctx = SceneStateApplyContext(
        use_volume=False,
        viewer=viewer,
        camera=camera,
        layer=layer,
        scene_source=source,
        active_ms_level=0,
        z_index=None,
        last_roi=None,
        preserve_view_on_switch=False,
        sticky_contrast=False,
        idr_on_z=False,
        data_wh=(64, 64),
        state_lock=threading.Lock(),
        ensure_scene_source=lambda: source,
        plane_scale_for_level=_plane_scale_for_level,
        load_slice=_load_slice,
        mark_render_tick_needed=lambda: None,
        ensure_plane_visual=lambda: plane_visual,
        ensure_volume_visual=lambda: SimpleNamespace(),
        request_encoder_idr=None,
    )

    queue = RenderUpdateMailbox()
    state = RenderLedgerSnapshot(
        current_step=(1, 0, 0),
        plane_center=(5.0, 6.0),
        plane_zoom=0.5,
    )

    result = SceneStateApplier.drain_updates(ctx, state=state, mailbox=queue)

    assert result.z_index == 1
    assert result.data_wh is None
    assert result.render_marked is True
    assert result.policy_refresh_needed is True
    assert camera.center == (5.0, 6.0)
    assert camera.zoom == 0.5
