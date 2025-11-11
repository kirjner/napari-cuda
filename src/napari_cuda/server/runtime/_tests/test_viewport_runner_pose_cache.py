from __future__ import annotations

from napari_cuda.server.runtime.render_loop.planning import staging as viewport_staging
from napari_cuda.server.runtime.render_loop.planning.viewport_planner import ViewportPlanner
from napari_cuda.server.scene import RenderLedgerSnapshot
from napari_cuda.server.scene.viewport import PlaneViewportCache


def test_update_camera_rect_ignores_none() -> None:
    plane = PlaneViewportCache()
    plane.update_pose(rect=(100.0, 200.0, 300.0, 400.0))
    runner = ViewportPlanner(plane_state=plane)

    # Sanity: explicit rect updates still apply.
    runner.update_camera_rect((10.0, 20.0, 30.0, 40.0))
    assert plane.pose.rect == (10.0, 20.0, 30.0, 40.0)

    # Regression: passing None (e.g. during volume mode) must preserve the cache.
    runner.update_camera_rect(None)
    assert plane.pose.rect == (10.0, 20.0, 30.0, 40.0)


def test_ingest_snapshot_handles_restored_plane_cache() -> None:
    plane = PlaneViewportCache()
    plane.target_level = 2
    plane.target_step = (4, 2, 0)
    plane.applied_level = 2
    plane.applied_step = (4, 2, 0)
    plane.update_pose(rect=(0.0, 0.0, 50.0, 60.0), center=(12.0, 14.0), zoom=2.0)

    runner = ViewportPlanner(plane_state=plane)
    snapshot = RenderLedgerSnapshot(
        plane_rect=plane.pose.rect,
        plane_center=plane.pose.center,
        plane_zoom=plane.pose.zoom,
        current_step=plane.applied_step,
        current_level=plane.applied_level,
        ndisplay=2,
    )

    runner.ingest_snapshot(snapshot)

    assert runner.state.request.level == plane.applied_level
    assert runner.state.request.step == plane.applied_step
    assert runner.state.pose.rect == plane.pose.rect


def test_plane_cache_from_snapshot_applies_pose() -> None:
    snapshot = RenderLedgerSnapshot(
        plane_rect=(1.0, 2.0, 3.0, 4.0),
        plane_center=(5.0, 6.0),
        plane_zoom=1.5,
        current_step=(3, 2, 1),
        current_level=2,
        ndisplay=2,
    )

    cache = viewport_staging._plane_cache_from_snapshot(snapshot)

    assert cache.applied_level == 2
    assert cache.applied_step == (3, 2, 1)
    assert cache.pose.rect == (1.0, 2.0, 3.0, 4.0)
    assert cache.pose.center == (5.0, 6.0)
    assert cache.pose.zoom == 1.5


def test_volume_cache_from_snapshot_applies_pose() -> None:
    snapshot = RenderLedgerSnapshot(
        volume_center=(4.0, 5.0, 6.0),
        volume_angles=(30.0, 10.0, 5.0),
        volume_distance=45.0,
        volume_fov=60.0,
        current_level=3,
    )

    cache = viewport_staging._volume_cache_from_snapshot(snapshot)

    assert cache.level == 3
    assert cache.pose.center == (4.0, 5.0, 6.0)
    assert cache.pose.angles == (30.0, 10.0, 5.0)
    assert cache.pose.distance == 45.0
    assert cache.pose.fov == 60.0
