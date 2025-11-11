from __future__ import annotations

from napari_cuda.server.runtime.render_loop.planning.viewport_planner import ViewportPlanner
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
