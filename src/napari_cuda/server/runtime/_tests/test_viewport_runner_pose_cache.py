from __future__ import annotations

from napari_cuda.server.runtime.viewport import PlaneState, ViewportRunner


def test_update_camera_rect_ignores_none() -> None:
    plane = PlaneState()
    plane.update_pose(rect=(100.0, 200.0, 300.0, 400.0))
    runner = ViewportRunner(plane_state=plane)

    # Sanity: explicit rect updates still apply.
    runner.update_camera_rect((10.0, 20.0, 30.0, 40.0))
    assert plane.pose.rect == (10.0, 20.0, 30.0, 40.0)

    # Regression: passing None (e.g. during volume mode) must preserve the cache.
    runner.update_camera_rect(None)
    assert plane.pose.rect == (10.0, 20.0, 30.0, 40.0)
