"""Camera snapshot application helpers."""

from __future__ import annotations

from typing import Any

from vispy.scene.cameras import TurntableCamera

from napari_cuda.server.scene import RenderLedgerSnapshot


def apply_camera_overrides(worker: Any, snapshot: RenderLedgerSnapshot) -> None:
    """Sync camera state from the snapshot when provided."""

    view = worker.view  # type: ignore[attr-defined]
    if view is None:
        return
    cam = view.camera
    if cam is None:
        return

    if snapshot.plane_center is not None and not isinstance(cam, TurntableCamera) and hasattr(cam, "center"):
        cam.center = tuple(float(v) for v in snapshot.plane_center)  # type: ignore[attr-defined]
    if snapshot.plane_zoom is not None and not isinstance(cam, TurntableCamera) and hasattr(cam, "zoom"):
        cam.zoom = float(snapshot.plane_zoom)  # type: ignore[attr-defined]

    if isinstance(cam, TurntableCamera):
        if snapshot.volume_center is not None:
            cx, cy, cz = snapshot.volume_center
            cam.center = (float(cx), float(cy), float(cz))
        if snapshot.volume_angles is not None:
            az, el, roll = snapshot.volume_angles
            cam.azimuth = float(az)  # type: ignore[attr-defined]
            cam.elevation = float(el)  # type: ignore[attr-defined]
            cam.roll = float(roll)  # type: ignore[attr-defined]
        if snapshot.volume_distance is not None:
            cam.distance = float(snapshot.volume_distance)
        if snapshot.volume_fov is not None:
            cam.fov = float(snapshot.volume_fov)


__all__ = ["apply_camera_overrides"]
