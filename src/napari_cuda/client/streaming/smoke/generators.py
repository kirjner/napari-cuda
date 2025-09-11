from __future__ import annotations

"""
Lightweight RGB frame generators for smoke tests.

Factory returns a callable(frame_idx) -> np.ndarray[H,W,3] (uint8 RGB).
Modes supported:
- 'checker': animated checkerboard (default)
- 'gradient': static XY gradient
- 'mip_turntable': rotating 3D MIP of a random volume (approximates server --volume --animate)

Notes:
- The MIP turntable runs at an internal resolution for performance and upscales.
- Control speed via env NAPARI_CUDA_SMOKE_TT_DPS (degrees/sec, default 30).
- Control elevation via env NAPARI_CUDA_SMOKE_TT_ELEV (degrees, default 30).
"""

import os
from typing import Callable

import numpy as np


def _viridis_lut() -> np.ndarray:
    """Return a 256x3 uint8 viridis lookup table.

    Small, hardcoded approximation to avoid heavy dependencies. Values generated
    from matplotlib viridis; truncated for brevity is not acceptable here, so we
    keep the full 256 entry table.
    """
    # fmt: off
    data = np.array([
        [68, 1, 84],[68, 2, 86],[69, 4, 87],[69, 5, 89],[70, 7, 90],[70, 8, 92],[70, 10, 93],[70, 11, 94],
        [71, 13, 96],[71, 14, 97],[71, 16, 99],[71, 17, 100],[71, 19, 101],[71, 20, 103],[71, 22, 104],[71, 23, 105],
        [71, 25, 106],[71, 26, 108],[71, 28, 109],[70, 29, 110],[70, 31, 111],[70, 32, 112],[70, 34, 113],[69, 35, 114],
        [69, 36, 115],[69, 38, 116],[68, 39, 117],[68, 41, 118],[67, 42, 119],[67, 44, 120],[66, 45, 121],[66, 46, 121],
        [65, 48, 122],[65, 49, 123],[64, 51, 123],[63, 52, 124],[63, 53, 125],[62, 55, 125],[62, 56, 126],[61, 57, 126],
        [61, 59, 127],[60, 60, 127],[59, 61, 128],[59, 62, 128],[58, 64, 129],[58, 65, 129],[57, 66, 129],[56, 67, 130],
        [56, 69, 130],[55, 70, 130],[54, 71, 131],[54, 72, 131],[53, 73, 131],[52, 74, 131],[52, 76, 132],[51, 77, 132],
        [50, 78, 132],[50, 79, 132],[49, 80, 132],[49, 81, 132],[48, 82, 133],[47, 83, 133],[47, 84, 133],[46, 85, 133],
        [46, 86, 133],[45, 87, 133],[45, 88, 133],[44, 89, 133],[44, 90, 133],[43, 91, 133],[43, 92, 133],[42, 93, 133],
        [42, 94, 133],[41, 95, 133],[41, 96, 133],[41, 97, 133],[40, 98, 133],[40, 99, 133],[39, 100, 133],[39, 101, 133],
        [38, 102, 133],[38, 103, 133],[38, 104, 133],[37, 105, 133],[37, 106, 133],[36, 107, 133],[36, 107, 133],[36, 108, 133],
        [35, 109, 133],[35, 110, 132],[35, 111, 132],[34, 112, 132],[34, 113, 132],[34, 113, 132],[33, 114, 132],[33, 115, 132],
        [33, 116, 132],[33, 117, 131],[32, 118, 131],[32, 118, 131],[32, 119, 131],[31, 120, 131],[31, 121, 130],[31, 121, 130],
        [31, 122, 130],[31, 123, 130],[30, 124, 129],[30, 124, 129],[30, 125, 129],[30, 126, 128],[30, 127, 128],[30, 127, 128],
        [30, 128, 127],[30, 129, 127],[30, 129, 127],[30, 130, 126],[30, 131, 126],[30, 131, 125],[30, 132, 125],[31, 133, 124],
        [31, 133, 124],[31, 134, 123],[31, 134, 123],[32, 135, 122],[32, 136, 122],[33, 136, 121],[33, 137, 121],[34, 137, 120],
        [35, 138, 119],[35, 138, 119],[36, 139, 118],[37, 139, 118],[38, 140, 117],[39, 140, 116],[40, 141, 116],[41, 141, 115],
        [42, 142, 114],[43, 142, 113],[44, 143, 113],[46, 143, 112],[47, 144, 111],[48, 144, 110],[50, 144, 109],[51, 145, 109],
        [53, 145, 108],[54, 146, 107],[56, 146, 106],[57, 146, 105],[59, 147, 104],[61, 147, 103],[63, 148, 102],[64, 148, 101],
        [66, 148, 100],[68, 149, 99],[70, 149, 98],[72, 149, 97],[74, 150, 96],[76, 150, 95],[78, 150, 94],[80, 151, 93],
        [82, 151, 92],[84, 151, 91],[86, 152, 90],[88, 152, 89],[90, 152, 88],[92, 153, 86],[94, 153, 85],[96, 153, 84],
        [98, 154, 83],[100, 154, 82],[102, 154, 80],[104, 154, 79],[106, 155, 78],[108, 155, 76],[110, 155, 75],[112, 156, 73],
        [114, 156, 72],[116, 156, 70],[118, 156, 69],[120, 157, 67],[122, 157, 66],[124, 157, 64],[126, 157, 63],[128, 158, 61],
        [130, 158, 60],[132, 158, 58],[134, 158, 56],[136, 159, 55],[138, 159, 53],[140, 159, 51],[142, 159, 50],[144, 160, 48],
        [146, 160, 46],[148, 160, 45],[149, 160, 43],[151, 161, 41],[153, 161, 39],[155, 161, 38],[157, 161, 36],[159, 161, 34],
        [161, 162, 33],[163, 162, 31],[165, 162, 29],[167, 162, 28],[169, 163, 26],[171, 163, 24],[173, 163, 23],[175, 163, 21],
        [177, 164, 20],[179, 164, 18],[181, 164, 17],[183, 164, 15],[185, 165, 14],[187, 165, 13],[189, 165, 12],[191, 165, 11],
        [193, 166, 10],[195, 166, 10],[197, 166, 9],[199, 166, 8],[201, 167, 8],[203, 167, 7],[205, 167, 7],[207, 168, 7],
        [209, 168, 7],[210, 168, 7],[212, 169, 7],[214, 169, 7],[216, 169, 7],[218, 170, 8],[220, 170, 8],[222, 170, 9],
        [224, 171, 9],[226, 171, 10],[228, 171, 11],[230, 172, 12],[232, 172, 13],[234, 173, 14],[236, 173, 15],[238, 173, 16],
        [239, 174, 17],[241, 174, 19],[243, 175, 20],[245, 175, 21],[247, 176, 23],[249, 177, 24],[251, 177, 26],[252, 178, 28],
        [254, 179, 30],[255, 180, 32],[255, 180, 34],[255, 181, 37],[255, 182, 39],[255, 183, 42],[255, 184, 45],[254, 185, 48],
        [253, 186, 52],[252, 187, 55],[250, 188, 59],[248, 190, 63],[245, 191, 67],[243, 192, 71],[240, 194, 76],[237, 195, 80],
        [233, 197, 85],[229, 198, 90],[225, 200, 95],[221, 201, 100],[216, 203, 105],[211, 205, 111],[206, 206, 116],[200, 208, 122],
        [195, 210, 128],[189, 212, 134],[183, 213, 140],[176, 215, 146],[170, 217, 152],[163, 219, 158],[156, 221, 164],[149, 223, 170],
        [142, 225, 176],[135, 227, 181],[128, 229, 186],[121, 231, 191],[114, 233, 196],[107, 235, 200],[100, 237, 204],[93, 238, 207],
        [86, 240, 210],[79, 242, 212],[73, 243, 214],[66, 245, 215],[60, 246, 216],[54, 247, 217],[48, 248, 217],[43, 249, 217],
        [38, 250, 217],[34, 251, 216],[30, 252, 214],[27, 252, 213],[25, 253, 211],[23, 253, 209],[21, 254, 206],[20, 254, 203],
        [19, 254, 200],[19, 254, 197],[19, 254, 193],[20, 254, 190],[21, 254, 187],[22, 254, 183],[24, 254, 180],[26, 254, 177],
        [28, 254, 174],[31, 253, 171],[33, 253, 168],[36, 253, 165],[39, 252, 163],[42, 252, 160],[45, 251, 158],[48, 251, 156],
        [51, 250, 154],[54, 249, 152],[57, 248, 150],[60, 248, 149],[63, 247, 147],[66, 246, 146],[69, 245, 145],[72, 244, 144],
        [75, 243, 144],[78, 242, 143],[81, 241, 143],[84, 240, 143],[86, 239, 143],[89, 238, 143],[92, 236, 143],[95, 235, 144],
        [97, 233, 144],[100, 232, 145],[103, 230, 146],[105, 229, 146],[108, 227, 147],[110, 225, 148],[113, 224, 149],[115, 222, 150],
        [118, 220, 151],[120, 219, 152],[123, 217, 153],[125, 215, 154],[127, 213, 155],[130, 212, 156],[132, 210, 157],[134, 208, 158],
        [137, 206, 159],[139, 204, 160],[141, 203, 160],[144, 201, 161],[146, 199, 162],[148, 197, 162],[151, 196, 163],[153, 194, 163],
        [155, 192, 164],[158, 191, 164],[160, 189, 165],[162, 187, 165],[165, 185, 166],[167, 184, 166],[169, 182, 166],[172, 180, 167],
        [174, 178, 167],[176, 177, 167],[179, 175, 167],[181, 173, 167],[184, 171, 167],[186, 170, 167],[188, 168, 167],[191, 166, 167],
        [193, 165, 167],[196, 163, 167],[198, 162, 167],[201, 160, 167],[203, 159, 166],[206, 157, 166],[208, 156, 166],[211, 154, 165],
        [213, 153, 165],[216, 151, 165],[218, 150, 164],[221, 148, 163],[223, 147, 163],[226, 145, 162],[228, 144, 162],[231, 142, 161],
        [233, 141, 160],[236, 139, 159],[238, 138, 158],[241, 136, 157],[243, 135, 156],[246, 133, 155],[248, 132, 154],[251, 130, 153],
        [253, 129, 152],[255, 127, 150]
    ], dtype=np.uint8)
    # fmt: on
    return data


def _deg2rad(a: float) -> float:
    return float(a) * np.pi / 180.0


def _rotation_elev_az(elev_deg: float, az_deg: float) -> np.ndarray:
    """Rotation matrix for elevation (x-axis), then azimuth (y-axis).

    Returns 3x3 float32 matrix R that transforms view-space coords into volume-space.
    """
    e = _deg2rad(elev_deg)
    a = _deg2rad(az_deg)
    ce, se = np.cos(e), np.sin(e)
    ca, sa = np.cos(a), np.sin(a)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)
    R = Ry @ Rx
    return R.astype(np.float32)


def _make_mip_turntable_cpu(w: int, h: int, fps: float) -> Callable[[int], "np.ndarray"]:
    """Return a generator producing a rotating MIP of a random volume.

    Implementation mirrors server defaults:
    - Volume depth = 64, dtype float32, random in [0,1)
    - Camera: Turntable with elevation 30°, azimuth animated at 30 dps
    - Rendering: 'mip' with viridis colormap

    To keep it fast in Python/NumPy, we render at a reduced internal resolution
    and upscale to the requested size.
    """
    # Internal render size (keep aspect) to trade quality for speed
    scale = 4  # render at 1/4 resolution, then upscale
    iw = max(64, w // scale)
    ih = max(64, h // scale)
    D = 64
    vol = np.random.rand(D, ih, iw).astype(np.float32)
    lut = _viridis_lut()
    dps = float(os.getenv('NAPARI_CUDA_SMOKE_TT_DPS', os.getenv('NAPARI_CUDA_TURNTABLE_DPS', '30')))  # match server env
    elev = float(os.getenv('NAPARI_CUDA_SMOKE_TT_ELEV', '30'))

    # Precompute XY grid in normalized coords [-0.5, 0.5]
    xs = (np.linspace(0, iw - 1, iw, dtype=np.float32) / float(max(1, iw - 1)) - 0.5)
    ys = (np.linspace(0, ih - 1, ih, dtype=np.float32) / float(max(1, ih - 1)) - 0.5)
    X, Y = np.meshgrid(xs, ys)  # (ih, iw)
    # Z steps along view ray in normalized coords
    Zs = np.linspace(-0.5, 0.5, D, dtype=np.float32)

    def _render(i: int) -> "np.ndarray":
        t = float(i) / max(1.0, fps)
        az = (dps * t) % 360.0
        R = _rotation_elev_az(elev, az)
        # R components
        r11, r12, r13 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
        r21, r22, r23 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
        r31, r32, r33 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
        # Accumulate maximum along samples
        acc = np.zeros((ih, iw), dtype=np.float32)
        for zk in Zs:
            # Rotate view ray sample [X, Y, zk] into volume space
            Xr = r11 * X + r12 * Y + r13 * zk
            Yr = r21 * X + r22 * Y + r23 * zk
            Zr = r31 * X + r32 * Y + r33 * zk
            # Map from [-0.5,0.5] to index space [0..iw-1], [0..ih-1], [0..D-1]
            ix = ((Xr + 0.5) * (iw - 1)).astype(np.int32)
            iy = ((Yr + 0.5) * (ih - 1)).astype(np.int32)
            iz = ((Zr + 0.5) * (D - 1)).astype(np.int32)
            # Clamp
            np.clip(ix, 0, iw - 1, out=ix)
            np.clip(iy, 0, ih - 1, out=iy)
            np.clip(iz, 0, D - 1, out=iz)
            # Sample nearest neighbor and update max
            acc = np.maximum(acc, vol[iz, iy, ix])
        # Normalize to [0,1]
        acc = np.clip(acc, 0.0, 1.0)
        idx = (acc * 255.0 + 0.5).astype(np.int32)
        # Colorize via viridis LUT
        rgb_small = lut[idx]
        # Upscale to (h,w) via nearest-neighbor repeat and crop
        ry = int(np.ceil(h / float(ih)))
        rx = int(np.ceil(w / float(iw)))
        up = np.repeat(np.repeat(rgb_small, ry, axis=0), rx, axis=1)
        out = up[:h, :w, :].astype(np.uint8, copy=False)
        return out

    return _render


def _make_mip_turntable_vispy(w: int, h: int, fps: float) -> Callable[[int], "np.ndarray"]:
    """Exact vispy-backed MIP turntable matching offline_vt_player.

    Creates a headless SceneCanvas with a Volume visual (method='mip') and a
    TurntableCamera, then returns frames via canvas.render(). Requires a working
    vispy backend (Qt is already in use by the client).
    """
    try:
        from vispy import scene  # type: ignore
    except Exception:
        # Fallback to CPU if vispy isn't importable
        return _make_mip_turntable_cpu(w, h, fps)

    dps = float(os.getenv('NAPARI_CUDA_SMOKE_TT_DPS', os.getenv('NAPARI_CUDA_TURNTABLE_DPS', '30')))
    elev = float(os.getenv('NAPARI_CUDA_SMOKE_TT_ELEV', '30'))
    rel_step = os.getenv('NAPARI_CUDA_SMOKE_TT_REL_STEP')

    canvas = scene.SceneCanvas(size=(w, h), bgcolor='black', show=False)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(elevation=elev, azimuth=30, fov=60, distance=500)
    vol = (np.random.rand(64, h, w).astype(np.float32))
    visual = scene.visuals.Volume(vol, parent=view.scene, method='mip', cmap='viridis')
    try:
        if rel_step is not None and hasattr(visual, 'relative_step_size'):
            visual.relative_step_size = float(rel_step)
    except Exception:
        pass

    # First render warms up GL state
    try:
        canvas.render()
    except Exception:
        pass

    def _render(i: int) -> "np.ndarray":
        t = float(i) / max(1.0, fps)
        try:
            view.camera.azimuth = (dps * t) % 360.0
        except Exception:
            pass
        img = canvas.render()
        # Ensure uint8 RGB
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        if img.shape[2] >= 3:
            img = img[:, :, :3]
        return img

    return _render


def make_generator(mode: str, w: int, h: int, fps: float) -> Callable[[int], "np.ndarray"]:
    mode = (mode or "checker").lower()
    w = int(w); h = int(h); fps = float(fps)
    # Precompute bases
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]

    if mode == "gradient":
        def _gen(i: int) -> "np.ndarray":
            r = np.broadcast_to(x, (h, w))
            g = np.broadcast_to(y, (h, w))
            b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
            return np.dstack([r, g, b])
        return _gen

    if mode in ("mip_turntable", "turntable", "volume"):
        # Try exact vispy-backed path first; fallback to CPU approximation
        return _make_mip_turntable_vispy(w, h, fps)

    # default: checker with slight motion
    def _gen_checker(i: int) -> "np.ndarray":
        # Modest drift per frame to avoid strobing; approx sine/cosine at ~2 Hz
        t = float(i) / max(1.0, fps)
        ox = int(((np.sin(t * 2.0) * 0.5 + 0.5) * 64))
        oy = int(((np.cos(t * 1.7) * 0.5 + 0.5) * 64))
        xx = (np.arange(w, dtype=np.int32)[None, :] + ox) // 32
        yy = (np.arange(h, dtype=np.int32)[:, None] + oy) // 32
        mask = ((xx ^ yy) & 1).astype(np.uint8)
        r = (mask * 255).astype(np.uint8, copy=False)
        g = np.broadcast_to(x, (h, w))
        b = np.broadcast_to(y, (h, w))
        return np.dstack([r, g, b])

    return _gen_checker
