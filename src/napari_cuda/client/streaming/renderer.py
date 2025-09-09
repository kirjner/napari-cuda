from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from vispy.gloo import Texture2D, Program

logger = logging.getLogger(__name__)


VERTEX_SHADER = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

FRAGMENT_SHADER = """
uniform sampler2D texture;
varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(texture, v_texcoord);
}
"""


class GLRenderer:
    """Minimal GL renderer for streaming frames.

    Owns a texture and shader program and draws an RGB frame.
    """

    def __init__(self, scene_canvas) -> None:
        self._scene_canvas = scene_canvas
        self._video_texture: Optional[Texture2D] = None
        self._video_program: Optional[Program] = None
        self._init_resources()

    def _init_resources(self) -> None:
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._video_texture = Texture2D(dummy_frame)
        self._video_program = Program(VERTEX_SHADER, FRAGMENT_SHADER)
        vertices = np.array([
            [-1, -1, 0, 1],
            [ 1, -1, 1, 1],
            [-1,  1, 0, 0],
            [ 1,  1, 1, 0],
        ], dtype=np.float32)
        self._video_program['position'] = np.ascontiguousarray(vertices[:, :2])
        self._video_program['texcoord'] = np.ascontiguousarray(vertices[:, 2:])
        self._video_program['texture'] = self._video_texture
        logger.debug("GLRenderer resources initialized")

    def draw(self, frame: Optional[np.ndarray]) -> None:
        ctx = self._scene_canvas.context
        if self._video_program is None or self._video_texture is None:
            self._init_resources()
        if frame is not None:
            try:
                if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
            except Exception:
                logger.debug("Frame normalization failed", exc_info=True)
            self._video_texture.set_data(frame)
        ctx.clear('black')
        self._video_program.draw('triangle_strip')

