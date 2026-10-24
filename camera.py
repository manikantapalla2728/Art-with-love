"""Orthographic portrait camera with zoom and a slow cinematic sway."""

from __future__ import annotations

import math

import numpy as np

import settings as S


class Camera:
    def __init__(self) -> None:
        self.zoom_index = 0
        self.zoom = S.ZOOM_LEVELS[0]
        self.sway = 0.0
        self._view_w = S.VIEW_WIDTH
        self._view_h = S.VIEW_HEIGHT

    def cycle_zoom(self) -> None:
        self.zoom_index = (self.zoom_index + 1) % len(S.ZOOM_LEVELS)
        self.zoom = S.ZOOM_LEVELS[self.zoom_index]

    def update(self, time_sec: float, paused: bool) -> None:
        if paused:
            return
        self.sway = S.CINEMATIC_SWAY * math.sin(time_sec * 0.07)

    def view_proj(self) -> np.ndarray:
        """Column-major 4x4: orthographic * Z-rotation (cinematic sway)."""
        z = max(self.zoom, 0.2)
        hw = (self._view_w * 0.5) / z
        hh = (self._view_h * 0.5) / z
        left, right, bottom, top = -hw, hw, -hh, hh
        ortho = np.array(
            [
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        c = math.cos(self.sway)
        s = math.sin(self.sway)
        rot = np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return (ortho @ rot).T.copy()  # upload as 16 floats, column-major via transpose of row-major product

    def letterbox(self, win_w: int, win_h: int) -> tuple[int, int, int, int]:
        """Viewport that preserves 9:16 inside the window."""
        target = S.ASPECT
        win_aspect = win_w / max(win_h, 1)
        if win_aspect > target:
            h = win_h
            w = int(h * target)
            x = (win_w - w) // 2
            y = 0
        else:
            w = win_w
            h = int(w / target)
            x = 0
            y = (win_h - h) // 2
        return x, y, w, h
