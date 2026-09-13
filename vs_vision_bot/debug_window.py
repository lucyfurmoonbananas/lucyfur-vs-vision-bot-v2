"""Parked Model Vision window — small, off the playfield, named for xdotool."""

from __future__ import annotations

import cv2
import numpy as np

from vs_vision_bot.capture import CaptureRegion, park_debug_origin
from vs_vision_bot.config import Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.vision import VisionResult

WINDOW_NAME = "Model Vision"


class DebugWindow:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.origin = (0, 0)
        self._created = False

    def ensure(self, capture: CaptureRegion, screen: tuple[int, int]) -> None:
        self.origin = park_debug_origin(
            screen[0],
            screen[1],
            capture,
            self.cfg.debug_w,
            self.cfg.debug_h,
            preferred=(self.cfg.debug_x, self.cfg.debug_y),
        )
        if not self._created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
            except cv2.error:
                pass
            self._created = True
        cv2.resizeWindow(WINDOW_NAME, self.cfg.debug_w, self.cfg.debug_h)
        cv2.moveWindow(WINDOW_NAME, self.origin[0], self.origin[1])

    def repark(self, capture: CaptureRegion, screen: tuple[int, int]) -> None:
        self.cfg.debug_x = None
        self.cfg.debug_y = None
        self.ensure(capture, screen)

    def render(
        self,
        frame: np.ndarray,
        result: VisionResult,
        paused: bool,
        mover: MoveController,
        vector: tuple[int, int],
    ) -> None:
        vis = frame.copy()
        px, py = result.player
        cv2.circle(vis, (px, py), 8, (80, 255, 80), 2)
        for det in result.monsters:
            color = (40, 40, 255)
            cv2.rectangle(vis, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 1)
            cv2.circle(vis, (det.cx, det.cy), 3, color, -1)
        if vector != (0, 0):
            end = (px + vector[0] * 48, py + vector[1] * 48)
            cv2.arrowedLine(vis, (px, py), end, (0, 220, 255), 2, tipLength=0.35)

        banner = "PAUSED — press p to run" if paused else "RUNNING — p pause  ESC quit"
        if result.menu:
            banner = "MENU DETECTED — movement held (level-up / pause cards)"
        color = (0, 200, 255) if paused or result.menu else (80, 220, 80)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(vis, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        footer = (
            f"{mover.status_line()}  vec={vector}  "
            f"mobs={len(result.monsters)}  {result.note}  "
            f"park={self.origin}"
        )
        cv2.rectangle(vis, (0, vis.shape[0] - 24), (vis.shape[1], vis.shape[0]), (20, 20, 20), -1)
        cv2.putText(
            vis,
            footer,
            (8, vis.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, vis)

    def poll_key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(WINDOW_NAME)
            self._created = False
