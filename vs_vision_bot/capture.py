"""Screen capture region. `q` snaps the top-left to the current mouse position."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.input_backend import get_mouse_position, screen_size


@dataclass
class CaptureRegion:
    x: int
    y: int
    w: int
    h: int

    def as_mss(self) -> dict[str, int]:
        return {"left": self.x, "top": self.y, "width": self.w, "height": self.h}

    def overlaps(self, other: CaptureRegion) -> bool:
        return not (
            self.x + self.w <= other.x
            or other.x + other.w <= self.x
            or self.y + self.h <= other.y
            or other.y + other.h <= self.y
        )


class ScreenCapture:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.region = CaptureRegion(cfg.capture_x, cfg.capture_y, cfg.capture_w, cfg.capture_h)
        self._sct = None

    def _mss(self):
        if self._sct is None:
            import mss

            self._sct = mss.mss()
        return self._sct

    def grab(self) -> np.ndarray:
        shot = self._mss().grab(self.region.as_mss())
        frame = np.array(shot)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        # mss is BGRA; after dropping A the channel order is BGR, which OpenCV expects.
        return frame

    def align_to_mouse(self) -> CaptureRegion | None:
        pos = get_mouse_position()
        if pos is None:
            print("[capture] could not read mouse position (need xdotool)", file=sys.stderr)
            return None
        self.region.x, self.region.y = pos
        self.cfg.capture_x, self.cfg.capture_y = pos
        print(f"[capture] aligned top-left to mouse ({pos[0]}, {pos[1]})")
        return self.region

    def display_size(self) -> tuple[int, int]:
        try:
            monitors = self._mss().monitors
            if len(monitors) > 1:
                primary = monitors[1]
                return int(primary["width"]), int(primary["height"])
        except Exception:  # noqa: BLE001
            pass
        return screen_size()


def park_debug_origin(
    screen_w: int,
    screen_h: int,
    capture: CaptureRegion,
    debug_w: int,
    debug_h: int,
    preferred: tuple[int | None, int | None] = (None, None),
    margin: int = 16,
) -> tuple[int, int]:
    """Place Model Vision off the playfield capture rect."""
    pref_x, pref_y = preferred
    if pref_x is not None and pref_y is not None:
        return pref_x, pref_y

    debug = CaptureRegion(0, 0, debug_w, debug_h)
    candidates = [
        (screen_w - debug_w - margin, margin),
        (margin, screen_h - debug_h - margin),
        (screen_w - debug_w - margin, screen_h - debug_h - margin),
        (margin, margin),
        (capture.x + capture.w + margin, capture.y),
        (max(margin, capture.x - debug_w - margin), capture.y),
    ]
    for x, y in candidates:
        x = max(0, min(x, max(0, screen_w - debug_w)))
        y = max(0, min(y, max(0, screen_h - debug_h)))
        debug.x, debug.y = x, y
        if not debug.overlaps(capture):
            return x, y
    return max(0, screen_w - debug_w - margin), margin
