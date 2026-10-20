"""No-progress watchdogs: end a run that has stopped moving instead of waiting forever.

Two independent guards, both on ``time.monotonic`` (a VM pause / host suspend does
not advance it, so a resumed box does not trip them spuriously):

* :class:`ProgressWatchdog` — fed once per main-loop frame. Trips when, during an
  unpaused run, the captured screen has not visibly changed for ``limit_s``
  (``frozen_frame``; menus included: a level-up / chest / GAME OVER screen that
  sits unchanged that long is stuck), or capture itself kept failing that long
  (``no_frames``). An exactly repeated gameplay frame trips sooner
  (``stale_frame``, ``identical_s``): a stuck grab is byte-identical, while a
  real menu still drifts a fraction of a grey level and keeps the long limit.

* :class:`LoopHeartbeat` — a daemon thread that fires ``on_stall`` when the main
  loop itself has not called :meth:`LoopHeartbeat.beat` for ``limit_s`` (a capture,
  OCR or input call that blocks, which the frame watchdog cannot see).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

WATCHDOG_LIMIT_S = 60.0
LOOP_STALL_LIMIT_S = 90.0
# Grey signature size and the mean absolute difference (0-255) that counts as motion.
# Live Mad Forest gameplay changes this by >>5 per second (sprites, gems, flashing
# weapons, ~1-5 per 0.5 s on the death black-box frames); a static GAME OVER screen
# drifts only ~0.1-0.2 and an identical frame differs by 0.
SIG_SIZE = (32, 18)
FRAME_CHANGE_MIN = 1.0


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Tiny grey thumbnail used to tell a frozen screen from a live one."""
    if frame.ndim == 3:
        grey = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        grey = frame
    return cv2.resize(grey, SIG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def signature_change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@dataclass(frozen=True)
class Stall:
    reason: str        # frozen_frame | stale_frame | no_frames | loop_blocked
    stalled_s: float
    detail: str = ""

    def describe(self) -> str:
        extra = f" {self.detail}" if self.detail else ""
        return f"reason={self.reason} stalled={self.stalled_s:.0f}s{extra}"


class ProgressWatchdog:
    def __init__(
        self,
        limit_s: float = WATCHDOG_LIMIT_S,
        clock: Callable[[], float] = time.monotonic,
        change_min: float = FRAME_CHANGE_MIN,
        identical_s: float | None = None,
    ) -> None:
        self.limit_s = float(limit_s)
        self.clock = clock
        self.change_min = change_min
        # 0 disables the fast path. None keeps older callers on the long limit only.
        self.identical_s = 0.0 if identical_s is None else float(identical_s)
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.limit_s > 0

    def reset(self) -> None:
        """Forget the baseline (paused, new run)."""
        self._ref_sig: np.ndarray | None = None
        self._frame_t: float | None = None

    def update(
        self,
        frame: np.ndarray | None,
        paused: bool = False,
        now: float | None = None,
        signature: np.ndarray | None = None,
        in_menu: bool = False,
    ) -> Stall | None:
        """Feed one frame (None = capture failed). Returns a :class:`Stall` once frozen."""
        if not self.enabled:
            return None
        if paused:
            self.reset()
            return None
        now = self.clock() if now is None else now
        sig = signature if signature is not None else (frame_signature(frame) if frame is not None else None)
        if self._frame_t is None:
            self._frame_t = now
        # Compare against the frame at the last change (not the previous frame), so slow
        # but real motion still accumulates past the threshold.
        changed = False
        if sig is not None and (
            self._ref_sig is None
            or self._ref_sig.shape != sig.shape
            or signature_change(sig, self._ref_sig) >= self.change_min
        ):
            self._ref_sig, self._frame_t = sig, now
            changed = True
        stalled = now - self._frame_t
        if not changed and sig is not None and self._ref_sig is not None:
            # A stuck grab repeats the same pixels. A quiet menu is not exact.
            exact = signature_change(sig, self._ref_sig) <= 1e-6
            if (
                exact
                and not in_menu
                and self.identical_s > 0
                and stalled >= self.identical_s
            ):
                return Stall("stale_frame", stalled, "identical capture")
        if not changed and stalled >= self.limit_s:
            return Stall("frozen_frame" if sig is not None else "no_frames", stalled)
        return None


class LoopHeartbeat:
    """Fire ``on_stall`` once if :meth:`beat` is not called for ``limit_s``."""

    def __init__(
        self,
        on_stall: Callable[[Stall], None],
        limit_s: float = LOOP_STALL_LIMIT_S,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 1.0,
    ) -> None:
        self.on_stall = on_stall
        self.limit_s = float(limit_s)
        self.clock = clock
        self.poll_s = poll_s
        self._last = clock()
        self.fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        self._last = self.clock()

    def check(self, now: float | None = None) -> Stall | None:
        """One poll (the thread calls this; tests call it directly)."""
        if self.fired or self.limit_s <= 0:
            return None
        now = self.clock() if now is None else now
        gap = now - self._last
        if gap < self.limit_s:
            return None
        self.fired = True
        stall = Stall("loop_blocked", gap, "main loop made no progress")
        self.on_stall(stall)
        return stall

    def start(self) -> None:
        if self.limit_s <= 0 or self._thread is not None:
            return
        self._last = self.clock()
        self._thread = threading.Thread(target=self._run, name="vs-loop-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            self.check()
