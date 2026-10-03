"""No-progress watchdogs: end a run that has stopped moving instead of waiting forever.

Two independent guards, both on ``time.monotonic`` (a VM pause / host suspend does
not advance it, so a resumed box does not trip them spuriously):

* :class:`ProgressWatchdog` — fed once per main-loop frame. Trips when, during an
  unpaused run,

  - ``frozen_frame``: the captured screen has not visibly changed for ``limit_s``
    (counts on menus too: a level-up / chest / GAME OVER screen that sits
    unchanged that long is stuck); ``no_frames`` when capture itself kept failing;
  - ``menu_stuck``: a menu screen (level-up, chest, revive, ...) has been up
    continuously for ``limit_s``;
  - ``timer_stalled``: outside menus, the OCR'd HUD timer kept reading the same
    value for ``limit_s`` (at least ``min_timer_reads`` confident reads). Menu
    time does not count: the game timer legitimately stops on level-up screens.

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
# A single non-menu frame inside a menu (detector flicker) does not end the menu.
MENU_EXIT_S = 2.0
# Timer stall needs this many confident identical HUD reads (OCR misses are not stalls).
MIN_TIMER_READS = 3


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
    reason: str        # frozen_frame | menu_stuck | timer_stalled | loop_blocked
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
        menu_exit_s: float = MENU_EXIT_S,
        min_timer_reads: int = MIN_TIMER_READS,
    ) -> None:
        self.limit_s = float(limit_s)
        self.clock = clock
        self.change_min = change_min
        self.menu_exit_s = menu_exit_s
        self.min_timer_reads = min_timer_reads
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.limit_s > 0

    def reset(self) -> None:
        """Forget all baselines (paused, new run)."""
        self._ref_sig: np.ndarray | None = None
        self._frame_t: float | None = None
        self._menu_since: float | None = None
        self._last_menu_t = 0.0
        self._timer_val: float | None = None
        self._timer_t = 0.0
        self._timer_reads = 0

    def update(
        self,
        frame: np.ndarray | None,
        timer_s: float | None = None,
        in_menu: bool = False,
        paused: bool = False,
        now: float | None = None,
        signature: np.ndarray | None = None,
    ) -> Stall | None:
        """Feed one frame. Returns a :class:`Stall` once the run has stopped progressing."""
        if not self.enabled:
            return None
        if paused:
            self.reset()
            return None
        now = self.clock() if now is None else now
        sig = signature if signature is not None else (frame_signature(frame) if frame is not None else None)

        # -- screen motion (menus included). No frame at all (capture failing) is no progress.
        if self._frame_t is None:
            self._frame_t = now
        if sig is not None and (
            self._ref_sig is None
            or self._ref_sig.shape != sig.shape
            or signature_change(sig, self._ref_sig) >= self.change_min
        ):
            self._ref_sig, self._frame_t = sig, now
        elif now - self._frame_t >= self.limit_s:
            reason = "frozen_frame" if sig is not None else "no_frames"
            return Stall(reason, now - self._frame_t, f"menu={int(self._menu_since is not None)}")

        # -- menu dwell (debounced so detector flicker does not restart it)
        if in_menu:
            self._last_menu_t = now
            if self._menu_since is None:
                self._menu_since = now
        elif self._menu_since is not None and now - self._last_menu_t >= self.menu_exit_s:
            self._menu_since = None
        if self._menu_since is not None:
            # The HUD timer is stopped on these screens; restart its stall clock.
            self._timer_t, self._timer_reads = now, 0
            if now - self._menu_since >= self.limit_s:
                return Stall("menu_stuck", now - self._menu_since)
            return None

        # -- HUD timer (outside menus)
        # Judged only on a fresh confident read: OCR misses are never evidence of a stall.
        if timer_s is not None:
            if self._timer_val is None or timer_s != self._timer_val:
                self._timer_val, self._timer_t, self._timer_reads = float(timer_s), now, 0
            else:
                self._timer_reads += 1
                if self._timer_reads >= self.min_timer_reads and now - self._timer_t >= self.limit_s:
                    return Stall("timer_stalled", now - self._timer_t, f"hud={self._timer_val:.0f}s")
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
