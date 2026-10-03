"""No-progress watchdog: frozen screen, stuck menu, stalled HUD timer, blocked loop."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot.watchdog import (
    FRAME_CHANGE_MIN,
    LoopHeartbeat,
    ProgressWatchdog,
    frame_signature,
    signature_change,
)

FIX = Path(__file__).parent / "fixtures"


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(54, 96, 3), dtype=np.uint8)


STILL = _frame(0)


def _run(wd: ProgressWatchdog, clock: FakeClock, seconds: float, step: float = 0.5, **kw):
    """Feed frames for ``seconds``; kw values may be callables of the tick index."""
    stall = None
    n = int(seconds / step)
    for i in range(n):
        clock.t += step
        args = {k: (v(i) if callable(v) else v) for k, v in kw.items()}
        stall = wd.update(**args)
        if stall is not None:
            return stall
    return stall


def test_signature_separates_live_and_frozen_frames():
    a = cv2.imread(str(FIX / "gameplay_full_hp.png"))
    b = cv2.imread(str(FIX / "gameplay_hurt.png"))
    assert signature_change(frame_signature(a), frame_signature(a.copy())) == 0.0
    assert signature_change(frame_signature(a), frame_signature(b)) > FRAME_CHANGE_MIN
    # Light blur / re-encode noise on the same frame is not "motion".
    blurred = cv2.GaussianBlur(a, (3, 3), 0)
    assert signature_change(frame_signature(a), frame_signature(blurred)) < FRAME_CHANGE_MIN


def test_live_gameplay_with_advancing_timer_never_trips():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    stall = _run(
        wd, clock, 300,
        frame=lambda i: _frame(i),
        timer_s=lambda i: float(i // 4) if i % 4 == 0 else None,  # a read every 2 s
    )
    assert stall is None


def test_frozen_frame_trips_after_limit_not_before():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 55, frame=STILL) is None
    stall = _run(wd, clock, 10, frame=STILL)
    assert stall is not None and stall.reason == "frozen_frame"
    assert 60 <= stall.stalled_s < 61
    assert "frozen_frame" in stall.describe()


def test_frozen_frame_counts_on_menu_screens_too():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    stall = _run(wd, clock, 70, frame=STILL, in_menu=True)
    assert stall is not None and stall.reason in ("frozen_frame", "menu_stuck")


def test_short_level_up_does_not_trip_but_a_stuck_one_does():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    # Level-up screens animate a little; 10 s of menu then back to play is fine.
    assert _run(wd, clock, 10, frame=lambda i: _frame(1000 + i), in_menu=True) is None
    assert _run(wd, clock, 30, frame=lambda i: _frame(2000 + i), timer_s=lambda i: float(i)) is None
    # A menu that never goes away (bot keeps failing to pick) trips even if it animates.
    stall = _run(wd, clock, 70, frame=lambda i: _frame(3000 + i), in_menu=True)
    assert stall is not None and stall.reason == "menu_stuck"


def test_menu_detector_flicker_does_not_reset_menu_dwell():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    stall = _run(wd, clock, 70, frame=lambda i: _frame(i), in_menu=lambda i: i % 3 != 0)
    assert stall is not None and stall.reason == "menu_stuck"


def test_timer_stall_trips_while_screen_still_moves():
    """E.g. the game paused itself (focus loss) behind an animated overlay."""
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 20, frame=lambda i: _frame(i), timer_s=lambda i: 100.0 + i // 2) is None
    stall = _run(wd, clock, 70, frame=lambda i: _frame(500 + i), timer_s=lambda i: 110.0 if i % 4 == 0 else None)
    assert stall is not None and stall.reason == "timer_stalled"
    assert "hud=110s" in stall.detail


def test_timer_ocr_misses_are_not_a_stall():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    _run(wd, clock, 4, frame=lambda i: _frame(i), timer_s=50.0)
    # No confident reading for 2 minutes (busy screen): no timer stall, screen is live.
    assert _run(wd, clock, 120, frame=lambda i: _frame(100 + i), timer_s=None) is None


def test_menu_time_does_not_count_toward_timer_stall():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    _run(wd, clock, 40, frame=lambda i: _frame(i), timer_s=lambda i: 200.0 if i % 4 == 0 else None)
    # Level-up (timer frozen on screen) for 15 s, then play resumes with a moving timer.
    assert _run(wd, clock, 15, frame=lambda i: _frame(900 + i), timer_s=200.0, in_menu=True) is None
    assert _run(wd, clock, 30, frame=lambda i: _frame(1900 + i), timer_s=lambda i: 201.0 + i) is None


def test_pause_resets_everything():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 50, frame=STILL) is None
    assert _run(wd, clock, 600, frame=STILL, paused=True) is None
    assert _run(wd, clock, 50, frame=STILL) is None  # baseline restarted after unpause


def test_disabled_watchdog_never_trips():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=0, clock=clock)
    assert not wd.enabled
    assert _run(wd, clock, 600, frame=STILL, in_menu=True) is None


def test_precomputed_signature_path():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=5, clock=clock)
    sig = frame_signature(STILL)
    stall = _run(wd, clock, 10, frame=None, signature=sig)
    assert stall is not None and stall.reason == "frozen_frame"


def test_heartbeat_fires_once_when_loop_blocks():
    clock = FakeClock()
    fired = []
    hb = LoopHeartbeat(fired.append, limit_s=90, clock=clock)
    hb.beat()
    clock.t += 89
    assert hb.check() is None
    hb.beat()
    clock.t += 89
    assert hb.check() is None and not fired
    clock.t += 2
    stall = hb.check()
    assert stall is not None and stall.reason == "loop_blocked"
    assert fired == [stall]
    clock.t += 500
    assert hb.check() is None and len(fired) == 1


def test_heartbeat_disabled_and_thread_lifecycle():
    clock = FakeClock()
    fired = []
    off = LoopHeartbeat(fired.append, limit_s=0, clock=clock)
    off.start()
    clock.t += 10_000
    assert off.check() is None and not fired

    real = LoopHeartbeat(fired.append, limit_s=0.05, poll_s=0.01)
    real.start()
    import time

    deadline = time.monotonic() + 2.0
    while not fired and time.monotonic() < deadline:
        time.sleep(0.01)
    real.stop()
    assert fired and fired[0].reason == "loop_blocked"


@pytest.mark.parametrize("limit", [30.0, 60.0])
def test_limit_is_configurable(limit):
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=limit, clock=clock)
    stall = _run(wd, clock, limit + 5, frame=STILL)
    assert stall is not None and limit <= stall.stalled_s < limit + 1
