"""No-progress watchdog: frozen screen / no frames, and a blocked main loop."""

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


def test_live_gameplay_never_trips():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 300, frame=lambda i: _frame(i)) is None


def test_slow_motion_accumulates_against_the_reference_frame():
    """Each step changes little, but the screen keeps drifting: not frozen."""
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    base = np.full((54, 96, 3), 60, np.uint8)
    stall = _run(wd, clock, 200, frame=lambda i: np.clip(base.astype(int) + (i // 4) % 120, 0, 255).astype(np.uint8))
    assert stall is None


def test_frozen_frame_trips_after_limit_not_before():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 55, frame=STILL) is None
    stall = _run(wd, clock, 10, frame=STILL)
    assert stall is not None and stall.reason == "frozen_frame"
    assert 60 <= stall.stalled_s < 61
    assert "frozen_frame" in stall.describe()


def test_static_menu_screen_counts_too():
    """A level-up / GAME OVER screen sitting unchanged for 60 s is stuck as well."""
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 10, frame=lambda i: _frame(i)) is None
    stall = _run(wd, clock, 70, frame=STILL)
    assert stall is not None and stall.reason == "frozen_frame"


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
    assert _run(wd, clock, 600, frame=STILL) is None


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

    clock2 = FakeClock()
    steps = {"n": 0}

    def wait(_poll: float) -> bool:
        steps["n"] += 1
        clock2.t += 1.0
        return steps["n"] >= 2

    real = LoopHeartbeat(fired.append, limit_s=0.05, poll_s=0.01, clock=clock2, wait=wait)
    real.start()
    assert real._thread is not None
    real._thread.join(timeout=2.0)
    real.stop()
    assert fired and fired[0].reason == "loop_blocked"


@pytest.mark.parametrize("limit", [30.0, 60.0])
def test_limit_is_configurable(limit):
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=limit, clock=clock)
    stall = _run(wd, clock, limit + 5, frame=STILL)
    assert stall is not None and limit <= stall.stalled_s < limit + 1


def test_capture_failures_count_as_no_progress():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    assert _run(wd, clock, 5, frame=lambda i: _frame(i)) is None
    stall = _run(wd, clock, 70, frame=None)
    assert stall is not None and stall.reason == "no_frames"


def test_capture_failing_from_the_start_also_trips():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, clock=clock)
    stall = _run(wd, clock, 70, frame=None)
    assert stall is not None and stall.reason == "no_frames"


def test_identical_gameplay_frames_trip_before_the_long_watchdog():
    """Run 4 repeated one capture from ~239s to ~295s (danger stuck at 8.52)
    until the 60s watchdog. An identical gameplay frame is a stuck grab.
    """
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, identical_s=4, clock=clock)
    assert _run(wd, clock, 3.5, step=0.5, frame=STILL) is None
    stall = _run(wd, clock, 2, step=0.5, frame=STILL)
    assert stall is not None and stall.reason == "stale_frame"
    assert 4 <= stall.stalled_s < 5
    assert "stale_frame" in stall.describe()


def test_identical_menu_waits_for_the_long_watchdog():
    """A still level-up or chest screen is not a stuck grab."""
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, identical_s=4, clock=clock)
    assert _run(wd, clock, 10, step=0.5, frame=STILL, in_menu=True) is None


def test_slow_drift_is_not_an_identical_capture():
    clock = FakeClock()
    wd = ProgressWatchdog(limit_s=60, identical_s=4, clock=clock)
    base = frame_signature(STILL)

    def sig(i: int) -> np.ndarray:
        out = base.copy()
        out[0, 0] = float(i)
        return out

    assert _run(wd, clock, 10, step=0.5, frame=None, signature=sig) is None
