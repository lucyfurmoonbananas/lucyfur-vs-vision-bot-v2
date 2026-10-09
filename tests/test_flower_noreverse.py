"""Flower-window "no reversal" rule (Tuning.flower_noreverse_s).

Oct 6 3 AM logs: inside the Man-eater window (game~300-340 s) consecutive breakouts
flipped v=(1,..) -> (-1,..) every ~3 s and walked back into the closing ring.
"""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import DIRS, Navigator, Tuning

from tests.test_movement import Clock, result, ring, run

GAP_RIGHT = (-1, 25)
GAP_LEFT = (160, 200)


def _nav(game_s: float, **kw):
    clock = Clock()
    logs: list[str] = []
    tune = Tuning(**{"flower_noreverse_s": 4.0, "ring_commit_s": 0.0, **kw})
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    nav.game_s = game_s
    return nav, clock, logs


def _break_then_end(nav, clock, gap):
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=gap)), 6, hp=1.0)
    assert nav.breakout is not None
    first = nav.breakout
    run(nav, clock, result([]), 16, hp=1.0)  # > breakout_min_s, not closing -> ends
    assert nav.breakout is None
    return first


def test_flower_breakout_does_not_reverse_the_last_one():
    nav, clock, logs = _nav(318.0)
    first = _break_then_end(nav, clock, GAP_RIGHT)
    assert first[0] > 0.5
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=1.0)
    assert nav.breakout is not None
    assert nav.breakout[0] >= -1e-6  # within 90 deg of "right", never back left
    assert all(v[0] >= 0 for v in out)
    assert any("flower no-reverse breakout" in line for line in logs)


def test_outside_flower_window_reversal_is_allowed():
    nav, clock, logs = _nav(100.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=1.0)
    assert nav.breakout is not None and nav.breakout[0] < -0.5
    assert not any("no-reverse" in line for line in logs)


def test_reversal_allowed_once_the_last_breakout_is_old():
    nav, clock, _ = _nav(318.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    nav.game_s += 5.0  # > flower_noreverse_s since the last breakout ended
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=1.0)
    assert nav.breakout is not None and nav.breakout[0] < -0.5


def test_disabled_rule_keeps_old_behaviour():
    nav, clock, logs = _nav(318.0, flower_noreverse_s=0.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=1.0)
    assert nav.breakout is not None and nav.breakout[0] < -0.5
    assert not any("no-reverse" in line for line in logs)


def test_running_flower_breakout_recheck_never_flips_back():
    nav, clock, _ = _nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_RIGHT)), 4, hp=1.0)
    assert nav.breakout is not None and nav.breakout[0] > 0.5
    # Right corridor now shut, left wide open: the 0.4 s re-check may turn, not reverse.
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 12, hp=1.0)
    assert nav.breakout is None or nav.breakout[0] >= -1e-6
    assert all(v[0] >= 0 for v in out)


def test_no_reverse_mask_keeps_five_headings():
    nav, _, _ = _nav(318.0)
    costs = np.ones(len(DIRS), np.float32)
    masked = nav._no_reverse(costs, (1.0, 0.0))
    assert int(np.isfinite(masked).sum()) == 5
    assert np.isinf(masked[DIRS.index((-1, 0))])
    nav.game_s = 100.0
    assert np.all(np.isfinite(nav._no_reverse(costs, (1.0, 0.0))))


def test_reset_clears_last_breakout():
    nav, clock, _ = _nav(318.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    assert nav._last_break_dir is not None
    nav.reset()
    assert nav._last_break_dir is None
