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
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=gap)), 6, hp=0.24)
    assert nav.breakout is not None
    first = nav.breakout
    run(nav, clock, result([]), 16, hp=0.24)  # > breakout_min_s, not closing -> ends
    assert nav.breakout is None
    return first


def test_flower_breakout_does_not_reverse_the_last_one():
    nav, clock, logs = _nav(318.0)
    first = _break_then_end(nav, clock, GAP_RIGHT)
    assert first[0] > 0.5
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None
    assert nav.breakout[0] >= -1e-6  # within 90 deg of "right", never back left
    assert all(v[0] >= 0 for v in out)
    assert any("flower no-reverse breakout" in line for line in logs)


def test_outside_flower_window_reversal_is_allowed():
    nav, clock, logs = _nav(100.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None and nav.breakout[0] < -0.5
    assert not any("no-reverse" in line for line in logs)


def test_flower_window_keeps_noreverse_past_ttl():
    """Oct 11: ring escapes ~6 s apart flipped 180 after flower_noreverse_s=4 expired."""
    nav, clock, _ = _nav(318.0)
    first = _break_then_end(nav, clock, GAP_RIGHT)
    assert first[0] > 0.5
    nav.game_s += 5.0  # still inside flower (280-360); old TTL would forget the heading
    assert nav._in_flower()
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None
    assert nav.breakout[0] >= -1e-6  # still anchored to last rightward breakout
    assert all(v[0] >= 0 for v in out)


def test_noreverse_expires_after_flower_window_ends():
    nav, clock, _ = _nav(318.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    nav.game_s = 370.0  # past flower_phase_hi; TTL has also elapsed
    assert not nav._in_flower()
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None and nav.breakout[0] < -0.5


def test_disabled_rule_keeps_old_behaviour():
    nav, clock, logs = _nav(318.0, flower_noreverse_s=0.0)
    _break_then_end(nav, clock, GAP_RIGHT)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None and nav.breakout[0] < -0.5
    assert not any("no-reverse" in line for line in logs)


def test_running_flower_breakout_recheck_never_flips_back():
    nav, clock, _ = _nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_RIGHT)), 4, hp=0.24)
    assert nav.breakout is not None and nav.breakout[0] > 0.5
    # Right corridor now shut, left wide open: the 0.4 s re-check may turn, not reverse.
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 12, hp=0.24)
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
    assert nav._breakout_origin is None


def test_running_breakout_recheck_anchors_to_origin_not_latest():
    """Successive ~90° re-aims must not gradually flip 180° (Bob Oct 8).

    Start right, pretend a re-check already turned us up, then open the left
    corridor. Without origin-anchor, no-reverse vs current (0,1) allows left;
    with origin at +x, heading must stay x>=0.
    """
    nav, clock, _ = _nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_RIGHT)), 4, hp=0.24)
    assert nav.breakout is not None and nav.breakout[0] > 0.5
    assert nav._breakout_origin is not None and nav._breakout_origin[0] > 0.5
    # Simulate a 90° re-aim already happened; force the next 0.4 s re-check.
    nav.breakout = (0.0, 1.0)
    nav._breakout_check = 0.4
    out = run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 12, hp=0.24)
    if nav.breakout is not None:
        assert nav.breakout[0] >= -1e-6
        assert nav._breakout_origin is not None and nav._breakout_origin[0] > 0.5
    assert all(v[0] >= 0 for v in out)


def test_end_breakout_clears_origin_keeps_last_dir():
    nav, clock, _ = _nav(318.0)
    first = _break_then_end(nav, clock, GAP_RIGHT)
    assert first[0] > 0.5
    assert nav.breakout is None
    assert nav._breakout_origin is None
    assert nav._last_break_dir is not None and nav._last_break_dir[0] > 0.5


def test_consecutive_ring_escapes_do_not_flip_after_ttl():
    """Oct 11 1 AM run1: (-1,-1) at ~281 then (1,1) at ~289 after 4 s TTL forgot heading."""
    clock = Clock()
    logs: list[str] = []
    # ring_commit on so closing flower ring uses committed escapes
    tune = Tuning(flower_noreverse_s=4.0, ring_commit_s=2.5)
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    nav.game_s = 281.0
    # First ring escape through right gap
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_RIGHT)), 4, hp=0.24)
    assert nav.escaping or nav.breakout is not None
    first = nav.breakout or nav._last_break_dir
    assert first is not None and first[0] > 0.5
    # Let the escape finish and age past the old 4 s TTL while still in flower
    run(nav, clock, result([]), 40, hp=0.24)  # clear field, end escape/breakout
    nav.game_s = 289.0
    assert nav._in_flower()
    assert nav._recent_break_dir() is not None
    assert nav._recent_break_dir()[0] > 0.5
    # New closing ring with gap on the LEFT — must NOT reverse into (-1, ...)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=GAP_LEFT)), 6, hp=0.24)
    assert nav.breakout is not None
    assert nav.breakout[0] >= -1e-6
