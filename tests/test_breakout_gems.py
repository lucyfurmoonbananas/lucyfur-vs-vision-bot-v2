"""Early-game plain breakouts keep the gem pull (Tuning.breakout_gem_until_s)."""

from __future__ import annotations

from vs_vision_bot.pathfind import Navigator, Tuning

from tests.test_movement import Clock, result, ring, run

GAP_RIGHT = (-1, 25)
GEMS_UP_RIGHT = [(560.0, 200.0), (575.0, 190.0), (590.0, 205.0)]


def _scores_in_breakout(game_s: float, gems, tune: Tuning | None = None, flee: bool = False):
    clock = Clock()
    nav = Navigator(tune=tune or Tuning(), clock=clock, log=lambda *_: None)
    nav.game_s = game_s
    if flee:
        nav.request_flee(3.0)
    run(nav, clock, result(ring(480, 256, 240, 32, gap_deg=GAP_RIGHT), gems=gems), 4, hp=1.0)
    assert nav.breakout is not None and not nav.escaping
    return nav.last.scores


def test_early_plain_breakout_is_pulled_toward_gems():
    with_g = _scores_in_breakout(150.0, GEMS_UP_RIGHT)
    without = _scores_in_breakout(150.0, None)
    assert with_g[(1, -1)] > without[(1, -1)]
    assert with_g[(-1, 1)] < without[(-1, 1)]


def test_flower_window_breakout_ignores_gems():
    assert _scores_in_breakout(318.0, GEMS_UP_RIGHT) == _scores_in_breakout(318.0, None)


def test_flee_breakout_ignores_gems_even_early():
    assert _scores_in_breakout(150.0, GEMS_UP_RIGHT, flee=True) == _scores_in_breakout(150.0, None, flee=True)


def test_knob_zero_restores_old_behaviour():
    off = Tuning(breakout_gem_until_s=0.0)
    assert _scores_in_breakout(150.0, GEMS_UP_RIGHT, off) == _scores_in_breakout(150.0, None, off)


def test_low_hp_breakout_ignores_gems():
    clock = Clock()
    a = Navigator(clock=clock, log=lambda *_: None)
    a.game_s = 150.0
    run(a, clock, result(ring(480, 256, 240, 32, gap_deg=GAP_RIGHT), gems=GEMS_UP_RIGHT), 4, hp=0.3)
    clock2 = Clock()
    b = Navigator(clock=clock2, log=lambda *_: None)
    b.game_s = 150.0
    run(b, clock2, result(ring(480, 256, 240, 32, gap_deg=GAP_RIGHT)), 4, hp=0.3)
    assert a.last.scores == b.last.scores
