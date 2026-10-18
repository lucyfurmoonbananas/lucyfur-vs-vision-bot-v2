"""Late-game gem seeking after the flower window (live run on cb74267).

From the level-8 pick at ~8:35 until the 11:04 quit, most samples were plain
breakouts at danger 1.8-3.7 with gems still on screen. Breakouts drop gems
after 280 s, and the roam seek fades to nothing at gem_danger 1.6, so the
seek counter kept climbing while the heading never went to the gems.
"""

from __future__ import annotations

import math

from vs_vision_bot.pathfind import Navigator, Tuning

from tests.test_movement import Clock, blob, result, ring, run

GAP_UP = (250, 290)
GEMS_RIGHT = [(700.0, 256.0), (712.0, 262.0), (690.0, 248.0)]
GEMS_DOWN = [(480.0, 420.0), (490.0, 430.0), (470.0, 415.0)]


def _breakout(game_s: float, gems, hp: float = 1.0):
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = game_s
    pts = ring(480, 256, 180, 28, gap_deg=GAP_UP)
    vecs = run(nav, clock, result(pts, gems=gems), 8, hp=hp)
    assert nav.breakouts >= 1 and nav.breakout is not None and not nav.escaping
    return nav, vecs


def test_late_breakout_steps_toward_side_gems():
    nav, vecs = _breakout(600.0, GEMS_RIGHT)
    assert 0.5 < nav.last.danger_here < 5.0
    assert vecs[-1][0] > 0
    assert vecs[-1][1] <= 0


def test_late_breakout_does_not_reverse_into_gems_behind_the_gap():
    _nav, vecs = _breakout(600.0, GEMS_DOWN)
    assert all(v[1] <= 0 for v in vecs)


def test_late_roam_seeks_gems_above_the_early_danger_cap():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 600.0
    nav.heading = -math.pi / 2  # kite says up; gems are to the right
    # Close pack below the player, not on the gems: danger above gem_danger 1.6.
    vecs = run(nav, clock, result(blob(480, 280, n=3, spread=6, seed=1), gems=GEMS_RIGHT), 6, hp=1.0)
    assert nav.breakouts == 0
    assert Tuning().gem_danger < nav.last.danger_here < 5.0
    assert vecs[-1][0] > 0


def test_low_hp_late_roam_still_ignores_gems():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 600.0
    nav.heading = -math.pi / 2
    vecs = run(nav, clock, result(blob(480, 280, n=3, spread=6, seed=1), gems=GEMS_RIGHT), 6, hp=0.3)
    assert all(v[0] <= 0 for v in vecs)
