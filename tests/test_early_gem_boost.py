"""Early denser XP: Tuning.early_gem_* boosts gem pull before the flower window."""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import Navigator, Tuning, gem_tune_for, score_directions

from tests.test_movement import Clock, P, result, run


FAR_LEFT_GEM = [(100.0, 256.0)]  # ~380 px; outside base gem_radius 300, inside early 460
NEAR_LEFT_GEM = [(280.0, 256.0)]  # ~200 px; inside both radii


def test_gem_tune_for_applies_only_before_cutoff():
    t = Tuning()
    early = gem_tune_for(t, 150.0)
    assert early.gem_gain == t.early_gem_gain and early.gem_radius == t.early_gem_radius
    assert gem_tune_for(t, t.early_gem_until_s) is t
    assert gem_tune_for(t, 400.0) is t
    off = Tuning(early_gem_until_s=0.0)
    assert gem_tune_for(off, 150.0) is off


def test_far_gem_pulls_only_with_early_radius():
    empty = np.zeros((0, 3), np.float32)
    base = score_directions(empty, P, None, 0.0, gems=FAR_LEFT_GEM, tune=Tuning())
    boosted = score_directions(
        empty, P, None, 0.0, gems=FAR_LEFT_GEM, tune=gem_tune_for(Tuning(), 120.0)
    )
    # Outside base radius: no left pull.
    assert base.scores[(-1, 0)] == 0.0
    assert boosted.vector[0] == -1
    assert boosted.scores[(-1, 0)] > base.scores[(-1, 0)]


def test_early_gain_beats_base_on_near_gem():
    empty = np.zeros((0, 3), np.float32)
    base = score_directions(empty, P, None, 0.0, gems=NEAR_LEFT_GEM, tune=Tuning())
    boosted = score_directions(
        empty, P, None, 0.0, gems=NEAR_LEFT_GEM, tune=gem_tune_for(Tuning(), 120.0)
    )
    assert boosted.scores[(-1, 0)] > base.scores[(-1, 0)]


def test_navigator_applies_boost_before_flower_not_after():
    """Roam step uses early knobs at game_s 120 and base knobs past the cutoff."""

    def scores_at(game_s: float, tune: Tuning | None = None):
        clock = Clock()
        nav = Navigator(tune=tune or Tuning(), clock=clock, log=lambda *_: None)
        nav.game_s = game_s
        # Freeze kite so the only score delta is the gem term.
        nav.heading = 0.0
        nav.orbit = 1
        run(nav, clock, result([], gems=FAR_LEFT_GEM), 1, hp=1.0)
        return nav.last.scores

    early = scores_at(120.0)
    late = scores_at(320.0)
    assert early[(-1, 0)] > late[(-1, 0)]
    # Disabling the boost: far gem stays outside radius both times (same left score).
    off = Tuning(early_gem_until_s=0.0)
    assert scores_at(120.0, off)[(-1, 0)] == scores_at(320.0, off)[(-1, 0)]
