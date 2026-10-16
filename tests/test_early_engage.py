"""Early Garlic engagement: Tuning.early_engage_* pulls toward weak early enemies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vs_vision_bot.pathfind import (
    DIRS, Navigator, Tuning, engage_pull, gem_tune_for, gems_in_range, score_directions, tuning_from_env,
)

from tests.test_movement import Clock, P, result, run

# One small enemy ~120 px to the left (4 grid points of 0.22, like vision.threat_points).
LEFT = [(360.0 + ox, 256.0 + oy, 0.22) for ox in (-6, 6) for oy in (-6, 6)]
LEFT_ARR = np.array(LEFT, dtype=np.float32)


ON = replace(Tuning(), early_engage_until_s=150.0)  # engage is off by default; tests opt in


def _left_score(dec) -> float:
    return max(dec.scores[(-1, 0)], dec.scores[(-1, -1)], dec.scores[(-1, 1)])


def test_defaults_and_gem_tune_for_window():
    assert Tuning().early_engage_until_s == 0.0 and gem_tune_for(Tuning(), 30.0).engage_gain == 0.0
    t = ON
    assert t.engage_gain == 0.0  # base knob off; only the early window turns it on
    assert t.early_engage_gain == 1.2
    assert gem_tune_for(t, 60.0).engage_gain == t.early_engage_gain
    assert gem_tune_for(t, 149.0).engage_gain == t.early_engage_gain
    assert gem_tune_for(t, 150.0).engage_gain == 0.0
    assert gem_tune_for(t, 400.0) is t  # past both early windows: untouched


def test_env_override_disables():
    for raw in ("early_engage_until_s=0", "early_engage_gain=0"):
        t = tuning_from_env(raw, log=lambda *_: None)
        assert gem_tune_for(t, 30.0).engage_gain == 0.0
    # Gem boost still applies when only the engage window is off.
    t = tuning_from_env("early_engage_until_s=0", log=lambda *_: None)
    assert gem_tune_for(t, 30.0).gem_gain == t.early_gem_gain


def test_engage_pull_points_at_enemy_and_gates():
    t = replace(Tuning(), engage_gain=1.2)
    pull = engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t)
    assert DIRS[int(np.argmax(pull))] == (-1, 0)
    assert pull[DIRS.index((1, 0))] < 0
    # Gates: unknown / low HP, danger at/above safe_danger, gain 0, nothing in range.
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=None, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=t.early_engage_hp - 0.01, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=t.safe_danger, hp=1.0, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=replace(t, engage_gain=0.0)).any()
    touching = np.array([(P[0] - 20.0, P[1], 0.22)], dtype=np.float32)  # inside standoff
    far = np.array([(P[0] - 400.0, P[1], 0.22)], dtype=np.float32)  # beyond engage_radius
    assert not engage_pull(touching, *P, here=0.1, hp=1.0, tune=t).any()
    assert not engage_pull(far, *P, here=0.1, hp=1.0, tune=t).any()
    # Fades with danger.
    lo = engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t).max()
    hi = engage_pull(LEFT_ARR, *P, here=0.4, hp=1.0, tune=t).max()
    assert 0 < hi < lo


def test_score_directions_lets_enemy_close_only_when_engaged():
    t = gem_tune_for(ON, 60.0)
    off = score_directions(LEFT, P, None, 1.0, hp=1.0, tune=replace(t, engage_gain=0.0))
    on = score_directions(LEFT, P, None, 1.0, hp=1.0, tune=t)
    assert off.vector[0] >= 0  # without engage: drifts away from the lone enemy
    assert on.vector[0] == -1  # engaged: steps toward it (Garlic range)
    low = score_directions(LEFT, P, None, 1.0, hp=0.5, tune=t)
    assert low.vector[0] >= 0  # low HP: no engage


def test_navigator_engages_early_not_later_and_not_in_breakout():
    clock = Clock()
    nav = Navigator(tune=ON, clock=clock, log=lambda *_: None)
    nav.game_s = 30.0
    res = result(LEFT)
    on = nav.step(res, hp=1.0)
    clock.t += 1 / 12
    assert _left_score(on) > max(on.scores[(1, 0)], on.scores[(1, -1)], on.scores[(1, 1)])

    clock2 = Clock()
    late = Navigator(tune=ON, clock=clock2, log=lambda *_: None)
    late.game_s = 200.0
    dec = late.step(result(LEFT), hp=1.0)
    off_nav = Navigator(tune=replace(Tuning(), early_engage_until_s=0), clock=Clock(), log=lambda *_: None)
    off_nav.game_s = 200.0
    ref = off_nav.step(result(LEFT), hp=1.0)
    assert dec.scores == ref.scores  # identical once the window has passed

    # During a breakout the engage pull is never applied.
    b = Navigator(tune=ON, clock=Clock(), log=lambda *_: None)
    b.game_s = 30.0
    b.breakout = (1.0, 0.0)
    b._breakout_origin = (1.0, 0.0)
    bon = b.step(result(LEFT), hp=1.0)
    boff = Navigator(tune=replace(Tuning(), early_engage_until_s=0), clock=Clock(), log=lambda *_: None)
    boff.game_s = 30.0
    boff.breakout = (1.0, 0.0)
    boff._breakout_origin = (1.0, 0.0)
    ref_b = boff.step(result(LEFT), hp=1.0)
    assert bon.mode.startswith("breakout")
    assert bon.scores == ref_b.scores


def test_engage_gated_below_safe_danger():
    t = replace(Tuning(), engage_gain=1.2)
    assert engage_pull(LEFT_ARR, *P, here=t.safe_danger * 0.5, hp=1.0, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=t.safe_danger + 0.5, hp=1.0, tune=t).any()


def test_direct_engage_gain_respects_time_limit():
    t = tuning_from_env("engage_gain=1.0,early_engage_until_s=150", log=lambda *_: None)
    assert gem_tune_for(t, 30.0).engage_gain > 0
    assert gem_tune_for(t, t.early_engage_until_s).engage_gain == 0.0
    off = tuning_from_env("engage_gain=1.0,early_engage_until_s=0", log=lambda *_: None)
    assert gem_tune_for(off, 30.0).engage_gain == 0.0


# Gems first, then hunt (early_engage_max_gems): a gem ~150 px to the right.
GEM_RIGHT = [(P[0] + 150.0, P[1])]


def test_gems_in_range_counts_only_within_radius():
    assert gems_in_range(None, *P, 300.0) == 0
    assert gems_in_range([], *P, 300.0) == 0
    gems = [(P[0] + 100.0, P[1]), (P[0], P[1] - 299.0), (P[0] - 301.0, P[1])]
    assert gems_in_range(gems, *P, 300.0) == 2


def test_engage_pull_off_while_gems_wait():
    t = replace(Tuning(), engage_gain=1.2)
    assert t.early_engage_max_gems == 0  # default: any gem in range pauses the engage pull
    assert engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t).any()
    assert engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t, gems=[]).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t, gems=GEM_RIGHT).any()
    # A gem beyond gem_radius does not count.
    far = [(P[0] + t.gem_radius + 10.0, P[1])]
    assert engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t, gems=far).any()
    # Threshold: up to early_engage_max_gems gems still engage; -1 disables the gate.
    two = GEM_RIGHT + [(P[0], P[1] + 120.0)]
    assert engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=replace(t, early_engage_max_gems=2), gems=two).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=replace(t, early_engage_max_gems=1), gems=two).any()
    assert engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=replace(t, early_engage_max_gems=-1), gems=two).any()


def test_score_directions_gem_beats_engage():
    t = gem_tune_for(ON, 60.0)
    gated = score_directions(LEFT, P, None, 1.0, gems=GEM_RIGHT, hp=1.0, tune=t)
    no_engage = score_directions(LEFT, P, None, 1.0, gems=GEM_RIGHT, hp=1.0, tune=replace(t, engage_gain=0.0))
    assert gated.scores == no_engage.scores  # engage contributes nothing while a gem waits
    assert gated.vector[0] == 1  # walks to the gem, away from the enemy
    ungated = score_directions(
        LEFT, P, None, 1.0, gems=GEM_RIGHT, hp=1.0, tune=replace(t, early_engage_max_gems=-1),
    )
    assert _left_score(ungated) > _left_score(gated)  # old behaviour: engage still pulls left
    # No gem: the engage pull is back (unchanged pre-gate behaviour).
    assert score_directions(LEFT, P, None, 1.0, hp=1.0, tune=t).vector[0] == -1


def test_navigator_early_engage_waits_for_gems():
    nav = Navigator(tune=ON, clock=Clock(), log=lambda *_: None)
    nav.game_s = 30.0
    on = nav.step(result(LEFT, gems=GEM_RIGHT), hp=1.0)
    off_nav = Navigator(tune=replace(Tuning(), early_engage_until_s=0), clock=Clock(), log=lambda *_: None)
    off_nav.game_s = 30.0
    ref = off_nav.step(result(LEFT, gems=GEM_RIGHT), hp=1.0)
    assert on.scores == ref.scores
    assert on.vector[0] == 1


def test_env_early_engage_max_gems():
    t = tuning_from_env("early_engage_max_gems=-1", log=lambda *_: None)
    assert t.early_engage_max_gems == -1 and isinstance(t.early_engage_max_gems, int)
    assert tuning_from_env("early_engage_max_gems=3", log=lambda *_: None).early_engage_max_gems == 3
