"""Early Garlic engagement: Tuning.early_engage_* pulls toward weak early enemies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vs_vision_bot.pathfind import (
    DIRS, Navigator, Tuning, engage_pull, gem_tune_for, score_directions, tuning_from_env,
)

from tests.test_movement import Clock, P, result, run

# One small enemy ~120 px to the left (4 grid points of 0.22, like vision.threat_points).
LEFT = [(360.0 + ox, 256.0 + oy, 0.22) for ox in (-6, 6) for oy in (-6, 6)]
LEFT_ARR = np.array(LEFT, dtype=np.float32)


def _left_score(dec) -> float:
    return max(dec.scores[(-1, 0)], dec.scores[(-1, -1)], dec.scores[(-1, 1)])


def test_defaults_and_gem_tune_for_window():
    t = Tuning()
    assert t.engage_gain == 0.0  # base knob off; only the early window turns it on
    assert t.early_engage_until_s == 150.0 and t.early_engage_gain == 1.2
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
    # Gates: unknown / low HP, danger at/above engage_danger, gain 0, nothing in range.
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=None, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=t.early_engage_hp - 0.01, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=t.engage_danger, hp=1.0, tune=t).any()
    assert not engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=replace(t, engage_gain=0.0)).any()
    touching = np.array([(P[0] - 20.0, P[1], 0.22)], dtype=np.float32)  # inside standoff
    far = np.array([(P[0] - 400.0, P[1], 0.22)], dtype=np.float32)  # beyond engage_radius
    assert not engage_pull(touching, *P, here=0.1, hp=1.0, tune=t).any()
    assert not engage_pull(far, *P, here=0.1, hp=1.0, tune=t).any()
    # Fades with danger.
    lo = engage_pull(LEFT_ARR, *P, here=0.1, hp=1.0, tune=t).max()
    hi = engage_pull(LEFT_ARR, *P, here=1.2, hp=1.0, tune=t).max()
    assert 0 < hi < lo


def test_score_directions_lets_enemy_close_only_when_engaged():
    t = gem_tune_for(Tuning(), 60.0)
    off = score_directions(LEFT, P, None, 1.0, hp=1.0, tune=replace(t, engage_gain=0.0))
    on = score_directions(LEFT, P, None, 1.0, hp=1.0, tune=t)
    assert off.vector[0] >= 0  # without engage: drifts away from the lone enemy
    assert on.vector[0] == -1  # engaged: steps toward it (Garlic range)
    low = score_directions(LEFT, P, None, 1.0, hp=0.5, tune=t)
    assert low.vector[0] >= 0  # low HP: no engage


def test_navigator_engages_early_not_later_and_not_in_breakout():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 30.0
    res = result(LEFT)
    on = nav.step(res, hp=1.0)
    clock.t += 1 / 12
    assert _left_score(on) > max(on.scores[(1, 0)], on.scores[(1, -1)], on.scores[(1, 1)])

    clock2 = Clock()
    late = Navigator(clock=clock2, log=lambda *_: None)
    late.game_s = 200.0
    dec = late.step(result(LEFT), hp=1.0)
    off_nav = Navigator(tune=replace(Tuning(), early_engage_until_s=0), clock=Clock(), log=lambda *_: None)
    off_nav.game_s = 200.0
    ref = off_nav.step(result(LEFT), hp=1.0)
    assert dec.scores == ref.scores  # identical once the window has passed

    # During a breakout the engage pull is never applied.
    b = Navigator(clock=Clock(), log=lambda *_: None)
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
