"""Early gem target: a non-cancelling pull at one gem clump (Tuning.early_gem_target_gain)."""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import Tuning, gem_target, gem_tune_for, score_directions, tuning_from_env

PX, PY = 480.0, 256.0
EMPTY = np.zeros((0, 3), np.float32)
# Two-gem clump to the right (~220 px), single gems up-left and down-left: the summed
# 1/(1+d/80) pull nearly cancels, so the kite bias would win without the target.
SCATTER = [(700.0, 256.0), (712.0, 262.0), (330.0, 120.0), (330.0, 392.0)]


def test_gem_target_prefers_clump_and_respects_radius():
    assert gem_target(None, PX, PY, 460, 70) is None
    assert gem_target([], PX, PY, 460, 70) is None
    assert gem_target([(PX + 500, PY)], PX, PY, 460, 70) is None  # outside radius
    tx, ty = gem_target(SCATTER, PX, PY, 460, 70)
    assert tx > PX and abs(ty - PY) < 20
    # A lone near gem beats a far pair.
    assert gem_target([(PX - 60, PY), (PX + 400, PY), (PX + 410, PY)], PX, PY, 460, 70) == (PX - 60, PY)
    # Accepts numpy arrays (score_directions passes the in-radius slice).
    assert gem_target(np.asarray(SCATTER, np.float32), PX, PY, 460, 70) is not None


def test_gem_tune_for_swaps_target_gain_only_early():
    t = Tuning()
    assert t.gem_target_gain == 0.0 and t.early_gem_target_gain > 0
    assert gem_tune_for(t, 100.0).gem_target_gain == t.early_gem_target_gain
    assert gem_tune_for(t, 400.0).gem_target_gain == 0.0
    off = tuning_from_env("early_gem_target_gain=0", log=lambda *_: None)
    assert gem_tune_for(off, 100.0).gem_target_gain == 0.0


def test_target_pull_beats_kite_when_scattered_gems_cancel():
    kite = (0.0, -1.0)  # kite says "up"
    # Kite band, without the defensive gem veto. Above pack_dive_hp the kite
    # term is off and the gem pull is what moves.
    band = dict(low_hp=0.10, critical_hp=0.10)
    early = gem_tune_for(Tuning(**band), 120.0)
    off = gem_tune_for(Tuning(early_gem_target_gain=0.0, **band), 120.0)
    no_target = score_directions(EMPTY, (PX, PY), kite, 1.0, gems=SCATTER, tune=off, hp=0.24)
    with_target = score_directions(EMPTY, (PX, PY), kite, 1.0, gems=SCATTER, tune=early, hp=0.24)
    assert no_target.vector[0] <= 0          # the cancelled sum leaves the kite in charge
    assert with_target.vector[0] == 1        # heads for the clump on the right
    assert with_target.scores[(1, 0)] > no_target.scores[(1, 0)]


def test_target_pull_fades_with_danger_and_off_at_low_hp():
    band = dict(low_hp=0.10, critical_hp=0.10)
    early = gem_tune_for(Tuning(**band), 120.0)
    # Enemy blob next to the player: danger here >= gem_danger -> no gem terms
    # once HP is in the kite band. Above that line the gems keep pulling.
    crowd = np.array([[PX + dx, PY + dy, 1.0] for dx in (-12, 0, 12) for dy in (-12, 0, 12)], np.float32)
    off = gem_tune_for(Tuning(early_gem_target_gain=0.0, **band), 120.0)
    a = score_directions(crowd, (PX, PY), None, 0.0, gems=SCATTER, tune=early, hp=0.24)
    b = score_directions(crowd, (PX, PY), None, 0.0, gems=SCATTER, tune=off, hp=0.24)
    assert a.danger_here >= early.gem_danger
    assert a.scores == b.scores
    farming = score_directions(crowd, (PX, PY), None, 0.0, gems=SCATTER, tune=early, hp=1.0)
    farming_off = score_directions(crowd, (PX, PY), None, 0.0, gems=SCATTER, tune=off, hp=1.0)
    assert farming.scores != farming_off.scores
    # Low HP (defensive): no gem pull either.
    lo_early = gem_tune_for(Tuning(), 120.0)
    lo_off = gem_tune_for(Tuning(early_gem_target_gain=0.0), 120.0)
    lo_a = score_directions(EMPTY, (PX, PY), None, 0.0, gems=SCATTER, hp=0.20, tune=lo_early)
    lo_b = score_directions(EMPTY, (PX, PY), None, 0.0, gems=SCATTER, hp=0.20, tune=lo_off)
    assert lo_a.scores == lo_b.scores
