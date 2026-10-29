"""High HP shrinks the kite and keeps a moderate-danger gem pull.

Garlic kills inside ~40 px (the sim aura) and the engage standoff is 34, but the
danger kernel is 42 and the kite's near band is 110, so a healthy bot still
arcs wide of the aura. Above dive_hp_hi the step kernel and danger_gain shrink
(42 -> ~29 at the default 0.7) and gems fade out later. The scale knob at 1.0
is today's scoring. Low, critical, blind, and unknown HP do not dive.
"""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot.menus import hp_for_nav
from vs_vision_bot.pathfind import Navigator, Tuning, dive_level, score_directions, tuning_from_env

from tests.test_movement import Clock, P, result, ring, run


def _pack_east(dist: float = 180.0, n: int = 12, spread: float = 12.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            (
                P[0] + dist + float(rng.uniform(-spread, spread)),
                P[1] + float(rng.uniform(-spread, spread)),
                1.0,
            )
            for _ in range(n)
        ],
        np.float32,
    )


def _south_crowd() -> np.ndarray:
    """Small pack below the player. Danger sits just above gem_danger 1.6."""
    rng = np.random.default_rng(0)
    return np.array(
        [
            (
                P[0] + float(rng.uniform(-12, 12)),
                P[1] + 50.0 + float(rng.uniform(-12, 12)),
                1.0,
            )
            for _ in range(4)
        ],
        np.float32,
    )


# Tangent is north. Today's field still prefers the outward diagonal (wide kite).
_KITE_N = (0.0, 1.0)


def _off() -> Tuning:
    return tuning_from_env("dive_radius_scale=1", log=lambda *_: None)


def test_defaults_match_the_garlic_standoff():
    tune = Tuning()
    # 0.7 * radius 42 = 29.4, inside the ~34 px standoff and the ~40 px aura.
    assert tune.dive_hp_hi == 0.75
    assert tune.dive_hp_lo == 0.5
    assert tune.dive_radius_scale == 0.7
    assert tune.dive_gem_gain == 1.5
    # 28px is the body floor inside the 34px aura. 0 walks onto sprite centers.
    assert tune.engage_standoff == 28.0
    assert tune.early_gem_target_gain >= 1.4
    assert tune.radius * tune.dive_radius_scale < Tuning(engage_standoff=34.0).engage_standoff
    assert dive_level(1.0, tune) == 1.0
    assert dive_level(0.75, tune) == 1.0
    assert math.isclose(dive_level(0.65, tune), 0.6)
    assert dive_level(0.5, tune) == 0.0
    assert dive_level(0.49, tune) == 0.0
    assert dive_level(0.12, tune) == 0.0
    assert dive_level(None, tune) == 0.0
    assert dive_level(1.0, _off()) == 0.0


def test_high_hp_accepts_a_closer_approach():
    pts = _pack_east()
    # Above pack_dive_hp the step goes into the pack. The radius scale does not
    # turn that off; the kite band still leaves.
    high = score_directions(pts, P, _KITE_N, 1.0, hp=1.0)
    assert high.vector[0] > 0
    assert high.dive == 1.0
    cautious = score_directions(pts, P, _KITE_N, 1.0, hp=1.0, tune=_off())
    assert cautious.vector[0] > 0
    assert cautious.dive == 0.0
    kite = score_directions(pts, P, _KITE_N, 1.0, hp=0.20)
    assert kite.vector[0] <= 0
    assert kite.dive == 0.0


def test_scale_one_matches_old_scoring():
    pts = _pack_east()
    gems = [(P[0] + 200.0, P[1])]
    off = _off()
    # Unknown HP farms, same as a full bar. Scale 1.0 does not disable that.
    # Below the kite line the scale-1 scores match the default.
    unknown = score_directions(pts, P, _KITE_N, 1.0, gems=gems, hp=None)
    restored = score_directions(pts, P, _KITE_N, 1.0, gems=gems, hp=1.0, tune=off)
    full = score_directions(pts, P, _KITE_N, 1.0, gems=gems, hp=1.0)
    assert unknown.vector[0] > 0
    assert restored.vector == unknown.vector == full.vector
    assert full.dive == 1.0
    assert restored.dive == 0.0
    kite = score_directions(pts, P, _KITE_N, 1.0, gems=gems, hp=0.20)
    kite_off = score_directions(pts, P, _KITE_N, 1.0, gems=gems, hp=0.20, tune=off)
    assert kite.scores == kite_off.scores
    assert kite.vector[0] <= 0


def test_mid_low_and_critical_hp_are_unchanged():
    pts = _pack_east()
    off = _off()
    for hp in (0.5, 0.3, 0.12):
        now = score_directions(pts, P, _KITE_N, 1.0, hp=hp)
        old = score_directions(pts, P, _KITE_N, 1.0, hp=hp, tune=off)
        assert now.scores == old.scores
        assert now.vector == old.vector
        assert now.dive == 0.0
    # Above 25% the pack is entered. Critical HP still leaves it.
    assert score_directions(pts, P, _KITE_N, 1.0, hp=0.5).vector[0] > 0
    assert score_directions(pts, P, _KITE_N, 1.0, hp=0.12).vector[0] <= 0
    mid = score_directions(pts, P, _KITE_N, 1.0, hp=0.625)
    assert mid.dive == 0.5
    assert mid.vector[0] > 0


def test_blind_and_unknown_hp_do_not_dive():
    pts = _pack_east()
    blind_hp = hp_for_nav(0.95, blind=True, low_held=False)
    assert blind_hp == 0.0
    blind = score_directions(pts, P, _KITE_N, 1.0, hp=blind_hp)
    critical = score_directions(pts, P, _KITE_N, 1.0, hp=0.0)
    unknown = score_directions(pts, P, _KITE_N, 1.0, hp=None)
    healthy = score_directions(pts, P, _KITE_N, 1.0, hp=0.95)
    assert blind.scores == critical.scores
    assert blind.dive == 0.0
    assert unknown.dive == 0.0
    assert unknown.scores == score_directions(pts, P, _KITE_N, 1.0, hp=None, tune=_off()).scores
    # Unknown and a full bar farm into the pack. A blind 0 read is critical and leaves.
    assert healthy.vector[0] > 0
    assert unknown.vector[0] > 0
    assert blind.vector[0] <= 0
    assert blind.vector != healthy.vector


def test_high_hp_seeks_a_gem_that_moderate_danger_used_to_refuse():
    """Danger just above gem_danger 1.6. Gems add nothing today; a full dive goes east."""
    pts = _south_crowd()
    gems = [(P[0] + 200.0, P[1])]

    def frames(hp, tune=None, with_gems=True):
        clock = Clock()
        nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
        nav.game_s = 100.0
        nav.heading = -math.pi / 2
        res = result(pts, gems=gems if with_gems else None)
        out = run(nav, clock, res, 6, hp=hp)
        return nav, out

    high, toward = frames(1.0)
    assert Tuning().gem_danger < high.last.danger_here < Tuning().gem_danger * Tuning().dive_gem_gain
    assert high.dive == 1.0
    assert toward[-1][0] > 0
    bare, away = frames(1.0, with_gems=False)
    assert away[-1][1] > 0  # into the south pack, not out to the gem
    assert away[-1][0] <= 0
    assert bare.last.danger_here == high.last.danger_here

    # At or below 25% the gem is refused. Above it, and with no HP read, it is taken.
    for hp in (0.5, 0.3, None):
        nav, vecs = frames(hp)
        old, old_vecs = frames(hp, tune=_off())
        assert vecs == old_vecs
        assert vecs[-1][0] > 0
    nav, vecs = frames(0.12)
    old, old_vecs = frames(0.12, tune=_off())
    assert vecs == old_vecs
    assert nav.dive == 0.0
    assert vecs[-1][0] <= 0

    blind_hp = hp_for_nav(0.95, blind=True)
    blind, blind_vecs = frames(blind_hp)
    critical, crit_vecs = frames(0.0)
    assert blind_vecs == crit_vecs
    assert blind.dive == 0.0


def test_kite_floor_emergencies_ignore_the_dive():
    pts = _pack_east()
    held = score_directions(pts, P, _KITE_N, 1.0, hp=1.0, kite_floor=True)
    old = score_directions(pts, P, _KITE_N, 1.0, hp=1.0, kite_floor=True, tune=_off())
    assert held.dive == 0.0
    assert held.scores == old.scores
    assert held.vector == (-1, 1)

    # A committed ring escape (cover high enough to wall-commit) stays on the gap.
    gap = ring(480, 256, 170, 40, gap_deg=(-1, 25))

    def escape(tune):
        clock = Clock()
        nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
        nav.game_s = 318.0
        vecs = run(nav, clock, result(gap), 8, hp=0.24)
        assert nav.escaping and nav.cover >= nav.tune.ring_commit_cover
        assert nav.dive == 0.0
        return vecs

    assert escape(Tuning()) == escape(_off())


def test_nav_line_reports_dive_level():
    """The periodic [nav] line carries the aggression actually applied."""
    import vs_vision_bot.main as main

    src = open(main.__file__, encoding="utf-8").read()
    assert "dive={nav.dive:.1f}" in src

    partial = score_directions(_pack_east(), P, _KITE_N, 1.0, hp=0.65)
    assert f"{partial.dive:.1f}" == "0.6"
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    clock.t += 1 / 12
    # Empty field: no breakout, so the logged level is the HP interpolation.
    nav.step(result([]), hp=0.65)
    assert f"{nav.dive:.1f}" == "0.6"
