"""Above 25% HP the bot farms packs and gems. At or below 25% it kites.

Leveling is first: run into the pack and into XP gems (red, then green, then
blue). A chest is second, and only when no gem is in range. ``hp > 0.25`` and
an unknown bar (full, before any damage read) do not kite or ring-break out.
``hp <= 0.25`` is the old flee.
"""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import (
    Tuning,
    gem_tune_for,
    score_directions,
    skeleton_phase,
    skeleton_plow_tune,
)

from tests.test_movement import Clock, P, blob, result, ring, run
from vs_vision_bot.pathfind import Navigator


def _east_pack() -> np.ndarray:
    return np.array(blob(680, 256, n=16, spread=28, seed=3), np.float32)


def _net_x(vecs) -> int:
    return sum(v[0] for v in vecs)


def test_score_above_quarter_walks_into_the_pack():
    pts = _east_pack()
    for hp in (0.26, 1.0, None):
        dec = score_directions(pts, P, None, 0.0, hp=hp)
        assert dec.vector[0] > 0, hp
        assert not dec.mode.startswith("breakout")
        assert "escape" not in dec.mode


def test_score_at_or_below_quarter_does_not_walk_into_the_pack():
    pts = _east_pack()
    into = score_directions(pts, P, None, 0.0, hp=0.26)
    assert into.vector[0] > 0
    for hp in (0.25, 0.24, 0.12):
        dec = score_directions(pts, P, None, 0.0, hp=hp)
        assert dec.vector[0] <= 0, hp
        assert dec.vector != into.vector


def test_nav_above_quarter_dives_the_pack_and_skips_the_ring():
    """A closed ring with an east gap is a breakout today. Above 25% it is a farm."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    pts = blob(680, 256, n=16, spread=28, seed=3)
    for hp in (0.26, 1.0, None):
        vecs = run(nav, clock, result(pts), 8, hp=hp)
        assert _net_x(vecs) > 0, hp
        assert all(v[0] > 0 for v in vecs), hp
        assert nav.breakouts == 0
        assert not nav.escaping
        assert "escape" not in nav.last.mode
        assert not nav.last.mode.startswith("breakout")

    gap = ring(480, 256, 160, 40, gap_deg=(-1, 25))
    clock2 = Clock()
    walled = Navigator(clock=clock2, log=lambda *_: None)
    walled.game_s = 660.0
    run(walled, clock2, result(gap), 8, hp=1.0)
    assert not walled.escaping
    assert walled.breakouts == 0
    assert not walled.last.mode.startswith("ring-escape")


def test_nav_at_or_below_quarter_kites_and_breaks_out():
    pts = blob(680, 256, n=16, spread=28, seed=3)
    for hp in (0.25, 0.24, 0.12):
        clock = Clock()
        nav = Navigator(clock=clock, log=lambda *_: None)
        vecs = run(nav, clock, result(pts), 8, hp=hp)
        assert all(v[0] <= 0 for v in vecs), hp
        assert _net_x(vecs) < 0

    gap = ring(480, 256, 160, 40, gap_deg=(-1, 25))
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 660.0
    vecs = run(nav, clock, result(gap), 8, hp=0.25)
    assert nav.escaping or nav.breakouts >= 1
    assert sum(1 for v in vecs if v[0] > 0) >= 4


def test_gems_beat_a_chest_and_red_beats_blue():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    pack = blob(480, 120, n=10, spread=24, seed=1)
    both = result(pack, gems=[(300.0, 256.0)])
    both.chest = (720.0, 256.0)
    vecs = run(nav, clock, both, 6, hp=1.0)
    assert all(v[0] < 0 for v in vecs)

    clock2 = Clock()
    nav2 = Navigator(clock=clock2, log=lambda *_: None)
    colored = result(pack, gems=[(700.0, 256.0), (260.0, 256.0)])
    colored.gem_colors = ["red", "blue"]
    vecs = run(nav2, clock2, colored, 6, hp=1.0)
    assert all(v[0] > 0 for v in vecs)


def test_chest_is_second_and_is_walked_through_the_pack():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    pack = blob(480, 80, n=12, spread=20, seed=2)
    res = result(pack)
    res.chest = (720.0, 256.0)
    vecs = run(nav, clock, res, 6, hp=1.0)
    assert all(v[0] > 0 for v in vecs)
    assert nav.breakouts == 0
    assert not nav.escaping


def test_pack_dive_hp_zero_restores_the_flee():
    pts = _east_pack()
    off = Tuning(pack_dive_hp=0.0)
    dec = score_directions(pts, P, None, 0.0, hp=1.0, tune=off)
    assert dec.vector[0] < 0


def test_above_quarter_runs_into_a_dense_pack():
    """Default engage_standoff is 0. Above 25% HP a pack on top of Poe is entered.

    VS_TUNE=engage_standoff=34 still grazes that ring instead of the center.
    """
    assert Tuning().engage_standoff == 0.0
    inside = np.array(blob(P[0] + 18, P[1], n=10, spread=6, seed=4), np.float32)
    outside = np.array(blob(P[0] + 120, P[1], n=10, spread=10, seed=5), np.float32)
    for hp in (0.26, 1.0, None):
        assert score_directions(inside, P, None, 0.0, hp=hp).vector[0] > 0, hp
        assert score_directions(outside, P, None, 0.0, hp=hp).vector[0] > 0, hp
    # A gem buried in the sprites is collected by walking into them.
    buried = score_directions(inside, P, None, 0.0, gems=[(P[0] + 18, P[1])], hp=1.0)
    assert buried.vector[0] > 0
    # A gem off the pack still wins.
    clear = score_directions(outside, P, None, 0.0, gems=[(P[0] - 160, P[1])], hp=1.0)
    assert clear.vector[0] < 0

    grazed = Tuning(engage_standoff=34.0)
    on_ring = np.array([(P[0] + 34.0, P[1], 1.0)], np.float32)
    for hp in (0.26, 1.0, None):
        assert score_directions(inside, P, None, 0.0, hp=hp, tune=grazed).vector[0] <= 0, hp
        assert score_directions(on_ring, P, None, 0.0, hp=hp, tune=grazed).vector[0] <= 0, hp

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    res = result(blob(P[0] + 18, P[1], n=10, spread=6, seed=4))
    res.chest = (P[0] + 40, P[1])
    vecs = run(nav, clock, res, 6, hp=1.0)
    assert all(v[0] > 0 for v in vecs)
    assert nav.breakouts == 0
    assert not nav.last.mode.startswith("ring-escape")


def test_skeleton_phase_plows_into_the_pack_and_aims_gems():
    """2:00–5:00 and 8:00–10:00, above 25% HP: no standoff, no breakout, into the pack.

    The 8:00 wave is past early_gem_until_s, so the plow puts the early gem
    target back. At or below 25% HP the kite returns. A tuned standoff of 34
    does not survive the skeleton window.
    """
    assert Tuning().early_gem_target_gain >= 1.4
    assert skeleton_phase(180.0) and skeleton_phase(299.0) and skeleton_phase(520.0)
    assert not skeleton_phase(300.0) and not skeleton_phase(400.0)
    assert not skeleton_phase(320.0)  # flower window, after the 5:00 skeleton cutoff

    inside = np.array(blob(P[0] + 18, P[1], n=10, spread=6, seed=4), np.float32)
    for game_s in (180.0, 520.0):
        clock = Clock()
        nav = Navigator(
            tune=Tuning(engage_standoff=34.0), clock=clock, log=lambda *_: None,
        )
        nav.game_s = game_s
        vecs = run(nav, clock, result(inside), 6, hp=1.0)
        assert all(v[0] > 0 for v in vecs), game_s
        assert nav.breakouts == 0
        assert not nav.escaping

    # Closed ring during the skeleton wave is still a pack, not a breakout.
    gap = ring(480, 256, 160, 40, gap_deg=(-1, 25))
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    run(nav, clock, result(gap), 8, hp=1.0)
    assert nav.breakouts == 0 and not nav.escaping

    # Same ring in the flower window is a wall: leave it even at full HP.
    clock = Clock()
    wall = Navigator(clock=clock, log=lambda *_: None)
    wall.game_s = 320.0
    run(wall, clock, result(gap), 8, hp=1.0)
    assert wall.escaping or wall.breakouts >= 1
    assert wall.last.mode.startswith("ring-escape") or wall.last.mode.startswith("breakout")

    # 8:00 restores the early gem clump pull. Below the kite line it does not.
    late = gem_tune_for(Tuning(engage_standoff=34.0), 520.0)
    assert late.gem_target_gain == 0.0
    assert late.engage_standoff == 34.0
    plowed = skeleton_plow_tune(late, 520.0, 1.0)
    assert plowed.engage_standoff == 0.0
    assert plowed.gem_target_gain >= 1.4
    assert plowed.gem_gain >= Tuning().early_gem_gain
    assert plowed.farm_gain > Tuning().farm_gain
    held = skeleton_plow_tune(late, 520.0, 0.25)
    assert held.gem_target_gain == 0.0
    assert held.engage_standoff == 34.0
