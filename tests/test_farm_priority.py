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


def test_touching_ring_breaks_out_above_quarter_hp():
    """test13 run 3, 1:11: ghouls and mudmen on Poe, danger 1.7, HP 1.00 to 0.47.

    A pack still gets dived. A distant ring still gets dived. A ring whose
    bodies are already touching, and which covers most directions, raises
    danger and leaves farm mode.
    """
    from vs_vision_bot.pathfind import _danger_at

    close = np.array(ring(P[0], P[1], 36, 24), np.float32)
    raw = float(_danger_at(close, np.array([P[0]]), np.array([P[1]]), Tuning().radius)[0])
    dec = score_directions(close, P, None, 0.0, hp=1.0)
    assert dec.danger_here > raw * 2
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 74.0
    run(nav, clock, result(close), 6, hp=1.0)
    assert nav.breakouts > 0 or nav.escaping
    assert "farm" not in nav.last.mode

    far = np.array(ring(P[0], P[1], 160, 24), np.float32)
    clock2 = Clock()
    held = Navigator(clock=clock2, log=lambda *_: None)
    held.game_s = 74.0
    run(held, clock2, result(far), 4, hp=1.0)
    assert held.breakouts == 0 and not held.escaping


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


def test_outline_contact_blocks_the_step_the_kernel_calls_safe():
    """test10 run2, ~67-70s: HP 1.00 to 0.50 in farm mode while danger stayed ~1.1.

    An outline ring at ~40px is outside the 28px floor, and the 42px kernel
    reads about 1. That mass is already a hit: do not step into it, report a
    higher danger, and do not shove backward. A far pack is still approached.
    engage_standoff=0 still walks onto the outline.
    """
    import math

    from vs_vision_bot.pathfind import OUTLINE_MASS, _danger_at

    pts = []
    for i in range(8):
        ang = math.radians(-50 + 100 * i / 7)
        pts.append((P[0] + 40 * math.cos(ang), P[1] + 40 * math.sin(ang), 0.25))
    outline = np.array(pts, np.float32)
    assert float(outline[:, 2].sum()) >= OUTLINE_MASS
    raw = float(_danger_at(outline, np.array([P[0]]), np.array([P[1]]), Tuning().radius)[0])
    assert raw < 1.5
    dec = score_directions(outline, P, None, 0.0, hp=1.0)
    assert dec.vector[0] <= 0
    assert dec.danger_here > raw
    opened = Tuning(engage_standoff=0.0)
    assert score_directions(outline, P, None, 0.0, hp=1.0, tune=opened).vector[0] > 0
    far = np.array(blob(P[0] + 120, P[1], n=10, spread=10, seed=5), np.float32)
    assert score_directions(far, P, None, 0.0, hp=1.0).vector[0] > 0
    speck = np.array([(P[0] + 40.0, P[1], 0.25)], np.float32)
    assert score_directions(speck, P, None, 0.0, hp=1.0).vector[0] > 0


def test_body_floor_refuses_the_sprite_without_a_backward_shove():
    """28px floor: do not step onto a body already inside it, and do not flee the pack.

    A gem beside that body still wins. engage_standoff=0 walks onto the center.
    engage_standoff=34 also pushes backward.
    """
    assert Tuning().engage_standoff == 28.0
    inside = np.array(blob(P[0] + 18, P[1], n=10, spread=6, seed=4), np.float32)
    outside = np.array(blob(P[0] + 120, P[1], n=10, spread=10, seed=5), np.float32)
    for hp in (0.26, 1.0, None):
        step = score_directions(inside, P, None, 0.0, hp=hp)
        assert step.vector[0] <= 0, hp
        assert score_directions(outside, P, None, 0.0, hp=hp).vector[0] > 0, hp
    # Gem north of a body on the east: take the gem. The 34px graze shoves west.
    beside = score_directions(
        inside, P, None, 0.0, gems=[(P[0], P[1] - 80)], hp=1.0,
    )
    assert beside.vector[1] < 0 and beside.vector[0] <= 0
    opened = Tuning(engage_standoff=0.0)
    assert score_directions(inside, P, None, 0.0, hp=1.0, tune=opened).vector[0] > 0
    grazed = Tuning(engage_standoff=34.0)
    shoved = score_directions(
        inside, P, None, 0.0, gems=[(P[0], P[1] - 80)], hp=1.0, tune=grazed,
    )
    assert shoved.vector[0] < 0

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    res = result(blob(P[0] + 18, P[1], n=10, spread=6, seed=4))
    res.gems = [(P[0], P[1] - 80)]
    vecs = run(nav, clock, res, 6, hp=1.0)
    assert all(v[1] < 0 for v in vecs)
    assert nav.breakouts == 0


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
    outside = np.array(blob(P[0] + 120, P[1], n=10, spread=10, seed=5), np.float32)
    for game_s in (180.0, 520.0):
        clock = Clock()
        nav = Navigator(clock=clock, log=lambda *_: None)
        nav.game_s = game_s
        held = run(nav, clock, result(inside), 6, hp=1.0)
        assert all(v[0] <= 0 for v in held), game_s
        assert nav.breakouts == 0 and not nav.escaping
        clock2 = Clock()
        far = Navigator(clock=clock2, log=lambda *_: None)
        far.game_s = game_s
        vecs = run(far, clock2, result(outside), 6, hp=1.0)
        assert all(v[0] > 0 for v in vecs), game_s

    # Closed ring during the skeleton wave is still a pack, not a breakout.
    gap = ring(480, 256, 160, 40, gap_deg=(-1, 25))
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    run(nav, clock, result(gap), 8, hp=1.0)
    assert nav.breakouts == 0 and not nav.escaping

    # Same ring in the flower window is a wall: graze it, do not ring-escape
    # and do not keep walking into the bodies.
    clock = Clock()
    wall = Navigator(clock=clock, log=lambda *_: None)
    wall.game_s = 320.0
    into = run(wall, clock, result(gap), 8, hp=1.0)
    assert wall.breakouts == 0 and not wall.escaping
    assert wall.cover >= wall.tune.ring_commit_cover
    # The east gap is the opening. Grazing does not spend the steps on it.
    assert sum(1 for v in into if v[0] > 0) < len(into)

    # 8:00 restores the early gem clump pull. Below the kite line it does not.
    late = gem_tune_for(Tuning(engage_standoff=34.0), 520.0)
    assert late.gem_target_gain == 0.0
    assert late.engage_standoff == 34.0
    plowed = skeleton_plow_tune(late, 520.0, 1.0)
    assert plowed.engage_standoff == 34.0
    assert skeleton_plow_tune(gem_tune_for(Tuning(), 520.0), 520.0, 1.0).engage_standoff == 28.0
    assert skeleton_plow_tune(
        gem_tune_for(Tuning(engage_standoff=0.0), 520.0), 520.0, 1.0,
    ).engage_standoff == 0.0
    assert plowed.gem_target_gain >= 1.4
    assert plowed.gem_gain >= Tuning().early_gem_gain
    assert plowed.farm_gain > Tuning().farm_gain
    held = skeleton_plow_tune(late, 520.0, 0.25)
    assert held.gem_target_gain == 0.0
    assert held.engage_standoff == 34.0


def test_skeleton_window_does_not_flee_a_chest_or_a_touching_ring():
    """test15 Run 1 at full HP, 2:00-5:00: escape mode, breakouts=0, dive=0.

    A chest leash orbited with kite_floor, which turns farm off. Run 3 also
    broke out of touching rings above 25% HP and was Lv9 at 4:00. test11 Run 2
    farmed danger 10 through the same window. The touch ring still breaks out
    outside the skeleton waves.
    """
    from vs_vision_bot.pathfind import TOUCH_MASS, _threat_mass

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    pack = blob(480, 120, n=20, spread=40, seed=1)
    res = result(pack, gems=[(300.0, 256.0), (320.0, 280.0)])
    res.chest = (233.0, 81.0)
    vecs = run(nav, clock, res, 6, hp=1.0)
    assert nav.breakouts == 0 and not nav.escaping
    assert nav.last.mode.startswith("farm")
    assert nav.dive > 0
    assert all(v[0] < 0 for v in vecs)

    touch = np.array(ring(P[0], P[1], 36, 24), np.float32)
    assert _threat_mass(touch, P[0], P[1], 42.0) >= TOUCH_MASS
    skel = Navigator(clock=Clock(), log=lambda *_: None)
    skel.game_s = 180.0
    run(skel, skel.clock, result(touch), 6, hp=0.44)
    assert skel.breakouts == 0 and not skel.escaping
    assert "farm" in skel.last.mode

    early = Navigator(clock=Clock(), log=lambda *_: None)
    early.game_s = 74.0
    run(early, early.clock, result(touch), 6, hp=1.0)
    assert early.breakouts > 0 or early.escaping
    assert "farm" not in early.last.mode

    # 4:20-5:00 is still the skeleton window, and also the flower graze.
    # A ring already on Poe has to be able to leave; the 4:30 horde sim dies
    # if this breakout is suppressed.
    flower = Navigator(clock=Clock(), log=lambda *_: None)
    flower.game_s = 270.0
    run(flower, flower.clock, result(touch), 6, hp=1.0)
    assert flower.breakouts > 0 or flower.escaping


def test_evolution_ready_walks_a_chest_past_gems():
    """test14 run 3: Soul Eater was ready and chest sightings were cleared for gems.

    The ~11:4x chest did open (OCR ``1id4s``, still ``Treasure Found!``, opened 3).
    The other sightings were seek-then-clear. With an evolution ready, the chest
    wins over a gem in range and the seek stays up.
    """
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    nav.set_build_state(True, False)
    nav.game_s = 700.0
    pack = blob(480, 120, n=10, spread=24, seed=1)
    both = result(pack, gems=[(300.0, 256.0)])
    both.chest = (720.0, 256.0)
    vecs = run(nav, clock, both, 6, hp=1.0)
    assert all(v[0] > 0 for v in vecs)
    assert any(ln.startswith("[chest] seek") for ln in lines)
    assert not any(ln.startswith("[chest] clear") for ln in lines)

    # No evolution: the same screen still walks the gem, not the chest.
    clock2 = Clock()
    plain = Navigator(clock=clock2, log=lambda *_: None)
    plain.game_s = 700.0
    vecs = run(plain, clock2, both, 6, hp=1.0)
    assert all(v[0] < 0 for v in vecs)


def test_late_closing_crowd_breaks_out_above_quarter_hp():
    """test14 run 3, 16:19: farmed at hp 0.36-0.50 with danger 4-6, then 17.

    The bodies were outside the 42px ring. After 10:00 that closed crowd
    breaks out. The same ring early, a 160px ring, and a one-sided pack do not.
    """
    from vs_vision_bot.pathfind import TOUCH_MASS, _danger_at, _threat_mass

    close = np.array(ring(P[0], P[1], 80, 20), np.float32)
    touch = _threat_mass(close, P[0], P[1], 42.0)
    danger = float(_danger_at(close, np.array([P[0]]), np.array([P[1]]), Tuning().radius)[0])
    assert touch < TOUCH_MASS
    assert danger >= 4.0

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 960.0
    run(nav, clock, result(close), 6, hp=0.45)
    assert nav.breakouts > 0 or nav.escaping
    assert "farm" not in nav.last.mode

    early = Navigator(clock=Clock(), log=lambda *_: None)
    early.game_s = 74.0
    run(early, early.clock, result(close), 6, hp=0.45)
    assert early.breakouts == 0 and not early.escaping
    assert "farm" in early.last.mode

    far = np.array(ring(P[0], P[1], 160, 24), np.float32)
    held = Navigator(clock=Clock(), log=lambda *_: None)
    held.game_s = 960.0
    run(held, held.clock, result(far), 4, hp=0.45)
    assert held.breakouts == 0 and not held.escaping

    pack = blob(680, 256, n=16, spread=28, seed=3)
    diver = Navigator(clock=Clock(), log=lambda *_: None)
    diver.game_s = 960.0
    vecs = run(diver, diver.clock, result(pack), 6, hp=0.45)
    assert diver.breakouts == 0 and not diver.escaping
    assert all(v[0] > 0 for v in vecs)
