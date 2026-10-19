"""A seen chest stays an anchor. The camera follows the player, so the anchor
is kept in world space and dead-reckoned from the bot's own steps.

Oct 18: 205 chest seeks and 0 reached. Between sightings the bot drifted off,
usually left. Past the leash radius it has to walk back when danger allows.
A committed escape still takes a clearly open gap, but a tied corridor prefers
the arrow side. Critical HP is not pulled off its heading.
"""

from __future__ import annotations

import math

from vs_vision_bot.pathfind import Navigator, Tuning, tuning_from_env
from vs_vision_bot.vision import PLAYFIELD_X1, REF_H_DEFAULT

from tests.test_movement import Clock, blob, result, ring, run

PLAYER = (480.0, 256.0)
CHEST_RIGHT = (720.0, 256.0)
ARROW_RIGHT = (900.0, 256.0)
ARROW_DOWN = (480.0, 500.0)


def _anchor(nav: Navigator, clock: Clock, *, chest=None, arrow=None) -> None:
    seen = result([])
    seen.chest = chest
    seen.chest_arrow = arrow
    run(nav, clock, seen, 1, hp=1.0)


def _park_west_of(nav: Navigator, dist: float) -> None:
    """Put the player ``dist`` px west of the anchor, heading further away.

    The short on-screen seek hold is a different mechanism (it keeps the last
    screen point for ``chest_hold_s``). These cases are about the world anchor
    after that hold has already ended.
    """
    nav._chest_hold = None
    nav._chest_latched = False
    nav._chest_miss_s = 0.0
    nav._chest_miss_n = 0
    wx, wy = nav._leash
    nav._world = (wx - dist, wy)
    nav.heading = math.pi
    nav.current = (-1, 0)


def test_arrow_anchor_is_past_the_playfield_edge():
    """An arrow is a direction plus 'beyond the edge', not the sprite itself."""
    tune = Tuning()
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, arrow=ARROW_RIGHT)
    ox = nav._leash[0] - nav._world[0]
    oy = nav._leash[1] - nav._world[1]
    assert nav._world == (0.0, 0.0)
    assert oy == 0.0
    assert ox == (PLAYFIELD_X1 - PLAYER[0]) + tune.chest_leash_beyond

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, arrow=ARROW_DOWN)
    ox = nav._leash[0] - nav._world[0]
    oy = nav._leash[1] - nav._world[1]
    assert ox == 0.0
    assert oy == (REF_H_DEFAULT - PLAYER[1]) + tune.chest_leash_beyond


def _anchor_logs(logs: list[str]) -> list[str]:
    return [ln for ln in logs if ln.startswith("[leash] chest ") or ln.startswith("[leash] arrow ")]


def test_leash_dead_reckons_from_our_own_steps():
    """The anchor stays in world space while the player moves.

    An open field steps in toward the chest, so the player closes the gap.
    The earlier pull-back contract (keep walking away inside the leash) was
    replaced by that step-in. Re-sights do not log a second anchor line.
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    anchor = nav._leash
    assert anchor is not None
    assert len(_anchor_logs(logs)) == 1
    _park_west_of(nav, 200.0)
    world_x = nav._world[0]
    run(nav, clock, result([]), 6, hp=1.0)
    assert nav._leash == anchor
    assert nav._world[0] > world_x + 30.0
    assert nav._leash_dist() < 170.0
    assert len(_anchor_logs(logs)) == 1


def test_drift_past_the_leash_radius_pulls_back():
    """Far west of an anchored chest, an away heading turns around."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, 2000.0)
    assert nav._leash_dist() > nav.tune.chest_leash_radius
    vecs = run(nav, clock, result([]), 6, hp=1.0)
    assert all(v[0] > 0 for v in vecs)
    assert "leash" in nav.last.mode


def test_arrow_only_anchor_biases_a_tied_escape():
    """Empty-field escape costs all tie. Prefer-left loses to an arrow on the right."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, arrow=ARROW_RIGHT)
    nav.heading = math.pi
    nav.request_revive_escape()
    vecs = run(nav, clock, result([]), 4, hp=1.0)
    assert nav.escaping and nav.breakout is not None
    assert nav.breakout[0] > 0.5
    assert all(v[0] > 0 for v in vecs)


def test_open_gap_beats_an_arrow_when_it_is_clearly_cheaper():
    """The leash only nudges a close call. A real gap the other way still wins."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, arrow=ARROW_RIGHT)
    assert nav._leash is not None
    nav.heading = 0.0  # prefer the arrow side; the gap is to the left
    nav.request_revive_escape()
    closing = result(ring(480, 256, 170, 40, gap_deg=(170, 200)))
    vecs = run(nav, clock, closing, 4, hp=1.0)
    assert nav.escaping and nav.breakout is not None and nav.breakout[0] < 0
    assert all(v[0] < 0 for v in vecs)


def test_critical_hp_keeps_a_far_leash_but_does_not_walk_to_it():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, 2000.0)
    vecs = run(nav, clock, result([]), 4, hp=0.12)
    assert nav._leash is not None
    assert nav.last.mode.endswith("critical")
    assert all(v[0] <= 0 for v in vecs)


def test_leash_drops_when_opened_stale_or_slipped():
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    nav.note_chest_opened()
    assert nav._leash is None
    assert any(ln.startswith("[leash] drop opened") for ln in logs)

    logs.clear()
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, 2000.0)
    nav._leash_seen = nav.game_s - (nav.tune.chest_leash_forget_s + 1.0)
    vecs = run(nav, clock, result([]), 4, hp=1.0)
    assert nav._leash is None
    assert any(ln.startswith("[leash] drop stale") for ln in logs)
    assert all(v[0] < 0 for v in vecs)

    logs.clear()
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, 2000.0)
    nav._leash_travel = nav.tune.chest_leash_slip + 1.0
    vecs = run(nav, clock, result([]), 3, hp=1.0)
    assert nav._leash is None
    assert any(ln.startswith("[leash] drop slip") for ln in logs)
    assert all(v[0] < 0 for v in vecs)


def test_chest_leash_radius_is_a_vs_tune_knob():
    tune = tuning_from_env("chest_leash_radius=0", log=lambda *_: None)
    assert tune.chest_leash_radius == 0.0
    clock = Clock()
    nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, 2000.0)
    vecs = run(nav, clock, result([]), 4, hp=1.0)
    assert all(v[0] < 0 for v in vecs)

    widened = tuning_from_env("chest_leash_forget_s=12.5,chest_leash_tie=0.2", log=lambda *_: None)
    assert widened.chest_leash_forget_s == 12.5
    assert widened.chest_leash_tie == 0.2


def _north_crowd():
    """Moderate danger to the north. Stays under a breakout, over chest_danger."""
    return blob(480, 156, n=12, spread=18, seed=1)


def _south_crowd():
    return blob(480, 356, n=12, spread=18, seed=1)


def test_orbit_holds_distance_under_moderate_danger():
    """Kiting around an anchor stays inside the radius band. Danger widens the ring."""
    plain = Navigator(log=lambda *_: None)
    assert plain._orbit_radius(2.0) == plain.tune.chest_orbit_radius + 2.0 * plain.tune.chest_orbit_swell

    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav.current = (0, 1)
    nav._since_change = 1.0
    crowd = result(_north_crowd())
    dists = []
    for _ in range(48):
        clock.t += 1 / 12
        nav.step(crowd, hp=1.0)
        dists.append(nav._leash_dist())
    assert nav.last.danger_here >= Tuning().chest_danger
    assert any(ln.startswith("[leash] orbit r=") for ln in logs)
    band = tune.chest_orbit_band
    assert all(abs(d - tune.chest_orbit_radius) <= band for d in dists[12:])


def test_orbit_steps_in_when_the_corridor_opens():
    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav.current = (0, 1)
    nav._since_change = 1.0
    run(nav, clock, result(_north_crowd()), 10, hp=1.0)
    dist = nav._leash_dist()
    run(nav, clock, result([]), 18, hp=1.0)
    assert nav._leash_dist() < dist - 100.0
    assert any(ln.startswith("[leash] orbit exit") for ln in logs)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)


def test_orbit_direction_flips_toward_the_open_side():
    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    south = run(nav, clock, result(_north_crowd()), 10, hp=1.0)
    sign = nav._orbit_sign
    assert sign != 0
    assert all(v[1] > 0 for v in south[-4:])

    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    north = run(nav, clock, result(_south_crowd()), 10, hp=1.0)
    assert nav._orbit_sign == -sign
    assert all(v[1] < 0 for v in north[-4:])
