"""A seen chest stays an anchor. The camera follows the player, so the anchor
is kept in world space and dead-reckoned from the bot's own steps.

Oct 18: 205 chest seeks and 0 reached. Between sightings the bot drifted off,
usually left. Past the leash radius it has to walk back when danger allows.
A committed escape still takes a clearly open gap, but a tied corridor prefers
the arrow side. Critical HP is not pulled off its heading.
"""

from __future__ import annotations

import math

from vs_vision_bot.pathfind import DIRS, Navigator, Tuning, tuning_from_env
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
    """Without the dive, kiting stays on the wide ring. Danger widens that ring.

    Full HP commits to the tight ring instead; see test_chest_commit.
    """
    plain = Navigator(log=lambda *_: None)
    assert plain._orbit_radius(2.0) == plain.tune.chest_orbit_radius + 2.0 * plain.tune.chest_orbit_swell

    tune = tuning_from_env("chest_orbit_swell=0,dive_radius_scale=1", log=lambda *_: None)
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
    """Dive off: moderate danger holds the wide ring, then a clear field steps in."""
    tune = tuning_from_env("chest_orbit_swell=0,dive_radius_scale=1", log=lambda *_: None)
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
    """The wide ring (dive off) still takes the open tangent."""
    tune = tuning_from_env("chest_orbit_swell=0,dive_radius_scale=1", log=lambda *_: None)
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


def _free_open(nav: Navigator, clock: Clock, logs: list[str]) -> None:
    """A chest dropped while the free quota remains is stepped onto."""
    start = len(logs)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    fresh = logs[start:]
    assert any(ln.startswith("[chest] open-allowed (always)") for ln in fresh)
    assert any(ln.startswith("[leash] step-in ") for ln in fresh)
    assert not any(ln.startswith("[chest] hold ") for ln in fresh)
    nav.on_chest_opened()
    assert nav._leash is None


def test_first_three_chests_open_and_the_fourth_does_too():
    """The default opens every chest. The old hold-after-3 is chest_open_always=0."""
    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    assert tune.chest_free_opens == 3
    assert tune.chest_open_always == 1
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(4):
        _free_open(nav, clock, logs)
    assert nav._chests_opened == 4
    assert not any(ln.startswith("[chest] hold ") for ln in logs)


def test_evolution_ready_holds_until_ten_minutes_then_steps_in():
    """With chest_open_always=0, evo-ready still waits for 10:00, then opens one."""
    tune = tuning_from_env("chest_open_always=0,chest_orbit_swell=0", log=lambda *_: None)
    assert tune.chest_evo_after_s == 600.0
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    nav.set_build_state(True, False)
    nav.set_build_state(True, False)
    assert [ln for ln in logs if ln.startswith("[build]")] == [
        "[build] evolution_ready=True all_maxed=False"
    ]

    nav.game_s = 540.0
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav.current = (0, 1)
    nav._since_change = 1.0
    held = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    assert any(
        ln.startswith("[chest] hold (opened=3 evo_ready=True t=540)") for ln in logs
    )
    assert not any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() > held - tune.chest_orbit_band

    nav.game_s = 600.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 18, hp=1.0)
    assert any(ln.startswith("[chest] open-allowed (evolution)") for ln in logs)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() < dist - 100.0

    # The same gate is re-checked for the next chest. Still ready past 10:00, so it opens.
    nav.on_chest_opened()
    assert nav._chests_opened == 4
    nav.game_s = 610.0
    start = len(logs)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    assert any(ln.startswith("[chest] open-allowed (evolution)") for ln in logs[start:])
    assert any(ln.startswith("[leash] step-in ") for ln in logs[start:])
    assert not any(ln.startswith("[chest] hold ") for ln in logs[start:])

    # The tracker clears the flag once that evolution is spent. The next chest waits again.
    nav.on_chest_opened()
    nav.set_build_state(False, False)
    nav.game_s = 620.0
    start = len(logs)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    fresh = logs[start:]
    assert any(ln.startswith("[chest] hold (opened=5 evo_ready=False ") for ln in fresh)
    assert not any(ln.startswith("[leash] step-in ") for ln in fresh)
    assert nav._leash_dist() > dist - tune.chest_orbit_band


def test_all_maxed_opens_a_chest_at_any_time():
    """The hold override still opens immediately once every weapon is maxed."""
    tune = tuning_from_env("chest_open_always=0,chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    nav.game_s = 90.0
    nav.set_build_state(False, True)
    assert "[build] evolution_ready=False all_maxed=True" in logs
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 18, hp=1.0)
    assert any(ln.startswith("[chest] open-allowed (all-maxed)") for ln in logs)
    assert not any(ln.startswith("[chest] hold ") for ln in logs)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() < dist - 100.0


def test_held_chest_keepout_blocks_a_step_in_but_not_an_escape():
    """The disc sits in the orbit costs. A committed escape may still walk across it."""
    tune = tuning_from_env("chest_open_always=0,chest_orbit_swell=0", log=lambda *_: None)
    assert 0 < tune.chest_hold_keepout < tune.chest_orbit_radius
    clock = Clock()
    nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_hold_keepout * 0.5)
    nav._since_change = 1.0
    costs = nav._keepout_costs()
    assert costs[DIRS.index((1, 0))] > costs[DIRS.index((-1, 0))]
    start = nav._leash_dist()
    vecs = run(nav, clock, result([]), 8, hp=1.0)
    assert all(v[0] <= 0 for v in vecs)
    assert nav._leash_dist() > start + 20.0

    _park_west_of(nav, tune.chest_hold_keepout * 0.5)
    nav._since_change = 1.0
    nav.heading = 0.0
    nav.request_revive_escape()
    gap = result(ring(480, 256, 170, 48, gap_deg=(340, 360)))
    vecs = run(nav, clock, gap, 6, hp=1.0)
    assert nav.escaping and nav.breakout is not None and nav.breakout[0] > 0
    assert all(v[0] > 0 for v in vecs)


def test_chest_free_opens_is_a_vs_tune_knob():
    """chest_open_always defaults on. With it off, free_opens still gates the hold."""
    assert Tuning().chest_free_opens == 3
    assert Tuning().chest_open_always == 1
    wide = tuning_from_env("chest_free_opens=9,chest_open_always=0", log=lambda *_: None)
    assert wide.chest_free_opens == 9
    assert wide.chest_open_always == 0
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=wide, clock=clock, log=logs.append)
    for _ in range(4):
        nav.on_chest_opened()
    _anchor(nav, clock, chest=CHEST_RIGHT)
    assert any(ln.startswith("[chest] open-allowed (free)") for ln in logs)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)
    assert not any(ln.startswith("[chest] hold ") for ln in logs)

    held = tuning_from_env(
        "chest_free_opens=0,chest_open_always=0,chest_orbit_swell=0", log=lambda *_: None,
    )
    clock = Clock()
    logs = []
    nav = Navigator(tune=held, clock=clock, log=logs.append)
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, held.chest_orbit_radius)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    assert any(ln.startswith("[chest] hold (opened=0 ") for ln in logs)
    assert not any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() > dist - held.chest_orbit_band


def test_pilot_evolution_ready_after_ten_minutes_opens_a_held_chest():
    """sync_build_state pushes the flags and does not count another chest.

    The default opens at 9:00. It does not wait for the evolution or for 10:00.
    """
    from vs_vision_bot.main import sync_build_state
    from vs_vision_bot.menus import MenuPilot

    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None)
    pilot.levels = {"Garlic": 8, "Pummarola": 1}
    assert pilot.evolution_ready and not pilot.all_maxed

    nav.game_s = 540.0
    sync_build_state(nav, pilot)
    assert nav._evolution_ready and not nav._all_maxed
    assert nav._chests_opened == tune.chest_free_opens
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    assert any(ln.startswith("[chest] open-allowed (always)") for ln in logs)
    assert not any(ln.startswith("[chest] hold ") for ln in logs)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() < dist - 60.0

    # A second chest-screen confirm updates flags only. It must not count again.
    pilot.chest_presses = 1
    sync_build_state(nav, pilot)
    assert nav._chests_opened == tune.chest_free_opens


def test_done_reward_does_not_spend_a_second_free_open():
    """OPEN and the DONE page are one chest. Run 1 counted both, so five
    'open-allowed (free)' lines never reached a hold.
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(
        tune=tuning_from_env("chest_orbit_swell=0", log=lambda *_: None),
        clock=clock,
        log=logs.append,
    )
    nav.on_chest_opened()
    nav.on_chest_opened(reward=True)
    assert nav._chests_opened == 1
    assert "[chest] opened 1" in logs
    assert any("not counted" in ln for ln in logs)
    clock.t += 60.0
    nav.on_chest_opened(reward=True)
    assert nav._chests_opened == 2


def test_set_build_state_logs_the_first_report_while_nothing_is_ready():
    """Live runs never printed [build]: the flags stayed false, and the log
    waited for a change. The first push has to be visible.
    """
    logs: list[str] = []
    nav = Navigator(clock=Clock(), log=logs.append)
    nav.set_build_state(False, False)
    assert logs == ["[build] evolution_ready=False all_maxed=False"]
    nav.set_build_state(False, False)
    assert len(logs) == 1
    nav.set_build_state(True, False)
    assert logs[-1] == "[build] evolution_ready=True all_maxed=False"


def test_plain_breakout_keeps_a_held_orbit_while_the_chest_is_visible():
    """run1 game~590s: orbit exit while chest=(210,465) was still detected.

    This is the opt-in hold (chest_open_always=0). Cover 0.62 starts a plain
    breakout. Danger stays under chest_clear_danger, so the orbit keeps the
    chest. A wall (cover >= 0.75) still leaves a held chest. An openable chest
    at dive HP stays with the chest instead; see test_chest_commit.
    """
    # low_hp stays under 0.24 so the danger field matches the old full-HP hold.
    # The kite line is pack_dive_hp, not the defensive radius.
    tune = tuning_from_env(
        "chest_open_always=0,chest_orbit_swell=0,critical_hp=0,low_hp=0.10",
        log=lambda *_: None,
    )
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    nav.game_s = 500.0
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav.current = (0, 1)
    nav._since_change = 1.0
    # Gap wide enough that cover stays under the wall commit.
    crowd = result(ring(480, 256, 160, 36, gap_deg=(0, 140)))
    crowd.chest = (210.0, 465.0)
    before = len(logs)
    # The anchor frame was full HP. Drop that sample so 0.24 is not an HP-crash.
    nav._hp_hist.clear()
    run(nav, clock, crowd, 8, hp=0.24)
    fresh = logs[before:]
    assert nav.breakouts >= 1 and not nav.escaping
    assert nav.cover < tune.ring_commit_cover
    assert nav._orbit_on
    assert nav.last.mode.startswith("leash-orbit")
    assert not any(ln.startswith("[leash] orbit exit") for ln in fresh)

    wall = Navigator(tune=tune, clock=Clock(), log=lambda *_: None)
    for _ in range(tune.chest_free_opens):
        wall.on_chest_opened()
    wall.game_s = 500.0
    _anchor(wall, wall.clock, chest=CHEST_RIGHT)
    _park_west_of(wall, tune.chest_orbit_radius)
    wall._since_change = 1.0
    closing = result(ring(480, 256, 160, 36, gap_deg=(0, 90)))
    closing.chest = (210.0, 465.0)
    wall._hp_hist.clear()
    run(wall, wall.clock, closing, 6, hp=0.24)
    assert wall.escaping and wall.last.mode.startswith("ring-escape")
    assert not wall._orbit_on


def test_critical_hp_and_a_wall_commit_override_a_held_chest():
    """Critical HP leaves the chest. A full-HP wall does not: the steps stay with it.

    The ring-escape heading can still be the gap. chest_open_always=0 is what
    lets that gap take the walked chord; see the plain-breakout hold test.
    """
    tune = tuning_from_env("chest_orbit_swell=0", log=lambda *_: None)
    clock = Clock()
    nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, tune.chest_orbit_radius)
    nav._since_change = 1.0
    vecs = run(nav, clock, result([]), 4, hp=0.12)
    assert nav._critical_on and nav.last.mode.endswith("critical")
    assert not nav.last.mode.startswith("leash-orbit")
    assert all(v[0] <= 0 for v in vecs)

    wall = Navigator(tune=tune, clock=Clock(), log=lambda *_: None)
    for _ in range(tune.chest_free_opens):
        wall.on_chest_opened()
    wall.game_s = 660.0
    _anchor(wall, wall.clock, chest=(240.0, 256.0))
    warm = result(blob(700, 256, n=8, spread=30))
    warm.chest = (240.0, 256.0)
    wall._hp_hist.clear()
    run(wall, wall.clock, warm, 8, hp=0.24)
    gap = result(ring(480, 256, 150, 48, gap_deg=(-1, 25)))
    gap.chest = (240.0, 256.0)
    vecs = run(wall, wall.clock, gap, 4, hp=0.24)
    assert wall.escaping and wall.breakout is not None and wall.breakout[0] > 0
    # At the kite line a held chest yields: the steps take the gap.
    assert wall.last.mode.startswith("ring-escape")
    assert all(v[0] > 0 for v in vecs)


def test_high_hp_keeps_a_visible_chest_past_the_old_clear():
    """Danger between chest_clear (3) and the commit fade (4.5) at full HP.

    The dive sits in that band. The old cutoff dropped the leash, and the
    heading matched the no-chest escape. Below the dive band, and with the
    scale knob at 1, the cutoff stays 3.
    """
    pts = blob(520, 180, n=16, spread=40, seed=1)
    chest = (700.0, 300.0)

    def headings(hp, scale=None):
        tune = Tuning() if scale is None else tuning_from_env(
            f"dive_radius_scale={scale}", log=lambda *_: None,
        )
        clock = Clock()
        nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
        nav.game_s = 200.0
        seen = result([])
        seen.chest = chest
        clock.t += 1 / 12
        nav.step(seen, hp=hp)
        crowd = result(pts)
        crowd.chest = chest
        return nav, run(nav, clock, crowd, 6, hp=hp)

    nav, vecs = headings(1.0)
    assert Tuning().chest_clear_danger < nav.last.danger_here < Tuning().chest_commit_fade
    # The lane through this blob is shut, so the commit kite-orbits on the tight
    # ring instead of stepping straight in. The heading still has a component
    # toward the chest. The no-chest field below flees the other way.
    assert nav._chase in ("step", "orbit")
    assert nav._orbit_radius(nav.last.danger_here) <= Tuning().chest_commit_orbit
    assert all(v[0] > 0 for v in vecs)

    clock = Clock()
    away = Navigator(clock=clock, log=lambda *_: None)
    away.game_s = 200.0
    # No chest: the pack (north of the player) is the farm, not a flee west.
    assert all(v[1] <= 0 for v in run(away, clock, result(pts), 6, hp=1.0))

    low, low_vecs = headings(0.20)
    assert low._chase != "step"
    assert all(v[0] < 0 for v in low_vecs)

    old, old_vecs = headings(1.0, scale="1")
    assert old.dive == 0.0
    # pack_dive_hp does not follow dive_radius_scale. The chest is still walked.
    assert all(v[0] > 0 for v in old_vecs)
