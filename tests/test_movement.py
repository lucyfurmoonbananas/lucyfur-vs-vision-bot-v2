"""Navigator / potential-field movement on synthetic monster layouts."""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot import pathfind
from vs_vision_bot.pathfind import DIRS, Decision, Navigator, Tuning, phase_for, score_directions
from vs_vision_bot.vision import VisionResult

P = (480, 256)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def ring(cx: float, cy: float, r: float, n: int, gap_deg: tuple[float, float] | None = None) -> list[tuple[float, float, float]]:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        deg = math.degrees(a) % 360
        if gap_deg and gap_deg[0] <= deg <= gap_deg[1]:
            continue
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), 1.0))
    return pts


def blob(cx: float, cy: float, n: int = 12, spread: float = 20.0, seed: int = 0) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    return [(cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread), 1.0) for _ in range(n)]


def result(points, gems=None, menu=False) -> VisionResult:
    arr = np.array(points, dtype=np.float32).reshape(-1, 3)
    return VisionResult(player=P, monsters=[], threats=arr, gems=list(gems or []), menu=menu)


def run(nav: Navigator, clock: Clock, res: VisionResult, frames: int, hp=None) -> list[tuple[int, int]]:
    out = []
    for _ in range(frames):
        clock.t += 1 / 12
        out.append(nav.step(res, hp=hp).vector)
    return out


def dot(v, ux, uy) -> float:
    n = math.hypot(*v) or 1.0
    return (v[0] * ux + v[1] * uy) / n


def test_one_sided_close_crowd_moves_away():
    dec = score_directions(np.array(blob(560, 256), np.float32), P, None, 0.0)
    assert dec.vector[0] == -1


def test_one_sided_crowd_with_kiting_goes_around_not_into_it():
    clock = Clock()
    nav = Navigator(clock=clock)
    vecs = run(nav, clock, result(blob(720, 256, n=20, spread=40)), 24)
    # Never walks toward the crowd; mostly tangential (vertical) or away.
    assert all(v[0] <= 0 or v[1] != 0 for v in vecs)
    assert all(v != (1, 0) for v in vecs)
    assert sum(1 for v in vecs if v[1] != 0) >= len(vecs) // 2


def test_surrounded_escapes_through_the_gap():
    # Ring of enemies 70 px out with a gap toward the lower-left (~135 deg).
    pts = ring(P[0], P[1], 70, 36, gap_deg=(110, 160))
    dec = score_directions(np.array(pts, np.float32), P, None, 0.0)
    assert dec.vector == (-1, 1)
    clock = Clock()
    nav = Navigator(clock=clock)
    vecs = run(nav, clock, result(pts), 6)
    assert vecs[-1] in ((-1, 1), (-1, 0), (0, 1))


def test_surrounded_never_stands_still():
    clock = Clock()
    nav = Navigator(clock=clock)
    vecs = run(nav, clock, result(ring(P[0], P[1], 30, 24) + ring(P[0], P[1], 60, 30)), 12)
    assert all(v != (0, 0) for v in vecs)


def test_empty_screen_walks_wide_circles():
    clock = Clock()
    nav = Navigator(clock=clock)
    vecs = run(nav, clock, result([]), 12 * 10)
    assert all(v != (0, 0) for v in vecs)
    # A full loop in ~9 s visits most headings...
    assert len(set(vecs)) >= 6
    # ...and changes direction smoothly, not every frame.
    changes = sum(1 for a, b in zip(vecs, vecs[1:]) if a != b)
    assert changes <= 20


def test_hysteresis_no_jitter_between_equal_options():
    # Two equal crowds up and down: left and right are equally good.
    pts = blob(480, 150, 10, seed=1) + blob(480, 362, 10, seed=2)
    clock = Clock()
    nav = Navigator(clock=clock)
    vecs = run(nav, clock, result(pts), 36)
    changes = sum(1 for a, b in zip(vecs, vecs[1:]) if a != b)
    assert changes <= 3


def test_prefers_least_dense_sector():
    # Dense on the right, a single bat on the left: go left-ish, not right.
    pts = blob(600, 256, n=30, spread=60) + [(380, 256, 1.0)]
    dec = score_directions(np.array(pts, np.float32), P, None, 0.0)
    assert dec.vector[0] <= 0 and dec.vector != (0, 0)
    # The lone bat blocks straight left; a diagonal around it is fine.
    assert dec.vector != (1, 0)


def test_gems_pull_only_when_safe():
    gem = [(400, 256)]
    safe = score_directions(np.zeros((0, 3), np.float32), P, None, 0.0, gems=gem)
    assert safe.vector == (-1, 0)
    # Crowd sitting right on the gem: don't go for it.
    pts = np.array(blob(420, 256, n=15, spread=15), np.float32)
    risky = score_directions(pts, P, None, 0.0, gems=gem)
    assert risky.vector[0] >= 0


def test_low_hp_is_more_defensive():
    # A pack at 120 px and a strong pull (kite) toward it.
    pts = np.array(blob(600, 256, n=12, spread=15), np.float32)
    full = score_directions(pts, P, (1.0, 0.0), 1.0, hp=1.0)
    low = score_directions(pts, P, (1.0, 0.0), 1.0, hp=0.3)
    right = (1, 0)
    assert low.scores[right] < full.scores[right]
    assert low.vector[0] <= 0


def test_menu_frames_hold_still():
    clock = Clock()
    nav = Navigator(clock=clock)
    assert run(nav, clock, result(blob(560, 256), menu=True), 3) == [(0, 0)] * 3


def test_phases_follow_lucyfurs_playbook():
    assert phase_for(60).name == "wide-circles"
    assert phase_for(7 * 60).name == "tight-weave"
    assert phase_for(7 * 60).orbit_period_s < phase_for(60).orbit_period_s
    assert phase_for(11 * 60).name == "wide-loops"
    assert phase_for(17 * 60).name == "slow-circles"
    assert phase_for(25 * 60).idle_ok


def test_late_game_stands_still_when_safe_but_dodges_when_close():
    clock = Clock()
    nav = Navigator(clock=clock)
    nav.game_s = 25 * 60
    assert run(nav, clock, result([]), 4)[-1] == (0, 0)
    assert run(nav, clock, result(blob(510, 256, n=10, spread=10)), 4)[-1] != (0, 0)


def test_orbit_flips_when_the_loop_is_blocked():
    clock = Clock()
    nav = Navigator(clock=clock)
    start = nav.orbit
    # Crowd to the right: the clockwise loop heads down. If the field keeps
    # overruling that (walking up), the orbit flips instead of fighting it.
    pts = blob(700, 256, n=15, spread=30)
    real = pathfind.score_directions
    pathfind.score_directions = lambda *a, **k: Decision((0, -1), {d: 0.0 for d in DIRS}, 0.0, "stub")
    try:
        run(nav, clock, result(pts), 12)
    finally:
        pathfind.score_directions = real
    assert nav.orbit == -start


def test_simulated_swarm_poe_survives_two_minutes():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "sim_movement.py"
    spec = importlib.util.spec_from_file_location("sim_movement", path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    # Chasing swarm with 1:00 / 2:00 bat bursts; Poe has 70 max HP.
    new = sim.simulate("new", 125, seed=0)
    assert new["damage"] < 35
    still = sim.simulate("still", 60, seed=0)
    assert still["damage"] > 10 * max(new["damage"], 1.0)


def test_all_decisions_are_valid_wasd_vectors():
    rng = np.random.default_rng(7)
    for _ in range(30):
        pts = np.column_stack([rng.uniform(50, 910, 40), rng.uniform(20, 530, 40), rng.uniform(0.1, 1, 40)]).astype(np.float32)
        dec = score_directions(pts, P, (rng.normal(), rng.normal()), 1.0)
        assert dec.vector in DIRS or dec.vector == (0, 0)


def test_tuning_is_overridable():
    t = Tuning(hysteresis=0.0, min_dwell_s=0.0)
    nav = Navigator(tune=t, clock=Clock())
    assert nav.tune.hysteresis == 0.0


# ---------------------------------------------------------------- breakout / rollout


def test_encirclement_measures_ring_coverage():
    full = np.array(ring(480, 256, 150, 32), np.float32)
    half = np.array([p for p in ring(480, 256, 150, 32) if p[0] > 480], np.float32)
    cf, nf = pathfind.encirclement(full, 480, 256)
    ch, nh = pathfind.encirclement(half, 480, 256)
    assert cf > 0.9
    assert 0.3 < ch < 0.65
    assert nf > nh
    assert pathfind.encirclement(np.zeros((0, 3), np.float32), 480, 256) == (0.0, 0.0)


def test_corridor_costs_find_the_thin_side():
    # Ring with a gap to the upper-left (225 deg on screen = dx<0, dy<0).
    pts = np.array(ring(480, 256, 140, 36, gap_deg=(205, 245)), np.float32)
    costs = pathfind.corridor_costs(pts, 480, 256)
    assert DIRS[int(np.argmin(costs))] == (-1, -1)


def test_closing_ring_triggers_breakout_through_the_gap():
    clock = Clock()
    nav = Navigator(clock=clock)
    nav.game_s = 300.0
    pts = ring(480, 256, 170, 40, gap_deg=(-1, 25))  # gap to the right
    out = run(nav, clock, result(pts), 10, hp=1.0)
    assert nav.breakouts >= 1
    assert nav.last.mode.startswith("breakout")
    assert sum(1 for v in out[-6:] if v[0] == 1) >= 4


def test_breakout_commits_then_resumes_loops():
    clock = Clock()
    nav = Navigator(clock=clock)
    pts = ring(480, 256, 170, 40, gap_deg=(170, 200))  # gap to the left
    run(nav, clock, result(pts), 6, hp=1.0)
    assert nav.breakout is not None
    heading = nav.breakout
    # Ring gone: keeps the breakout for its minimum time, then drops it.
    run(nav, clock, result([]), 3)
    assert nav.breakout == heading
    run(nav, clock, result([]), 30)
    assert nav.breakout is None
    assert abs(math.atan2(math.sin(nav.heading - math.pi), math.cos(nav.heading - math.pi))) < 1.6


def test_sparse_or_one_sided_crowd_does_not_trigger_breakout():
    clock = Clock()
    nav = Navigator(clock=clock)
    run(nav, clock, result(blob(600, 256, n=14)), 12, hp=1.0)
    assert nav.breakouts == 0


def test_rollout_avoids_walking_into_an_incoming_wall():
    # A wall of enemies ahead-right at medium range: the static field barely
    # notices, the chase rollout does.
    wall = [(700, y, 1.0) for y in range(120, 400, 14)]
    costs = pathfind.rollout_costs(np.array(wall, np.float32), 480, 256, Tuning())
    right = costs[DIRS.index((1, 0))]
    left = costs[DIRS.index((-1, 0))]
    assert right > 2 * left


def test_menu_frames_do_not_advance_game_clock():
    clock = Clock()
    nav = Navigator(clock=clock)
    run(nav, clock, result([], menu=True), 24)
    assert nav.game_s == 0.0
    run(nav, clock, result([]), 12)
    assert 0.8 < nav.game_s < 1.1


def test_denser_horde_sim_survives():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "sim_movement.py"
    spec = importlib.util.spec_from_file_location("sim_movement", path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    r = sim.simulate_horde("new", 60, seed=1)
    assert r["damage"] < 70  # alive after the first minute of ~4:30 waves
    still = sim.simulate_horde("still", 30, seed=1)
    assert still["damage"] > 5 * max(r["damage"], 1.0)


def test_gems_still_pull_at_moderate_danger():
    # A few enemies at mid range (here ~0.5-1.0): the old code ignored gems then.
    pts = np.array(blob(480, 420, n=6, spread=20) + blob(700, 256, n=4, spread=15), np.float32)
    here = score_directions(pts, P, None, 0.0).danger_here
    assert 0.3 < here < 1.6
    dec = score_directions(pts, P, None, 0.0, gems=[(330, 150), (340, 160), (320, 140)])
    assert dec.vector[0] == -1 and dec.vector[1] == -1
    # Not when HP is low.
    low = score_directions(pts, P, None, 0.0, gems=[(330, 150), (340, 160), (320, 140)], hp=0.3)
    assert low.scores[(-1, -1)] < dec.scores[(-1, -1)]


def test_request_flee_forces_breakout_without_encirclement():
    """After REVIVE the navigator must flee even when the ring is not closing."""
    clock = Clock()
    nav = Navigator(clock=clock)
    # One-sided sparse crowd: normal logic must NOT break out.
    sparse = result(blob(600, 256, n=10, spread=30))
    run(nav, clock, sparse, 8, hp=1.0)
    assert nav.breakouts == 0
    nav.request_flee(2.0)
    # Attractive gem toward the crowd: flee must ignore gem pull.
    gems = [(700.0, 256.0)]
    out = run(nav, clock, result(blob(600, 256, n=10, spread=30), gems=gems), 6, hp=0.5)
    assert nav.breakouts >= 1
    assert nav.breakout is not None
    assert nav.last.mode.startswith("breakout")
    # Corridor away from the right-side blob: mostly leftward.
    assert sum(1 for v in out if v[0] == -1) >= 3


def test_request_flee_holds_for_configured_duration():
    clock = Clock()
    nav = Navigator(clock=clock)
    nav.request_flee(1.8)
    # Empty field: still commits to a flee corridor.
    run(nav, clock, result([]), 2)
    assert nav.breakout is not None
    assert nav._flee_remaining > 0
    run(nav, clock, result([]), 30)  # ~2.5 s at 12 fps
    assert nav.breakout is None
    assert nav._flee_remaining == 0.0


def test_request_flee_drops_pre_death_breakout():
    clock = Clock()
    nav = Navigator(clock=clock)
    nav.game_s = 300.0
    ring_pts = result(ring(480, 256, 170, 40, gap_deg=(-1, 25)))  # gap right
    while nav.breakout is None or nav._breakout_s < 2.3:
        run(nav, clock, ring_pts, 1, hp=1.0)
    clock.t += 3.0  # death + REVIVE screen
    nav.request_flee(2.0)
    out = run(nav, clock, result(blob(600, 256, n=20, spread=40)), 20, hp=1.0)
    assert nav.breakout is not None  # full ~2 s flee, not the stale leftover
    assert all(v[0] == -1 for v in out)
