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
