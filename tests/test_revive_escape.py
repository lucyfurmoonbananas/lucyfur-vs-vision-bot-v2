"""Committed post-REVIVE escape (Navigator.request_revive_escape) and the
heuristic-menu gate used while it runs (main.gate_heuristic_menu)."""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot import main as m
from vs_vision_bot.pathfind import DIRS, Navigator, escape_costs, Tuning

from tests.test_movement import Clock, blob, result, ring, run


def _dot(v, h) -> float:
    n = math.hypot(*v) or 1.0
    return (v[0] * h[0] + v[1] * h[1]) / n


def _quiet_nav(clock):
    logs: list[str] = []
    return Navigator(clock=clock, log=logs.append), logs


def test_escape_costs_point_at_least_dense_side():
    # Heavy crowd right and below, light above, nothing left: left side is cheapest.
    pts = np.array(blob(650, 256, n=30, spread=50) + blob(480, 420, n=20, spread=40) + blob(480, 120, n=4), np.float32)
    costs = escape_costs(pts, 480, 256, Tuning())
    best = DIRS[int(np.argmin(costs))]
    assert best[0] == -1
    assert costs[DIRS.index((1, 0))] > costs[DIRS.index((-1, 0))]


def test_escape_costs_empty_field_is_corridor_only():
    t = Tuning()
    pts = np.zeros((0, 3), np.float32)
    assert np.all(np.isfinite(escape_costs(pts, 480, 256, t)))


def test_revive_escape_runs_full_window_toward_open_side_and_logs():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.game_s = 318.0
    crowd = result(blob(620, 256, n=30, spread=45) + blob(560, 380, n=10, spread=30))
    dur = nav.request_revive_escape()
    assert dur == nav.tune.revive_escape_s == 3.0
    assert nav.escaping
    out = run(nav, clock, crowd, 12, hp=0.5)  # ~1 s
    assert nav.breakout is not None and nav.escaping
    assert nav.last.mode.startswith("revive-escape")
    assert any("revive escape" in line for line in logs)
    # Every frame inside the ~45 deg cone of the heading, and away from the crowd.
    h = nav.breakout
    assert all(_dot(v, h) >= 0.7 for v in out)
    assert all(v[0] <= 0 for v in out)
    run(nav, clock, crowd, 26, hp=0.5)  # past 3 s
    assert not nav.escaping
    assert nav.breakout is None or not nav.last.mode.startswith("revive-escape")
    assert any("revive escape done" in line for line in logs)


def test_revive_escape_ignores_small_gap_reevaluation():
    """A gap that opens behind us after the early window must not cause a reversal."""
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    first = result(blob(680, 256, n=30, spread=40))  # crowd right -> go left
    run(nav, clock, first, 8, hp=0.5)  # > escape_repick_s (0.5 s)
    h0 = nav.breakout
    assert h0[0] < -0.5
    # A few stragglers ahead and the right now empty: the plain breakout re-check
    # (1.6x + 0.5) would flip to the right; the committed escape keeps going.
    flipped = result(blob(250, 256, n=5, spread=50))
    pathfind_costs = escape_costs(flipped.threats, 480, 256, nav.tune)
    right, left = pathfind_costs[DIRS.index((1, 0))], pathfind_costs[DIRS.index((-1, 0))]
    assert left > 1.6 * right + 0.5  # old logic would have switched
    out = run(nav, clock, flipped, 12, hp=0.5)
    assert nav.breakout[0] < -0.5
    assert all(v[0] <= 0 for v in out)
    # At most a 45 deg side-step around the stragglers, never a turn back.
    assert _dot(nav.breakout, h0) >= 0.7


def test_revive_escape_never_turns_back_even_when_ahead_fills():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.request_revive_escape()
    run(nav, clock, result(blob(680, 256, n=30, spread=40)), 8, hp=0.5)
    h0 = nav.breakout
    out = run(nav, clock, result(blob(330, 256, n=25, spread=40)), 14, hp=0.5)
    assert _dot(nav.breakout, h0) >= -1e-6
    assert all(_dot(v, h0) >= -1e-6 for v in out)


def test_revive_escape_early_repick_when_first_frame_was_stale():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    run(nav, clock, result([]), 1, hp=None)  # first frame: flash, nothing seen
    h0 = nav.breakout
    # Real crowd appears right where we were heading: re-aim inside the early window.
    hx, hy = h0
    crowd = result(blob(480 + 160 * hx, 256 + 160 * hy, n=40, spread=40))
    run(nav, clock, crowd, 3, hp=0.5)
    assert any("re-aim" in line for line in logs)
    assert _dot(_nearest(nav.breakout), h0) < 0.7


def _nearest(h):
    return max(DIRS, key=lambda d: _dot(d, h))


def test_revive_escape_late_switch_only_when_corridor_shut_and_not_reversal():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    run(nav, clock, result(blob(700, 256, n=30, spread=40)), 8, hp=0.5)  # head left
    h0 = nav.breakout
    assert h0[0] < -0.5
    # A solid wall straight ahead (left) and up-left; down-left stays open.
    wall = blob(380, 256, n=40, spread=25, seed=1) + blob(400, 170, n=40, spread=25, seed=2) + blob(700, 256, n=30, spread=40)
    run(nav, clock, result(wall), 8, hp=0.5)
    assert any("re-aim" in line for line in logs)
    assert _dot(nav.breakout, h0) >= -1e-6  # never a reversal


def test_hp_crash_during_escape_keeps_heading():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    crowd = result(blob(680, 256, n=30, spread=40))
    run(nav, clock, crowd, 4, hp=0.5)
    h0 = nav.breakout
    run(nav, clock, crowd, 3, hp=0.15)  # 0.5 -> 0.15: a crash
    assert nav._hp_crash_latched
    assert nav.breakout == h0
    assert not any("HP-crash flee" in line for line in logs)
    assert nav.escaping


def test_repeated_revive_frames_restart_not_stack():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    for _ in range(10):  # REVIVE screen visible for several frames
        nav.request_revive_escape()
    assert nav._escape_remaining == nav.tune.revive_escape_s
    assert nav._flee_remaining == nav.tune.revive_escape_s


def test_reset_clears_escape():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.request_revive_escape()
    run(nav, clock, result(blob(680, 256)), 3, hp=0.5)
    nav.reset()
    assert not nav.escaping and nav._escape_remaining == 0.0


def test_plain_request_flee_unchanged_by_escape_code():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.request_flee(2.0)
    assert not nav.escaping
    run(nav, clock, result(blob(600, 256, n=10, spread=30)), 4, hp=0.5)
    assert nav.last.mode.startswith("breakout")


def test_escape_in_ring_leaves_through_gap():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.game_s = 320.0
    pts = result(ring(480, 256, 150, 48, gap_deg=(160, 200)))  # gap on the left
    nav.request_revive_escape()
    out = run(nav, clock, pts, 24, hp=0.5)
    assert sum(1 for v in out if v[0] == -1) >= 20


def test_revive_escape_holds_while_the_ring_stays_closed_and_ends_when_it_opens():
    """r3 at 641s and 651s: the 3s heading ended with cover still 1.0, dead ~10s later."""
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.game_s = 641.0
    wall = result(ring(480, 256, 140, 64))
    nav.request_revive_escape()
    run(nav, clock, wall, 40, hp=0.5)  # ~3.3 s, past the old fixed window
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    assert nav.last.mode.startswith("revive-escape")
    assert any("revive escape holding" in line for line in logs)
    assert not any("escape done" in line for line in logs)
    h0 = nav.breakout
    # Open ground: the hold stops and the committed heading ends.
    run(nav, clock, result([]), 10, hp=0.5)
    assert not nav.escaping
    assert any("revive escape done" in line for line in logs)
    assert h0 is not None and _dot(nav._last_break_dir or h0, h0) >= -1e-6


def test_revive_escape_hold_stops_at_the_cap_inside_a_closed_ring():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    wall = result(ring(480, 256, 140, 64))
    nav.request_revive_escape()
    run(nav, clock, wall, 12 * 12 + 8, hp=0.5)  # 12.7 s inside a ring that never opens
    assert any("revive escape holding" in line for line in logs)
    assert not nav.escaping
    assert any("revive escape done" in line for line in logs)
    assert nav.tune.revive_escape_max_s > nav.tune.revive_escape_s


# -- main.gate_heuristic_menu


def test_gate_passes_confirmed_menu_and_non_escape():
    assert m.gate_heuristic_menu(True, "levelup", True, 0) == (True, 0)
    assert m.gate_heuristic_menu(True, None, False, 2) == (True, 0)
    assert m.gate_heuristic_menu(False, None, True, 2) == (False, 0)


def test_gate_needs_agreeing_frames_during_escape():
    menu, s = m.gate_heuristic_menu(True, None, True, 0)
    assert (menu, s) == (False, 1)
    menu, s = m.gate_heuristic_menu(True, None, True, s)
    assert (menu, s) == (False, 2)
    menu, s = m.gate_heuristic_menu(True, None, True, s)
    assert (menu, s) == (True, 3)
    assert m.ESCAPE_HEURISTIC_MENU_FRAMES == 3


# -- flower-window ring commit (same committed escape, reason "ring")


def _ring_nav(game_s, tune=None):
    clock = Clock()
    logs: list[str] = []
    tune = tune or Tuning(ring_commit_s=2.5)  # explicit so off-by-zero tests can override
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    nav.game_s = game_s
    return nav, clock, logs


def test_flower_window_closing_ring_commits_escape_through_gap():
    nav, clock, logs = _ring_nav(318.0)
    pts = result(ring(480, 256, 170, 40, gap_deg=(-1, 25)))  # gap right
    out = run(nav, clock, pts, 10, hp=0.24)
    assert nav.escaping and nav.last.mode.startswith("ring-escape")
    assert any(line.startswith("[nav] ring escape (1, 0)") for line in logs)
    assert sum(1 for v in out[-6:] if v[0] == 1) >= 4


def test_ring_escape_is_not_abandoned_when_cover_dips():
    """Plain breakouts stop after 1.1 s once 'closing' is false; the ring commit holds."""
    nav, clock, logs = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 2, hp=0.24)
    assert nav.escaping
    h0 = nav.breakout
    out = run(nav, clock, result(blob(330, 256, n=8, spread=30)), 18, hp=0.24)  # ~1.5 s, sparse
    assert nav.escaping and nav.breakout is not None
    assert all(_dot(v, h0) >= -1e-6 for v in out)
    run(nav, clock, result([]), 14, hp=0.24)
    assert not nav.escaping
    assert any("ring escape done" in line for line in logs)


def test_outside_flower_window_plain_breakout_unchanged():
    nav, clock, logs = _ring_nav(100.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 10, hp=0.24)
    assert nav.breakouts >= 1 and not nav.escaping
    assert nav.last.mode.startswith("breakout")
    assert not any("ring escape" in line for line in logs)


def test_ring_commit_defaults_on_and_explicit_zero_disables():
    assert Tuning().ring_commit_s == 2.5
    assert Tuning().flower_noreverse_s == 4.0
    nav, clock, logs = _ring_nav(318.0, tune=Tuning(ring_commit_s=0.0))
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 10, hp=0.24)
    assert nav.breakouts >= 1 and not nav.escaping
    assert nav.last.mode.startswith("breakout")


def test_revive_overrides_running_ring_escape():
    nav, clock, logs = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 4, hp=0.24)
    assert nav._escape_reason == "ring"
    nav.request_revive_escape()
    assert nav._escape_reason == "revive" and nav._escape_remaining == nav.tune.revive_escape_s
    run(nav, clock, result(blob(620, 256, n=30, spread=40)), 2, hp=0.5)
    assert nav.last.mode.startswith("revive-escape")


def test_plain_flee_cancels_committed_escape():
    nav, clock, _ = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 4, hp=0.24)
    assert nav.escaping
    nav.request_flee(2.0)
    assert not nav.escaping and nav._flee_remaining >= 2.0  # extends, never shortens


# -- ring escapes: no early (stale-frame) re-pick; that window is REVIVE-only


def _start_ring_escape_right():
    nav, clock, logs = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 1, hp=0.24)  # gap right
    assert nav.escaping and nav._escape_reason == "ring"
    assert _nearest(nav.breakout) == (1, 0)
    assert nav._escape_s <= nav.tune.escape_repick_s
    return nav, clock, logs


def test_ring_escape_never_reverses_inside_early_window():
    """run1.log L141/L156-157: a ring escape flipped 180 deg at +0.3-0.5 s."""
    nav, clock, logs = _start_ring_escape_right()
    h0 = nav.breakout
    # Crowd piles up ahead (right); left is empty. Old early re-pick turned to (-1, 0).
    crowd = result(blob(620, 256, n=40, spread=40) + blob(600, 150, n=20, spread=30) + blob(600, 360, n=20, spread=30))
    costs = escape_costs(crowd.threats, 480, 256, nav.tune)
    assert DIRS[int(np.argmin(costs))][0] == -1  # unconstrained pick would be a reversal
    out = run(nav, clock, crowd, 4, hp=0.24)  # still within escape_repick_s
    assert nav._escape_s <= nav.tune.escape_repick_s
    assert _dot(nav.breakout, h0) >= -1e-6
    assert all(_dot(v, h0) >= -1e-6 for v in out)
    assert not any("re-aim (1, 0)->(-1" in line for line in logs)


def test_ring_escape_early_window_does_not_reset_origin():
    nav, clock, _ = _start_ring_escape_right()
    origin = nav._escape_origin
    crowd = result(blob(620, 256, n=40, spread=40) + blob(600, 150, n=20, spread=30) + blob(600, 360, n=20, spread=30))
    run(nav, clock, crowd, 4, hp=0.24)
    assert nav._escape_origin == origin


def test_tied_north_and_south_do_not_default_north():
    """DIRS starts with (0, -1). An east heading equidistant from N and S used to pick north."""
    from vs_vision_bot.pathfind import _pick_min

    costs = np.array([1, 1, 5, 5, 4, 4, 4, 4], np.float32)
    assert DIRS[_pick_min(costs, (1.0, 0.0))] == (0, 1)
    # West is the other orthogonal heading: clockwise from west is north, not a fixed default.
    assert DIRS[_pick_min(costs, (-1.0, 0.0))] == (0, -1)
    # A heading that already faces south keeps south.
    assert DIRS[_pick_min(costs, (0.0, 1.0))] == (0, 1)


def _walls_leaving_north_and_south_open():
    pts = []
    for deg in (0, 180, 45, 135, -45, -135):
        for r in (160, 260, 360):
            a = math.radians(deg)
            x, y = 480 + r * math.cos(a), 256 + r * math.sin(a)
            pts.append((x, y, 1.0))
            px, py = -math.sin(a), math.cos(a)
            for s in (-22, 22):
                pts.append((x + px * s, y + py * s, 1.0))
    return pts


def test_side_heavy_revive_does_not_escape_north():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.heading = 0.0
    nav.request_revive_escape()
    run(nav, clock, result(_walls_leaving_north_and_south_open()), 2, hp=0.5)
    assert nav.escaping and nav.breakout is not None
    assert _nearest(nav.breakout) != (0, -1)


def test_revive_hold_repicks_the_open_corridor():
    """run6: the hold kept (0, -1) into the 15:00 bat wave after the opening moved."""
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.heading = -math.pi / 2
    nav.request_revive_escape()
    gap_north = result(ring(480, 256, 150, 48, gap_deg=(250, 290)))
    run(nav, clock, gap_north, 8, hp=0.5)
    assert _nearest(nav.breakout) == (0, -1)
    run(nav, clock, gap_north, 36, hp=0.5)
    assert nav.escaping and any("revive escape holding" in line for line in logs)
    assert _nearest(nav.breakout) == (0, -1)
    gap_south = result(ring(480, 256, 150, 48, gap_deg=(70, 110)))
    run(nav, clock, gap_south, 12, hp=0.5)
    assert nav.escaping
    assert _nearest(nav.breakout) == (0, 1)


def _arc(deg: float, n: int, radius: float, weight: float, spread: float):
    pts = []
    base = math.radians(deg)
    for i in range(n):
        ang = base + (i - (n - 1) / 2) * spread
        pts.append((480 + radius * math.cos(ang), 256 + radius * math.sin(ang), weight))
    return pts


def _pack(extra):
    return ring(480, 256, 150, 48) + ring(480, 256, 210, 40) + extra


def _arm_east_hold():
    """Revive through an east gap until the surrounded hold is on, still facing east."""
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    gap = result(ring(480, 256, 150, 48, gap_deg=(-20, 20)))
    run(nav, clock, gap, 8, hp=0.5)
    assert _nearest(nav.breakout) == (1, 0)
    run(nav, clock, gap, 30, hp=0.5)
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    assert _nearest(nav.breakout) == (1, 0)
    assert any("revive escape holding" in line for line in logs)
    return nav, clock, logs


def test_revive_hold_does_not_reverse_on_a_slight_bias():
    """Oct 26 1 AM: past 3s the hold re-aimed 135-180 deg every few tenths
    (run1 game~733s, run3 ~128s, run4 ~447s) and never left the pack.
    A closed ring with a small east/west bias is the step just taken, not a new opening.
    """
    nav, clock, logs = _arm_east_hold()
    n0 = len(logs)
    seen = []
    for i in range(48):  # ~4s; the cheap side swaps every two frames (~0.17s)
        cheap = 0 if (i // 2) % 2 == 0 else 180
        crowd = _pack(_arc(cheap + 180, n=8, radius=160, weight=1.2, spread=0.12))
        run(nav, clock, result(crowd), 1, hp=0.5)
        seen.append(_nearest(nav.breakout))
        assert nav.cover >= nav.tune.cover_trigger
    assert nav.escaping
    assert all(h == (1, 0) for h in seen)
    assert not any("re-aim" in line for line in logs[n0:])


def test_revive_hold_reverses_a_shut_corridor_but_not_every_frame():
    """A wall that closes the corridor still turns the hold. Swapping that wall
    to the other side every couple of frames must not flip the heading each time.
    """
    nav, clock, _ = _arm_east_hold()
    headings = [(1, 0)]
    covers = []
    for i in range(36):
        wall = 0 if (i // 2) % 2 == 0 else 180
        extra = []
        for radius in (120.0, 160.0, 200.0):
            extra += _arc(wall, n=9, radius=radius, weight=1.5, spread=0.18)
        run(nav, clock, result(_pack(extra)), 1, hp=0.5)
        headings.append(_nearest(nav.breakout))
        covers.append(nav.cover)
    assert min(covers) >= nav.tune.cover_trigger
    changes = [i for i in range(1, len(headings)) if headings[i] != headings[i - 1]]
    assert changes, "a shut corridor is still a re-aim"
    assert changes[0] <= 2
    min_gap = nav.tune.revive_hold_reaim_s * 12
    assert all(b - a >= min_gap - 1 for a, b in zip(changes, changes[1:]))


def test_revive_hold_sidesteps_a_wall_ahead():
    """An impassable wall on the heading turns into the open side, not back through the ring."""
    nav, clock, logs = _arm_east_hold()
    pts = ring(480, 256, 150, 48, gap_deg=(250, 290))
    pts += blob(640, 256, n=40, spread=30, seed=1)
    pts += blob(600, 180, n=20, spread=25, seed=2)
    pts += blob(600, 340, n=20, spread=25, seed=3)
    n0 = len(logs)
    run(nav, clock, result(pts), 6, hp=0.5)
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    assert _nearest(nav.breakout) == (0, -1)
    assert not any("->(-1," in line for line in logs[n0:])


def _arm_south_hold():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.heading = math.pi / 2
    nav.request_revive_escape()
    gap = result(ring(480, 256, 150, 48, gap_deg=(70, 110)))
    run(nav, clock, gap, 8, hp=0.5)
    assert _nearest(nav.breakout) == (0, 1)
    run(nav, clock, gap, 30, hp=0.5)
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    assert _nearest(nav.breakout) == (0, 1)
    assert _nearest(nav._escape_origin) == (0, 1)
    assert any("revive escape holding" in line for line in logs)
    return nav, clock, logs


def _south_wall():
    """Closed ring with the south corridor shut. West is a 90 deg side-step."""
    wall = []
    for radius in (120.0, 160.0, 200.0, 240.0):
        wall += _arc(90, n=11, radius=radius, weight=1.6, spread=0.16)
    return ring(480, 256, 150, 48) + ring(480, 256, 210, 40) + wall


def _north_nudge():
    """North is a bit cheaper than west, and not a shut wall behind the anchor.

    From a west heading this is a forward 90 deg step. From the original south
    heading it is a 180 deg reversal, and west itself is still under 2x north.
    """
    pts = ring(480, 256, 150, 48, gap_deg=(250, 290)) + ring(480, 256, 220, 36)
    for deg, weight in ((0, 1.5), (45, 1.3), (-45, 1.3), (90, 1.2), (180, 1.6), (135, 1.1), (-135, 1.05)):
        for radius in (140.0, 190.0):
            pts += _arc(deg, n=5, radius=radius, weight=weight, spread=0.12)
    return pts


def test_revive_hold_side_steps_do_not_add_up_to_a_reversal():
    """run2.log 271-276: the hold was on (0, 1). A 135 deg re-aim to (-1, -1)
    overwrote the origin, and the next re-aim to (0, -1) was only a forward
    step from that new origin — 180 deg back through the pack.

    A 90 deg side-step off a shut corridor is still allowed. The following
    90 deg step must not finish the reversal while the original hemisphere
    still has a corridor.
    """
    nav, clock, _ = _arm_south_hold()
    origin = nav._escape_origin
    run(nav, clock, result(_south_wall()), 12, hp=0.5)
    assert nav.cover >= nav.tune.cover_trigger
    assert _dot(nav.breakout, origin) >= -1e-6
    assert _nearest(nav.breakout) == (-1, 0)
    run(nav, clock, result(_north_nudge()), 18, hp=0.5)
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    assert _dot(nav.breakout, origin) >= -1e-6
    assert _dot(nav._escape_origin, origin) >= -1e-6
    assert _nearest(nav.breakout) != (0, -1)


def test_revive_escape_early_window_still_allows_reversal():
    """REVIVE behaviour unchanged: stale first frame may be fully re-picked."""
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    run(nav, clock, result(blob(340, 256, n=30, spread=40)), 1, hp=0.5)  # crowd left -> go right
    h0 = nav.breakout
    assert h0[0] > 0.5
    crowd = result(blob(620, 256, n=40, spread=40) + blob(600, 150, n=20, spread=30) + blob(600, 360, n=20, spread=30))
    run(nav, clock, crowd, 3, hp=0.5)
    assert nav.breakout[0] < -0.5
    assert nav._escape_origin == nav.breakout
