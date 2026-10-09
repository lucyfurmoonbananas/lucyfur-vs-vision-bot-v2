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
    out = run(nav, clock, pts, 10, hp=1.0)
    assert nav.escaping and nav.last.mode.startswith("ring-escape")
    assert any(line.startswith("[nav] ring escape (1, 0)") for line in logs)
    assert sum(1 for v in out[-6:] if v[0] == 1) >= 4


def test_ring_escape_is_not_abandoned_when_cover_dips():
    """Plain breakouts stop after 1.1 s once 'closing' is false; the ring commit holds."""
    nav, clock, logs = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 2, hp=1.0)
    assert nav.escaping
    h0 = nav.breakout
    out = run(nav, clock, result(blob(330, 256, n=8, spread=30)), 18, hp=1.0)  # ~1.5 s, sparse
    assert nav.escaping and nav.breakout is not None
    assert all(_dot(v, h0) >= -1e-6 for v in out)
    run(nav, clock, result([]), 14, hp=1.0)
    assert not nav.escaping
    assert any("ring escape done" in line for line in logs)


def test_outside_flower_window_plain_breakout_unchanged():
    nav, clock, logs = _ring_nav(100.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 10, hp=1.0)
    assert nav.breakouts >= 1 and not nav.escaping
    assert nav.last.mode.startswith("breakout")
    assert not any("ring escape" in line for line in logs)


def test_ring_commit_defaults_on_and_explicit_zero_disables():
    assert Tuning().ring_commit_s == 2.5
    assert Tuning().flower_noreverse_s == 4.0
    nav, clock, logs = _ring_nav(318.0, tune=Tuning(ring_commit_s=0.0))
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 10, hp=1.0)
    assert nav.breakouts >= 1 and not nav.escaping
    assert nav.last.mode.startswith("breakout")


def test_revive_overrides_running_ring_escape():
    nav, clock, logs = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 4, hp=1.0)
    assert nav._escape_reason == "ring"
    nav.request_revive_escape()
    assert nav._escape_reason == "revive" and nav._escape_remaining == nav.tune.revive_escape_s
    run(nav, clock, result(blob(620, 256, n=30, spread=40)), 2, hp=0.5)
    assert nav.last.mode.startswith("revive-escape")


def test_plain_flee_cancels_committed_escape():
    nav, clock, _ = _ring_nav(318.0)
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 4, hp=1.0)
    assert nav.escaping
    nav.request_flee(2.0)
    assert not nav.escaping and nav._flee_remaining >= 2.0  # extends, never shortens
