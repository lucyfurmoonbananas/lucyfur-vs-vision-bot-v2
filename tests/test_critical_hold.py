"""test3 (10:52, one revive): critical let go too early, the bar lied under the
horde, the revive hold ended at 3.3s, and ring commits fired on ordinary cover.

Critical enters under 0.20 and must not leave on a 0.25 median or on one
full-bar flash. The anchored bar is right when it is visible; the fallback
search is what reads a damage slash as 1.0 and a 2px speck as 0.06.
"""

from __future__ import annotations

import numpy as np
import pytest

from vs_vision_bot import menus
from vs_vision_bot.pathfind import Navigator, Tuning

from tests.test_movement import Clock, blob, result, ring, run


def test_quarter_hp_and_a_full_bar_flash_stay_critical():
    """10:17 exited at hp 0.25. 10:50 exited after 0.95, 0.96, 1.00."""
    assert menus.HP_CRITICAL < menus.HP_CRITICAL_EXIT
    assert menus.HP_CRITICAL_EXIT >= 0.40

    low = menus.HpTracker()
    for v in (0.06, 0.06, 0.00):
        low.add(v)
    assert low.low_held
    for _ in range(4):
        low.add(0.25)
    assert low.value == pytest.approx(0.25)
    assert low.low_held
    assert menus.hp_for_nav(low.value, low.blind, low.low_held) < menus.HP_CRITICAL

    flash = menus.HpTracker()
    for _ in range(3):
        flash.add(0.03)
    flash.add(0.96)  # one frame never counts as a heal
    assert flash.low_held
    assert flash.value < menus.HP_CRITICAL
    for v in (0.95, 0.96, 1.00):
        flash.add(v)
    assert flash.low_held
    assert menus.hp_for_nav(flash.value, flash.blind, flash.low_held) < menus.HP_CRITICAL

    # The same three-frame flash, then the bar disappears. Still critical.
    for _ in range(4):
        flash.add(None)
    acted = menus.hp_for_nav(
        flash.value, flash.blind, flash.low_held, trusted=flash._trusted, blind_s=0.0,
    )
    assert acted is not None and acted < menus.HP_CRITICAL


def _green_over_bar() -> np.ndarray:
    frame = np.full((540, 960, 3), (40, 90, 50), np.uint8)
    x0, y0, x1, y1 = menus.HP_BAR_RECT_960
    # Sprite on the bar: not red, not the dark empty track, so the anchor gives up.
    frame[y0:y1, x0:x1] = (30, 150, 40)
    return frame


def test_fallback_ignores_a_damage_slash_and_a_red_speck():
    """quit 02_06: anchor lost, a 2-row red slash above the bar read as 1.0.
    revive 01_06 / 01_08: a 2px speck in that window read as 0.06 and stuck.
    """
    slash = _green_over_bar()
    slash[266:268, 469:498] = (0, 0, 254)
    slash[268:270, 476:510] = (0, 0, 254)
    assert menus.read_hp_bar(slash) is None
    assert menus.read_hp_fraction(slash) is None

    speck = _green_over_bar()
    speck[269:270, 480:482] = (0, 0, 254)
    assert menus.read_hp_bar(speck) is None
    assert menus.read_hp_fraction(speck) is None

    # A visible bar is unchanged.
    x0, y0, x1, y1 = menus.HP_BAR_RECT_960
    shown = np.full((540, 960, 3), (40, 90, 50), np.uint8)
    shown[y0:y1, x0:x1] = (10, 10, 10)
    shown[y0:y1, x0 : x0 + 18] = (0, 0, 255)
    assert menus.read_hp_fraction(shown) == pytest.approx(0.5)


def test_revive_hold_keeps_going_while_the_crowd_is_still_close():
    """test3 revive: cover fell 0.94 -> 0.31 inside the 3s window, so the 12s
    hold never armed, and kite resumed at danger 0.7. A ring that opens while
    enemies are still close is not clear ground. A crowd that was never a ring
    still ends at 3s (test_revive_escape_runs_full_window_toward_open_side_and_logs).
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    nav.request_revive_escape()
    run(nav, clock, result(ring(480, 256, 140, 64)), 24, hp=0.5)
    assert nav.escaping and nav.cover >= nav.tune.cover_trigger
    # Ring opens. Cover stays under the trigger; danger stays above safe.
    opened = result(blob(560, 256, n=6, spread=80, seed=1))
    run(nav, clock, opened, 24, hp=0.5)
    assert nav.cover < nav.tune.cover_trigger
    assert nav.last.danger_here >= nav.tune.safe_danger
    assert nav.escaping
    assert any("revive escape holding" in line for line in logs)
    assert nav._escape_s < nav.tune.revive_escape_max_s

    run(nav, clock, result([]), 16, hp=0.5)
    assert not nav.escaping
    assert any("revive escape done" in line for line in logs)


def test_cover_044_density_noise_does_not_commit_a_ring_escape():
    """test3: 87 ring commits in 11 minutes, 39 of them at cover 0.44, from
    game~1s. That field is a handful of scattered cells, not a closing wall.
    The full-ring surge (cover ~0.94) still commits; see test_wall_breakout.
    """
    assert Tuning().ring_commit_cover > Tuning().cover_rising
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    nav.game_s = 100.0  # outside the flower window
    run(nav, clock, result(blob(700, 256, n=3, spread=20, seed=0)), 16, hp=1.0)
    run(nav, clock, result(blob(520, 280, n=8, spread=140, seed=1)), 6, hp=1.0)
    assert not any(line.startswith("[nav] ring escape") for line in logs)
    assert not nav.escaping

    # The wall this was built for: sparse, then a closed ring with a gap.
    wall = Navigator(clock=Clock(), log=logs.append)
    wall.game_s = 660.0
    clock2 = wall.clock
    run(wall, clock2, result(blob(700, 256, n=8, spread=30)), 8, hp=0.24)
    run(wall, clock2, result(ring(480, 256, 150, 48, gap_deg=(-1, 25))), 4, hp=0.24)
    assert wall.escaping and wall.last.mode.startswith("ring-escape")
