"""A low HP read must stay critical while the bar is missing, and say so in the log.

Death 1 (run1, game~400s): raw hp=0.06, then no reading for ~4.6s. The median
was still 0.92 at -4.4s and the mode was chest/tight-weave. Both low stretches
(hp under 0.2) ended in a death with no critical-HP line.
"""

from __future__ import annotations

import numpy as np
import pytest

from vs_vision_bot import menus
from vs_vision_bot.pathfind import Navigator, Tuning, score_directions

from tests.test_movement import Clock, P, blob, result, run


def test_low_read_then_a_miss_stays_critical_until_a_trusted_heal():
    t = menus.HpTracker()
    for _ in range(5):
        t.add(0.92)
    t.add(None)
    assert t.low_held is False
    assert menus.hp_for_nav(t.value, t.blind, t.low_held) == pytest.approx(0.92)

    # Downward jump is real. The median is still the old high window.
    t.add(0.06)
    assert t.value > 0.5
    assert t.low_held is True
    t.add(None)
    t.add(None)
    assert t.blind is False and t.value > 0.5
    assert menus.HP_CRITICAL == Tuning().critical_hp
    nav_hp = menus.hp_for_nav(t.value, t.blind, t.low_held)
    assert nav_hp is not None and nav_hp < Tuning().critical_hp

    enemies = np.array(blob(640, 256, n=6, spread=12), np.float32)
    chest = (720.0, 256.0)
    chased = score_directions(
        enemies, P, (1.0, 0.0), 1.0, gems=[(710.0, 250.0)], hp=nav_hp, chest=chest,
    )
    assert chased.mode != "chest"
    assert chased.vector[0] <= 0

    # run1 quit: 0.00, 0.97, 0.97, 0.06. The spikes must not clear the hold.
    spike = menus.HpTracker()
    for v in (0.03, 0.03, 0.03, 0.00, 0.97, 0.97, 0.06):
        spike.add(v)
    assert spike.value < 0.2
    assert spike.low_held is True
    assert menus.hp_for_nav(spike.value, spike.blind, spike.low_held) < 0.2

    for _ in range(3):
        spike.add(0.55)
    assert spike.value >= 0.2 and spike.low_held is False
    assert menus.hp_for_nav(spike.value, spike.blind, spike.low_held) >= 0.2


def test_critical_hp_logs_on_off_and_tags_the_nav_mode():
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    run(nav, clock, result([]), 2, hp=0.90)
    assert not any("critical HP" in line for line in logs)
    assert "critical" not in nav.last.mode

    run(nav, clock, result(blob(640, 256, n=4)), 2, hp=0.12)
    assert any("critical HP on" in line and "hp=0.12" in line for line in logs)
    assert nav.last.mode.endswith("critical")
    assert nav.last.mode.startswith("kite") or nav.last.mode.startswith("escape") or nav.last.mode.startswith("breakout")

    run(nav, clock, result([]), 2, hp=0.80)
    assert any("critical HP off" in line and "hp=0.80" in line for line in logs)
    assert "critical" not in nav.last.mode
