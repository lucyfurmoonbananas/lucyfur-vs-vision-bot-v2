"""test4: critical dropped on hp=? after REVIVE, the escape was a fixed 3s,
and the latch waited for a 0.00 reading inside a swarm that was already
hurting."""

from __future__ import annotations

import pytest

from vs_vision_bot import menus
from vs_vision_bot.pathfind import Navigator

from tests.test_movement import Clock, blob, result, ring, run
from tests.test_revive_escape import _quiet_nav


def test_unknown_hp_does_not_clear_critical():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    run(nav, clock, result(blob(640, 256, n=6)), 2, hp=0.12)
    assert nav._critical_on
    run(nav, clock, result(blob(640, 256, n=6)), 3, hp=None)
    assert nav._critical_on
    assert not any("critical HP off" in line for line in logs)


def test_revive_stays_critical_until_agreed_readings():
    """The revive click used to reset the tracker, so the next frame was hp=?
    and nav left critical. One restored-bar sample is not yet trusted, and it
    must not look like a full bar to a high-HP dive.
    """
    t = menus.HpTracker()
    for v in (0.2, 0.19, 0.1):
        t.add(v)
    t.forget_for_revive()
    assert t.value is None and t.low_held
    assert menus.hp_for_nav(t.value, t.blind, t.low_held) == 0.0

    t.add(1.0)  # one flash, or the first look at a refilling bar
    assert t.low_held
    assert menus.hp_for_nav(t.value, t.blind, t.low_held) == 0.0

    fresh = menus.HpTracker()
    fresh.forget_for_revive()
    for _ in range(3):
        fresh.add(0.50)
    assert fresh.low_held is False
    assert menus.hp_for_nav(fresh.value, fresh.blind, fresh.low_held) == pytest.approx(0.50)


def test_falling_hp_in_a_crowd_is_critical_before_zero():
    """Nav lines sit at 0.31–0.50 with danger already high, then the latch
    logs hp=0.00 about 2s before death. The bar is readable in that band.
    """
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    wall = result(ring(480, 256, 140, 64))
    run(nav, clock, wall, 8, hp=0.70)
    assert not nav._critical_on
    run(nav, clock, wall, 4, hp=0.36)
    assert nav._critical_on
    assert any("critical HP on" in line for line in logs)
    assert not any("hp=0.00" in line for line in logs)


def test_critical_revive_keeps_escaping_until_the_crowd_clears():
    """A one-sided crowd at a trusted 0.50 still ends at 3s. The same crowd
    while HP is critical (the post-revive blind) does not.
    """
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    crowd = result(blob(620, 256, n=30, spread=45) + blob(560, 380, n=10, spread=30))
    nav.request_revive_escape()
    run(nav, clock, crowd, 38, hp=0.0)  # past the old 3s window
    assert nav.escaping
    assert nav.last.danger_here >= nav.tune.safe_danger
    assert nav._escape_s < nav.tune.revive_escape_max_s
    run(nav, clock, result([]), 12, hp=0.0)
    assert not nav.escaping
