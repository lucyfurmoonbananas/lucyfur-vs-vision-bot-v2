"""A uniform crowd must not send a revive north or south just because those
rays are shorter. Silo's equal view, with each direction stopped at its own
screen edge."""

from __future__ import annotations

import numpy as np

from vs_vision_bot.menus import HpTracker, hp_for_nav
from vs_vision_bot.pathfind import Tuning
from vs_vision_bot.vision import VisionResult

from tests.test_movement import Clock, result, run
from tests.test_revive_escape import _nearest, _quiet_nav


def _uniform(px: float, py: float, step: int = 28) -> np.ndarray:
    rows = []
    for x in range(16, 944, step):
        for y in range(16, 524, step):
            if (x - px) ** 2 + (y - py) ** 2 < 40 ** 2:
                continue
            rows.append((x, y, 1.0))
    return np.array(rows, np.float32)


def _at(points: np.ndarray, px: int, py: int) -> VisionResult:
    res = result(points)
    res.player = (px, py)
    return res


def test_uniform_crowd_revive_takes_a_long_edge():
    """Centered on an even field, north and south used to win because the
    vertical rays contain fewer enemies. East and west are the long runways.
    """
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.heading = -np.pi / 2  # the death that used to repeat north
    nav.request_revive_escape()
    run(nav, clock, _at(_uniform(480, 256), 480, 256), 2, hp=0.5)
    assert _nearest(nav.breakout) in ((1, 0), (-1, 0))


def test_blind_revive_stays_critical_on_that_heading():
    """The death-trace line at the revive click is the menu sample
    (hp=none med=none danger=none mode=revive). The next gameplay frame
    still has no median, and nav must treat that as critical.
    """
    tracker = HpTracker()
    tracker.add(0.12)
    tracker.forget_for_revive()
    acted = hp_for_nav(tracker.value, tracker.blind, tracker.low_held)
    assert tracker.value is None and acted == 0.0

    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.request_revive_escape()
    crowd = _at(_uniform(480, 256), 480, 256)
    run(nav, clock, crowd, 8, hp=acted)
    assert nav._critical_on
    assert any("critical HP on" in line for line in logs)
    assert _nearest(nav.breakout) in ((1, 0), (-1, 0))
    assert "critical" in nav.last.mode


def test_wall_past_the_near_edge_is_not_the_runway():
    """Player against the west border, enemies only to the east beyond 160px.

    Capping every direction at the nearest edge hides that wall and the
    runway tie then walks east into it. Each direction keeps its own reach.
    """
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.request_revive_escape()
    wall = [
        (x, y, 1.0)
        for x in range(340, 600, 18)
        for y in range(40, 500, 22)
    ]
    run(nav, clock, _at(np.array(wall, np.float32), 120, 270), 2, hp=0.5)
    heading = _nearest(nav.breakout)
    assert heading[0] != 1


def test_equal_view_switch_restores_the_short_axis():
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.tune = Tuning(escape_equal_view=0)
    nav.heading = -np.pi / 2
    nav.request_revive_escape()
    run(nav, clock, _at(_uniform(480, 256), 480, 256), 2, hp=0.5)
    assert _nearest(nav.breakout)[0] == 0
