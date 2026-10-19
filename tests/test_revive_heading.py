"""Revive escape heading: a flat cost vector must not repeat north, and a
corridor that fills in the first seconds must be left before the 3s hold."""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot.pathfind import DIRS, escape_costs, _UNIT

from tests.test_movement import Clock, blob, result, ring, run
from tests.test_revive_escape import _dot, _nearest, _quiet_nav


def test_flat_revive_tie_does_not_repeat_north():
    """An empty frame is an 8-way tie. DIRS starts at north, and so does the
    heading left over from a northward death, so the old pick repeats (0, -1).
    """
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.heading = -math.pi / 2
    nav.request_revive_escape()
    run(nav, clock, result([]), 1, hp=0.5)
    assert nav.breakout is not None
    # East and west are the long runways (480 px). The tie breaks toward east.
    assert _nearest(nav.breakout) == (1, 0)


def test_flat_tie_prefers_a_long_edge_arrow_and_a_real_gap_ignores_it():
    """A west arrow on a flat frame is an open long edge, so it wins the tie.
    The same arrow beside a real northern gap is the dense side and is ignored.
    """
    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.heading = -math.pi / 2
    nav.request_revive_escape()
    blank = result([])
    blank.chest_arrow = (80, 256)
    run(nav, clock, blank, 1, hp=0.5)
    assert _nearest(nav.breakout) == (-1, 0)

    clock = Clock()
    nav, _ = _quiet_nav(clock)
    nav.heading = -math.pi / 2
    nav.request_revive_escape()
    gap = result(ring(480, 256, 150, 48, gap_deg=(250, 290)))
    gap.chest_arrow = (80, 256)
    run(nav, clock, gap, 1, hp=0.5)
    assert _nearest(nav.breakout) == (0, -1)


def _north_filling():
    """North a bit fuller than the forward diagonals, not a wall.

    North ~3.1, best forward ~1.5: above the 1.6x re-aim and below the 3x lock.
    South stays cheap and is a reversal, so it must not be the pick.
    """
    return (
        blob(480, 80, n=4, spread=30, seed=1)
        + blob(620, 120, n=2, spread=35, seed=2)
        + blob(340, 120, n=2, spread=35, seed=3)
        + blob(700, 256, n=4, spread=40, seed=4)
        + blob(260, 256, n=4, spread=40, seed=5)
    )


def test_revive_reaims_a_filling_corridor_before_three_seconds():
    clock = Clock()
    nav, logs = _quiet_nav(clock)
    nav.heading = -math.pi / 2
    nav.request_revive_escape()
    gap = result(ring(480, 256, 150, 48, gap_deg=(250, 290)))
    run(nav, clock, gap, 8, hp=0.5)
    assert _nearest(nav.breakout) == (0, -1)
    assert nav.tune.escape_repick_s < nav._escape_s < nav.tune.revive_escape_s

    fill = result(_north_filling())
    costs = escape_costs(fill.threats, 480, 256, nav.tune)
    cur = DIRS.index((0, -1))
    ahead = (_UNIT @ _UNIT[cur]) >= -1e-6
    best = int(np.argmin(np.where(ahead, costs, np.inf)))
    assert best != cur
    assert costs[cur] > 1.6 * costs[best] + 0.5
    assert costs[cur] <= 3.0 * costs[best] + 1.5

    run(nav, clock, fill, 6, hp=0.5)
    assert nav._escape_s < nav.tune.revive_escape_s
    assert _nearest(nav.breakout) != (0, -1)
    assert _dot(nav.breakout, (0, -1)) >= -1e-6
    assert any("re-aim" in line and "(0, -1)->" in line for line in logs)
