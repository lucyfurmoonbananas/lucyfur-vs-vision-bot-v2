"""A closing wall outside the flower window must punch the thin side and stay there.

run1 at ~11:00 (game~676-686s, wide-loops): cells 105 -> 782 -> 1746 and cover
0.25 -> 0.50 -> 0.88 in a few seconds. That is outside the flower window, so it
was only a plain breakout (dropped once cover dipped, free to reverse). The
final frames are hp 0.03, danger 21-28, boxed in.

Also: a tied escape cone used to return the first DIRS entry, which is north.
"""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import Navigator, Tuning, score_directions

from tests.test_movement import Clock, blob, result, ring, run


def test_late_density_surge_commits_through_the_gap_and_holds():
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    nav.game_s = 660.0  # wide-loops, flower window already closed
    run(nav, clock, result(blob(700, 256, n=8, spread=30)), 8, hp=0.24)
    assert nav.escaping is False

    gap_east = result(ring(480, 256, 150, 48, gap_deg=(-1, 25)))
    run(nav, clock, gap_east, 4, hp=0.24)
    assert nav.escaping
    assert nav.last.mode.startswith("ring-escape")
    assert nav.breakout is not None and nav.breakout[0] > 0.5
    assert any(line.startswith("[nav] ring escape (1, 0)") for line in logs)

    # Plain breakouts end after 1.1s once the ring is gone. The commit must hold.
    out = run(nav, clock, result(blob(300, 256, n=6, spread=20)), 18, hp=0.24)
    assert nav.escaping
    assert all(v[0] >= 0 for v in out)


def test_density_surge_stays_with_a_latched_chest():
    """A closing wall still picks the gap, but dive HP does not walk it or clear the chest.

    The chest is west and the gap is east. Critical HP and a chest_open_always=0
    hold still yield; this is the default open.
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    nav.game_s = 660.0
    quiet = result(blob(700, 256, n=4, spread=12))
    quiet.chest = (300.0, 256.0)  # west; the gap below is east
    run(nav, clock, quiet, 6, hp=1.0)
    assert nav._chest_latched
    assert any(line.startswith("[chest] seek") for line in logs)

    gap_east = result(ring(480, 256, 150, 48, gap_deg=(-1, 25)))
    gap_east.chest = (300.0, 256.0)
    out = run(nav, clock, gap_east, 4, hp=1.0)
    # Above 25% the wall is not a ring-escape. The chest to the west is walked.
    assert not nav.escaping
    assert not nav.last.mode.startswith("ring-escape")
    assert nav._chest_latched
    assert not any(line.startswith("[chest] clear") for line in logs)
    assert all(v[0] <= 0 for v in out)


def test_surge_picks_the_south_gap_when_that_side_is_thinnest():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 660.0
    nav.heading = 0.0
    run(nav, clock, result(blob(480, 80, n=6, spread=20)), 8, hp=0.24)
    run(nav, clock, result(ring(480, 256, 150, 48, gap_deg=(70, 110))), 4, hp=0.24)
    assert nav.escaping and nav.breakout is not None
    assert nav.breakout[1] > 0.5


def test_tied_breakout_cone_follows_the_heading_not_north():
    """DIRS starts at (0, -1). Equal scores in an east 90-degree cone used to pick north."""
    tune = Tuning(
        breakout_weight=0.0, hysteresis=0.0, rollout_gain=0.0, min_dwell_s=0.0,
        ring_commit_s=0.0, flower_noreverse_s=4.0,
    )
    clock = Clock()
    nav = Navigator(tune=tune, clock=clock, log=lambda *_: None)
    nav.game_s = 318.0
    nav.heading = 0.0
    run(nav, clock, result(ring(480, 256, 170, 40, gap_deg=(-1, 25))), 2, hp=0.24)
    assert not nav.escaping and nav.breakout is not None and nav.breakout[0] > 0.5
    clock.t += 1 / 12
    dec = nav.step(result([]), hp=0.24)
    assert dec.vector != (0, -1)
    assert dec.vector[0] > 0


def test_full_screen_crowd_does_not_score_north_as_the_opening():
    """Camera is at y=256 of 540. A uniform crowd's short northern ray is the edge."""
    xs = np.arange(16, 960, 32)
    ys = np.arange(16, 540, 32)
    grid = np.array([(float(x), float(y), 1.0) for x in xs for y in ys], np.float32)
    old = Tuning(pack_dive_hp=0.0)
    dec = score_directions(grid, (480.0, 256.0), None, 0.0, tune=old)
    # Before the short-side fill this was about +6 (north scored as the opening).
    advantage = dec.scores[(0, -1)] - dec.scores[(0, 1)]
    assert advantage < 2.0

    gap_north = np.array(ring(480, 256, 140, 36, gap_deg=(250, 290)), np.float32)
    assert score_directions(gap_north, (480.0, 256.0), None, 0.0, tune=old).vector == (0, -1)
    # Above 25% the same gap is not the exit. The mass of the ring is south of it.
    farmed = score_directions(gap_north, (480.0, 256.0), None, 0.0, hp=1.0)
    assert farmed.vector != (0, -1)
