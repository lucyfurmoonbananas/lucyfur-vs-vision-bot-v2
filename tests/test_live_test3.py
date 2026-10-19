"""Live test3 (10:52, Lv12): latch, false arrow, clock, opening gems.

These cases are taken from run1.log and the periodic shots. Each one fails on
the pre-fix navigator or detector and passes after the small guard.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vs_vision_bot.pathfind import Navigator, Tuning, gem_tune_for, phase_for, score_directions
from vs_vision_bot.vision import detect_chest, detect_chest_arrow

from tests.test_movement import Clock, P, result, run, timer_frame

EMPTY = np.zeros((0, 3), np.float32)

FALSE = Path(__file__).parent / "fixtures" / "chest_false"


def test_slow_frame_miss_does_not_drop_a_latched_arrow():
    """One missed sample is held. Further misses last until chest_hold_s (0.6).

    The 5 fps cases live in test_chest_hold. Here a 0.25 s frame stays latched,
    and a second miss that pushes the gap past 0.6 s clears.
    """
    lines: list[str] = []
    clock = Clock()
    nav = Navigator(clock=clock, log=lines.append)
    arrow = result([])
    arrow.chest_arrow = (600.0, 500.0)
    run(nav, clock, arrow, 3, hp=1.0)
    assert any(ln.startswith("[chest] seek arrow") for ln in lines)
    lines.clear()

    gone = result([])
    clock.t += 0.25
    nav.step(gone, hp=1.0)
    assert not any(ln.startswith("[chest] clear") for ln in lines)
    assert nav._chest_latched

    clock.t += 0.45  # miss time 0.70 s, past chest_hold_s
    nav.step(gone, hp=1.0)
    assert any(ln.startswith("[chest] clear arrow") for ln in lines)


def test_pale_chevron_without_a_chest_icon_is_not_an_arrow():
    """Bottom-edge pale chevron from the test3 streak color, no white chest icon.

    The periodic stills fragment that streak under the area gate, so this
    fixture is the same yellow at a size the detector already accepts. The
    real r1_t302 crop keeps its top chest arrow and must not prefer the
    bottom heart streak.
    """
    false = cv2.imread(str(FALSE / "false_bottom_chevron.png"))
    assert false is not None and false.shape[:2] == (540, 960)
    assert detect_chest_arrow(false) is None
    assert detect_chest(false) is None

    shot = cv2.imread(str(FALSE / "r1_t302_bottom_heart.png"))
    assert shot is not None and shot.shape[:2] == (540, 960)
    arrow = detect_chest_arrow(shot)
    assert arrow is not None and arrow[1] < 160, arrow
    assert detect_chest(shot) is None


def test_dropped_ten_minute_digit_does_not_rewind_the_phase():
    """HUD 10:14 read as 0:14 must not resync, even when the misread ticks."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 10 * 60 + 14.0
    timer_frame(nav, clock, 10 * 60 + 14.0)
    assert nav.phase.name == "wide-loops"
    for reading in (14.0, 16.0, 18.0):
        game_s = timer_frame(nav, clock, reading, 2.0)
        assert game_s > 600.0
        assert phase_for(game_s).name == "wide-loops"


def test_opening_minute_seeks_a_gem_the_early_radius_misses():
    """First minute, danger ~0: a gem just outside early_gem_radius is still sought.

    Past the opening window the early radius is unchanged, so the same gem
    does not pull.
    """
    # 470 px left of the player. early_gem_radius is 460.
    far = [(10.0, 256.0)]
    kite = (0.0, -1.0)
    opening = score_directions(
        EMPTY, P, kite, 1.0, gems=far, tune=gem_tune_for(Tuning(), 20.0),
    )
    later = score_directions(
        EMPTY, P, kite, 1.0, gems=far, tune=gem_tune_for(Tuning(), 120.0),
    )
    assert opening.vector[0] == -1
    assert later.vector == (0, -1)
    assert opening.danger_here < 0.05
