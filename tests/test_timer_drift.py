"""A HUD timer that keeps ticking behind the wall clock corrects game~.

Live run on cb74267: the results screen said 11:04 and the quit frame's HUD
read 11:00, while [gems] printed game~689s. Stuck OCR misreads must still
not drag the clock backwards.
"""

from __future__ import annotations

from vs_vision_bot.pathfind import Navigator

from tests.test_movement import Clock, timer_frame


def test_ticking_hud_behind_wall_clock_resyncs_game_time():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 100.0
    timer_frame(nav, clock, 100.0)
    # Level-up pauses left the wall clock ~30 s ahead. The HUD is real and ticking.
    assert timer_frame(nav, clock, 70.0, 2.0) > 90.0
    assert timer_frame(nav, clock, 72.0, 2.0) > 90.0
    assert abs(timer_frame(nav, clock, 74.0, 2.0) - 74.0) < 1.0
