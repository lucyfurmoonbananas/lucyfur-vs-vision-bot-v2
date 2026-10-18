"""Gem-seek activations and the XP-bar pickup proxy.

Pickups are rises of the top XP bar. A level-up drops the bar and is not a
pickup. Seek activations are rising edges of the roam gem-seek boost, so a
stretch of seeking counts once.
"""

from __future__ import annotations

from pathlib import Path

from vs_vision_bot.pathfind import Navigator, format_gem_summary, gem_status, tuning_from_env
from vs_vision_bot.vision import read_xp_fraction

from tests.test_chest_real import _load
from tests.test_movement import Clock, result, run
from tests.test_vision_real import load


def test_xp_bar_fill_on_gameplay_frames():
    # Blue run x=51..290, 51..311, 51..548 on the 51..868 track (818 px).
    assert read_xp_fraction(load("gameplay_bats_bushes.png")) == 240 / 818
    assert read_xp_fraction(load("gameplay_full_hp.png")) == 261 / 818
    assert read_xp_fraction(load("gameplay_hurt.png")) == 498 / 818
    assert read_xp_fraction(load("main_menu.png")) is None
    assert read_xp_fraction(load("results_screen.png")) is None


def test_gem_seek_counts_each_activation_not_every_frame():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    gems = result([], gems=[(300.0, 256.0)])
    run(nav, clock, gems, 8, hp=1.0)
    assert nav.gem_seeks == 1
    run(nav, clock, result([]), 2, hp=1.0)
    run(nav, clock, gems, 4, hp=1.0)
    assert nav.gem_seeks == 2


def test_chest_or_disabled_seek_does_not_activate():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    armed = result([], gems=[(300.0, 256.0)])
    armed.chest = (720.0, 256.0)
    run(nav, clock, armed, 6, hp=1.0)
    assert nav.gem_seeks == 0

    off = tuning_from_env("gem_seek_gain=0", log=lambda *_: None)
    nav = Navigator(tune=off, clock=clock, log=lambda *_: None)
    run(nav, clock, result([], gems=[(300.0, 256.0)]), 6, hp=1.0)
    assert nav.gem_seeks == 0


def test_xp_rise_counts_a_pickup_and_a_levelup_drop_does_not():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    res = result([])
    res.xp = 0.20
    run(nav, clock, res, 1, hp=1.0)
    assert nav.gem_pickups == 0  # first reading is the baseline

    res.xp = 0.20 + 1 / 818  # one-pixel flicker
    run(nav, clock, res, 1, hp=1.0)
    assert nav.gem_pickups == 0

    res.xp = 0.20 + 5 / 818
    run(nav, clock, res, 1, hp=1.0)
    assert nav.gem_pickups == 1

    res.xp = 0.02  # level-up reset
    run(nav, clock, res, 1, hp=1.0)
    assert nav.gem_pickups == 1

    res.xp = 0.08
    run(nav, clock, res, 1, hp=1.0)
    assert nav.gem_pickups == 2

    # A menu hides the bar. A full-looking reading there is not a pickup.
    menu = result([], menu=True)
    menu.xp = 1.0
    run(nav, clock, menu, 1, hp=1.0)
    assert nav.gem_pickups == 2
    assert nav._xp == 0.08


def test_nav_line_and_run_summary_carry_the_counts():
    nav = Navigator(log=lambda *_: None)
    nav.gem_seeks = 4
    nav.gem_pickups = 10
    nav.game_s = 120
    assert gem_status(nav) == "gem_seek=4 gem_pickups=10"
    assert format_gem_summary(nav) == "[gems] seeks=4 pickups=10 game~120s 5.0/min"

    lines: list[str] = []
    nav = Navigator(log=lines.append)
    nav.game_s = 60
    nav.gem_pickups = 3
    nav.log_gem_summary()
    nav.log_gem_summary()
    assert lines == ["[gems] seeks=0 pickups=3 game~60s 3.0/min"]
    nav.reset()
    nav.game_s = 30
    nav.gem_pickups = 1
    nav.log_gem_summary()
    assert lines[-1] == "[gems] seeks=0 pickups=1 game~30s 2.0/min"


def test_real_frames_still_have_a_readable_xp_bar():
    # Sanity: the chest fixtures are gameplay, so the bar is there and not full.
    frame = _load(
        Path(__file__).parent / "fixtures" / "chest_real" / "REAL_arrow_left_run6_20261018_051535.png"
    )
    xp = read_xp_fraction(frame)
    assert xp is not None and 0.05 < xp < 0.95
