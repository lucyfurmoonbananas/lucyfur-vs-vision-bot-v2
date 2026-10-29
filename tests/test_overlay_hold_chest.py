"""test6: overlay HP, revive-hold hemisphere, chest reward card.

Run 1 reached 28:40 with Soul Eater and Unholy Vespers evolved, and the log
still treated chest / level-up frames as a real HP crash and let the post-revive
hold walk from +D to -D and back.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vs_vision_bot import main as main_mod
from vs_vision_bot import menus
from vs_vision_bot.menus import (
    Card,
    MenuPilot,
    evolutions_from_chest_text,
    hp_for_nav,
    levels_from_chest_text,
    read_chest_text,
)
from vs_vision_bot.pathfind import Navigator

from tests.test_movement import Clock, blob, result, run
from tests.test_revive_escape import _arc, _arm_east_hold, _dot, _nearest, _pack

needs_tesseract = pytest.mark.skipif(not menus.tesseract_available(), reason="tesseract not installed")


def test_one_frame_zero_from_a_healthy_bar_is_not_critical():
    """Chest / level-up darkens the bar for one frame. hp=1.00 then 0.00 is not a crash.

    A second 0.00 is a real empty bar. A drop to 0.06 still latches immediately,
    and a bar that is already low still accepts 0.00.
    """
    t = menus.HpTracker()
    for _ in range(5):
        t.add(1.0)
    t.add(0.0)
    assert t.low_held is False
    assert t.value == pytest.approx(1.0)
    assert hp_for_nav(t.value, t.blind, t.low_held) == pytest.approx(1.0)

    t.add(0.0)
    assert t.low_held is True
    assert hp_for_nav(t.value, t.blind, t.low_held) == 0.0

    real = menus.HpTracker()
    for _ in range(5):
        real.add(0.92)
    real.add(0.06)
    assert real.low_held is True
    assert hp_for_nav(real.value, real.blind, real.low_held) == 0.0

    already = menus.HpTracker()
    already.add(0.03)
    already.add(0.00)
    assert already.low_held is True
    assert already.value < 0.2


def test_chest_overlay_clears_a_confirmed_zero_and_the_crash_flee():
    """Two dark frames latch before the chest scan confirms. The menu drops both."""
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append)
    for _ in range(5):
        pilot._hp.add(1.0)
    pilot._hp.add(0.0)
    pilot._hp.add(0.0)
    assert pilot.hp_low_held is True
    acted = hp_for_nav(pilot.hp, pilot.hp_blind, pilot.hp_low_held)
    sparse = result(blob(600, 256, n=8, spread=30))
    run(nav, clock, sparse, 4, hp=1.0)
    run(nav, clock, sparse, 2, hp=acted)
    assert nav._critical_on
    assert any("HP-crash flee" in line for line in logs)
    assert nav._flee_remaining > 0

    armed = main_mod.note_menu_overlay(nav, pilot, "chest", False)
    assert armed is True
    # The tracker is already holding a real low read. The chest screen must not
    # drop it; test7 re-logged critical HP on hp=0.00 every time it did.
    assert nav._critical_on is True
    assert not any("cleared (overlay)" in line for line in logs)
    for _ in range(6):
        pilot._hp.add(0.0)
    assert pilot.hp_low_held is True  # still the pre-menu latch; zeros during the menu do not stick
    armed = main_mod.note_menu_overlay(nav, pilot, None, armed)
    assert armed is False
    assert pilot.hp_low_held is False
    assert hp_for_nav(pilot.hp, pilot.hp_blind, pilot.hp_low_held) == pytest.approx(1.0)
    assert nav._critical_on is True
    assert not any("cleared (overlay)" in line for line in logs)


def test_overlay_still_clears_a_false_latch_when_hp_is_not_held_low():
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append)
    for _ in range(5):
        pilot._hp.add(1.0)
    assert pilot.hp_low_held is False
    run(nav, clock, result(blob(600, 256, n=8, spread=30)), 4, hp=0.12)
    assert nav._critical_on
    armed = main_mod.note_menu_overlay(nav, pilot, "levelup", False)
    assert armed is True
    assert nav._critical_on is False
    assert any("cleared (overlay)" in line for line in logs)
    main_mod.note_menu_overlay(nav, pilot, None, armed)


def test_revive_hold_does_not_ping_pong_across_the_hemisphere():
    """test6 run1 +8.2/+9.2/+11.7 and run2 +4.3/+6.9/+10.0.

    A moderate wall (~2.2x) shuts the corridor we are on, so the hold may leave
    that hemisphere once. The same wall on the far side is not a reason to walk
    back. The heavy-wall swap (~3.3x) still returns; this one must not.
    """
    nav, clock, logs = _arm_east_hold()
    frozen = tuple(nav.breakout)
    n0 = len(logs)
    headings = []
    covers = []
    for i in range(40):
        wall = 0 if (i // 2) % 2 == 0 else 180
        extra = []
        for radius in (130.0, 170.0, 210.0):
            extra += _arc(wall, n=7, radius=radius, weight=0.8, spread=0.14)
        run(nav, clock, result(_pack(extra)), 1, hp=0.5)
        headings.append(_nearest(nav.breakout))
        covers.append(nav.cover)
    assert min(covers) >= nav.tune.cover_trigger
    assert headings[0] == (-1, 0)
    assert all(_dot(h, frozen) <= 1e-6 for h in headings)
    assert _dot(nav._revive_hemi, frozen) > 0.9
    back = [line for line in logs[n0:] if "->(1, 0)" in line]
    assert back == []


def _reward_frame(name: str, level: int) -> np.ndarray:
    """Dim reward card under bright HUD noise. Full-screen thr=200 OCR misses the name."""
    img = np.zeros((540, 960, 3), np.uint8)
    img[:] = (6, 8, 6)
    rng = np.random.default_rng(2)
    junk = ["Venda", "Feet", "18304", "oF]", "Ce", "en", "wr", "res", "oe"]
    for i in range(30):
        x, y = int(rng.integers(30, 900)), int(rng.integers(30, 120))
        cv2.putText(img, junk[i % len(junk)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(img, "28:28", (40, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    cv2.rectangle(img, (260, 140), (700, 340), (16, 16, 18), -1)
    col = (180, 180, 180)
    cv2.putText(img, name, (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2, cv2.LINE_AA)
    cv2.putText(img, f"Level: {level}", (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
    cv2.putText(img, "DONE", (420, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
    return img


@needs_tesseract
def test_chest_card_crop_reads_evolution_and_level_the_full_screen_misses():
    evo = _reward_frame("Soul Eater", 8)
    full = read_chest_text(evo)
    assert evolutions_from_chest_text(full) == set()
    card = menus.read_chest_card_text(evo)
    assert evolutions_from_chest_text(card) == {"Garlic"}

    wand = _reward_frame("Magic Wand", 2)
    assert levels_from_chest_text(read_chest_text(wand)) == {}
    assert levels_from_chest_text(menus.read_chest_card_text(wand)) == {"Magic Wand": 2}

    logs: list[str] = []
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append, async_chest=False)
    pilot.levels = {"Garlic": 8, "Pummarola": 1, "Magic Wand": 1}
    pilot._scan(evo)
    pilot._scan(evo)
    now = pilot.clock()
    pilot._next_action = 0
    assert pilot._handle_chest(evo, now) == "chest"
    assert pilot.evolved == {"Garlic"}
    assert pilot.levels["Garlic"] == 8

    pilot._scan(wand)
    pilot._scan(wand)
    pilot._next_action = 0
    assert pilot._handle_chest(wand, pilot.clock()) == "chest"
    assert pilot.levels["Magic Wand"] == 2


def test_levelup_card_that_shows_an_evolution_marks_the_base():
    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None)
    pilot.levels = {"Garlic": 8, "Pummarola": 1, "King Bible": 4}
    screen = menus.LevelUpScreen(
        panel=(0, 0, 10, 10),
        cards=[
            Card(0, 10, 0, 10, name="Soul Eater", new=False, level=8),
            Card(20, 30, 0, 10, name="King Bible", new=False, level=5),
        ],
        selected=None,
    )
    pilot._note_shown_cards(screen)
    assert pilot.evolved == {"Garlic"}
    assert pilot.levels["Garlic"] == 8
    assert pilot.levels["King Bible"] == 4  # gap of 1 is still the next-level tag

    pilot._names = ["Unholy Vespers"]
    pilot._chosen = "Unholy Vespers"
    pilot._pick_reason = "main"
    pilot._inventory_certain = True
    pilot._record_pick(0)
    assert "King Bible" in pilot.evolved
    assert "Unholy Vespers" not in pilot.levels
    assert pilot.levels["King Bible"] == 8
