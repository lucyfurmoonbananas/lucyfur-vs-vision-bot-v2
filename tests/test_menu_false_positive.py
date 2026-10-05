"""Heuristic menu detector must not freeze the bot inside a Man-eater flower ring.

Oct 4 2026 blackboxes: in ~60% of the gameplay frames saved before a death
(168 of 267 death events had >=50% of them), ``detect_level_up_menu`` said
"menu", so main.py released WASD and the character stood still in the ring
(log: ``[revive+1.5s] v=(0, 0) keys=- held=- menu=- heuristic``). It also
paused ``Navigator.game_s`` (game~309s while the HUD read 05:21).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot.config import Config
from vs_vision_bot.vision import (
    MENU_GRASS_MAX,
    Vision,
    detect_level_up_menu,
    draw_synthetic_arena,
    playfield_grass_fraction,
)

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIX / name))
    assert frame is not None, name
    return frame


def center(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[h // 5 : 4 * h // 5, w // 8 : 7 * w // 8]


@pytest.mark.parametrize(
    "name",
    [
        "gameplay_flower_ring_menu_fp.png",
        "gameplay_flower_ring_menu_fp2.png",
        "gameplay_full_hp.png",
        "gameplay_hurt.png",
        "gameplay_skeleton_swarm.png",
        "gameplay_bats_bushes.png",
    ],
)
def test_flower_ring_gameplay_is_not_a_menu(name):
    frame = load(name)
    assert playfield_grass_fraction(center(frame)) >= MENU_GRASS_MAX
    assert detect_level_up_menu(frame) is False


def test_vision_keeps_threats_and_moves_in_flower_ring():
    res = Vision(Config()).analyze(load("gameplay_flower_ring_menu_fp.png"))
    assert res.menu is False
    assert res.threats is not None and len(res.threats) > 50


@pytest.mark.parametrize(
    "name",
    [
        "levelup_4cards_sel4.png",
        "game_over_revive.png",
        "game_over_quit.png",
        "game_over_after_flower_ring.png",
        "main_menu.png",
        "results_screen.png",
    ],
)
def test_menu_screens_show_little_grass(name):
    assert playfield_grass_fraction(center(load(name))) < MENU_GRASS_MAX


def test_game_over_after_ring_still_flagged():
    # Dim GAME OVER overlay: the gate must not hide real overlays.
    assert detect_level_up_menu(load("game_over_after_flower_ring.png")) is True


def test_synthetic_cards_over_grass_edge_cases():
    # Cards on a dark overlay (no grass) are still a menu.
    frame = np.full((540, 960, 3), 18, dtype=np.uint8)
    for x in (180, 400, 620):
        frame[140:400, x : x + 160] = (70, 90, 120)
    assert playfield_grass_fraction(center(frame)) == 0.0
    assert detect_level_up_menu(frame) is True
    # Same card shapes drawn over open grass are treated as gameplay.
    grass = draw_synthetic_arena(960, 540, monsters=[])
    for x in (180, 400, 620):
        cv2.rectangle(grass, (x, 140), (x + 160, 400), (10, 10, 10), 6)
    assert detect_level_up_menu(grass) is False


def test_grass_fraction_empty_region():
    assert playfield_grass_fraction(np.zeros((0, 0, 3), np.uint8)) == 0.0
