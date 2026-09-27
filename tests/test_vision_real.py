"""Enemy / gem detection on real Mad Forest frames (HUD, player, pickups masked)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot.config import Config
from vs_vision_bot.vision import (
    HUD_RECTS,
    PLAYER_RECT,
    Vision,
    detect_gems,
    enemy_mask,
    threat_points,
)

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIX / name))
    assert frame is not None, name
    return frame


def _near(points: np.ndarray, x: float, y: float, r: float = 16.0) -> float:
    if len(points) == 0:
        return 0.0
    d = np.hypot(points[:, 0] - x, points[:, 1] - y)
    return float(points[d < r, 2].sum())


def _inside(points: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = rect
    return int(((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)).sum())


# Black bats hand-labelled on gameplay_bats_bushes.png (0:20, Poe on Mad Forest).
BATS = [(643, 100), (490, 125), (700, 165), (650, 185), (550, 258), (460, 70)]
BLUE_GEMS = [(605, 183), (592, 251), (386, 256)]


def test_black_bats_are_threats():
    pts = threat_points(enemy_mask(load("gameplay_bats_bushes.png")))
    for x, y in BATS:
        assert _near(pts, x, y) > 0.12, (x, y)


def test_hud_player_and_grass_are_not_threats():
    for name in ("gameplay_bats_bushes.png", "gameplay_hurt.png", "gameplay_skeleton_swarm.png"):
        pts = threat_points(enemy_mask(load(name)))
        for rect in HUD_RECTS:
            assert _inside(pts, rect) == 0, (name, rect)
        assert _inside(pts, PLAYER_RECT) == 0, name
    # Open grass right of the bush ring and the dirt path hold nothing.
    pts = threat_points(enemy_mask(load("gameplay_bats_bushes.png")))
    assert _inside(pts, (600, 300, 900, 430)) == 0
    assert _inside(pts, (180, 390, 560, 430)) == 0  # clovers sit left of x=180


def test_blue_gems_found_and_not_counted_as_threats():
    frame = load("gameplay_bats_bushes.png")
    mask = enemy_mask(frame)
    gems = detect_gems(frame, mask)
    for gx, gy in BLUE_GEMS:
        assert any(abs(gx - x) < 10 and abs(gy - y) < 12 for x, y in gems), (gx, gy)
    pts = threat_points(mask)
    for gx, gy in BLUE_GEMS:
        assert _near(pts, gx, gy, 8) < 0.08


def test_glowing_bat_is_an_enemy_not_a_gem():
    frame = load("gameplay_hurt.png")  # blue-outlined boss bat at ~(642, 350)
    mask = enemy_mask(frame)
    gems = detect_gems(frame, mask)
    assert not any(abs(x - 642) < 20 and abs(y - 350) < 20 for x, y in gems)
    assert _near(threat_points(mask), 642, 350, 20) > 0.12


def test_ghouls_detected_on_hurt_frame():
    pts = threat_points(enemy_mask(load("gameplay_hurt.png")))
    for x, y in [(645, 283), (555, 353), (537, 435), (512, 108), (240, 287)]:
        assert _near(pts, x, y, 20) > 0.12, (x, y)


def test_skeleton_swarm_density_is_up_top():
    pts = threat_points(enemy_mask(load("gameplay_skeleton_swarm.png")))
    top = pts[pts[:, 1] < 200, 2].sum()
    bottom = pts[pts[:, 1] > 330, 2].sum()
    assert top > 3 * bottom


def test_vision_analyze_fills_threats_and_gems():
    result = Vision(Config()).analyze(load("gameplay_bats_bushes.png"))
    assert result.threats is not None and len(result.threats) > 20
    assert len(result.gems) >= 3
    assert result.player == (480, 256)
    assert result.monsters


@pytest.mark.parametrize("name", ["main_menu.png", "results_screen.png"])
def test_non_gameplay_frames_do_not_crash(name):
    result = Vision(Config()).analyze(load(name))
    assert result.threats is not None


# Vegetation hand-checked on gameplay_bats_bushes.png: berry bushes, a bramble
# clump, a flower, clovers, and a small tree. Their near-black outlines must not
# register as threats (bats above still must).
VEGETATION = [
    (345, 269),  # bramble / bush clump left of player
    (222, 88),   # purple flower
    (272, 202),  # berry bush band
    (203, 353),  # berry bush
    (840, 176),  # small tree / leafy bush on the right
    (131, 377),  # clover
    (155, 404),  # clover
]


def test_bushes_and_trees_are_not_threats():
    pts = threat_points(enemy_mask(load("gameplay_bats_bushes.png")))
    for x, y in VEGETATION:
        assert _near(pts, x, y, 14) < 0.08, (x, y)
    # Broader bands that used to leak bush rim cells into the threat grid.
    assert _inside(pts, (150, 185, 450, 215)) == 0
    assert _inside(pts, (100, 360, 180, 420)) == 0
