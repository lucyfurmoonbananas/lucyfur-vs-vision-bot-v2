"""Real Mad Forest frames: pale-yellow chest arrows and on-screen chests.

The saturated-gold detector (b < 80, 56 px frame-edge band) misses every frame
in tests/fixtures/chest_real. Coordinates are the 960x540 centroids from the
2026-10-18 session notes. An arrow's seek point has to land on the side the
arrow points; a chest has to land on the wooden body, not the chevron above it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from vs_vision_bot.vision import detect_chest, detect_chest_arrow

FIX = Path(__file__).parent / "fixtures"
REAL = FIX / "chest_real"

# (filename, labeled centroid, which side of the player the seek point must be on)
ARROWS = [
    ("REAL_arrow_bottom_run1_g0517_20261018_030909.png", (361, 498), "bottom"),
    ("REAL_arrow_bottomright_run1_20261018_031013.png", (674, 475), "bottomright"),
    ("REAL_arrow_bottom_run1_late_20261018_031848.png", (605, 496), "bottom"),
    ("REAL_arrow_bottomleft_run4_20261018_034423.png", (214, 444), "bottomleft"),
    ("REAL_arrow_left_run4_20261018_034441.png", (158, 395), "left"),
    ("REAL_arrow_right_run4_20261018_034922.png", (862, 251), "right"),
    ("REAL_arrow_bottomleft_run6_20261018_051142.png", (294, 481), "bottomleft"),
    ("REAL_arrow_right_run6_20261018_051350.png", (862, 237), "right"),
    ("REAL_arrow_left_run6_20261018_051535.png", (126, 173), "left"),
    ("REAL_arrow_top_run6_20261018_051602.png", (664, 58), "top"),
    ("REAL_arrow_top_run6_b_20261018_051910.png", (698, 71), "top"),
]

# Chest body, not the chevron. g0416 only labels the marker (579, 175); the body
# sits ~28 px under it, the same offset as the frames that label both.
CHESTS = [
    ("REAL_chest_onscreen_run2_g0416_20261018_032441.png", (579, 203)),
    ("REAL_chest_onscreen_run2_g0421_20261018_032446.png", (548, 218)),
    ("REAL_chest_onscreen_run2_g0609_20261018_032908.png", (365, 405)),
    ("REAL_chest_onscreen_run2_g0640_20261018_032942.png", (365, 405)),
    ("REAL_chest_onscreen_run6_20261018_051053.png", (532, 125)),
]


def _load(path: Path):
    frame = cv2.imread(str(path))
    assert frame is not None and frame.shape[:2] == (540, 960), path.name
    return frame


def _on_side(xy: tuple[int, int], side: str) -> bool:
    x, y = xy
    if side == "left":
        return x < 220
    if side == "right":
        return x > 780
    if side == "top":
        return y < 130
    if side == "bottom":
        return y > 430
    if side == "bottomleft":
        return x < 360 and y > 400
    if side == "bottomright":
        return x > 600 and y > 420
    raise AssertionError(side)


@pytest.mark.parametrize("name,expected,side", ARROWS, ids=[a[2] + "-" + a[0][11:22] for a in ARROWS])
def test_real_arrow_points_the_labeled_way(name, expected, side):
    frame = _load(REAL / name)
    arrow = detect_chest_arrow(frame)
    assert arrow is not None, name
    assert _on_side(arrow, side), (name, side, arrow)
    assert abs(arrow[0] - expected[0]) < 40 and abs(arrow[1] - expected[1]) < 40, (name, arrow, expected)
    # The off-screen arrow has no wooden body. A chest hit here would steal the seek.
    assert detect_chest(frame) is None, name


@pytest.mark.parametrize("name,expected", CHESTS, ids=[c[0][18:28] for c in CHESTS])
def test_real_chest_is_near_the_labeled_body(name, expected):
    frame = _load(REAL / name)
    chest = detect_chest(frame)
    assert chest is not None, name
    assert abs(chest[0] - expected[0]) < 24 and abs(chest[1] - expected[1]) < 24, (name, chest, expected)
    # The on-screen chevron is the marker above the chest, not an edge arrow.
    assert detect_chest_arrow(frame) is None, name


def test_existing_fixtures_and_firewand_are_not_chests():
    """Saturated projectiles and the old gameplay set still return nothing."""
    for path in sorted(FIX.glob("*.png")):
        frame = cv2.imread(str(path))
        assert frame is not None
        assert detect_chest(frame) is None, path.name
        assert detect_chest_arrow(frame) is None, path.name
