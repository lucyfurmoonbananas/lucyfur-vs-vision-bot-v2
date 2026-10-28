"""Green ghouls / mantises on Mad Forest (Oct 27 2026): their thin dark outlines
sit on a lime body, so the bush lime-band rule used to drop them as hedge
leaves. A cool, even green body (red ~ blue) marks them as enemies."""
from pathlib import Path

import cv2
import numpy as np

from vs_vision_bot import vision as V
from vs_vision_bot.vision import enemy_mask, threat_points

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    f = cv2.imread(str(FIX / name))
    assert f is not None, name
    return f


def _near(pts, x, y, r=20):
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    return float(pts[d < r, 2].sum()) if len(pts) else 0.0


# Hand-picked green ghouls (6:11, Poe on Mad Forest, open grass).
OPEN_GHOULS = [(383, 152), (295, 252), (432, 248), (617, 168), (845, 152), (850, 355), (703, 435)]
# Green ghouls among the Man-eater flower ring (5:14). The three close to Poe
# sit on the ring edge; (465, 220) is the outline the chroma rule used to drop.
RING_GHOULS = [
    (545, 165), (897, 125), (850, 460), (610, 140),
    (465, 220), (550, 210), (740, 210),
]


def test_green_ghouls_on_open_grass_are_threats():
    pts = threat_points(enemy_mask(_load("gameplay_green_ghouls_open.png")))
    hits = sum(_near(pts, x, y) > 0.12 for x, y in OPEN_GHOULS)
    assert hits >= len(OPEN_GHOULS) - 1, hits


def test_green_ghouls_in_flower_ring_are_threats():
    pts = threat_points(enemy_mask(_load("gameplay_green_ghouls_flower_ring.png")))
    for x, y in RING_GHOULS:
        assert _near(pts, x, y) > 0.12, (x, y)


def test_rule_adds_ghouls_without_turning_hedges_into_threats():
    frame = _load("gameplay_bats_bushes.png")
    old = V.GHOUL_GREEN_MIN
    try:
        V.GHOUL_GREEN_MIN = 10**9
        before = enemy_mask(frame)
    finally:
        V.GHOUL_GREEN_MIN = old
    after = enemy_mask(frame)
    # The hedge ring (140..450, 140..360) and the fern gain nothing.
    added = after & ~before
    assert int(added[140:360, 140:450].sum()) == 0
    assert int(added[410:450, 555:600].sum()) == 0
    ghouls = _load("gameplay_green_ghouls_open.png")
    try:
        V.GHOUL_GREEN_MIN = 10**9
        b2 = enemy_mask(ghouls)
    finally:
        V.GHOUL_GREEN_MIN = old
    assert int(enemy_mask(ghouls).sum()) > int(b2.sum()) + 500
