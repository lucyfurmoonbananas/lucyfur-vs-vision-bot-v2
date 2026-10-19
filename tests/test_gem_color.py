"""Gem colour: red (highest XP) beats green, green beats blue, distance still counts.

Blue detection is unchanged. Green and red are the same crystal with the hue
moved; anything that is not that crystal stays out of those classes, and a
missing colour scores as blue.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vs_vision_bot.pathfind import Tuning, format_gem_summary, gem_status, gem_target, gem_tune_for, score_directions
from vs_vision_bot.vision import detect_gems, enemy_mask

from tests.test_movement import Clock, result
from tests.test_vision_real import load
from vs_vision_bot.pathfind import Navigator

PX, PY = 480.0, 256.0
EMPTY = np.zeros((0, 3), np.float32)
# Weights the seek uses. Tests import nothing extra so a flat scorer fails them.
BLUE, GREEN, RED = 1.0, 1.7, 2.4
RADIUS, CLUSTER = 460.0, 70.0

# Bright rim of a real blue gem (gameplay_bats_bushes), area 17, 6x8.
_RIM = np.array(
    [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    np.uint8,
)

FIX = Path(__file__).parent / "fixtures"
_REAL = (
    "gameplay_bats_bushes.png",
    "gameplay_full_hp.png",
    "gameplay_hurt.png",
    "gameplay_skeleton_swarm.png",
    "gameplay_firewand_false_chest.png",
    "gameplay_flower_ring_menu_fp.png",
    "gameplay_flower_ring_menu_fp2.png",
    "chest_false/r1_t302_bottom_heart.png",
)


def _grass(w=960, h=540):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = (40, 90, 50)
    return frame


def _paint_rim(frame, cx, cy, bgr):
    y0, x0 = cy - 3, cx - 2
    patch = frame[y0 : y0 + _RIM.shape[0], x0 : x0 + _RIM.shape[1]]
    patch[_RIM.astype(bool)] = bgr


def _colored(frame, enemies=None):
    from vs_vision_bot.vision import detect_colored_gems

    return detect_colored_gems(frame, enemies)


def _near(gems, x, y, color, tol=8):
    return any(c == color and abs(gx - x) <= tol and abs(gy - y) <= tol for gx, gy, c in gems)


def test_red_at_moderate_distance_beats_a_near_blue():
    # d=40 blue scores 1/1.267; d=180 red scores 2.4/2.2. Red wins.
    # A red at d=400 scores 2.4/3.67 and loses to the blue: distance still counts.
    gems = [(PX + 40, PY, BLUE), (PX + 180, PY, RED), (PX - 400, PY, RED)]
    assert gem_target(gems, PX, PY, RADIUS, CLUSTER) == (PX + 180, PY)


def test_green_beats_blue_and_red_beats_green_when_close():
    # Opposite sides so they are not one cluster (cluster_r is 70).
    green_blue = [(PX + 40, PY, BLUE), (PX - 100, PY, GREEN)]
    assert gem_target(green_blue, PX, PY, RADIUS, CLUSTER) == (PX - 100, PY)
    # Similar distance: d=110 green vs d=130 red. Equal weights would take green.
    red_green = [(PX - 110, PY, GREEN), (PX + 130, PY, RED)]
    assert gem_target(red_green, PX, PY, RADIUS, CLUSTER) == (PX + 130, PY)


def test_danger_and_low_hp_still_veto_a_red_gem():
    early = gem_tune_for(Tuning(), 120.0)
    gems = [(PX - 40, PY, BLUE), (PX + 180, PY, RED)]
    calm = score_directions(EMPTY, (PX, PY), None, 0.0, gems=gems, tune=early)
    assert calm.vector[0] == 1  # the red, to the east
    crowd = np.array(
        [[PX + dx, PY + dy, 1.0] for dx in (-12, 0, 12) for dy in (-12, 0, 12)],
        np.float32,
    )
    hot = score_directions(crowd, (PX, PY), None, 0.0, gems=gems, tune=early)
    off = score_directions(crowd, (PX, PY), None, 0.0, gems=None, tune=early)
    assert hot.danger_here >= early.gem_danger
    assert hot.scores == off.scores
    lo = score_directions(EMPTY, (PX, PY), None, 0.0, gems=gems, hp=0.3, tune=early)
    lo_off = score_directions(EMPTY, (PX, PY), None, 0.0, gems=None, hp=0.3, tune=early)
    assert lo.scores == lo_off.scores


def test_blue_only_scores_match_uncoloured_gems():
    early = gem_tune_for(Tuning(), 120.0)
    plain = [(PX - 60, PY), (PX + 400, PY), (PX + 410, PY)]
    blue = [(x, y, BLUE) for x, y in plain]
    assert gem_target(plain, PX, PY, RADIUS, CLUSTER) == gem_target(blue, PX, PY, RADIUS, CLUSTER)
    a = score_directions(EMPTY, (PX, PY), None, 0.0, gems=plain, tune=early)
    b = score_directions(EMPTY, (PX, PY), None, 0.0, gems=blue, tune=early)
    assert a.scores == b.scores and a.vector == b.vector

    clock_a, clock_b = Clock(), Clock()
    nav_a = Navigator(clock=clock_a, log=lambda *_: None)
    nav_b = Navigator(clock=clock_b, log=lambda *_: None)
    ra = result([], gems=[(700.0, 256.0), (330.0, 200.0)])
    rb = result([], gems=[(700.0, 256.0), (330.0, 200.0)])
    rb.gem_colors = ["blue", "blue"]
    for _ in range(6):
        clock_a.t += 1 / 12
        clock_b.t += 1 / 12
        assert nav_a.step(ra, hp=1.0).vector == nav_b.step(rb, hp=1.0).vector
    assert nav_b.gem_blue == 1 and nav_b.gem_red == 0 and nav_b.gem_green == 0
    assert gem_status(nav_b) == "gem_seek=1 gem_pickups=0 gem_target=blue"
    assert format_gem_summary(nav_b) == "[gems] seeks=1 pickups=0 game~0s 0.0/min red=0 green=0 blue=1"


def test_seek_logs_red_and_counts_it():
    from vs_vision_bot.pathfind import gem_value

    assert gem_value("nope") == gem_value(None) == gem_value("blue")
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    res = result([], gems=[(PX + 40, PY), (PX + 180, PY)])
    res.gem_colors = ["blue", "red"]
    clock.t += 1 / 12
    nav.step(res, hp=1.0)
    assert nav.gem_target_color == "red"
    assert nav.gem_red == 1 and nav.gem_green == 0 and nav.gem_blue == 0
    assert "gem_target=red" in gem_status(nav)
    assert format_gem_summary(nav).endswith("red=1 green=0 blue=0")


def test_synthetic_red_and_green_rims_are_classified():
    frame = _grass()
    # Real red crystal, Oct 19 frames: BGR (156, 38, 225), not pure red.
    _paint_rim(frame, 300, 300, (156, 38, 225))
    _paint_rim(frame, 700, 300, (38, 228, 38))  # green rim
    gems = _colored(frame, enemy_mask(frame))
    assert _near(gems, 300, 300, "red")
    assert _near(gems, 700, 300, "green")
    assert not any(c == "blue" for *_, c in gems)


def test_recolored_real_gem_is_found_and_blue_mask_ignores_it():
    frame = load("gameplay_bats_bushes.png")
    y, x = 249, 591
    win = frame[y - 16 : y + 16, x - 16 : x + 16].copy()
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    sel = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 110) & (hsv[:, :, 1] > 80)
    # Blue gems sit at OpenCV hue ~98. The real red crystal is about +63.
    hsv[:, :, 0] = np.where(sel, (hsv[:, :, 0].astype(np.int16) + 63) % 180, hsv[:, :, 0]).astype(np.uint8)
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    red = frame.copy()
    red[y - 16 : y + 16, x - 16 : x + 16] = np.where(sel[:, :, None], shifted, win)
    colored = _colored(red, enemy_mask(red))
    assert _near(colored, 591, 249, "red")
    assert _near(colored, 605, 183, "blue") and _near(colored, 386, 256, "blue")
    assert not any(c == "green" for *_, c in colored)
    # The blue mask does not claim the recolored crystal.
    blues = detect_gems(red, enemy_mask(red))
    assert not any(abs(gx - 591) < 12 and abs(gy - 249) < 12 for gx, gy in blues)


def _assert_only_crystal_reds(shot, name: str) -> None:
    found = _colored(shot, enemy_mask(shot))
    assert not any(c == "green" for *_, c in found), name
    h, w = shot.shape[:2]
    for x, y, color in found:
        if color != "red":
            continue
        # The centroid can sit in the dark hole. A rim pixel next to it is the crystal.
        y0, y1 = max(0, y - 4), min(h, y + 5)
        x0, x1 = max(0, x - 4), min(w, x + 5)
        win = shot[y0:y1, x0:x1]
        bb = win[:, :, 0].astype(np.int16)
        gg = win[:, :, 1].astype(np.int16)
        rr = win[:, :, 2].astype(np.int16)
        crystal = (rr >= 180) & (bb >= 100) & (bb <= 220) & (gg <= 100)
        assert crystal.any(), (name, x, y)


def test_heart_foliage_chicken_blood_and_red_sprites_are_not_gems():
    frame = _grass()
    # Winged-heart red, copied from the real pickup body and rim.
    cv2.circle(frame, (200, 400), 8, (0, 38, 165), -1)
    cv2.circle(frame, (200, 400), 8, (10, 69, 254), 1)
    # Floor Chicken: orange/gold body, not a pure rim.
    cv2.ellipse(frame, (400, 400), (16, 12), 0, 0, 360, (40, 160, 220), -1)
    # Mad Forest leaf highlight and a 2 px pure-green speck.
    frame[120:140, 200:230] = (107, 242, 79)
    frame[150:152, 240:242] = (46, 235, 46)
    # Blood splatter and a thin damage streak.
    cv2.circle(frame, (650, 200), 14, (20, 20, 160), -1)
    frame[220:223, 640:680] = (0, 0, 254)
    # Red enemy: dark body, red edge.
    cv2.circle(frame, (800, 300), 18, (8, 8, 12), -1)
    cv2.circle(frame, (800, 300), 18, (20, 20, 200), 2)
    gems = _colored(frame, enemy_mask(frame))
    assert gems == []

    # Older gameplay shots do contain the red crystal (same BGR as the Oct 19
    # frames). They must not contain a green gem or a pure-red sprite.
    for name in _REAL:
        shot = cv2.imread(str(FIX / name))
        assert shot is not None, name
        _assert_only_crystal_reds(shot, name)
    for path in sorted((FIX / "chest_real").glob("*.png")):
        shot = cv2.imread(str(path))
        _assert_only_crystal_reds(shot, path.name)


def test_oct19_frames_keep_red_crystals_and_drop_enemies():
    """Poe on Mad Forest, 19 Oct. Green hits in the runner notes are Mantichana.

    frame_g140405 has a Red Death-like sprite on the player; none of it is a gem.
    Red crystals are BGR about (156, 38, 225). There is no green crystal in these five.
    """
    root = FIX / "gems"
    expect = {
        "frame_g134558.png": {"blue": 14, "green": 0, "red": 20},
        "frame_g134659.png": {"blue": 0, "green": 0, "red": 0},
        "frame_g140204.png": {"blue": 5, "green": 0, "red": 15},
        "frame_g140304.png": {"blue": 3, "green": 0, "red": 0},
        "frame_g140405.png": {"blue": 3, "green": 0, "red": 0},
    }
    for name, counts in expect.items():
        shot = cv2.imread(str(root / name))
        assert shot is not None and shot.shape[:2] == (540, 960), name
        found = _colored(shot, enemy_mask(shot))
        got = {c: sum(col == c for *_, col in found) for c in ("blue", "green", "red")}
        assert got == counts, (name, got)
        _assert_only_crystal_reds(shot, name)
    # A known crystal, and the Red Death wing that used to pass the pure-red gate.
    first = cv2.imread(str(root / "frame_g134558.png"))
    assert _near(_colored(first, enemy_mask(first)), 293, 272, "red", tol=12)
    death = cv2.imread(str(root / "frame_g140405.png"))
    found = _colored(death, enemy_mask(death))
    assert not any(c == "red" and abs(x - 469) < 40 and abs(y - 208) < 40 for x, y, c in found)
    assert not any(c == "red" and abs(x - 480) < 50 and abs(y - 245) < 50 for x, y, c in found)


def test_real_blue_gems_keep_their_centers():
    frame = load("gameplay_bats_bushes.png")
    plain = detect_gems(frame, enemy_mask(frame))
    colored = _colored(frame, enemy_mask(frame))
    assert [(x, y) for x, y, c in colored if c == "blue"] == plain
    assert all(c == "blue" for *_, c in colored)
