"""Treasure chests outrank gems. Escapes still win when danger is high.

No fixture in tests/fixtures shows a chest or the yellow off-screen arrow
(checked against every captured frame). Detection is pinned to the game's
look: a solid yellow edge triangle, and a brown chest with a gold band.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.pathfind import Navigator, Tuning, tuning_from_env
from vs_vision_bot.vision import Vision, detect_chest, detect_chest_arrow, draw_synthetic_arena

from tests.test_movement import Clock, blob, result, ring, run

FIX = Path(__file__).parent / "fixtures"
# Gem to the left, chest / arrow to the right: they disagree, so the winner is obvious.
GEM_LEFT = [(300.0, 256.0)]
CHEST_RIGHT = (720.0, 256.0)
ARROW_RIGHT = (900.0, 256.0)


def _paint_arrow(frame: np.ndarray, cx: int, cy: int, size: int = 28) -> None:
    """Yellow edge arrow, tip to the right. Fill matches in-game coin yellow (b ~ 30)."""
    tip = (cx + int(size * 0.55), cy)
    b1 = (cx - int(size * 0.35), cy - int(size * 0.5))
    b2 = (cx - int(size * 0.35), cy + int(size * 0.5))
    pts = np.array([tip, b1, b2], np.int32)
    cv2.fillConvexPoly(frame, pts, (32, 205, 250))
    cv2.polylines(frame, [pts], True, (8, 16, 20), 1)


def _paint_chest(frame: np.ndarray, cx: int, cy: int) -> None:
    """Wooden chest: brown body, gold band, gold lock. About 30x26 at 960x540."""
    x0, y0 = cx - 15, cy - 13
    cv2.rectangle(frame, (x0, y0), (x0 + 30, y0 + 26), (18, 32, 48), -1)
    cv2.rectangle(frame, (x0 + 2, y0 + 2), (x0 + 28, y0 + 12), (42, 96, 158), -1)
    cv2.rectangle(frame, (x0 + 2, y0 + 14), (x0 + 28, y0 + 24), (36, 82, 140), -1)
    cv2.rectangle(frame, (x0 + 2, y0 + 11), (x0 + 28, y0 + 16), (36, 186, 236), -1)
    cv2.rectangle(frame, (cx - 3, cy - 4), (cx + 3, cy + 5), (40, 210, 252), -1)


def test_fixtures_have_no_chest_and_detector_finds_a_painted_one():
    for path in sorted(FIX.glob("*.png")):
        frame = cv2.imread(str(path))
        assert frame is not None
        assert detect_chest(frame) is None, path.name
        assert detect_chest_arrow(frame) is None, path.name

    arrow_frame = draw_synthetic_arena(960, 540, monsters=[])
    _paint_arrow(arrow_frame, 930, 270)
    arrow = detect_chest_arrow(arrow_frame)
    assert arrow is not None and arrow[0] > 880 and abs(arrow[1] - 270) < 20
    assert detect_chest(arrow_frame) is None

    chest_frame = draw_synthetic_arena(960, 540, monsters=[])
    _paint_chest(chest_frame, 640, 380)
    chest = detect_chest(chest_frame)
    assert chest is not None
    assert abs(chest[0] - 640) < 12 and abs(chest[1] - 380) < 12
    assert detect_chest_arrow(chest_frame) is None
    # A gold coin sitting on brown is not a chest (real coins are ~8 px).
    coin = draw_synthetic_arena(960, 540, monsters=[])
    cv2.rectangle(coin, (200, 200), (236, 236), (40, 90, 150), -1)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx * dx + dy * dy <= 8:
                coin[220 + dy, 218 + dx] = (48, 198, 254)
    assert detect_chest(coin) is None

    found = Vision(Config()).analyze(chest_frame)
    assert found.chest is not None
    assert abs(found.chest[0] - 640) < 12 and abs(found.chest[1] - 380) < 12
    assert found.chest_arrow is None


def test_navigator_prefers_chest_over_gem_target_when_safe():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    offscreen = result([], gems=GEM_LEFT)
    offscreen.chest_arrow = ARROW_RIGHT
    assert run(nav, clock, offscreen, 6, hp=1.0) == [(1, 0)] * 6
    assert nav.last.mode.startswith("chest")

    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    onscreen = result([], gems=GEM_LEFT)
    onscreen.chest = CHEST_RIGHT
    assert run(nav, clock, onscreen, 4, hp=1.0)[-1] == (1, 0)

    # Tuning toggle: chest_gain 0 (VS_TUNE=chest_gain=0) leaves the gem target in charge.
    clock = Clock()
    off = tuning_from_env("chest_gain=0", log=lambda *_: None)
    nav = Navigator(tune=off, clock=clock, log=lambda *_: None)
    assert nav.tune.chest_gain == 0.0
    assert run(nav, clock, onscreen, 4, hp=1.0)[-1][0] == -1


def test_high_danger_escapes_then_resumes_chest():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    # Crowd and chest both to the right. Danger here is above chest_danger.
    crowd = result(blob(600, 256, n=16, spread=16), gems=GEM_LEFT)
    crowd.chest = CHEST_RIGHT
    vecs = run(nav, clock, crowd, 8, hp=1.0)
    assert nav.last.danger_here >= Tuning().chest_danger
    assert nav.last.mode.startswith("escape")
    assert all(v[0] < 0 for v in vecs)

    # HP-crash flee (same path as a flower-ring survival flee) ignores the chest.
    run(nav, clock, crowd, 2, hp=1.0)
    flee = run(nav, clock, crowd, 6, hp=0.65)
    assert nav._hp_crashing and nav.breakout is not None
    assert nav.last.mode.startswith("breakout")
    assert all(v[0] < 0 for v in flee)

    # Flower-window ring escape goes through the gap, not toward a chest behind it.
    clock_r = Clock()
    ring_nav = Navigator(clock=clock_r, log=lambda *_: None)
    ring_nav.game_s = 318.0
    closing = result(ring(480, 256, 170, 40, gap_deg=(-1, 25)))  # gap to the right
    closing.chest = (200.0, 256.0)  # chest back into the ring
    out = run(ring_nav, clock_r, closing, 8, hp=1.0)
    assert ring_nav.escaping and ring_nav.last.mode.startswith("ring-escape")
    assert all(v[0] == 1 for v in out)

    # Once the crash flee ends and the field is clear, chest-seeking resumes.
    for _ in range(80):
        clock.t += 1 / 12
        nav.step(crowd, hp=0.7)
        if nav.breakout is None and nav._flee_remaining <= 0 and not nav.escaping:
            break
    assert nav.breakout is None and nav._flee_remaining == 0.0
    safe = result([], gems=GEM_LEFT)
    safe.chest = CHEST_RIGHT
    resumed = run(nav, clock, safe, 6, hp=0.95)
    assert resumed[-1] == (1, 0)
    assert nav.last.mode.startswith("chest")
