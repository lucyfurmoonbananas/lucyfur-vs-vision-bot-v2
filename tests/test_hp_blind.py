"""A bar lost under a swarm must not leave nav on the last high HP, and a static
GAME OVER screen must not be what deaths/ keeps."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vs_vision_bot import menus
from vs_vision_bot.main import DeathTrace
from vs_vision_bot.pathfind import score_directions


def test_sustained_bar_loss_drops_stale_hp_and_nav_treats_it_as_empty():
    t = menus.HpTracker()
    for _ in range(4):
        t.add(0.89)
    t.add(None)
    assert t.value == pytest.approx(0.89)
    assert getattr(t, "blind", False) is False

    # run1: raw hp stayed none for ~4.6s and the median kept 0.92. Three misses drop it.
    t.add(None)
    t.add(None)
    assert t.value is None and t.blind is True
    assert menus.hp_for_nav(t.value, t.blind) == 0.0

    t.add(0.4)
    assert t.blind is False
    assert t.value == pytest.approx(0.4)
    assert menus.hp_for_nav(None, False) is None

    # Never seen a bar: still unknown, so nav does not invent an empty bar.
    fresh = menus.HpTracker()
    for _ in range(5):
        fresh.add(None)
    assert fresh.value is None and fresh.blind is False
    assert menus.hp_for_nav(fresh.value, fresh.blind) is None

    threat = np.array([[520.0, 270.0, 1.0]], np.float32)
    healthy = score_directions(threat, (480.0, 270.0), None, 0.0, hp=0.92)
    blind = score_directions(
        threat, (480.0, 270.0), None, 0.0, hp=menus.hp_for_nav(None, True),
    )
    assert blind.danger_here > healthy.danger_here


def _solid(value: int) -> np.ndarray:
    return np.full((24, 32, 3), value, np.uint8)


def _pixels(paths: list[str]) -> list[int]:
    return [int(cv2.imread(p)[0, 0, 0]) for p in paths]


def test_static_death_screen_saves_earlier_snaps(tmp_path):
    tr = DeathTrace(str(tmp_path), log=lambda _s: None)
    play = [_solid(v) for v in (30, 80, 140, 200)]
    screen = _solid(10)
    for i, fr in enumerate(play):
        tr.observe(i * 0.5, 0.89, 0.89, 9.0, "breakout", fr)
    # OCR sits on the revive screen; every immediate prior is that screen.
    for k in range(6):
        tr.observe(2.0 + k * 0.4, None, None, 3.57, "escape", screen)
    paths = tr.dump("revive", screen, 2.0 + 5 * 0.4, game_s=681.0)
    assert _pixels(paths) == [30, 80, 140, 200]


def test_snaps_older_than_the_keep_window_are_dropped(tmp_path):
    tr = DeathTrace(str(tmp_path), log=lambda _s: None)
    # 12s before the dump is outside the 8s snap window; 7s before it is inside.
    tr.observe(0.0, 0.9, 0.9, 1.0, "kite", _solid(200))
    tr.observe(5.0, 0.9, 0.9, 1.0, "kite", _solid(80))
    dump_t = 12.0
    for k in range(4):
        tr.observe(dump_t - 2.0 + k * 0.5, None, None, 3.57, "escape", _solid(15))
    paths = tr.dump("revive", _solid(15), dump_t, game_s=100.0)
    assert _pixels(paths) == [80]
