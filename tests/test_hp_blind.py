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


def _red_sprite_over_hidden_bar() -> np.ndarray:
    """Anchored bar hidden; a pure-red run in the fallback search window.

    death_quit_053751_02_06 and death_revive_053457_01_05: the bar rect is
    occluded (read_hp_bar is None) and read_hp_fraction returns ~1.0 from a
    red sprite / damage flash above the bar.
    """
    frame = np.full((540, 960, 3), (40, 90, 50), np.uint8)
    frame[260:268, 500:539] = (0, 0, 254)
    return frame


def test_upward_hp_spike_cannot_cancel_low_or_blind():
    """run6 quit trace: 0.00, 0.80, 0.53, 1.00, 0.06 while the median was ~0.03."""
    spike = menus.read_hp_fraction(_red_sprite_over_hidden_bar())
    assert spike == pytest.approx(1.0)

    t = menus.HpTracker()
    for _ in range(5):
        t.add(0.03)
    for v in (0.00, 0.80, 0.53, spike, 0.06):
        t.add(v)
    assert t.value < 0.2

    # One high frame is not a heal.
    healed = menus.HpTracker()
    for _ in range(4):
        healed.add(0.12)
    healed.add(0.55)
    assert healed.value < 0.3
    healed.add(0.55)
    healed.add(0.52)
    assert healed.value == pytest.approx(0.55, abs=0.05)

    for _ in range(3):
        t.add(None)
    assert t.blind is True
    t.add(spike)
    assert t.blind is True
    assert menus.hp_for_nav(t.value, t.blind) == 0.0


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
