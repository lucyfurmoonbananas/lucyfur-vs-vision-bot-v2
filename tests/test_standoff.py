"""Keep a standoff from bodies Garlic cannot one-shot, and from a closing ring.

Weak packs (skeletons, bats) are still dived, including in the skeleton
windows. A mixed pack is dived only on the side that has no tanky body.
test18 died farming a ring that was already closing, and a blind bar was
passed to nav as 0.00 for the rest of the escape. A revive must not walk
back into that same ring.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vs_vision_bot.menus import HpTracker, decay_blind_hp, hp_for_nav
from vs_vision_bot.pathfind import Navigator, score_directions
from vs_vision_bot.vision import GarlicTicks, enemy_mask, tag_threat_points, threat_points

from tests.test_movement import Clock, P, blob, result, ring, run


def _tagged(points, kind: float) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    flag = np.full((arr.shape[0], 1), kind, np.float32)
    return np.hstack([arr, flag])


def test_weak_pack_is_dived():
    pts = _tagged(blob(P[0] + 160, P[1], n=12, spread=16, seed=1), 0.0)
    dec = score_directions(pts, P, None, 0.0, hp=1.0)
    assert dec.vector[0] > 0


def test_tanky_pack_holds_a_standoff():
    pts = _tagged(blob(P[0] + 70, P[1], n=10, spread=8, seed=2), 1.0)
    dec = score_directions(pts, P, None, 0.0, hp=1.0)
    assert dec.vector[0] <= 0


def test_boss_holds_a_wider_standoff():
    pts = _tagged(blob(P[0] + 110, P[1], n=8, spread=10, seed=3), 2.0)
    dec = score_directions(pts, P, None, 0.0, hp=1.0)
    assert dec.vector[0] <= 0
    weak = _tagged(blob(P[0] + 110, P[1], n=8, spread=10, seed=3), 0.0)
    assert score_directions(weak, P, None, 0.0, hp=1.0).vector[0] > 0


def test_mixed_pack_dives_only_where_the_tanky_bodies_are_not():
    weak = _tagged(blob(P[0] + 170, P[1], n=10, spread=12, seed=4), 0.0)
    tank = _tagged(blob(P[0] - 60, P[1], n=8, spread=8, seed=5), 1.0)
    dec = score_directions(np.vstack([weak, tank]), P, None, 0.0, hp=1.0)
    assert dec.vector[0] > 0
    blocked = np.vstack([
        _tagged(blob(P[0] + 170, P[1], n=8, spread=8, seed=6), 0.0),
        _tagged(blob(P[0] + 55, P[1], n=8, spread=6, seed=7), 1.0),
    ])
    assert score_directions(blocked, P, None, 0.0, hp=1.0).vector[0] <= 0


def test_skeleton_window_still_dives_a_weak_pack():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    res = result([])
    res.threats = _tagged(blob(P[0] + 160, P[1], n=12, spread=14, seed=8), 0.0)
    vecs = run(nav, clock, res, 8, hp=1.0)
    assert all(v[0] > 0 for v in vecs)
    assert nav.breakouts == 0


def test_thickening_ring_is_left_through_the_gap():
    """test18 runs 1 and 2 farmed a closing crowd between 5:00 and 6:30."""
    pts = ring(P[0], P[1], 80, 40, gap_deg=(160, 200))
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 330.0
    vecs = run(nav, clock, result(pts), 16, hp=1.0)
    assert sum(1 for v in vecs[-6:] if v[0] < 0) >= 4


def test_skeleton_window_closed_ring_is_not_dived():
    """test18 run 3, about 3:21: cover 0.88, still farming, then the bar crashed.

    A one-sided skeleton pack is the other test. This ring is already around Poe.
    """
    pts = ring(P[0], P[1], 70, 36, gap_deg=(160, 200))
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    vecs = run(nav, clock, result(pts), 16, hp=1.0)
    assert nav.breakouts == 0
    assert sum(1 for v in vecs[-6:] if v[0] < 0) >= 3


def test_blind_bar_decays_toward_the_kite_line_instead_of_zero():
    t = HpTracker()
    for _ in range(4):
        t.add(0.80)
    for _ in range(3):
        t.add(None)
    assert t.blind is True
    fresh = hp_for_nav(t.value, t.blind, trusted=t._trusted, blind_s=0.0)
    assert fresh == pytest.approx(0.80)
    late = decay_blind_hp(t._trusted, 6.0)
    assert late == pytest.approx(0.25)
    assert late > 0.20
    # Already critical: the decay does not heal it.
    assert decay_blind_hp(0.10, 6.0) == pytest.approx(0.10)


def test_revive_does_not_walk_back_into_the_same_ring():
    crowd = ring(P[0], P[1], 90, 28, gap_deg=(150, 210))
    crowd += blob(P[0] + 70, P[1], n=16, spread=18, seed=9)
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    run(nav, clock, result(crowd), 8, hp=1.0)
    assert nav._into_crowd is not None and nav._into_crowd[0] > 0.2
    nav.request_revive_escape()
    vecs = run(nav, clock, result(crowd), 6, hp=0.50)
    assert all(v[0] <= 0 for v in vecs)


def test_sprite_color_and_size_classify_tanky_and_boss():
    frame = np.full((540, 960, 3), (40, 80, 50), np.uint8)
    cv2.circle(frame, (200, 200), 14, (12, 12, 16), -1)
    cv2.circle(frame, (400, 200), 18, (20, 20, 20), -1)
    cv2.circle(frame, (400, 200), 14, (70, 180, 80), -1)
    cv2.circle(frame, (600, 200), 16, (12, 12, 14), -1)
    cv2.circle(frame, (600, 200), 11, (100, 45, 35), -1)
    cv2.circle(frame, (760, 320), 48, (18, 18, 22), -1)
    mask = enemy_mask(frame)
    pts = tag_threat_points(threat_points(mask), frame, mask)

    def kind_near(x, y) -> float:
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        close = d < 36
        return float(pts[close, 3].max()) if np.any(close) else -1.0

    assert kind_near(200, 200) == 0.0
    assert kind_near(400, 200) >= 1.0
    assert kind_near(600, 200) >= 1.0
    assert kind_near(760, 320) == 2.0


def test_garlic_ticks_mark_a_body_that_survives_two_ticks():
    player = (480.0, 256.0)
    near = [(500.0, 256.0)]
    ticks = GarlicTicks()
    assert ticks.update(near, player, 0.4) == [False]
    assert ticks.update(near, player, 0.4) == [False]
    assert ticks.update(near, player, 0.4) == [True]
    far = GarlicTicks()
    assert far.update([(700.0, 256.0)], player, 2.0) == [False]
