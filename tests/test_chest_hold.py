"""Chest/arrow seeks must survive detection misses at the live frame rate.

test3 on 3c8c971 logged 205 [chest] seek/clear pairs in one 10:52 run; 168 clears
came right after their seek with no [nav] tick between, and no chest was ever
opened. Live frames are ~0.2 s apart, so chest_hold_s=0.10 never held a single
missed frame. These tests step at 5 fps.
"""

from __future__ import annotations

from vs_vision_bot.pathfind import Navigator, Tuning, tuning_from_env

from tests.test_movement import Clock, result

ARROW_DOWN = (500.0, 510.0)
LIVE_DT = 0.2


def _step(nav, clock, arrow):
    clock.t += LIVE_DT
    res = result([], gems=[])
    res.chest_arrow = arrow
    return nav.step(res, hp=1.0)


def test_arrow_seek_survives_flickering_detection_at_live_fps():
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    nav.game_s = 300.0
    vecs = []
    # Detected on every other frame, like the pale arrow at the HUD edge.
    for i in range(20):
        dec = _step(nav, clock, ARROW_DOWN if i % 2 == 0 else None)
        vecs.append(dec.vector)
    seeks = [ln for ln in lines if ln.startswith("[chest] seek")]
    clears = [ln for ln in lines if ln.startswith("[chest] clear")]
    assert len(seeks) == 1
    assert clears == []
    assert all(v[1] > 0 for v in vecs[1:])


def test_two_missed_frames_are_held_then_a_long_gap_clears():
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    nav.game_s = 300.0
    _step(nav, clock, ARROW_DOWN)
    for _ in range(2):
        _step(nav, clock, None)
    assert not any(ln.startswith("[chest] clear") for ln in lines)
    for _ in range(4):  # 0.6 s hold is long gone
        _step(nav, clock, None)
    assert sum(ln.startswith("[chest] clear") for ln in lines) == 1


def test_first_miss_is_held_even_with_a_long_frame_gap():
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda _s: None)
    nav.game_s = 300.0
    _step(nav, clock, ARROW_DOWN)
    res = result([], gems=[])
    clock.t += 1.5  # one slow frame (OCR stall) must not drop the target by itself
    assert nav._chest_seen(res, False, 1.5) is not None
    assert nav._chest_seen(res, False, 0.2) is None


def test_chest_hold_s_is_a_vs_tune_knob():
    """VS_TUNE=chest_hold_s=0.10 restores the short window after the first miss."""
    assert Tuning().chest_hold_s == 0.6
    short = tuning_from_env("chest_hold_s=0.10", log=lambda *_: None)
    assert short.chest_hold_s == 0.10
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(tune=short, clock=clock, log=lines.append)
    nav.game_s = 300.0
    _step(nav, clock, ARROW_DOWN)
    _step(nav, clock, None)
    assert not any(ln.startswith("[chest] clear") for ln in lines)
    _step(nav, clock, None)  # 0.4 s of misses, past the restored 0.10 s window
    assert any(ln.startswith("[chest] clear") for ln in lines)
