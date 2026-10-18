"""Chest seeks must hold through moderate danger (live run on cb74267).

The run logged 48 seeks and 48 clears. 46 clears were below chest_danger
(1.6), so the seek died on the next tick whenever danger jittered around
1.0-1.5 or the heading flicked off the arrow. A bottom-edge arrow then sat
under the player for the rest of the run and the chest was never reached.
"""

from __future__ import annotations

from vs_vision_bot.pathfind import Navigator

from tests.test_movement import Clock, blob, result, run


ARROW_DOWN = (500.0, 510.0)
# Two threat fields from the live pattern: danger ~0.6 (seek) and ~1.4 (clear).
_LOW = blob(500, 340, n=3, spread=18, seed=2)
_HIGH = blob(500, 320, n=5, spread=18, seed=1)


def test_chest_seek_holds_through_moderate_danger_jitter():
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    nav.game_s = 600.0
    vecs = []
    highs = []
    for i in range(16):
        clock.t += 1 / 12
        # Arm on a quiet frame, then the live jitter (danger ~1.4) must not drop it.
        res = result(_LOW if i == 0 else (_HIGH if i % 2 == 0 else _LOW))
        res.chest_arrow = ARROW_DOWN
        dec = nav.step(res, hp=1.0)
        vecs.append(dec.vector)
        if i > 0 and dec.danger_here >= 1.2:
            highs.append(dec.vector)
    seeks = [ln for ln in lines if ln.startswith("[chest] seek")]
    clears = [ln for ln in lines if ln.startswith("[chest] clear")]
    assert len(seeks) == 1
    assert clears == []
    assert highs and all(v[1] > 0 for v in highs)
    assert vecs[-1][1] > 0
    # Still the moderate band from the live log (clears were 0.5-2.2), not a real flee.
    assert nav.last.danger_here < 3.0


def test_truly_high_danger_clears_a_chest_seek():
    """Once danger is a real flee, the chest commit ends. The live clears were at 0.5-2.2."""
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    safe = result([], gems=[(300.0, 256.0)])
    safe.chest = (720.0, 256.0)
    assert run(nav, clock, safe, 3, hp=1.0) == [(1, 0)] * 3
    packed = result([(480 + dx, 256 + dy, 1.0) for dx in (-12, 0, 12) for dy in (-12, 0, 12)])
    packed.chest = (720.0, 256.0)
    run(nav, clock, packed, 4, hp=1.0)
    assert nav.last.danger_here >= 3.0
    assert any(ln.startswith("[chest] clear chest") for ln in lines)
