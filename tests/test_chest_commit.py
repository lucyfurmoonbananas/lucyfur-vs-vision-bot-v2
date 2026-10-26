"""Every spotted chest opens immediately, and a crowd does not drop it.

Oct 26 1 AM / 3 AM: after the free opens the leash held the chest, and a
surrounded chest was orbited at r=351-399 or lost on a ring-escape
(``[chest] seek`` then ``[chest] clear``) while it was still on screen.
"""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot.pathfind import Navigator, Tuning, tuning_from_env

from tests.test_chest_leash import CHEST_RIGHT, _anchor, _park_west_of
from tests.test_movement import Clock, blob, result, ring, run

ARROW_DOWN = (500.0, 510.0)
# Two threat fields from the live pattern: danger ~0.6 (seek) and ~1.4 (clear).
_LOW = blob(500, 340, n=3, spread=18, seed=2)
_HIGH = blob(500, 320, n=5, spread=18, seed=1)


def test_chest_seek_holds_through_moderate_danger_jitter():
    """A seek armed under chest_danger stays up while danger jitters around 1.0-1.5."""
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


def test_truly_high_danger_keeps_a_full_hp_chest_seek():
    """Danger past 3 used to clear the seek. Full HP keeps it.

    ``dive_radius_scale=1`` is the old cutoff: that seek still clears.
    """
    clock = Clock()
    lines: list[str] = []
    nav = Navigator(clock=clock, log=lines.append)
    safe = result([], gems=[(300.0, 256.0)])
    safe.chest = (720.0, 256.0)
    assert run(nav, clock, safe, 3, hp=1.0) == [(1, 0)] * 3
    packed = result([(480 + dx, 256 + dy, 1.0) for dx in (-12, 0, 12) for dy in (-12, 0, 12)])
    packed.chest = (720.0, 256.0)
    vecs = run(nav, clock, packed, 4, hp=1.0)
    assert nav.last.danger_here >= 3.0
    assert not any(ln.startswith("[chest] clear chest") for ln in lines)
    assert nav._chase in ("step", "orbit")
    assert all(v[0] >= 0 for v in vecs)

    off = tuning_from_env("dive_radius_scale=1", log=lambda *_: None)
    clock = Clock()
    lines = []
    nav = Navigator(tune=off, clock=clock, log=lines.append)
    assert run(nav, clock, safe, 3, hp=1.0) == [(1, 0)] * 3
    run(nav, clock, packed, 4, hp=1.0)
    assert nav.last.danger_here >= 3.0
    assert any(ln.startswith("[chest] clear chest") for ln in lines)


def test_chests_past_the_free_quota_open_immediately():
    """Chest 4+ is open-allowed before 10:00 with no evolution ready.

    ``chest_open_always=0`` restores the old hold until an evolution at 10:00.
    """
    tune = Tuning()
    assert tune.chest_free_opens == 3
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    for _ in range(tune.chest_free_opens):
        nav.on_chest_opened()
    nav.game_s = 120.0
    assert not nav._evolution_ready
    _anchor(nav, clock, chest=CHEST_RIGHT)
    assert nav._chest_open_allowed()
    assert nav._open_reason() == "always"
    assert any(ln.startswith("[chest] open-allowed (always)") for ln in logs)
    assert not any(ln.startswith("[chest] hold ") for ln in logs)
    _park_west_of(nav, 220.0)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    assert any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() < dist - 60.0

    held = tuning_from_env("chest_open_always=0,chest_orbit_swell=0", log=lambda *_: None)
    assert held.chest_open_always == 0
    clock = Clock()
    logs = []
    nav = Navigator(tune=held, clock=clock, log=logs.append)
    for _ in range(held.chest_free_opens):
        nav.on_chest_opened()
    nav.game_s = 120.0
    _anchor(nav, clock, chest=CHEST_RIGHT)
    _park_west_of(nav, held.chest_orbit_radius)
    nav._since_change = 1.0
    dist = nav._leash_dist()
    run(nav, clock, result([]), 12, hp=1.0)
    assert any(ln.startswith("[chest] hold (opened=3 ") for ln in logs)
    assert not any(ln.startswith("[leash] step-in ") for ln in logs)
    assert nav._leash_dist() > dist - held.chest_orbit_band


def _pursue(nav: Navigator, clock: Clock, pts, chest, frames: int, hp: float):
    """Camera-follow: threats and the chest shift opposite each step."""
    pts = np.array(pts, dtype=np.float32).reshape(-1, 3).copy()
    cx, cy = float(chest[0]), float(chest[1])
    speed = nav.tune.player_speed / 12.0
    vecs = []
    dists = []
    for _ in range(frames):
        clock.t += 1 / 12
        res = result(pts)
        res.chest = (cx, cy)
        dec = nav.step(res, hp=hp)
        vecs.append(dec.vector)
        dists.append(nav._leash_dist() if nav._leash is not None else 0.0)
        n = math.hypot(*dec.vector) or 1.0
        dx = dec.vector[0] / n * speed
        dy = dec.vector[1] / n * speed
        pts[:, 0] -= dx
        pts[:, 1] -= dy
        cx -= dx
        cy -= dy
    return vecs, dists


def test_crowded_chest_is_closed_on_instead_of_orbited_away():
    """High danger and cover keep the seek and close on the chest.

    The 1 AM run1 chest sat at 96-280 px with danger 3-15 and cover >= 0.55
    while the leash orbited at 350-400. run4 paired every seek with a clear.
    Full HP either steps a lane or kite-orbits on ``chest_commit_orbit``.
    """
    tune = Tuning()
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune, clock=clock, log=logs.append)
    nav.game_s = 800.0
    chest = (660.0, 256.0)
    pts = blob(620, 256, n=18, spread=50, seed=2) + ring(480, 256, 150, 40, gap_deg=(80, 110))
    seen = result(pts)
    seen.chest = chest
    clock.t += 1 / 12
    nav.step(seen, hp=1.0)
    assert nav.last.danger_here >= 3.0
    assert nav.cover >= 0.55
    assert nav._chest_open_allowed()

    vecs, dists = _pursue(nav, clock, pts, chest, 36, hp=1.0)
    fresh = [ln for ln in logs if ln.startswith("[chest] ") or ln.startswith("[leash] orbit ")]
    assert any(ln.startswith("[chest] seek chest ") for ln in fresh)
    assert not any(ln.startswith("[chest] clear ") for ln in fresh)
    radii = []
    for ln in fresh:
        if ln.startswith("[leash] orbit r="):
            radii.append(float(ln.split("r=", 1)[1].split()[0]))
    # Wide swell (280 + 40 * danger, about 350-400) is the old ring.
    assert radii == [] or max(radii) < 200.0
    # Net closure onto the tight ring. A ring-escape off the chest grows it.
    assert dists[-1] < dists[0] - 20.0
    assert dists[-1] <= tune.chest_commit_orbit + 15.0
    assert nav._leash is not None or any(ln.startswith("[leash] drop reached") for ln in logs)
    assert max(dists) < tune.chest_orbit_radius


def test_ring_escape_does_not_walk_off_an_openable_chest():
    """The wall commit (cover >= 0.75 after a density rise) used to take the gap.

    Chest to the left, gap to the right, HP still in the dive band. The walked
    chord stays with the chest. Critical HP still leaves; a held chest
    (chest_open_always=0) still yields to the wall.
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(clock=clock, log=logs.append)
    nav.game_s = 660.0
    chest = (240.0, 256.0)
    _anchor(nav, clock, chest=chest)
    warm = result(blob(700, 256, n=8, spread=30))
    warm.chest = chest
    run(nav, clock, warm, 8, hp=0.80)
    gap = result(ring(480, 256, 150, 48, gap_deg=(-1, 25)))
    gap.chest = chest
    vecs = run(nav, clock, gap, 6, hp=0.80)
    assert nav.cover >= 0.55
    assert any(ln.startswith("[nav] ring escape") for ln in logs)
    assert nav.escaping and nav.breakout is not None and nav.breakout[0] > 0
    assert nav._chase in ("step", "orbit")
    assert nav.last.mode.startswith("leash-")
    # The escape heading is the east gap. The steps are not.
    assert all(v[0] <= 0 for v in vecs)
    assert not any(ln.startswith("[chest] clear ") for ln in logs)
