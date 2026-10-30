"""Solid obstacles are whatever the background refuses to scroll through.

A commanded step that does not move the ground paints a short-lived world cell.
The heading rotates to a perpendicular, then the other side, and that heading
stays refused. Packs are still dived above 25% HP, including in the skeleton
windows. Mobs are not written into the obstacle grid.
"""

from __future__ import annotations

import math

import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.pathfind import Navigator, Tuning
from vs_vision_bot.vision import Vision, estimate_scroll

from tests.test_movement import Clock, P, blob, result


Block = tuple[float, float, float, float]
SPEED = 100.0


def _hits(x: float, y: float, blocks: list[Block]) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in blocks)


def _advance(player: tuple[float, float], vec: tuple[int, int], dt: float, blocks: list[Block]):
    """Stop at the first solid. Scroll is the background shift, opposite the step."""
    if vec == (0, 0) or dt <= 0:
        return player, (0.0, 0.0)
    n = math.hypot(vec[0], vec[1])
    dx = vec[0] / n * SPEED * dt
    dy = vec[1] / n * SPEED * dt
    x, y = player
    for _ in range(8):
        nx, ny = x + dx / 8.0, y + dy / 8.0
        if _hits(nx, ny, blocks):
            break
        x, y = nx, ny
    return (x, y), (player[0] - x, player[1] - y)


# The live default is off. Geometry tests opt in at the old window.
_ON = Tuning(obstacle_window_s=0.40)


def _drive(
    blocks: list[Block],
    pack,
    frames: int,
    hp: float = 1.0,
    game_s: float = 30.0,
    tune: Tuning | None = None,
    frozen: bool = False,
):
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=tune or _ON, clock=clock, log=logs.append)
    nav.game_s = game_s
    player = (0.0, 0.0)
    vec = (0, 0)
    path = []
    for i in range(frames):
        clock.t += 1.0 / 12.0
        scroll = None
        if frozen:
            if i:
                scroll = (0.0, 0.0)
        elif i:
            player, scroll = _advance(player, vec, 1.0 / 12.0, blocks)
        res = result(pack)
        res.scroll = scroll
        vec = nav.step(res, hp=hp).vector
        path.append((nav._world[0], nav._world[1], vec))
    return nav, path, logs


def _east_pack():
    return blob(P[0] + 220, P[1], n=12, spread=16, seed=1)


def test_background_scroll_follows_a_shifted_frame():
    """Content moving left is a rightward step. The center disk is not the ground."""
    rng = np.random.default_rng(2)
    prev = rng.integers(0, 255, (540, 960), np.uint8)
    frame = np.roll(prev, shift=-12, axis=1)
    shift = estimate_scroll(prev, frame)
    assert shift is not None
    assert abs(shift[0] + 12) < 1.5
    assert abs(shift[1]) < 1.5
    # A second identical frame is a held background, not a missing measurement.
    held = estimate_scroll(frame, frame.copy())
    assert held is not None and abs(held[0]) < 1.0 and abs(held[1]) < 1.0
    # A body sliding across a still background is not the ground moving.
    still = rng.integers(0, 255, (540, 960), np.uint8)
    moved = still.copy()
    still[250:310, 450:510] = 255
    moved[250:310, 410:470] = 255
    body = estimate_scroll(still, moved)
    assert body is None or abs(body[0]) < 20.0
    # The live tracker keeps the downscaled frame. The second call must still
    # see the shift; comparing a thumbnail to a full frame would drop it.
    vision = Vision(Config())
    assert vision.background_scroll(prev) is None
    tracked = vision.background_scroll(frame)
    assert tracked is not None and abs(tracked[0] + 12) < 1.5


def test_wall_stops_the_step_and_the_heading_turns():
    """An infinite wall to the east. The pack stays east. The body slides along."""
    wall = [(48.0, -400.0, 64.0, 400.0)]
    nav, path, logs = _drive(wall, _east_pack(), 36, hp=1.0)
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    assert max(xs) < 48.0
    assert max(abs(y) for y in ys) > 25.0
    assert any(v[0] <= 0 or v[1] != 0 for _, _, v in path[-8:])
    assert any(line.startswith("[nav] stuck") for line in logs)
    assert nav.breakouts == 0


def test_corner_is_left_along_the_open_side():
    east = [(30.0, -200.0, 48.0, 200.0)]
    south = [(-200.0, 30.0, 200.0, 48.0)]
    pack = blob(P[0] + 180, P[1] + 180, n=12, spread=16, seed=2)
    nav, path, _logs = _drive(east + south, pack, 48, hp=1.0)
    x, y, _v = path[-1]
    assert not (x > 24.0 and y > 24.0)
    assert math.hypot(x, y) > 40.0
    assert nav.breakouts == 0


def test_u_pocket_tries_both_sides_and_backs_out():
    blocks = [
        (22.0, -80.0, 40.0, 80.0),    # east
        (-80.0, -46.0, 40.0, -28.0),  # north
        (-80.0, 28.0, 40.0, 46.0),    # south
    ]
    _nav, path, logs = _drive(blocks, _east_pack(), 72, hp=1.0)
    # The opening is to the west. The refusal is short-lived, so a pack that
    # is still east can be approached again after it expires.
    assert min(p[0] for p in path) < -30.0
    stuck = [line for line in logs if line.startswith("[nav] stuck")]
    assert len(stuck) >= 2


def test_thin_pole_is_sidestepped():
    pole = [(36.0, -8.0, 48.0, 8.0)]
    _nav, path, _logs = _drive(pole, _east_pack(), 48, hp=1.0)
    assert path[-1][0] > 70.0
    while_passing = [abs(y) for x, y, _v in path if 20.0 < x < 70.0]
    assert while_passing and max(while_passing) > 14.0


def test_skeleton_window_still_dives_a_pack_and_only_turns_for_a_wall():
    """2:00-4:20, above 25% HP: the pack is dived. A wall is not a mob."""
    clock = Clock()
    nav = Navigator(clock=clock, log=lambda *_: None)
    nav.game_s = 180.0
    pack = _east_pack()
    player = (0.0, 0.0)
    vec = (0, 0)
    vecs = []
    for i in range(10):
        clock.t += 1.0 / 12.0
        scroll = None
        if i:
            player, scroll = _advance(player, vec, 1.0 / 12.0, [])
        res = result(pack)
        res.scroll = scroll
        vec = nav.step(res, hp=1.0).vector
        vecs.append(vec)
    assert all(v[0] > 0 for v in vecs)
    assert nav.breakouts == 0
    assert nav._world[0] > 40.0

    walled, path, _logs = _drive([(48.0, -400.0, 70.0, 400.0)], pack, 30, hp=1.0, game_s=180.0)
    assert max(p[0] for p in path) < 48.0
    assert walled.breakouts == 0


def test_open_field_stall_does_not_fire_when_the_detector_is_off():
    """test17: phase correlation stalled in the open, 270-420 stucks a run.

    The default window is 0. A frozen background with a pack to the east is
    still a dive, and it does not log a terrain block.
    """
    nav, path, logs = _drive([], _east_pack(), 36, hp=1.0, tune=Tuning(), frozen=True)
    assert not any(line.startswith("[nav] stuck") for line in logs)
    assert not any(line.startswith("[nav] obstacle") for line in logs)
    assert all(v[0] > 0 for _, _, v in path)
    assert nav.breakouts == 0


def test_crowd_stall_does_not_reroute_a_dive_above_quarter_hp():
    """Body-blocked scroll is not a wall. Flag on, skeleton window, hp 1.

    The pack sits just east of Poe. danger_here is hot, so the stall must not
    ban the dive heading.
    """
    pack = blob(P[0] + 70, P[1], n=40, spread=18, seed=3)
    nav, path, logs = _drive(
        [], pack, 24, hp=1.0, game_s=180.0, tune=_ON, frozen=True,
    )
    assert nav.last is not None and nav.last.danger_here >= 2.0
    assert not any(line.startswith("[nav] stuck") for line in logs)
    assert not any(line.startswith("[nav] obstacle") for line in logs)
    assert all(v[0] > 0 for _, _, v in path)
    assert nav.breakouts == 0


def test_menu_sequence_does_not_carry_a_stuck_timer():
    """Level-up and chest screens freeze the background. That is not terrain.

    Four stalled steps sit just under the window. A menu clears the timer, so
    four more stalled steps still do not fire. The same eight steps without
    the menu would.
    """
    clock = Clock()
    logs: list[str] = []
    nav = Navigator(tune=_ON, clock=clock, log=logs.append)
    nav.game_s = 40.0
    pack = _east_pack()

    def stall(menu: bool) -> None:
        clock.t += 1.0 / 12.0
        res = result(pack, menu=menu)
        res.scroll = (0.0, 0.0)
        vec = nav.step(res, hp=1.0).vector
        if menu:
            assert vec == (0, 0)

    for _ in range(5):
        stall(False)
    assert not any(line.startswith("[nav] stuck") for line in logs)
    for _ in range(4):
        stall(True)
    for _ in range(5):
        stall(False)
    assert not any(line.startswith("[nav] stuck") for line in logs)
    assert not any(line.startswith("[nav] obstacle") for line in logs)
