"""
Tiny offline swarm simulator for tuning pathfind.Navigator.

Enemies spawn just off-screen and chase the player; a garlic-like aura kills
what it touches over time. Reports contact damage taken per policy.

    python scripts/sim_movement.py [--seconds 240] [--seeds 5]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vs_vision_bot.pathfind import Navigator, choose_vector  # noqa: E402
from vs_vision_bot.vision import Detection, VisionResult  # noqa: E402

W, H = 960, 540
PX, PY = 480, 256


def legacy_vector(dets: list[Detection], last: tuple[int, int]) -> tuple[int, int]:
    """The pre-rewrite pathfind: maximise nearest-blob distance 36 px ahead, idle if empty."""
    if not dets:
        return (0, 0)
    vecs = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1))

    def score(v):
        n = math.hypot(*v) or 1.0
        sx, sy = PX + v[0] / n * 36, PY + v[1] / n * 36
        near = min(math.hypot(sx - d.cx, sy - d.cy) for d in dets)
        return near * 0.02 + (0.08 if v == last and v != (0, 0) else 0.0)

    best = max(vecs, key=score)
    if best != (0, 0) and score(best) < score((0, 0)) + 0.05:
        return (0, 0)
    return best


def simulate(policy: str, seconds: float, seed: int, fps: float = 12.0) -> dict:
    rng = random.Random(seed)
    t = 0.0
    dt = 1.0 / fps
    player = np.array([0.0, 0.0])
    enemies: list[list[float]] = []  # x, y, speed, hp
    clock = [0.0]
    nav = Navigator(clock=lambda: clock[0])
    vec = (0, 0)
    damage = 0.0
    kills = 0
    pspeed = 105.0
    death_t = None
    while t < seconds:
        if death_t is None and damage >= 140.0:
            death_t = t
        # Spawn: rate grows over time; bursts every 60 s (bat swarms).
        rate = 2.0 + t / 40.0 + (12.0 if int(t) % 60 < 4 and t > 20 else 0.0)
        for _ in range(np.random.default_rng(rng.randrange(1 << 30)).poisson(rate * dt)):
            ang = rng.uniform(0, 2 * math.pi)
            r = 560.0
            ex = player[0] + math.cos(ang) * r
            ey = player[1] + math.sin(ang) * r * 0.62
            speed = rng.choice([45.0, 60.0, 95.0])
            enemies.append([ex, ey, speed, 1.2 + t / 120.0])
        # Observe
        dets = []
        for ex, ey, *_ in enemies:
            sx, sy = ex - player[0] + PX, ey - player[1] + PY
            if 48 <= sx < 912 and 18 <= sy < 540:
                dets.append(Detection(int(sx) - 10, int(sy) - 8, 20, 16))
        # Like vision.threat_points: each sprite covers ~4 grid cells of ~0.22.
        pts = np.array(
            [(d.cx + ox, d.cy + oy, 0.22) for d in dets for ox in (-6, 6) for oy in (-6, 6)], dtype=np.float32
        ).reshape(-1, 3)
        res = VisionResult(player=(PX, PY), monsters=dets, threats=pts)
        hp = max(0.0, 1.0 - damage / 70.0)
        if policy == "new":
            vec = nav.step(res, hp=hp).vector
        elif policy == "old":
            vec = legacy_vector(dets, vec)
        elif policy == "field":  # new potential field, no kiting memory
            vec = choose_vector(res, last=vec)
        elif policy == "still":
            vec = (0, 0)
        n = math.hypot(*vec) or 1.0
        player += np.array(vec) / n * pspeed * dt
        # Enemies chase with a little jitter; overlap resolution.
        alive = []
        for e in enemies:
            dx, dy = player[0] - e[0], player[1] - e[1]
            d = math.hypot(dx, dy) or 1.0
            if d > 16:
                e[0] += dx / d * e[2] * dt + rng.uniform(-8, 8) * dt
                e[1] += dy / d * e[2] * dt + rng.uniform(-8, 8) * dt
            if d < 18:
                damage += 5.0 * dt  # ~5 dmg/s per touching enemy (armor ignored)
            if d < 40:
                e[3] -= 1.6 * dt  # garlic
            if e[3] <= 0:
                kills += 1
                continue
            if d > 1400:
                continue
            alive.append(e)
        enemies = alive
        t += dt
        clock[0] = t
    return {"damage": damage, "kills": kills, "death": death_t if death_t is not None else seconds}


def _observe(rel: np.ndarray) -> tuple[list[Detection], np.ndarray]:
    """Enemies relative to the player (N,2) -> on-screen detections + threat grid points."""
    sx = rel[:, 0] + PX
    sy = rel[:, 1] + PY
    on = (sx >= 48) & (sx < 912) & (sy >= 18) & (sy < 540)
    xs, ys = sx[on], sy[on]
    dets = [Detection(int(x) - 10, int(y) - 8, 20, 16) for x, y in zip(xs, ys)]
    if xs.size:
        # Like vision.threat_points: overlapping sprites merge into grid cells.
        cells = {}
        for x, y in zip(xs, ys):
            for ox in (-6, 6):
                for oy in (-6, 6):
                    k = (int((x + ox) // 12), int((y + oy) // 12))
                    cells[k] = min(0.25, cells.get(k, 0.0) + 0.22)
        pts = np.array([((k[0] + 0.5) * 12, (k[1] + 0.5) * 12, w) for k, w in cells.items()], dtype=np.float32)
    else:
        pts = np.zeros((0, 3), np.float32)
    return dets, pts


def simulate_horde(policy: str, seconds: float, seed: int, fps: float = 12.0, start_s: float = 270.0,
                   density: float = 1.0) -> dict:
    """Denser ~4:30-8:30 Mad Forest waves: steady chasers from every side, straight-flying
    bat flocks, and encircling rings of ghouls that close in on the player (the pattern that
    boxed the live bot in at ~5:25). Enemies push each other apart so crowds form walls.

    ``death`` is when 70 damage (Poe's max HP) was taken, ``death2`` at 140 (one revive).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    t = 0.0
    player = np.zeros(2)
    pos = np.zeros((0, 2))
    vel = np.zeros((0, 2))       # straight flyers keep a fixed velocity; chasers use speed
    speed = np.zeros(0)
    hp = np.zeros(0)
    chase = np.zeros(0, dtype=bool)
    kind = np.zeros(0, dtype=int)  # 0 chaser, 1 ring ghoul, 2 bat flock
    dmg_by = [0.0, 0.0, 0.0]
    clock = [0.0]
    nav = Navigator(clock=lambda: clock[0])
    nav.game_s = start_s
    vec = (0, 0)
    damage = 0.0
    kills = 0
    pspeed = 105.0
    death = death2 = None
    next_ring = 12.0
    next_flock = 6.0
    shot_acc = 0.0

    def add(p, v, s, h, c, k):
        nonlocal pos, vel, speed, hp, chase, kind
        kind = np.concatenate([kind, np.full(len(s), k)])
        pos = np.vstack([pos, p])
        vel = np.vstack([vel, v])
        speed = np.concatenate([speed, s])
        hp = np.concatenate([hp, h])
        chase = np.concatenate([chase, c])

    while t < seconds:
        T = start_s + t
        if death is None and damage >= 70.0:
            death = t
        if death2 is None and damage >= 140.0:
            death2 = t
        ehp = 1.2 + T / 120.0
        # Steady chasers from all sides, off-screen.
        n = rng.poisson(density * (2.0 + T / 40.0) * dt)
        if n:
            ang = rng.uniform(0, 2 * math.pi, n)
            p = player + np.column_stack([np.cos(ang) * 560, np.sin(ang) * 340])
            add(p, np.zeros((n, 2)), rng.choice([45.0, 60.0, 80.0], n), np.full(n, ehp), np.ones(n, bool), 0)
        # Encircling ring: a closed ellipse of ghouls converging on the player.
        if t >= next_ring:
            next_ring += rng.uniform(25, 40) / density
            m = int(40 * density)
            ang = np.linspace(0, 2 * math.pi, m, endpoint=False) + rng.uniform(0, 1)
            p = player + np.column_stack([np.cos(ang) * 470, np.sin(ang) * 300])
            add(p, np.zeros((m, 2)), np.full(m, 42.0), np.full(m, ehp * 2), np.ones(m, bool), 1)
        # Bat flock: flies straight across the screen through the player's area.
        if t >= next_flock:
            next_flock += rng.uniform(15, 25)
            m = int(22 * density)
            a = rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(a), math.sin(a)])
            normal = np.array([-d[1], d[0]])
            off = rng.uniform(-40, 40)
            base = player - d * 600 + normal * off
            p = base + np.outer(rng.uniform(-60, 60, m), normal) + np.outer(rng.uniform(-80, 80, m), d)
            add(p, np.tile(d * 150.0, (m, 1)), np.full(m, 150.0), np.full(m, ehp * 0.5), np.zeros(m, bool), 2)

        rel = pos - player
        dets, pts = _observe(rel)
        res = VisionResult(player=(PX, PY), monsters=dets, threats=pts)
        hpf = max(0.0, 1.0 - (damage % 70.0) / 70.0)
        if policy == "new":
            vec = nav.step(res, hp=hpf).vector
        elif policy == "old":
            vec = legacy_vector(dets, vec)
        elif policy == "field":
            vec = choose_vector(res, last=vec)
        elif policy == "still":
            vec = (0, 0)
        nrm = math.hypot(*vec) or 1.0
        player = player + np.array(vec) / nrm * pspeed * dt

        if pos.shape[0]:
            rel = player - pos
            d = np.hypot(rel[:, 0], rel[:, 1]) + 1e-6
            move = np.where(chase[:, None], rel / d[:, None] * speed[:, None], vel)
            move = move * (d[:, None] > 14) + rng.uniform(-8, 8, move.shape)
            pos = pos + move * dt
            # Separation: crowds form walls instead of stacking on one point.
            if pos.shape[0] > 1:
                close = d < 260
                idx = np.flatnonzero(close)
                if idx.size > 1:
                    q = pos[idx]
                    diff = q[:, None, :] - q[None, :, :]
                    dist = np.hypot(diff[..., 0], diff[..., 1]) + 1e-6
                    push = np.clip(16.0 - dist, 0, None) / dist
                    np.fill_diagonal(push, 0)
                    pos[idx] += 0.5 * (diff * push[..., None]).sum(axis=1)
            rel = player - pos
            d = np.hypot(rel[:, 0], rel[:, 1])
            touching = d < 18
            damage += 5.0 * dt * int(touching.sum())
            for k in range(3):
                dmg_by[k] += 5.0 * dt * int((touching & (kind == k)).sum())
            hp = hp - 1.6 * dt * (d < 40)  # garlic
            # Other weapons (wand / cross / peachone): kill-rate on nearest on-screen enemies.
            shot_acc += 2.5 * dt
            while shot_acc >= 1.0:
                shot_acc -= 1.0
                vis = np.flatnonzero((np.abs(rel[:, 0]) < 430) & (np.abs(rel[:, 1]) < 250))
                if vis.size:
                    hp[vis[np.argmin(d[vis])]] -= 2.0
            dead = hp <= 0
            kills += int(dead.sum())
            keep = ~dead & (d < 1400)
            pos, vel, speed, hp, chase, kind = pos[keep], vel[keep], speed[keep], hp[keep], chase[keep], kind[keep]
        t += dt
        clock[0] = t
    return {
        "damage": damage,
        "kills": kills,
        "death": death if death is not None else seconds,
        "death2": death2 if death2 is not None else seconds,
        "damage_by": {"chasers": dmg_by[0], "rings": dmg_by[1], "bats": dmg_by[2]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policies", default="still,old,field,new")
    ap.add_argument("--horde", action="store_true", help="dense ~5:00 Mad Forest waves (rings, flocks)")
    ap.add_argument("--density", type=float, default=1.0)
    a = ap.parse_args()
    if a.horde:
        for pol in a.policies.split(","):
            res = [simulate_horde(pol, a.seconds, s, density=a.density) for s in range(a.seeds)]
            print(
                f"{pol:6s} horde damage mean {np.mean([r['damage'] for r in res]):7.1f}  "
                f"kills {np.mean([r['kills'] for r in res]):.0f}  "
                f"death@70 {[round(r['death']) for r in res]} mean {np.mean([r['death'] for r in res]):.0f}s  "
                f"death@140 mean {np.mean([r['death2'] for r in res]):.0f}s",
                flush=True,
            )
        return
    for pol in a.policies.split(","):
        res = [simulate(pol, a.seconds, s) for s in range(a.seeds)]
        dmg = [r["damage"] for r in res]
        print(f"{pol:6s} damage mean {np.mean(dmg):8.1f}  per-seed {[round(x) for x in dmg]}  kills {np.mean([r['kills'] for r in res]):.0f}  death(140dmg) {[round(r['death']) for r in res]}")


if __name__ == "__main__":
    main()
