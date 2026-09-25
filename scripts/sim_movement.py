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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policies", default="still,old,field,new")
    a = ap.parse_args()
    for pol in a.policies.split(","):
        res = [simulate(pol, a.seconds, s) for s in range(a.seeds)]
        dmg = [r["damage"] for r in res]
        print(f"{pol:6s} damage mean {np.mean(dmg):8.1f}  per-seed {[round(x) for x in dmg]}  kills {np.mean([r['kills'] for r in res]):.0f}  death(140dmg) {[round(r['death']) for r in res]}")


if __name__ == "__main__":
    main()
