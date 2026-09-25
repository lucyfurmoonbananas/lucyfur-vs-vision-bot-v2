"""
Survival movement: potential field over 8 directions + circular kiting.

Coordinates are reference 960x540 capture pixels with the player at
``VisionResult.player`` (the camera keeps the character there).

Per frame, every one of the 8 WASD directions is scored:

* danger  - sum of enemy-cell weights around the spot the player would reach
  after ~0.35 s and ~0.8 s of walking that way (a potential field sampled at
  future positions), plus a heavy term for cells touching the path itself.
  Nearby cells dominate, so the least-dense sector wins when threatened.
* kiting  - bonus for heading tangentially around the crowd's centroid
  (Lucyfur's "wide circles": herd the swarm behind you instead of running into
  one side). With no crowd the heading keeps rotating, so the character walks
  big loops and never stands idle in the open.
* gems    - small pull toward blue XP gems when nothing is close.
* inertia - hysteresis: the current direction gets a bonus and a minimum
  dwell time so the chord does not jitter frame to frame.

Behaviour switches by game phase (tight weaving for the ~7:00 skeleton swarm,
slow circles later, near-still after ~21:00) and by HP (more defensive when low).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.vision import VisionResult

VECTORS_8 = (
    (0, 0),
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)
DIRS = VECTORS_8[1:]
_UNIT = np.array([(dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)) for dx, dy in DIRS], dtype=np.float32)


@dataclass
class Phase:
    name: str
    orbit_period_s: float  # time for one full loop with nothing around
    kite_weight: float     # tangential bias strength
    idle_ok: bool = False  # may stand still when nothing is close


def phase_for(game_s: float) -> Phase:
    """Movement style by game time (Mad Forest, Lucyfur's playbook)."""
    if game_s < 6 * 60 + 30:
        return Phase("wide-circles", 9.0, 1.0)
    if game_s < 9 * 60 + 30:
        return Phase("tight-weave", 4.0, 0.55)
    if game_s < 15 * 60:
        return Phase("wide-loops", 11.0, 1.0)
    if game_s < 21 * 60:
        return Phase("slow-circles", 6.0, 0.7)
    return Phase("hold-center", 6.0, 0.35, idle_ok=True)


@dataclass
class Tuning:
    step_near: float = 30.0     # px walked in ~0.35 s (Poe base speed)
    step_far: float = 75.0      # px walked in ~0.8 s
    radius: float = 42.0        # potential falloff radius
    body: float = 26.0          # enemy + player half size: contact distance
    danger_gain: float = 1.0
    contact_gain: float = 3.0
    idle_penalty: float = 0.6   # standing still only when truly safe
    hysteresis: float = 0.35
    min_dwell_s: float = 0.22
    gem_gain: float = 0.5
    safe_danger: float = 0.6    # below this, gems may pull
    panic_danger: float = 2.5   # above this, kiting bias fades out
    low_hp: float = 0.5


@dataclass
class Decision:
    vector: tuple[int, int]
    scores: dict[tuple[int, int], float] = field(default_factory=dict)
    danger_here: float = 0.0
    mode: str = ""


def _danger_at(points: np.ndarray, sx: np.ndarray, sy: np.ndarray, radius: float) -> np.ndarray:
    """Potential at sample positions (arrays) from weighted threat points."""
    if points.shape[0] == 0:
        return np.zeros_like(sx)
    dx = points[None, :, 0] - sx[:, None]
    dy = points[None, :, 1] - sy[:, None]
    d2 = dx * dx + dy * dy
    return (points[None, :, 2] / (1.0 + d2 / (radius * radius))).sum(axis=1)


def _path_contact(points: np.ndarray, px: float, py: float, length: float, body: float) -> np.ndarray:
    """Weighted count of cells within ``body`` of each direction's path segment."""
    out = np.zeros(len(DIRS), dtype=np.float32)
    if points.shape[0] == 0:
        return out
    rel = points[:, :2] - np.array([px, py], dtype=np.float32)
    for i, (ux, uy) in enumerate(_UNIT):
        t = np.clip(rel[:, 0] * ux + rel[:, 1] * uy, 0.0, length)
        cx = rel[:, 0] - t * ux
        cy = rel[:, 1] - t * uy
        dist = np.sqrt(cx * cx + cy * cy)
        near = dist < body
        # Closer along the path = sooner = worse.
        out[i] = float((points[near, 2] * (1.0 - 0.5 * t[near] / length)).sum())
    return out


def score_directions(
    threats: np.ndarray,
    player: tuple[float, float],
    kite_dir: tuple[float, float] | None,
    kite_weight: float,
    gems: list[tuple[float, float]] | None = None,
    current: tuple[int, int] = (0, 0),
    hp: float | None = None,
    tune: Tuning | None = None,
    idle_ok: bool = False,
) -> Decision:
    """Pure scoring: returns the best 8-way (or idle) vector plus diagnostics."""
    tune = tune or Tuning()
    pts = np.asarray(threats, dtype=np.float32).reshape(-1, 3) if threats is not None else np.zeros((0, 3), np.float32)
    px, py = float(player[0]), float(player[1])
    defensive = hp is not None and hp < tune.low_hp
    danger_gain = tune.danger_gain * (1.8 if defensive else 1.0)
    radius = tune.radius * (1.25 if defensive else 1.0)

    here = float(_danger_at(pts, np.array([px]), np.array([py]), radius)[0])
    nx = px + _UNIT[:, 0] * tune.step_near
    ny = py + _UNIT[:, 1] * tune.step_near
    fx = px + _UNIT[:, 0] * tune.step_far
    fy = py + _UNIT[:, 1] * tune.step_far
    danger = _danger_at(pts, nx, ny, radius) + 0.6 * _danger_at(pts, fx, fy, radius * 1.4)
    contact = _path_contact(pts, px, py, tune.step_far, tune.body)
    scores = -(danger_gain * danger + tune.contact_gain * contact)

    # Kiting fades out when things get tight: pure escape then.
    panic = min(1.0, max(0.0, (here - tune.safe_danger) / max(tune.panic_danger - tune.safe_danger, 1e-6)))
    kite_scale = kite_weight * (1.0 - 0.75 * panic) * (0.7 if defensive else 1.0)
    if kite_dir is not None and kite_scale > 0:
        kx, ky = kite_dir
        norm = math.hypot(kx, ky) or 1.0
        scores = scores + kite_scale * (_UNIT[:, 0] * kx / norm + _UNIT[:, 1] * ky / norm)

    if gems and here < tune.safe_danger and not defensive:
        g = np.asarray(gems, dtype=np.float32).reshape(-1, 2)
        gd = np.hypot(g[:, 0] - px, g[:, 1] - py)
        close = g[gd < 220]
        if close.shape[0]:
            pull = np.zeros(len(DIRS), dtype=np.float32)
            for gx, gy in close:
                vx, vy = gx - px, gy - py
                d = math.hypot(vx, vy) or 1.0
                pull += (_UNIT[:, 0] * vx / d + _UNIT[:, 1] * vy / d) / (1.0 + d / 80.0)
            scores = scores + tune.gem_gain * np.clip(pull, -1.5, 1.5)

    if current in DIRS:
        scores[DIRS.index(current)] += tune.hysteresis * (1.0 - 0.6 * panic)

    table = {d: float(s) for d, s in zip(DIRS, scores)}
    best_i = int(np.argmax(scores))
    best = DIRS[best_i]
    # Standing still gets swarmed: only allowed late game ("stand in the
    # middle, weapons clear it") and only while nothing is close.
    mode = "escape" if panic > 0.5 else ("kite" if kite_dir is not None else "roam")
    if idle_ok and here < tune.safe_danger * 0.5:
        table[(0, 0)] = table[best] + 0.1
        return Decision((0, 0), table, here, "hold")
    return Decision(best, table, here, mode)


class Navigator:
    """Stateful wrapper: orbit direction, rotating heading, dwell, game clock."""

    def __init__(self, tune: Tuning | None = None, clock=time.monotonic) -> None:
        self.tune = tune or Tuning()
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.game_s = 0.0
        self._last_t: float | None = None
        self.heading = 0.0          # radians, roam heading when nothing is around
        self.orbit = 1              # +1 clockwise on screen, -1 counter-clockwise
        self.current: tuple[int, int] = (0, 0)
        self._since_change = 0.0
        self._blocked_s = 0.0
        self.last: Decision | None = None

    @property
    def phase(self) -> Phase:
        return phase_for(self.game_s)

    def _kite_direction(self, pts: np.ndarray, px: float, py: float, dt: float) -> tuple[float, float]:
        phase = self.phase
        self.heading += self.orbit * dt * 2.0 * math.pi / phase.orbit_period_s
        roam = (math.cos(self.heading), math.sin(self.heading))
        if pts.shape[0] == 0:
            return roam
        rel = pts[:, :2] - np.array([px, py], dtype=np.float32)
        d = np.hypot(rel[:, 0], rel[:, 1]) + 1e-3
        w = pts[:, 2] / (1.0 + (d / 160.0) ** 2)
        cx = float((w * rel[:, 0] / d).sum())
        cy = float((w * rel[:, 1] / d).sum())
        mag = math.hypot(cx, cy)
        total = float(w.sum()) + 1e-6
        lopsided = mag / total  # 0 = balanced ring (or nothing), 1 = all on one side
        if mag < 1e-6:
            return roam
        ux, uy = cx / mag, cy / mag
        # Tangent around the crowd, leaning outward the closer / denser it is.
        tx, ty = -uy * self.orbit, ux * self.orbit
        near = float((pts[:, 2] * (d < 110)).sum())
        out = min(0.9, 0.15 + 0.08 * near)
        kx, ky = tx - out * ux, ty - out * uy
        # Balanced surroundings: blend with the roaming loop.
        a = min(1.0, lopsided * 2.0)
        kx = a * kx + (1 - a) * roam[0]
        ky = a * ky + (1 - a) * roam[1]
        self.heading = math.atan2(ky, kx)  # keep the roam loop continuous
        return (kx, ky)

    def step(self, result: VisionResult, hp: float | None = None) -> Decision:
        now = self.clock()
        dt = 0.0 if self._last_t is None else min(0.5, max(0.0, now - self._last_t))
        self._last_t = now
        self.game_s += dt
        self._since_change += dt
        if result.menu:
            self.current = (0, 0)
            self.last = Decision((0, 0), {}, 0.0, "menu")
            return self.last

        pts = result.threats
        if pts is None:
            pts = _points_from_detections(result)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        px, py = _ref_player(result)
        phase = self.phase
        kite = self._kite_direction(pts, px, py, dt)
        dec = score_directions(
            pts, (px, py), kite, phase.kite_weight,
            gems=result.gems, current=self.current, hp=hp,
            tune=self.tune, idle_ok=phase.idle_ok,
        )

        # Flip the orbit when the field keeps overruling it (the loop runs into
        # a crowd): kiting the other way round stops the two fighting.
        v = dec.vector
        if v != (0, 0):
            kn = math.hypot(*kite) or 1.0
            agree = (v[0] * kite[0] + v[1] * kite[1]) / (math.hypot(*v) * kn)
            self._blocked_s = self._blocked_s + dt if agree < -0.3 else 0.0
            if self._blocked_s > 0.6:
                self.orbit *= -1
                self.heading += math.pi
                self._blocked_s = 0.0

        choice = dec.vector
        if choice != self.current and self.current != (0, 0) and self._since_change < self.tune.min_dwell_s:
            cur_score = dec.scores.get(self.current, -1e9)
            # Keep the current chord for a moment unless it is clearly worse.
            if cur_score > dec.scores[choice] - 1.0:
                choice = self.current
        if choice != self.current:
            self._since_change = 0.0
        self.current = choice
        dec = Decision(choice, dec.scores, dec.danger_here, f"{dec.mode}/{phase.name}")
        self.last = dec
        return dec


def _ref_player(result: VisionResult) -> tuple[float, float]:
    w, h = result.size
    return result.player[0] * 960.0 / w, result.player[1] * 540.0 / h


def _points_from_detections(result: VisionResult) -> np.ndarray:
    w, h = result.size
    sx, sy = 960.0 / w, 540.0 / h
    rows = []
    for det in result.monsters:
        area = max(det.w * det.h, 1)
        rows.append((det.cx * sx, det.cy * sy, min(3.0, max(0.5, area / 400.0))))
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def _nearest_dir(v: tuple[float, float]) -> tuple[int, int]:
    x, y = v
    n = math.hypot(x, y) or 1.0
    dots = _UNIT[:, 0] * x / n + _UNIT[:, 1] * y / n
    return DIRS[int(np.argmax(dots))]


def choose_vector(
    result: VisionResult,
    last: tuple[int, int] = (0, 0),
    hp: float | None = None,
) -> tuple[int, int]:
    """Stateless one-shot pick (no kiting memory). Idle with nothing around."""
    if result.menu:
        return (0, 0)
    pts = result.threats if result.threats is not None else _points_from_detections(result)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return (0, 0)
    dec = score_directions(pts, _ref_player(result), None, 0.0, gems=result.gems, current=last, hp=hp)
    return dec.vector


def keys_for_result(
    result: VisionResult,
    cfg: Config,
    last: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], tuple[str, ...]]:
    vector = choose_vector(result, last=last)
    return vector, cfg.keys_for_vector(vector)
