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

Encirclement breakout: the crowd is binned into 16 angular sectors around the
player. When most sectors are occupied (the ring is closing) or nearby density
keeps rising while half the sectors fill, the navigator commits to a breakout:
it casts corridors in the 8 directions, picks the thinnest one that leads to
open space (preferring not to reverse), and follows it for a second or more
with a strong bias, re-checking the gap as it goes. Afterwards the roam loop
continues from that heading, so the character keeps sweeping in big loops
instead of being boxed in. After a REVIVE, ``request_flee`` forces the same
breakout for ~2 s even when encirclement metrics are still low (no gem pull).
A fast HP crash (e.g. Man-eater flower ring ~5:00) also forces that flee, and
during the mid-run flower window the cover trigger is lowered so breakouts
start earlier and last longer.

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
        return Phase("wide-circles", 13.0, 1.0)
    if game_s < 9 * 60 + 30:
        # Skeleton swarm (~7:00): tighter, less tangential so circles stay dense.
        return Phase("tight-weave", 3.0, 0.45)
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
    gem_gain: float = 1.6
    gem_danger: float = 1.6     # gems pull below this danger, fading out toward it
    gem_radius: float = 300.0
    safe_danger: float = 0.6
    panic_danger: float = 2.5   # above this, kiting bias fades out
    low_hp: float = 0.5
    # Encirclement breakout.
    sector_radius: float = 260.0   # crowd within this radius counts for encirclement
    sector_thr: float = 0.30       # per-sector density that counts as "occupied"
    cover_trigger: float = 0.55    # occupied fraction of sectors that forces a breakout
    cover_rising: float = 0.40     # ... or this much while density keeps rising
    rise_trigger: float = 0.8      # fast-minus-slow EMA of nearby density
    near_min: float = 2.0          # ignore sparse screens
    breakout_weight: float = 1.5
    breakout_min_s: float = 1.1
    breakout_max_s: float = 3.0
    flee_after_revive_s: float = 2.0  # forced breakout after taking REVIVE
    # Man-eater flower ring (~4:40–6:00): HP can drop 1.0→0.1 in 2–3 s.
    hp_crash_delta: float = -0.3       # peak-to-now drop that counts as crashing
    hp_crash_window_s: float = 1.5     # look-back for that drop
    flower_phase_lo: float = 280.0     # game_s start of flower-ring window
    flower_phase_hi: float = 360.0     # game_s end
    cover_trigger_crash: float = 0.35  # cover fraction when HP is crashing
    cover_trigger_mid: float = 0.45    # cover fraction during flower window
    breakout_max_crash_s: float = 4.5  # longer commit while HP crashes / mid phase
    flee_on_hp_crash_s: float = 2.5    # forced flee length on a fresh HP crash
    corridor_len: float = 320.0
    corridor_width: float = 34.0
    # Short look-ahead rollout: enemies chase while we walk each direction.
    rollout_gain: float = 0.6
    player_speed: float = 100.0    # reference px / s
    enemy_speed: float = 70.0
    rollout_s: float = 1.2         # per leg; two legs (turn allowed after the first)
    rollout_steps: int = 4


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


N_SECTORS = 16


def sector_profile(points: np.ndarray, px: float, py: float, radius: float = 260.0) -> np.ndarray:
    """Crowd weight per angular sector (16) around the player, nearer cells count more."""
    out = np.zeros(N_SECTORS, dtype=np.float32)
    if points.shape[0] == 0:
        return out
    rx = points[:, 0] - px
    ry = points[:, 1] - py
    d = np.hypot(rx, ry)
    keep = (d < radius) & (d > 1e-3)
    if not keep.any():
        return out
    ang = np.arctan2(ry[keep], rx[keep])
    idx = ((ang + math.pi) / (2 * math.pi) * N_SECTORS).astype(int) % N_SECTORS
    w = points[keep, 2] / (1.0 + (d[keep] / 140.0) ** 2)
    np.add.at(out, idx, w)
    return out


def encirclement(points: np.ndarray, px: float, py: float, radius: float = 260.0, thr: float = 0.3) -> tuple[float, float]:
    """(fraction of occupied sectors, total nearby weight). 1.0 = fully surrounded."""
    prof = sector_profile(points, px, py, radius)
    return float((prof > thr).mean()), float(prof.sum())


def corridor_costs(points: np.ndarray, px: float, py: float, length: float = 320.0, width: float = 34.0) -> np.ndarray:
    """Crowd weight inside a widening corridor along each of the 8 directions (low = open)."""
    out = np.zeros(len(DIRS), dtype=np.float32)
    if points.shape[0] == 0:
        return out
    rel = points[:, :2] - np.array([px, py], dtype=np.float32)
    for i, (ux, uy) in enumerate(_UNIT):
        t = rel[:, 0] * ux + rel[:, 1] * uy
        perp = np.abs(rel[:, 0] * uy - rel[:, 1] * ux)
        inside = (t > -10) & (t < length) & (perp < width + 0.3 * np.clip(t, 0, None))
        out[i] = float((points[inside, 2] / (1.0 + np.clip(t[inside], 0, None) / 110.0)).sum())
    # The visible playfield is shorter vertically: rays that leave the screen early see
    # less, so a little unseen-space penalty keeps breakouts biased toward long runs.
    vis = np.array([min(abs(430 / ux) if ux else 1e9, abs(250 / uy) if uy else 1e9) for ux, uy in _UNIT])
    out += 0.4 * np.clip((length - vis) / length, 0, None)
    return out


def rollout_costs(
    points: np.ndarray, px: float, py: float, tune: "Tuning",
) -> np.ndarray:
    """Predicted crowding for each of the 8 directions over two short legs.

    Enemies are assumed to chase the player at ``enemy_speed``. Leg 1 walks a
    direction for ``rollout_s``; leg 2 takes the best follow-up direction, so a
    move that leads into a dead end scores badly even if its first step is clear.
    """
    D = len(DIRS)
    if points.shape[0] == 0:
        return np.zeros(D, dtype=np.float32)
    # Only cells that can reach us within the horizon matter.
    rel = points[:, :2] - np.array([px, py], dtype=np.float32)
    reach = (tune.player_speed + tune.enemy_speed) * 2 * tune.rollout_s + 60
    keep = np.hypot(rel[:, 0], rel[:, 1]) < reach
    pts = points[keep]
    rel = rel[keep]
    if pts.shape[0] == 0:
        return np.zeros(D, dtype=np.float32)
    w = pts[:, 2]
    dt = tune.rollout_s / tune.rollout_steps
    body2 = tune.body * tune.body

    # Leg 1: shape (D, N)
    p = np.zeros((D, 2), dtype=np.float32)
    e = np.broadcast_to(rel, (D,) + rel.shape).copy()
    c1 = np.zeros(D, dtype=np.float32)
    for k in range(tune.rollout_steps):
        p = p + _UNIT * tune.player_speed * dt
        v = p[:, None, :] - e
        d = np.hypot(v[..., 0], v[..., 1]) + 1e-3
        e = e + v / d[..., None] * np.minimum(tune.enemy_speed * dt, d)[..., None]
        d = np.hypot(p[:, None, 0] - e[..., 0], p[:, None, 1] - e[..., 1])
        c1 += ((w / (1.0 + d * d / body2)) * (1.0 + 2.0 * (d < tune.body))).sum(axis=1) * (1.0 - 0.1 * k)
    # Leg 2: (D, D, N) best follow-up per first direction.
    p2 = np.repeat(p[:, None, :], D, axis=1)
    e2 = np.repeat(e[:, None, :, :], D, axis=1)
    c2 = np.zeros((D, D), dtype=np.float32)
    for k in range(tune.rollout_steps):
        p2 = p2 + _UNIT[None, :, :] * tune.player_speed * dt
        v = p2[:, :, None, :] - e2
        d = np.hypot(v[..., 0], v[..., 1]) + 1e-3
        e2 = e2 + v / d[..., None] * np.minimum(tune.enemy_speed * dt, d)[..., None]
        d = np.hypot(p2[:, :, None, 0] - e2[..., 0], p2[:, :, None, 1] - e2[..., 1])
        c2 += ((w / (1.0 + d * d / body2)) * (1.0 + 2.0 * (d < tune.body))).sum(axis=2) * 0.6
    return (c1 + c2.min(axis=1)) / tune.rollout_steps


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
    kite_floor: bool = False,
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
    if tune.rollout_gain > 0:
        scores = scores - tune.rollout_gain * (1.8 if defensive else 1.0) * rollout_costs(pts, px, py, tune)

    # Kiting fades out when things get tight: pure escape then.
    panic = min(1.0, max(0.0, (here - tune.safe_danger) / max(tune.panic_danger - tune.safe_danger, 1e-6)))
    if kite_floor:
        # Breakout heading: stays strong under pressure (contact terms still veto bodies).
        kite_scale = kite_weight * (1.0 - 0.3 * panic)
    else:
        kite_scale = kite_weight * (1.0 - 0.75 * panic) * (0.7 if defensive else 1.0)
    if kite_dir is not None and kite_scale > 0:
        kx, ky = kite_dir
        norm = math.hypot(kx, ky) or 1.0
        scores = scores + kite_scale * (_UNIT[:, 0] * kx / norm + _UNIT[:, 1] * ky / norm)

    gem_scale = max(0.0, 1.0 - here / tune.gem_danger) if tune.gem_danger > 0 else 0.0
    if gems and gem_scale > 0 and not defensive:
        # XP matters as much as dodging (live run8: level 6 at 3:53 with dozens of
        # gems left on screen), so gems pull at moderate danger too, fading out as
        # the crowd gets closer.
        g = np.asarray(gems, dtype=np.float32).reshape(-1, 2)
        gd = np.hypot(g[:, 0] - px, g[:, 1] - py)
        close = g[gd < tune.gem_radius]
        if close.shape[0]:
            pull = np.zeros(len(DIRS), dtype=np.float32)
            for gx, gy in close:
                vx, vy = gx - px, gy - py
                d = math.hypot(vx, vy) or 1.0
                pull += (_UNIT[:, 0] * vx / d + _UNIT[:, 1] * vy / d) / (1.0 + d / 80.0)
            scores = scores + tune.gem_gain * gem_scale * np.clip(pull, -1.5, 1.5)

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


# HUD timer OCR plausibility (see Navigator._sync_timer).
TIMER_JUMP_TOL_S = 8.0       # accept readings this close to the wall-clock estimate
TIMER_RESYNC_N = 3           # consecutive agreeing out-of-range readings to resync
TIMER_RESYNC_AGREE_S = 2.0   # max spread of those readings' offsets to the estimate


class Navigator:
    """Stateful wrapper: orbit direction, rotating heading, dwell, game clock."""

    def __init__(self, tune: Tuning | None = None, clock=time.monotonic) -> None:
        self.tune = tune or Tuning()
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.game_s = 0.0
        self._last_t: float | None = None
        self._timer_cands: list[float] = []  # offsets (reading - game_s) of rejected OCR jumps
        self.heading = 0.0          # radians, roam heading when nothing is around
        self.orbit = 1              # +1 clockwise on screen, -1 counter-clockwise
        self.current: tuple[int, int] = (0, 0)
        self._since_change = 0.0
        self._blocked_s = 0.0
        self.last: Decision | None = None
        self._near_fast = 0.0
        self._near_slow = 0.0
        self._near_init = False
        self.breakout: tuple[float, float] | None = None
        self._breakout_s = 0.0
        self._breakout_check = 0.0
        self.breakouts = 0
        self.cover = 0.0
        self._flee_remaining = 0.0  # >0: force breakout (post-revive flee)
        self._hp_hist: list[tuple[float, float]] = []  # (wall_t, hp) for crash detect
        self._hp_crashing = False
        self._hp_crash_latched = False  # edge-trigger flee once per crash bout

    @property
    def phase(self) -> Phase:
        return phase_for(self.game_s)

    def request_flee(self, duration_s: float | None = None) -> None:
        """Force a breakout for ~1.5–2.5 s after REVIVE (ignore gem pull).

        Picks the open corridor on the next ``step`` even when encirclement
        metrics are still low (the crowd has not been re-measured yet).
        """
        t = self.tune
        dur = t.flee_after_revive_s if duration_s is None else float(duration_s)
        self._flee_remaining = max(t.breakout_min_s, min(dur, t.breakout_max_s))
        self.breakout = None  # pre-death corridor and its elapsed time are stale

    def _start_breakout(self, pts: np.ndarray, px: float, py: float) -> None:
        t = self.tune
        costs = corridor_costs(pts, px, py, t.corridor_len, t.corridor_width)
        if self.current in DIRS:
            cu = _UNIT[DIRS.index(self.current)]
            align = _UNIT @ cu
            costs = costs + 0.25 * (1.0 - align)
        best = int(np.argmin(costs))
        self.breakout = tuple(float(v) for v in _UNIT[best])
        self._breakout_s = 0.0
        self._breakout_check = 0.0
        self.breakouts += 1

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

    def _note_hp_crash(self, hp: float | None, now: float) -> bool:
        """Update HP history; return True while HP is crashing hard.

        A crash is a peak-to-now drop of ``hp_crash_delta`` (default -0.3) inside
        ``hp_crash_window_s``. Used to force / extend breakouts for the Man-eater
        flower ring that can dump HP from ~1.0 to ~0.1 in a couple of seconds.
        """
        t = self.tune
        if hp is None:
            return self._hp_crashing
        win = t.hp_crash_window_s
        self._hp_hist = [(ts, h) for ts, h in self._hp_hist if now - ts <= win]
        self._hp_hist.append((now, float(hp)))
        if len(self._hp_hist) < 2:
            self._hp_crashing = False
            self._hp_crash_latched = False
            return False
        peak = max(h for _, h in self._hp_hist)
        drop = float(hp) - peak
        crashing = drop <= t.hp_crash_delta
        if not crashing:
            self._hp_crash_latched = False
        self._hp_crashing = crashing
        return crashing

    def _update_breakout(self, pts: np.ndarray, px: float, py: float, dt: float, hp: float | None) -> None:
        t = self.tune
        now = self.clock()
        crashing = self._note_hp_crash(hp, now)
        flower = t.flower_phase_lo <= self.game_s <= t.flower_phase_hi
        # Fresh HP crash: force a flee corridor (ignore gem pull via breakout path).
        if crashing and not self._hp_crash_latched:
            dur = max(t.breakout_min_s, min(t.flee_on_hp_crash_s, t.breakout_max_crash_s))
            self._flee_remaining = max(self._flee_remaining, dur)
            self.breakout = None  # pick a fresh corridor on the next start
            self._hp_crash_latched = True
        cover, near = encirclement(pts, px, py, t.sector_radius, t.sector_thr)
        self.cover = cover
        if not self._near_init:
            # First frame: no trend yet (a full screen at unpause is not "rising").
            self._near_fast = self._near_slow = near
            self._near_init = True
        a_fast = min(1.0, dt / 0.35) if dt > 0 else 0.0
        a_slow = min(1.0, dt / 2.5) if dt > 0 else 0.0
        self._near_fast += a_fast * (near - self._near_fast)
        self._near_slow += a_slow * (near - self._near_slow)
        rising = self._near_fast - self._near_slow
        low = hp is not None and hp < t.low_hp
        if crashing:
            trigger_cover = t.cover_trigger_crash
        elif flower:
            trigger_cover = t.cover_trigger_mid - (0.05 if low else 0.0)
        else:
            trigger_cover = t.cover_trigger - (0.1 if low else 0.0)
        max_s = t.breakout_max_crash_s if (crashing or flower) else t.breakout_max_s
        closing = near >= t.near_min and (cover >= trigger_cover or (cover >= t.cover_rising and rising >= t.rise_trigger))

        if self.breakout is not None:
            self._breakout_s += dt
            self._breakout_check += dt
            if self._flee_remaining > 0:
                # Post-revive / HP-crash flee: hold the corridor for the full window.
                self._flee_remaining = max(0.0, self._flee_remaining - dt)
                done = self._flee_remaining <= 0 or self._breakout_s >= max_s
            else:
                done = self._breakout_s >= max_s or (
                    self._breakout_s >= t.breakout_min_s and not closing
                )
            if done:
                self.heading = math.atan2(self.breakout[1], self.breakout[0])
                self.breakout = None
                self._flee_remaining = 0.0
                return
            if self._breakout_check >= 0.4:
                # Re-check the gap: switch only if the chosen corridor clearly closed.
                self._breakout_check = 0.0
                costs = corridor_costs(pts, px, py, t.corridor_len, t.corridor_width)
                cur = DIRS.index(_nearest_dir(self.breakout))
                best = int(np.argmin(costs))
                if costs[cur] > 1.6 * costs[best] + 0.5:
                    self.breakout = tuple(_UNIT[best])
            return
        if self._flee_remaining > 0 and self.breakout is None:
            # Post-revive / HP-crash: commit to a corridor even if the ring is not closing yet.
            self._start_breakout(pts, px, py)
            return
        if not closing:
            return
        self._start_breakout(pts, px, py)

    def _sync_timer(self, reading: float) -> None:
        """Plausibility-check an OCR'd HUD timer reading against the wall clock.

        ``self.game_s`` already holds the expected time (last accepted reading
        plus elapsed unpaused wall time). The minute digit is sometimes misread
        (e.g. 05:27 read as 06:27, 01:34 as 11:34), which would flip the phase, so:

        * a reading within ``TIMER_JUMP_TOL_S`` of the expectation is accepted;
        * anything else is rejected and the wall-clock estimate keeps running;
        * ``game_s`` never moves backwards;
        * a real drift (e.g. a stall/pause of the loop) is resynced only when
          ``TIMER_RESYNC_N`` consecutive readings agree with each other.
        """
        expected = self.game_s
        if abs(reading - expected) <= TIMER_JUMP_TOL_S:
            self.game_s = max(expected, reading)
            self._timer_cands.clear()
            return
        if reading < expected:
            # Behind the estimate: moving back is never allowed, so it cannot
            # seed a resync either.
            self._timer_cands.clear()
            return
        # Ahead by more than the tolerance. Track the offset to our own clock:
        # successive readings of one true clock keep that offset constant.
        off = reading - expected
        if self._timer_cands and abs(off - self._timer_cands[0]) > TIMER_RESYNC_AGREE_S:
            self._timer_cands.clear()
        self._timer_cands.append(off)
        if len(self._timer_cands) >= TIMER_RESYNC_N:
            self.game_s = max(expected, reading)
            self._timer_cands.clear()

    def step(self, result: VisionResult, hp: float | None = None) -> Decision:
        now = self.clock()
        dt = 0.0 if self._last_t is None else min(0.5, max(0.0, now - self._last_t))
        self._last_t = now
        # Advance on the wall clock (menus pause the in-game clock), then let a
        # plausible HUD timer OCR reading correct the estimate.
        if not result.menu:
            self.game_s += dt
        if result.timer_s is not None:
            self._sync_timer(float(result.timer_s))
        if result.menu:
            # Level-up / chest screens pause the game clock too.
            self.current = (0, 0)
            self.last = Decision((0, 0), {}, 0.0, "menu")
            return self.last
        self._since_change += dt

        pts = result.threats
        if pts is None:
            pts = _points_from_detections(result)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        px, py = _ref_player(result)
        phase = self.phase
        kite = self._kite_direction(pts, px, py, dt)
        self._update_breakout(pts, px, py, dt, hp)
        if self.breakout is not None:
            dec = score_directions(
                pts, (px, py), self.breakout, self.tune.breakout_weight,
                gems=None, current=self.current, hp=hp, tune=self.tune, idle_ok=False,
                kite_floor=True,
            )
            dec.mode = "breakout"
            kite = self.breakout
        else:
            dec = score_directions(
                pts, (px, py), kite, phase.kite_weight,
                gems=result.gems, current=self.current, hp=hp,
                tune=self.tune, idle_ok=phase.idle_ok,
            )

        # Flip the orbit when the field keeps overruling it (the loop runs into
        # a crowd): kiting the other way round stops the two fighting.
        v = dec.vector
        if v != (0, 0) and self.breakout is None:
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
