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
* gems    - pull toward blue XP gems when nothing is close (stronger / farther before the flower).
* inertia - hysteresis: the current direction gets a bonus and a minimum
  dwell time so the chord does not jitter frame to frame.

Encirclement breakout: the crowd is binned into 16 angular sectors around the
player. When most sectors are occupied (the ring is closing) or nearby density
keeps rising while half the sectors fill, the navigator commits to a breakout:
it casts corridors in the 8 directions, picks the thinnest one that leads to
open space (preferring not to reverse), and follows it for a second or more
with a strong bias, re-checking the gap as it goes. Afterwards the roam loop
continues from that heading, so the character keeps sweeping in big loops
instead of being boxed in. After a REVIVE, ``request_revive_escape`` (built on
``request_flee``) runs a committed ~3 s escape toward the least threat-dense
side: steering stays within ~45 deg of that heading, the heading is only re-aimed
early or when its corridor is clearly shut, and it never turns back.
A fast HP crash (e.g. Man-eater flower ring ~5:00) also forces that flee, and
during the mid-run flower window the cover trigger is lowered so breakouts
start earlier and last longer; there a ring-closing breakout is also run as the
same committed escape when ``ring_commit_s`` > 0 (reason "ring"; default 2.5 s)
so it is not dropped when cover dips and never turns back, and
``flower_noreverse_s`` (default 4 s) blocks walking back into the ring.

Behaviour switches by game phase (tight weaving for the ~7:00 skeleton swarm,
slow circles later, near-still after ~21:00) and by HP (more defensive when low).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields, replace

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
    # Committed post-REVIVE escape (request_revive_escape): HP 0.50 inside the crowd
    # that just killed us. Oct 5 logs: the plain 2 s flee re-picked its corridor every
    # 0.4 s and reversed (w -> s, d -> a), walking back into the ring.
    revive_escape_s: float = 3.0      # length of the committed escape
    escape_len: float = 420.0         # corridor length when picking the escape heading
    escape_wedge_radius: float = 480.0  # crowd within this radius counts for the wedge term
    escape_wedge_gain: float = 0.5    # weight of whole-screen wedge density vs. corridor
    escape_repick_s: float = 0.5      # early window (REVIVE escapes only): first frames may be stale
    escape_repick_ratio: float = 1.25  # early re-pick only when clearly better
    escape_switch_ratio: float = 3.0  # later: switch only when the corridor is clearly shut
    escape_cone_dot: float = 0.70     # steer only within ~45 deg of the escape heading
    # Flower window: a ring-closing breakout becomes the same committed escape, so it
    # is not abandoned after 1.1 s when cover dips and never reverses into the ring
    # (Oct 5: cells 150 -> 400-600 at game~315-335 s, HP-crash flee fired too late).
    # Sim --horde (density 1.5) did not help vs off (damage ~69 off vs ~75 at 2.5 s) because
    # sim rings are converging ghouls, not the static Man-eater. Oct 8 live A/B with
    # flower_noreverse_s=4 alone still died ~5:36–5:42 in the real ring; enable both
    # as defaults (ring_commit_s=2.5, flower_noreverse_s=4). 0 disables either knob.
    ring_commit_s: float = 2.5        # 0 disables
    # Flower window "no reversal" (Bob, Oct 6 3 AM logs: breakouts at game~300-340 s
    # flipped v=(1,..) -> (-1,..) every 3 s, walking back into the ring). Inside the
    # window a new breakout started within this many seconds of the previous one, and
    # the 0.4 s corridor re-check, may only pick headings within 90 deg of the last
    # breakout heading, and the per-frame vector of a flower breakout stays within
    # 90 deg of its heading (side-steps OK, never back). 0 disables.
    # Sim --horde showed no gain (density 1.0: 30.8 off vs 32.8 at 3 s; 1.5: 75.9 vs 79.3);
    # Oct 8 live: noreverse alone was not enough (~5:36–5:42 deaths). Default on with
    # ring_commit_s for the real Man-eater ring; override via VS_TUNE if A/B needed.
    flower_noreverse_s: float = 4.0
    # Early-game XP: plain encirclement breakouts used to ignore gems (gems=None), and in
    # the Oct 6 3 AM logs 40-50 % of game 120-280 s samples were breakouts while gems on
    # screen piled up 5 -> 9.5 (level 6-8 at 5:00). Before this game_s a plain breakout
    # (not a REVIVE / HP-crash flee or committed escape) keeps the normal gem pull, which
    # still fades out with danger (gem_danger) and is off at low HP. 0 disables.
    breakout_gem_until_s: float = 280.0
    # Stronger / longer-range gem pull before the flower (Oct 9 live: gems piled to avg
    # ~8-13 on screen at 2-4 min while runs entered the ring at Lv6-8). While game_s is
    # below this, score_directions uses early_gem_gain / early_gem_radius instead of the
    # base knobs (still gated by gem_danger and low-HP). 0 disables the boost.
    early_gem_until_s: float = 280.0
    early_gem_gain: float = 2.8       # vs gem_gain 1.6
    early_gem_radius: float = 420.0   # vs gem_radius 300; reach gems left in the wake
    # Man-eater flower ring (~4:40–6:00): HP can drop 1.0→0.1 in 2–3 s.
    hp_crash_delta: float = -0.3       # peak-to-now drop that counts as crashing
    hp_crash_window_s: float = 1.5     # look-back for that drop
    flower_phase_lo: float = 280.0     # game_s start of flower-ring window
    flower_phase_hi: float = 360.0     # game_s end
    cover_trigger_crash: float = 0.35  # cover fraction when HP is crashing
    cover_trigger_mid: float = 0.45    # cover fraction during flower window
    breakout_max_crash_s: float = 4.5  # longer commit while HP crashes / mid phase
    flee_on_hp_crash_s: float = 3.0    # forced flee length on a fresh HP crash (was 2.5; flower ring ~5:15)
    corridor_len: float = 320.0
    corridor_width: float = 34.0
    # Short look-ahead rollout: enemies chase while we walk each direction.
    rollout_gain: float = 0.6
    player_speed: float = 100.0    # reference px / s
    enemy_speed: float = 70.0
    rollout_s: float = 1.2         # per leg; two legs (turn allowed after the first)
    rollout_steps: int = 4



def gem_tune_for(tune: Tuning, game_s: float) -> Tuning:
    """Return ``tune`` with early gem_gain/radius applied when still before the flower.

    No-op when ``early_gem_until_s`` is 0 or ``game_s`` is already past it, so late-game
    and VS_TUNE=early_gem_until_s=0 keep the base gem knobs unchanged.
    """
    if tune.early_gem_until_s <= 0 or game_s >= tune.early_gem_until_s:
        return tune
    return replace(tune, gem_gain=tune.early_gem_gain, gem_radius=tune.early_gem_radius)


def tuning_from_env(raw: str | None, base: Tuning | None = None, log=print) -> Tuning:
    """Parse ``VS_TUNE`` ("ring_commit_s=2.5,flower_noreverse_s=4") into a Tuning.

    Lets a session script A/B these knobs live without editing the repo. Only
    known numeric fields are accepted; unknown names or bad values are logged and
    skipped, so a typo can never stop the bot.
    """
    base = base or Tuning()
    if not raw or not raw.strip():
        return base
    types = {f.name: f.type for f in fields(Tuning)}
    over: dict[str, float | int] = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, val = item.partition("=")
        name, val = name.strip(), val.strip()
        kind = types.get(name)
        if not sep or kind not in ("float", "int", float, int):
            log(f"[nav] VS_TUNE: ignored {item!r} (unknown or non-numeric field)")
            continue
        try:
            over[name] = int(val) if kind in ("int", int) else float(val)
        except ValueError:
            log(f"[nav] VS_TUNE: ignored {item!r} (bad value)")
    if over:
        log("[nav] VS_TUNE overrides: " + ", ".join(f"{k}={v}" for k, v in over.items()))
    return replace(base, **over)


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


def escape_costs(points: np.ndarray, px: float, py: float, tune: "Tuning") -> np.ndarray:
    """Per-direction cost for the post-REVIVE escape (low = least threat density).

    A long corridor cast (what is directly in the way) plus the crowd weight in a
    wide 3-sector (~67 deg) wedge out to most of the screen, so the heading points
    at the emptiest side / edge rather than at a thin gap between two walls.
    """
    costs = corridor_costs(points, px, py, tune.escape_len, tune.corridor_width)
    prof = sector_profile(points, px, py, tune.escape_wedge_radius)
    if prof.sum() <= 0:
        return costs
    wedge = np.zeros(len(DIRS), dtype=np.float32)
    for i, (ux, uy) in enumerate(_UNIT):
        k = int(((math.atan2(uy, ux) + math.pi) / (2 * math.pi) * N_SECTORS)) % N_SECTORS
        wedge[i] = prof[(k - 1) % N_SECTORS] + prof[k] + prof[(k + 1) % N_SECTORS]
    return costs + tune.escape_wedge_gain * wedge


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

    def __init__(self, tune: Tuning | None = None, clock=time.monotonic, log=print) -> None:
        self.tune = tune or Tuning()
        self.clock = clock
        self.log = log
        self.reset()

    def reset(self) -> None:
        self.game_s = 0.0
        self._last_t: float | None = None
        self._timer_cands: list[float] = []  # offsets (reading - wall clock) of rejected OCR jumps
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
        self._breakout_origin: tuple[float, float] | None = None  # heading at breakout start (flower no-reverse)
        self._breakout_s = 0.0
        self._breakout_check = 0.0
        self.breakouts = 0
        self.cover = 0.0
        self._flee_remaining = 0.0  # >0: force breakout (post-revive flee)
        self._hp_hist: list[tuple[float, float]] = []  # (wall_t, hp) for crash detect
        self._hp_crashing = False
        self._hp_crash_latched = False  # edge-trigger flee once per crash bout
        self._hp_crash_peak = 0.0       # peak HP of the current crash look-back (for the log)
        self._flower_on = False         # edge-trigger the flower-window log line
        self._escape_remaining = 0.0    # >0: committed post-REVIVE escape running
        self._escape_s = 0.0            # time spent in the current escape
        self._escape_pending = False    # escape requested, heading not picked yet
        self._escape_origin: tuple[float, float] | None = None  # heading after the early window
        self._escape_reason = "revive"  # "revive" or "ring" (flower-window commit)
        self._last_break_dir: tuple[float, float] | None = None  # heading of the last breakout
        self._last_break_end = -1e9     # game_s when it ended

    @property
    def escaping(self) -> bool:
        """True while a committed post-REVIVE escape is requested or running."""
        return self._escape_pending or self._escape_remaining > 0

    @property
    def phase(self) -> Phase:
        return phase_for(self.game_s)

    def request_flee(
        self, duration_s: float | None = None, *, max_s: float | None = None
    ) -> float:
        """Force a breakout after REVIVE or an HP crash (ignore gem pull).

        Picks the open corridor on the next ``step`` even when encirclement
        metrics are still low (the crowd has not been re-measured yet).

        ``max_s`` caps the flee window (defaults to ``breakout_max_s``; HP-crash
        passes ``breakout_max_crash_s``). Extends an existing flee rather than
        shortening it. Returns the clamped duration applied this call.
        """
        t = self.tune
        dur = t.flee_after_revive_s if duration_s is None else float(duration_s)
        cap = t.breakout_max_s if max_s is None else float(max_s)
        clamped = max(t.breakout_min_s, min(dur, cap))
        self._flee_remaining = max(self._flee_remaining, clamped)
        self._end_breakout()  # pre-death / stale corridor; pick fresh on next start
        # A plain flee replaces any committed escape (request_revive_escape and the
        # flower-ring commit re-arm it right after this call).
        self._escape_remaining = 0.0
        self._escape_pending = False
        return clamped

    def request_revive_escape(self, duration_s: float | None = None) -> float:
        """Committed escape after REVIVE (built on ``request_flee``).

        For ``revive_escape_s`` (~3 s) of gameplay the navigator walks toward the
        lowest-threat-density direction (``escape_costs``), steering only within
        ~45 deg of that heading. The heading may be re-picked during the first
        ``escape_repick_s`` (stale first frames), afterwards only if that corridor
        is clearly shut and never into a reversal. An HP crash during the escape
        does not drop the heading. Calling it again while the REVIVE screen is up
        just restarts the same window. Returns the applied duration.
        """
        t = self.tune
        dur = t.revive_escape_s if duration_s is None else float(duration_s)
        return self._request_escape(dur, "revive")

    def _request_escape(self, dur: float, reason: str) -> float:
        t = self.tune
        dur = self.request_flee(dur, max_s=max(dur, t.breakout_min_s))
        self._escape_reason = reason
        self._flee_remaining = dur  # restart, do not stack repeated REVIVE frames
        self._escape_remaining = dur
        self._escape_s = 0.0
        self._escape_pending = True
        return dur

    def _end_breakout(self) -> None:
        """Drop the current breakout, remembering its heading for the no-reversal rule."""
        if self.breakout is not None:
            self._last_break_dir = self.breakout
            self._last_break_end = self.game_s
        self.breakout = None
        self._breakout_origin = None

    def _in_flower(self) -> bool:
        t = self.tune
        return t.flower_phase_lo <= self.game_s <= t.flower_phase_hi

    def _no_reverse(self, costs: np.ndarray, ref: tuple[float, float] | None) -> np.ndarray:
        """Flower window: forbid headings more than 90 deg from ``ref`` (inf cost)."""
        if ref is None or self.tune.flower_noreverse_s <= 0 or not self._in_flower():
            return costs
        ok = (_UNIT @ np.asarray(ref, dtype=np.float32)) >= -1e-6
        return np.where(ok, costs, np.inf)

    def _recent_break_dir(self) -> tuple[float, float] | None:
        """Last breakout heading if it ended within ``flower_noreverse_s`` (game time)."""
        if self._last_break_dir is None:
            return None
        if self.game_s - self._last_break_end > self.tune.flower_noreverse_s:
            return None
        return self._last_break_dir

    def _start_breakout(self, pts: np.ndarray, px: float, py: float) -> None:
        t = self.tune
        if self._escape_pending:
            self._escape_pending = False
            ecost = escape_costs(pts, px, py, t)
            if self._escape_reason == "ring":
                ecost = self._no_reverse(ecost, self._recent_break_dir())
            self.breakout = tuple(float(v) for v in _UNIT[int(np.argmin(ecost))])
            self._escape_origin = self.breakout
            self._breakout_origin = self.breakout
            self._breakout_s = 0.0
            self._breakout_check = 0.0
            self.breakouts += 1
            self.log(
                f"[nav] {self._escape_reason} escape {_nearest_dir(self.breakout)} for {self._escape_remaining:.1f}s "
                f"game~{self.game_s:.0f}s cover={self.cover:.2f}"
            )
            return
        costs = corridor_costs(pts, px, py, t.corridor_len, t.corridor_width)
        if self.current in DIRS:
            cu = _UNIT[DIRS.index(self.current)]
            align = _UNIT @ cu
            costs = costs + 0.25 * (1.0 - align)
        prev = self._recent_break_dir()
        costs = self._no_reverse(costs, prev)
        best = int(np.argmin(costs))
        if prev is not None and self._in_flower() and t.flower_noreverse_s > 0:
            self.log(
                f"[nav] flower no-reverse breakout {DIRS[best]} (last {_nearest_dir(prev)}) game~{self.game_s:.0f}s"
            )
        self.breakout = tuple(float(v) for v in _UNIT[best])
        self._breakout_origin = self.breakout
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
        self._hp_crash_peak = peak
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
        if flower != self._flower_on:
            self._flower_on = flower
            if flower:
                self.log(
                    f"[nav] flower window ON game~{self.game_s:.0f}s "
                    f"({t.flower_phase_lo:.0f}-{t.flower_phase_hi:.0f}s): cover trigger {t.cover_trigger_mid:.2f}, "
                    f"breakouts up to {t.breakout_max_crash_s:.1f}s"
                )
            else:
                self.log(f"[nav] flower window OFF game~{self.game_s:.0f}s")
        # Fresh HP crash: same flee path as post-revive (shared request_flee).
        # A committed revive escape keeps its heading (no fresh corridor pick).
        if crashing and not self._hp_crash_latched and self.escaping:
            self._hp_crash_latched = True
        if crashing and not self._hp_crash_latched:
            dur = self.request_flee(t.flee_on_hp_crash_s, max_s=t.breakout_max_crash_s)
            self._hp_crash_latched = True
            self.log(
                f"[nav] HP-crash flee: hp {self._hp_crash_peak:.2f}->{hp:.2f} within "
                f"{t.hp_crash_window_s:.1f}s, flee {dur:.1f}s game~{self.game_s:.0f}s"
            )
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

        if self.breakout is not None and self._escape_remaining > 0:
            self._update_escape(pts, px, py, dt)
            return
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
                self._end_breakout()
                self._flee_remaining = 0.0
                return
            if self._breakout_check >= 0.4:
                # Re-check the gap: switch only if the chosen corridor clearly closed.
                self._breakout_check = 0.0
                costs = corridor_costs(pts, px, py, t.corridor_len, t.corridor_width)
                cur = DIRS.index(_nearest_dir(self.breakout))
                best = int(np.argmin(self._no_reverse(costs, self._breakout_origin or self.breakout)))
                if costs[cur] > 1.6 * costs[best] + 0.5:
                    self.breakout = tuple(_UNIT[best])
            return
        if self._flee_remaining > 0 and self.breakout is None:
            # Post-revive / HP-crash: commit to a corridor even if the ring is not closing yet.
            self._start_breakout(pts, px, py)
            return
        if not closing:
            return
        if flower and t.ring_commit_s > 0:
            # Flower ring closing: commit (least-dense side, cone, no reversal).
            self._request_escape(t.ring_commit_s, "ring")
        self._start_breakout(pts, px, py)

    def _update_escape(self, pts: np.ndarray, px: float, py: float, dt: float) -> None:
        """Advance a committed post-REVIVE escape (no small-gap re-evaluation)."""
        t = self.tune
        self._breakout_s += dt
        self._escape_s += dt
        self._escape_remaining = max(0.0, self._escape_remaining - dt)
        self._flee_remaining = self._escape_remaining
        if self._escape_remaining <= 0:
            self.heading = math.atan2(self.breakout[1], self.breakout[0])
            self.log(f"[nav] {self._escape_reason} escape done {_nearest_dir(self.breakout)} game~{self.game_s:.0f}s")
            self._end_breakout()
            self._flee_remaining = 0.0
            return
        costs = escape_costs(pts, px, py, t)
        cur_d = _nearest_dir(self.breakout)
        cur = DIRS.index(cur_d)
        # The free early re-pick exists for stale frames right after REVIVE only;
        # ring (flower-window) escapes start from a live frame, so they are locked
        # from the first frame (no reversal, origin kept).
        early = self._escape_reason == "revive" and self._escape_s <= t.escape_repick_s
        if early:
            best = int(np.argmin(costs))
            switch = best != cur and costs[cur] > t.escape_repick_ratio * costs[best] + 0.3
        else:
            # Locked: only a clearly shut corridor, and never a reversal (the
            # replacement stays within 90 deg of the current heading).
            org = np.asarray(self._escape_origin or self.breakout, dtype=np.float32)
            ahead = ((_UNIT @ _UNIT[cur]) >= -1e-6) & ((_UNIT @ org) >= -1e-6)
            best = int(np.argmin(np.where(ahead, costs, np.inf)))
            switch = best != cur and costs[cur] > t.escape_switch_ratio * costs[best] + 1.5
        if switch:
            self.breakout = tuple(float(v) for v in _UNIT[best])
            if early:
                self._escape_origin = self.breakout
            self.log(f"[nav] {self._escape_reason} escape re-aim {cur_d}->{DIRS[best]} at +{self._escape_s:.1f}s")

    def _sync_timer(self, reading: float, now: float) -> None:
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
        # Ahead by more than the tolerance. Track the offset to the wall clock:
        # successive readings of one true clock keep that offset constant even
        # while game_s keeps lagging (slow frames hit the dt cap).
        off = reading - now
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
            self._sync_timer(float(result.timer_s), now)
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
        escape = self.breakout is not None and self._escape_remaining > 0
        # Denser early XP: boost gem pull before the flower without changing late knobs.
        score_tune = gem_tune_for(self.tune, self.game_s)
        cone: list[tuple[int, int]] | None = None
        if self.breakout is not None:
            t = self.tune
            plain = not escape and self._flee_remaining <= 0
            bgems = result.gems if plain and self.game_s < t.breakout_gem_until_s else None
            dec = score_directions(
                pts, (px, py), self.breakout, self.tune.breakout_weight,
                gems=bgems, current=self.current, hp=hp, tune=score_tune, idle_ok=False,
                kite_floor=True,
            )
            dec.mode = "breakout"
            kite = self.breakout
            if escape:
                # Committed: best-scoring direction inside the ~45 deg cone only
                # (the field can still side-step a body, never turn back).
                bx, by = self.breakout
                ox, oy = self._escape_origin or self.breakout
                cone = [
                    d for i, d in enumerate(DIRS)
                    if _UNIT[i, 0] * bx + _UNIT[i, 1] * by >= self.tune.escape_cone_dot
                    and _UNIT[i, 0] * ox + _UNIT[i, 1] * oy >= -1e-6
                ]
                dec.vector = max(cone, key=lambda d: dec.scores[d])
                dec.mode = f"{self._escape_reason}-escape"
            elif self.tune.flower_noreverse_s > 0 and self._in_flower():
                # Flower window: the field may side-step (<= 90 deg of the *start*
                # heading), never gradually flip 180 via successive re-aims.
                ox, oy = self._breakout_origin or self.breakout
                cone = [d for i, d in enumerate(DIRS) if _UNIT[i, 0] * ox + _UNIT[i, 1] * oy >= -1e-6]
                dec.vector = max(cone, key=lambda d: dec.scores[d])
        else:
            dec = score_directions(
                pts, (px, py), kite, phase.kite_weight,
                gems=result.gems, current=self.current, hp=hp,
                tune=score_tune, idle_ok=phase.idle_ok,
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
        # During the escape the dwell may only keep a chord that is inside the cone.
        keep_ok = cone is None or self.current in cone
        if keep_ok and choice != self.current and self.current != (0, 0) and self._since_change < self.tune.min_dwell_s:
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
