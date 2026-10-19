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
* gems    - pull toward blue XP gems when nothing is close (stronger / farther before the flower),
  plus a non-cancelling pull at one target gem (densest clump per distance). While
  roaming with no chest, ``gem_seek_gain`` keeps that clump ahead of the kite heading.
  After the flower, a plain breakout does the same, fading at ``late_gem_danger``
  and only forward of the breakout so it cannot turn back into the ring.
  Each time that boost turns on is a gem-seek activation. Gem pickups are counted
  from rises in the XP bar (a level-up resets the bar and is not a pickup).
* chests  - walk toward an on-screen treasure chest, or the yellow edge arrow that
  points at an off-screen one. Outranks gems and roaming, including a plain
  breakout. A seek arms below ``chest_danger`` and then holds up to
  ``chest_clear_danger`` (the pull fades at ``chest_commit_fade``). The first
  missed sample is always kept, then further misses up to ``chest_hold_s``
  (0.6 s, about three frames at the live ~5 fps). Flees and committed escapes
  skip it. ``chest_gain`` 0
  disables. Seek transitions are logged as ``[chest]``.
  A sighting is also a world anchor (``[leash]``). The camera follows the
  player, so the anchor is dead-reckoned from the bot's own steps and
  re-anchored on every re-sight. An arrow is a direction past the playfield
  edge. While the anchor is pending and danger is moderate, the bot orbits
  it: tangent to the circle, plus a radial correction toward
  ``chest_orbit_radius`` (wider as danger rises). The orbit sign is whichever
  tangent corridor is more open, and it flips when the other side opens up.
  A low-danger corridor toward the chest (the same gate as a chest seek)
  steps in, unless the chest is being held. The first ``chest_free_opens``
  chests (default 3) open immediately. After that the bot keeps orbiting and
  treats the chest as a small keep-out disc until a weapon is evolution-ready
  and the clock is past ``chest_evo_after_s`` (10:00), or every weapon is
  maxed or evolved. ``set_build_state`` reports those two flags; both default
  to false, so an unwired tracker holds every chest after the free ones.
  Committed escapes, flees, and critical HP are not pulled and may cross the
  disc. When escape costs are close, the corridor toward the anchor wins by
  ``chest_leash_tie``. The anchor drops once the chest is reached, after
  ``chest_leash_forget_s`` without a re-sight, or after ``chest_leash_slip``
  of travel since the last sight. Orbit enter/exit, step-in, hold, and
  open-allowed are logged.
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
early or when its corridor is clearly shut, and it never turns back. If the ring
is still closed after that, or the crowd is still close, the escape continues
(up to 12 s) but the heading is re-picked toward the open corridor only while
the ring stays closed.
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
from vs_vision_bot.vision import PLAYFIELD_X0, PLAYFIELD_X1, REF_H_DEFAULT, VisionResult

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
# Symmetric corridor costs match to float noise. A looser band would hide a real gap.
_COST_TIE = 1e-3
# On-screen chest this close to the player counts as reached.
_LEASH_REACH = 48.0
# Revive corridors this close are one choice. An empty frame is all zeros, and a
# short northern ray a few tenths above an open diagonal is the same kind of tie.
# A real gap (about 2 vs the next corridor near 5) stays unique.
_REVIVE_CLOSE_ABS = 0.40
_REVIVE_CLOSE_REL = 0.20
# Chest or arrow must already lie along a tied runway (~30 deg). A hint on the
# short side, or on a corridor outside the tie, does not pull the heading.
_REVIVE_HINT_DOT = 0.85


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
    # Below this, wider berth and no gem or chest chase. 0 disables.
    critical_hp: float = 0.20
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
    # run3 at 641s and 651s: the 3s heading ended with cover still 1.0 (danger 12–19)
    # and the next death was 9–10s later. Hold the heading while the ring stays
    # closed, and stop once it opens or this cap is reached.
    revive_escape_max_s: float = 12.0
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
    # Surge commit only once the ring is mostly closed. cover_rising (0.40) plus
    # a small EMA rise is ordinary density noise: test3 logged 87 ring commits
    # in 11 minutes, 39 of them at cover 0.44, starting at game~1s. The wall
    # this is for reached cover 0.88. Plain breakouts still start at cover_rising.
    ring_commit_cover: float = 0.75
    # Flower window "no reversal" (Bob, Oct 6 3 AM logs: breakouts at game~300-340 s
    # flipped v=(1,..) -> (-1,..) every 3 s, walking back into the ring). Inside the
    # window a new breakout / ring escape, the 0.4 s corridor re-check, and the
    # per-frame vector may only pick headings within 90 deg of the last breakout
    # heading (side-steps OK, never back). While still in the flower phase the last
    # heading does not expire (Oct 11: ring escapes ~6 s apart flipped 180 after the
    # old 4 s TTL); outside the phase the TTL below still applies. 0 disables.
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
    early_gem_gain: float = 3.2       # vs gem_gain 1.6 (was 2.8; Oct 11 still under-leveled into flower)
    early_gem_radius: float = 460.0   # vs gem_radius 300; was 420 — reach gems left in the wake
    # First minute, while the field is still thin: look past early_gem_radius so a
    # gem near the screen edge is sought when danger is ~0. 0 disables.
    opening_gem_until_s: float = 75.0
    opening_gem_radius: float = 560.0
    # Gem target (Oct 16, from Oct 14-15 logs, 19 runs): at game 200-280 s 4-25 gems sat on
    # screen on average even in samples with danger < 0.8, so the summed gem pull is not
    # collecting them. That pull is a sum of unit vectors weighted 1/(1+d/80): a gem 300 px
    # away adds ~0.2, and gems scattered around the wide-circle trail cancel out, so the
    # kite bias (1.0) + hysteresis (0.35) win. On top of it, aim at ONE gem: the one with
    # the most gems within gem_cluster_r per distance (score count / (1 + d/150)), with a
    # pull that does not decay with distance or cancel. Same gates as the gem pull (inside
    # gem_radius, fades with gem_danger, off at low HP / in flee-escapes). Only before
    # early_gem_until_s via gem_tune_for (late game unchanged). 0 disables.
    early_gem_target_gain: float = 1.4
    gem_target_gain: float = 0.0      # base (late) value; gem_tune_for swaps in the early one
    # Roam only, and only when no chest is armed: raise the clump pull to this if
    # gem_target_gain is lower, so scattered gems still beat the kite heading once
    # danger is off zero (gem_danger still fades it; 0 disables).
    gem_seek_gain: float = 6.0
    gem_cluster_r: float = 70.0
    # After the flower window, plain breakouts seek gems again. The roam seek
    # uses the same higher fade, so a late crowd (danger ~2-4, still under this)
    # does not zero the pull the way gem_danger 1.6 does. 0 disables.
    late_gem_after_s: float = 360.0
    late_gem_danger: float = 5.0
    # On-screen chest or the yellow edge arrow. The pull fades out by chest_danger
    # until a seek arms; a latched seek then fades at chest_commit_fade and is
    # not dropped until chest_clear_danger. Not applied during a flee or
    # committed escape. chest_gain 0 disables.
    chest_gain: float = 8.0
    chest_danger: float = 1.6
    chest_clear_danger: float = 3.0
    chest_commit_fade: float = 4.5
    chest_commit_s: float = 0.75   # minimum hold once armed, while danger stays under chest_clear_danger
    # Ignore a chest/arrow detection miss up to this long. The first missed
    # frame is always held: live frames are ~0.2 s apart, so 0.10 never held
    # one (test3 on 3c8c971: 205 seek/clear pairs, 168 with no tick between).
    # VS_TUNE=chest_hold_s=0.10 restores that short window.
    chest_hold_s: float = 0.6
    # World anchor so a seen chest is not walked away from between sightings.
    # The hard pull starts past this (~1.5 reference-frame widths). 0 disables
    # the pull and the heading bias. VS_TUNE=chest_leash_radius=0.
    chest_leash_radius: float = 1440.0
    # Heading component added at the radius (linear in distance, capped at 1).
    # Kept under 1 so a breakout aimed away does not flip sign.
    chest_leash_bias: float = 0.55
    # An arrow anchor sits this far past the playfield edge, along the ray.
    chest_leash_beyond: float = 160.0
    # Drop the anchor after this long with no re-sight. 0 disables.
    chest_leash_forget_s: float = 90.0
    # Drop it after this much travel since the last sight (dead-reckon went stale).
    chest_leash_slip: float = 2880.0
    # Escape-cost shaved off corridors aligned with the anchor. A clearly
    # cheaper corridor still wins. 0 disables.
    chest_leash_tie: float = 0.4
    # Circle this far from the anchor while danger is between chest_danger and
    # chest_clear_danger. 0 disables the orbit (the straight leash pull remains).
    chest_orbit_radius: float = 280.0
    # Radial error that counts as a full correction. Also the hold band.
    chest_orbit_band: float = 60.0
    # Extra px of radius per unit of danger, capped at chest_clear_danger. 0 holds the radius fixed.
    chest_orbit_swell: float = 40.0
    # Weight of the orbit heading. High enough to beat a moderate crowd, under chest_gain.
    chest_orbit_gain: float = 6.0
    # The other tangent takes the orbit when its corridor cost is below this
    # fraction of the current side. Ties keep the way the body is already pointed.
    chest_orbit_switch: float = 0.7
    # Chests taken as soon as they drop. Later ones are held. A large value
    # disables the hold. VS_TUNE=chest_free_opens=9.
    chest_free_opens: int = 3
    # A held chest opens once a weapon is evolution-ready and the clock is here
    # (10:00). all_maxed ignores the clock. VS_TUNE=chest_evo_after_s=600.
    chest_evo_after_s: float = 600.0
    # Steps that would enter this disc around a held chest are charged in the
    # orbit costs. 0 disables. Emergencies do not consult it.
    chest_hold_keepout: float = 110.0
    # Early Garlic engage: before early_engage_until_s, HP >= early_engage_hp, danger < safe_danger
    # and no gems waiting to be collected (early_engage_max_gems), pull toward enemies.
    engage_gain: float = 0.0
    early_engage_until_s: float = 0.0     # off: Oct 14-16 A/B ON lv@2:00 2 vs OFF 3; VS_TUNE=early_engage_until_s=150 to test
    early_engage_gain: float = 1.2
    early_engage_hp: float = 0.8
    # Gems first, then hunt: the engage pull is off while more than this many gems are
    # within gem_radius (early_gem_radius before 280 s). Oct 14 live A/B (10 runs): engage
    # ON first level-up median ~81 s, lv@2:00 = 2 and only 0.3-0.9 gems on screen on
    # average vs OFF ~56 s, lv@2:00 = 3-5, 2-7 gems. Hypothesis: walking at the crowd
    # overrules the gem pull and leaves drops behind, so gems take priority here.
    # -1 disables the gate (engage regardless of gems).
    early_engage_max_gems: int = 0
    engage_radius: float = 260.0
    engage_standoff: float = 34.0
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
    """Return ``tune`` with early gem_gain/radius (and early Garlic engage) applied.

    No-op when ``early_gem_until_s`` is 0 or ``game_s`` is already past it, so late-game
    and VS_TUNE=early_gem_until_s=0 keep the base gem knobs unchanged. Before
    ``opening_gem_until_s`` the radius reaches the screen edge.
    """
    over: dict[str, float] = {}
    if tune.early_gem_until_s > 0 and game_s < tune.early_gem_until_s:
        radius = tune.early_gem_radius
        if tune.opening_gem_until_s > 0 and game_s < tune.opening_gem_until_s:
            radius = max(radius, tune.opening_gem_radius)
        over.update(
            gem_gain=tune.early_gem_gain, gem_radius=radius,
            gem_target_gain=tune.early_gem_target_gain,
        )
    if tune.early_engage_until_s > 0 and game_s < tune.early_engage_until_s:
        over["engage_gain"] = tune.early_engage_gain or tune.engage_gain
    elif tune.engage_gain:
        over["engage_gain"] = 0.0
    return replace(tune, **over) if over else tune


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
    return _fill_short_side(out, px, py)


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
    return _fill_short_side((c1 + c2.min(axis=1)) / tune.rollout_steps, px, py)


def gems_in_range(gems, px: float, py: float, radius: float) -> int:
    """Number of gem positions within ``radius`` px of the player (None / empty -> 0)."""
    if not gems:
        return 0
    g = np.asarray(gems, dtype=np.float32).reshape(-1, 2)
    return int((np.hypot(g[:, 0] - px, g[:, 1] - py) < radius).sum())


def gem_target(gems, px: float, py: float, radius: float, cluster_r: float) -> tuple[float, float] | None:
    """Gem to walk to: most gems within ``cluster_r`` of it per distance, inside ``radius``.

    Score = cluster count / (1 + d / 150): a near single gem beats a far one, a clump of
    gems a bit farther beats a lone near one. None when no gem is within ``radius``.
    """
    if gems is None or len(gems) == 0:
        return None
    g = np.asarray(gems, dtype=np.float32).reshape(-1, 2)
    d = np.hypot(g[:, 0] - px, g[:, 1] - py)
    g, d = g[d < radius], d[d < radius]
    if g.shape[0] == 0:
        return None
    pair = np.hypot(g[:, None, 0] - g[None, :, 0], g[:, None, 1] - g[None, :, 1])
    count = (pair < cluster_r).sum(axis=1).astype(np.float32)
    i = int(np.argmax(count / (1.0 + d / 150.0)))
    return float(g[i, 0]), float(g[i, 1])


def engage_pull(
    pts: np.ndarray, px: float, py: float, here: float, hp: float | None, tune: Tuning,
    gems=None,
) -> np.ndarray:
    """Early Garlic engagement: per-direction bonus toward nearby enemies.

    Zero unless ``tune.engage_gain`` > 0, HP is known and >= ``early_engage_hp``, and
    the danger here is below ``safe_danger`` (fades linearly to 0 at it). Only points
    between ``engage_standoff`` and ``engage_radius`` px pull, closer ones more, so the
    player lets weak early enemies into the aura instead of out-running them.
    Gems first: also zero while more than ``early_engage_max_gems`` gems are within
    ``gem_radius`` (the gem pull collects them; engage resumes once they are picked up).
    """
    out = np.zeros(len(DIRS), dtype=np.float32)
    if tune.engage_gain <= 0 or hp is None or hp < tune.early_engage_hp or tune.safe_danger <= 0:
        return out
    if tune.early_engage_max_gems >= 0 and gems_in_range(gems, px, py, tune.gem_radius) > tune.early_engage_max_gems:
        return out
    fade = 1.0 - here / tune.safe_danger
    if fade <= 0 or pts.shape[0] == 0:
        return out
    rel = pts[:, :2] - np.array([px, py], dtype=np.float32)
    d = np.hypot(rel[:, 0], rel[:, 1])
    m = (d > tune.engage_standoff) & (d < tune.engage_radius)
    if not m.any():
        return out
    w = pts[m, 2] / (1.0 + d[m] / 80.0)
    vx = float((w * rel[m, 0] / d[m]).sum())
    vy = float((w * rel[m, 1] / d[m]).sum())
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return out
    return (tune.engage_gain * fade * (_UNIT[:, 0] * vx / n + _UNIT[:, 1] * vy / n)).astype(np.float32)


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
    chest: tuple[float, float] | None = None,
    gem_forward: bool = False,
) -> Decision:
    """Pure scoring: returns the best 8-way (or idle) vector plus diagnostics."""
    tune = tune or Tuning()
    pts = np.asarray(threats, dtype=np.float32).reshape(-1, 3) if threats is not None else np.zeros((0, 3), np.float32)
    px, py = float(player[0]), float(player[1])
    critical = hp is not None and tune.critical_hp > 0 and hp < tune.critical_hp
    defensive = critical or (hp is not None and hp < tune.low_hp)
    if critical:
        danger_gain = tune.danger_gain * 3.2
        radius = tune.radius * 1.9
        body = tune.body * 1.7
        kite_mul = 0.35
        rollout_mul = 3.0
    else:
        danger_gain = tune.danger_gain * (1.8 if defensive else 1.0)
        radius = tune.radius * (1.25 if defensive else 1.0)
        body = tune.body
        kite_mul = 0.7 if defensive else 1.0
        rollout_mul = 1.8 if defensive else 1.0

    here = float(_danger_at(pts, np.array([px]), np.array([py]), radius)[0])
    nx = px + _UNIT[:, 0] * tune.step_near
    ny = py + _UNIT[:, 1] * tune.step_near
    fx = px + _UNIT[:, 0] * tune.step_far
    fy = py + _UNIT[:, 1] * tune.step_far
    danger = _danger_at(pts, nx, ny, radius) + 0.6 * _danger_at(pts, fx, fy, radius * 1.4)
    contact = _path_contact(pts, px, py, tune.step_far, body)
    scores = -(danger_gain * danger + tune.contact_gain * contact)
    if critical:
        scores = scores - corridor_costs(pts, px, py, tune.corridor_len, tune.corridor_width)
    if tune.rollout_gain > 0:
        scores = scores - tune.rollout_gain * rollout_mul * rollout_costs(pts, px, py, tune)

    # Kiting fades out when things get tight: pure escape then.
    panic = min(1.0, max(0.0, (here - tune.safe_danger) / max(tune.panic_danger - tune.safe_danger, 1e-6)))
    if kite_floor:
        # Breakout heading: stays strong under pressure (contact terms still veto bodies).
        kite_scale = kite_weight * (1.0 - 0.3 * panic)
    else:
        kite_scale = kite_weight * (1.0 - 0.75 * panic) * kite_mul
    if kite_dir is not None and kite_scale > 0:
        kx, ky = kite_dir
        norm = math.hypot(kx, ky) or 1.0
        scores = scores + kite_scale * (_UNIT[:, 0] * kx / norm + _UNIT[:, 1] * ky / norm)

    gem_scale = max(0.0, 1.0 - here / tune.gem_danger) if tune.gem_danger > 0 else 0.0
    gem_mask = None
    if gem_forward and kite_dir is not None and gem_scale > 0:
        kx, ky = float(kite_dir[0]), float(kite_dir[1])
        kn = math.hypot(kx, ky) or 1.0
        # Breakout: gems may bias the open half, not pull a U-turn back into the ring.
        gem_mask = (_UNIT[:, 0] * (kx / kn) + _UNIT[:, 1] * (ky / kn)) >= -1e-6
    if gems and gem_scale > 0 and not defensive and not critical:
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
            bonus = tune.gem_gain * gem_scale * np.clip(pull, -1.5, 1.5)
            if gem_mask is not None:
                bonus = np.where(gem_mask, np.maximum(bonus, 0.0), 0.0)
            scores = scores + bonus
            if tune.gem_target_gain > 0:
                tgt = gem_target(close, px, py, tune.gem_radius, tune.gem_cluster_r)
                if tgt is not None:
                    vx, vy = tgt[0] - px, tgt[1] - py
                    d = math.hypot(vx, vy)
                    if d > 1e-3:
                        bonus = tune.gem_target_gain * gem_scale * (
                            _UNIT[:, 0] * vx / d + _UNIT[:, 1] * vy / d
                        )
                        if gem_mask is not None:
                            bonus = np.where(gem_mask, np.maximum(bonus, 0.0), 0.0)
                        scores = scores + bonus

    if tune.engage_gain > 0 and pts.shape[0]:
        scores = scores + engage_pull(pts, px, py, here, hp, tune, gems=gems)

    chest_scale = 0.0
    chest_vx = chest_vy = 0.0
    if not critical and tune.chest_gain > 0 and tune.chest_danger > 0 and chest is not None:
        chest_scale = max(0.0, 1.0 - here / tune.chest_danger)
        chest_vx, chest_vy = float(chest[0]) - px, float(chest[1]) - py
        cd = math.hypot(chest_vx, chest_vy)
        if chest_scale > 0 and cd > 1e-3:
            scores = scores + tune.chest_gain * chest_scale * (
                _UNIT[:, 0] * chest_vx / cd + _UNIT[:, 1] * chest_vy / cd
            )
        else:
            chest_scale = 0.0

    if current in DIRS:
        scores[DIRS.index(current)] += tune.hysteresis * (1.0 - 0.6 * panic)

    table = {d: float(s) for d, s in zip(DIRS, scores)}
    prefer = current if current in DIRS else kite_dir
    best = DIRS[_pick_min(-scores, prefer)]
    # Standing still gets swarmed: only allowed late game ("stand in the
    # middle, weapons clear it") and only while nothing is close.
    toward_chest = False
    if chest_scale > 0:
        cd = math.hypot(chest_vx, chest_vy)
        if cd > 1e-3:
            toward_chest = (best[0] * chest_vx + best[1] * chest_vy) / (math.hypot(*best) * cd) > 0.5
    if panic > 0.5:
        mode = "escape"
    elif toward_chest:
        mode = "chest"
    elif kite_dir is not None:
        mode = "kite"
    else:
        mode = "roam"
    if idle_ok and not toward_chest and here < tune.safe_danger * 0.5:
        table[(0, 0)] = table[best] + 0.1
        return Decision((0, 0), table, here, "hold")
    return Decision(best, table, here, mode)


# How far to extend a ray's own density across the part of the screen the
# opposite vertical heading can still see. The camera sits at y=256 of 540, so
# a full-frame crowd has a shorter northern ray and that ray was scoring as the
# opening (run1 breakouts: up 59, down 13). 0 disables.
_SHORT_SIDE_FILL = 1.0


def _playfield_edge(px: float, py: float, ux: float, uy: float) -> float:
    """Pixels from (px, py) to the playfield border along a unit ray."""
    hits: list[float] = []
    if ux > 1e-8:
        hits.append((PLAYFIELD_X1 - px) / ux)
    elif ux < -1e-8:
        hits.append((PLAYFIELD_X0 - px) / ux)
    if uy > 1e-8:
        hits.append((float(REF_H_DEFAULT) - py) / uy)
    elif uy < -1e-8:
        hits.append((0.0 - py) / uy)
    ahead = [t for t in hits if t > 1.0]
    return min(ahead) if ahead else 0.0


def _edge_distance(px: float, py: float, ux: float, uy: float) -> float:
    """Pixels from (px, py) to the 960x540 border along a unit heading."""
    tx = 1e9 if abs(ux) < 1e-8 else ((-px) / ux if ux < 0 else (960.0 - px) / ux)
    ty = 1e9 if abs(uy) < 1e-8 else ((-py) / uy if uy < 0 else (540.0 - py) / uy)
    return float(min(tx, ty))


def _fill_short_side(costs: np.ndarray, px: float, py: float) -> np.ndarray:
    """Charge a short ray for the density it cannot see on the longer opposite side."""
    if _SHORT_SIDE_FILL <= 0:
        return costs
    out = np.asarray(costs, dtype=np.float32).copy()
    for i, (ux, uy) in enumerate(_UNIT):
        if out[i] <= 0:
            continue
        here = _edge_distance(px, py, float(ux), float(uy))
        there = _edge_distance(px, py, float(ux), -float(uy))
        short = there - here
        if short > 1.0 and here > 1.0:
            out[i] += out[i] / here * short * _SHORT_SIDE_FILL
    return out


def _pick_best(
    scores: dict[tuple[int, int], float],
    candidates: list[tuple[int, int]],
    prefer: tuple[float, float] | None,
) -> tuple[int, int]:
    """Highest score. Exact ties follow ``prefer``, not candidate order.

    ``candidates`` is a DIRS slice. ``max`` on that slice returns the first
    entry, and DIRS starts with north, so a tied east cone used to go north.
    """
    best = max(scores[d] for d in candidates)
    tied = [d for d in candidates if scores[d] >= best - 1e-5]
    if len(tied) == 1 or prefer is None:
        return tied[0]
    px, py = float(prefer[0]), float(prefer[1])
    n = math.hypot(px, py)
    if n < 1e-6:
        return tied[0]
    dots = [((d[0] * px + d[1] * py) / ((math.hypot(*d) or 1.0) * n)) for d in tied]
    top = max(dots)
    aligned = [d for d, dot in zip(tied, dots) if dot >= top - 1e-5]
    if len(aligned) == 1:
        return aligned[0]
    # Same clockwise rule as _pick_min.
    cross = [
        (d[0] / (math.hypot(*d) or 1.0)) * (py / n) - (d[1] / (math.hypot(*d) or 1.0)) * (px / n)
        for d in aligned
    ]
    return aligned[int(np.argmin(cross))]


def _pick_min(costs: np.ndarray, prefer: tuple[float, float] | None) -> int:
    """Index of the cheapest direction. Exact ties follow ``prefer``, not DIRS order.

    DIRS is up, down, left, then right. On a symmetric ring the vertical penalty
    leaves the other six corridors equal, and ``argmin`` then always returns left.
    When ``prefer`` is equally aligned with several tied headings (east, with
    north and south both open), DIRS order would still return north.
    """
    costs = np.asarray(costs, dtype=np.float32)
    best = float(np.min(costs))
    tied = np.flatnonzero(np.isfinite(costs) & (costs <= best + _COST_TIE))
    if tied.size == 0:
        return int(np.argmin(costs))
    if tied.size == 1 or prefer is None:
        return int(tied[0])
    px, py = float(prefer[0]), float(prefer[1])
    n = math.hypot(px, py)
    if n < 1e-6:
        return int(tied[0])
    dots = _UNIT[tied, 0] * (px / n) + _UNIT[tied, 1] * (py / n)
    best_dot = float(dots.max())
    cand = np.flatnonzero(dots >= best_dot - 1e-5)
    if cand.size == 1:
        return int(tied[int(cand[0])])
    # Clockwise on screen from prefer, so an east/west tie is not always north.
    sub = tied[cand]
    cross = _UNIT[sub, 0] * (py / n) - _UNIT[sub, 1] * (px / n)
    return int(sub[int(np.argmin(cross))])


def _toward_point(vec: tuple[float, float], pos: tuple[float, float], px: float, py: float) -> bool:
    """True when ``vec`` is within ~60 deg of the direction from the player to ``pos``."""
    if vec == (0, 0):
        return False
    vx, vy = float(pos[0]) - px, float(pos[1]) - py
    cd = math.hypot(vx, vy)
    vn = math.hypot(vec[0], vec[1])
    if cd < 1e-3 or vn < 1e-3:
        return False
    return (vec[0] * vx + vec[1] * vy) / (vn * cd) > 0.5


def chest_status(result: VisionResult) -> str:
    """Raw chest/arrow detection for the periodic ``[nav]`` line."""
    spot = result.chest
    label = "chest"
    if spot is None:
        spot = result.chest_arrow
        label = "arrow"
    if spot is None:
        return "chest=-"
    return f"{label}=({spot[0]:.0f},{spot[1]:.0f})"


def gem_status(nav: Navigator) -> str:
    """Running gem-seek activations and XP-bar pickup proxy for the ``[nav]`` line."""
    return f"gem_seek={nav.gem_seeks} gem_pickups={nav.gem_pickups}"


def format_gem_summary(nav: Navigator) -> str:
    """One line for the end of a run: counts plus pickups per minute of game time."""
    minutes = nav.game_s / 60.0
    rate = nav.gem_pickups / minutes if minutes >= 1.0 / 60.0 else 0.0
    return (
        f"[gems] seeks={nav.gem_seeks} pickups={nav.gem_pickups} "
        f"game~{nav.game_s:.0f}s {rate:.1f}/min"
    )


# HUD timer OCR plausibility (see Navigator._sync_timer).
TIMER_JUMP_TOL_S = 8.0       # accept readings this close to the wall-clock estimate
TIMER_RESYNC_N = 3           # consecutive agreeing out-of-range readings to resync
TIMER_RESYNC_AGREE_S = 2.0   # max spread of those readings' offsets to the estimate
TIMER_BEHIND_MAX_S = 45.0    # pause drift can lag ~30s; a dropped minute digit is 60s+


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
        self._timer_behind: list[float] = []  # ticking HUD readings that sit well behind game_s
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
        self._revive_held = False       # logged the surrounded-hold extension once
        self._revive_ring = False       # this revive escape has seen a closed ring
        self._aim_hint = None           # chest or arrow on the frame that picks a heading
        self._last_break_dir: tuple[float, float] | None = None  # heading of the last breakout
        self._last_break_end = -1e9     # game_s when it ended
        self._seek: tuple[str, tuple[float, float]] | None = None  # ("chest"|"arrow", xy) while seeking
        self._chest_latched = False
        self._chest_until = 0.0       # game_s: minimum hold of the current chest seek
        self._chest_hold: tuple[str, tuple[float, float]] | None = None
        self._chest_miss_s = 0.0
        self._chest_miss_n = 0
        self.gem_seeks = 0          # times the roam gem-seek boost turned on
        self.gem_pickups = 0        # XP-bar rises (pickup proxy; a level-up drop does not count)
        self._gem_seeking = False
        self._xp: float | None = None
        self._gem_summary_logged = False
        self._tie_bias = 0.0  # orbit radians kept across breakout-heading snaps
        self._spun = 0.0      # orbit radians since the last snap
        self._critical_on = False
        self._world = (0.0, 0.0)  # player position; the camera follows, so this is dead-reckoned
        self._leash: tuple[float, float] | None = None  # last chest, same coordinates
        self._leash_kind = ""
        self._leash_seen = 0.0
        self._leash_travel = 0.0  # px moved since the last sight
        self._chase = ""  # "" | "orbit" | "step" for this frame
        self._orbit_on = False
        self._orbit_sign = 0  # +1 / -1 around the anchor, 0 until a side is chosen
        self._orbit_aim: tuple[float, float] | None = None
        self._step_in = False
        self._chests_opened = 0
        # Conservative until the level-up tracker calls set_build_state.
        self._evolution_ready = False
        self._all_maxed = False
        self._policy = ""  # last logged open/hold decision for the current anchor

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
        ~45 deg of that heading. A flat cost vector takes the longest open edge
        (east on a full tie) instead of repeating the pre-death heading. The
        heading may be re-picked during the first ``escape_repick_s`` (stale
        first frames). Until ``revive_escape_s`` a forward corridor that has
        filled in (the plain breakout's 1.6x bar) also wins, still with no
        reversal. While the ring stays closed,
        or this escape has seen a closed ring and danger is still at or above
        ``safe_danger``, the heading is held up to ``revive_escape_max_s``. An HP
        crash during the escape does not drop the heading. Calling it again while
        the REVIVE screen is up just restarts the same window. Returns the applied
        duration.
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
        self._revive_held = False
        self._revive_ring = False
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
        """Last breakout heading remembered for flower no-reverse.

        Inside the Man-eater flower window the heading stays for the whole phase so
        consecutive ring escapes cannot flip 180 after ``flower_noreverse_s`` expires
        (Oct 11 1 AM run1: (-1,-1) at ~281 then (1,1) at ~289). Outside the window the
        ``flower_noreverse_s`` TTL still applies.
        """
        if self._last_break_dir is None or self.tune.flower_noreverse_s <= 0:
            return None
        if self._in_flower():
            return self._last_break_dir
        if self.game_s - self._last_break_end > self.tune.flower_noreverse_s:
            return None
        return self._last_break_dir

    def _start_breakout(self, pts: np.ndarray, px: float, py: float) -> None:
        t = self.tune
        if self._escape_pending:
            self._escape_pending = False
            ecost = escape_costs(pts, px, py, t)
            if self._escape_reason == "ring":
                # Wall commit: the open corridor wins. The chest leash does not vote.
                ecost = self._no_reverse(ecost, self._recent_break_dir())
                pick = _pick_min(ecost, self._roam_vec())
            else:
                pick = self._revive_pick(ecost, px, py)
            self.breakout = tuple(float(v) for v in _UNIT[pick])
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
        prev = self._recent_break_dir()
        costs = self._no_reverse(costs, prev)
        # A symmetric crowd ties. The 0.25 "don't reverse" nudge would then lock
        # the first DIRS winner (left) forever, so only apply it when one corridor
        # is actually thinner.
        finite = costs[np.isfinite(costs)]
        unique = finite.size > 0 and int(np.sum(finite <= float(finite.min()) + _COST_TIE)) == 1
        if unique and self.current in DIRS:
            cu = _UNIT[DIRS.index(self.current)]
            costs = costs + 0.25 * (1.0 - (_UNIT @ cu))
            costs = self._no_reverse(costs, prev)
        # Ties follow the remembered breakout inside the flower window (no-reverse
        # is already masking the back half). Outside it, follow the roam loop,
        # including the turn a breakout snap would otherwise throw away — DIRS
        # order used to make that tie left, and keeping the snap heading made
        # the first side stick for the rest of the run.
        if prev is not None and self._in_flower():
            prefer = prev
        else:
            ang = self.heading + self._tie_bias
            prefer = (math.cos(ang), math.sin(ang))
        best = _pick_min(costs, prefer)
        if prev is not None and self._in_flower() and t.flower_noreverse_s > 0:
            self.log(
                f"[nav] flower no-reverse breakout {DIRS[best]} (last {_nearest_dir(prev)}) game~{self.game_s:.0f}s"
            )
        self.breakout = tuple(float(v) for v in _UNIT[best])
        self._breakout_origin = self.breakout
        self._breakout_s = 0.0
        self._breakout_check = 0.0
        self.breakouts += 1

    def _roam_vec(self) -> tuple[float, float]:
        return (math.cos(self.heading), math.sin(self.heading))

    def _revive_pick(self, costs: np.ndarray, px: float, py: float) -> int:
        """Revive heading. Near-equal costs are not broken by DIRS order.

        Directions within ``max(0.40, 20% of the best)`` are one corridor. The
        longest runway to the screen edge wins, so a short northern ray does not
        beat an equally open east or west. Exact ties on that runway prefer
        east, unless this frame's chest or arrow — or, with nothing on screen,
        the dead-reckoned chest anchor — already lies along one of them.
        A unique gap stays the minimum.
        """
        costs = np.asarray(costs, dtype=np.float32)
        finite = costs[np.isfinite(costs)]
        if finite.size == 0:
            return _pick_min(costs, self._roam_vec())
        best = float(finite.min())
        band = max(_REVIVE_CLOSE_ABS, _REVIVE_CLOSE_REL * max(best, 0.0))
        close = np.flatnonzero(np.isfinite(costs) & (costs <= best + band))
        if close.size <= 1:
            return _pick_min(costs, self._roam_vec())
        edges = np.array(
            [_edge_distance(px, py, float(ux), float(uy)) for ux, uy in _UNIT],
            dtype=np.float32,
        )
        longest = float(edges[close].max())
        runway = close[edges[close] >= longest - 1.0]
        cheap = float(costs[runway].min())
        runway = runway[costs[runway] <= cheap + _COST_TIE]
        prefer = (1.0, 0.0)
        hint = self._revive_hint(px, py)
        if hint is not None and runway.size > 1:
            hx, hy = float(hint[0]) - px, float(hint[1]) - py
            hn = math.hypot(hx, hy)
            if hn > 1e-3:
                dots = _UNIT[runway, 0] * (hx / hn) + _UNIT[runway, 1] * (hy / hn)
                if float(dots.max()) >= _REVIVE_HINT_DOT:
                    prefer = (hx / hn, hy / hn)
        masked = np.full(len(DIRS), np.inf, dtype=np.float32)
        masked[runway] = 0.0
        return _pick_min(masked, prefer)

    def _revive_hint(self, px: float, py: float) -> tuple[float, float] | None:
        """Screen point that may break a revive tie. A live sight wins over the anchor."""
        if self._aim_hint is not None:
            return self._aim_hint
        if not self._leash_enabled():
            return None
        return self._leash_screen(px, py)

    def _keep_orbit(self, vec: tuple[float, float]) -> None:
        """Aim the roam along ``vec`` without dropping the loop's accumulated turn."""
        self._tie_bias += self._spun
        self._spun = 0.0
        self.heading = math.atan2(vec[1], vec[0])

    def _note_chest(self, kind: str | None, pos: tuple[float, float] | None, danger: float) -> None:
        """Log each transition into or out of an arrow-seek or chest-seek target."""
        prev = self._seek
        if kind is None or pos is None:
            if prev is not None:
                k, p = prev
                self.log(f"[chest] clear {k} ({p[0]:.0f},{p[1]:.0f}) danger={danger:.2f}")
            self._seek = None
            return
        spot = (float(pos[0]), float(pos[1]))
        if prev is not None and prev[0] == kind:
            self._seek = (kind, spot)
            return
        if prev is not None:
            k, p = prev
            self.log(f"[chest] clear {k} ({p[0]:.0f},{p[1]:.0f}) danger={danger:.2f}")
        self.log(f"[chest] seek {kind} ({spot[0]:.0f},{spot[1]:.0f}) danger={danger:.2f}")
        self._seek = (kind, spot)

    def _note_xp(self, xp: float | None) -> None:
        """Count an XP-bar rise as one pickup. A drop (level-up) only resets the baseline."""
        if xp is None:
            return
        prev = self._xp
        self._xp = xp
        if prev is None:
            return
        # ~2 px on the 818 px track. A 1 px flicker is not a gem.
        if xp > prev + (1.5 / 818.0):
            self.gem_pickups += 1

    def log_gem_summary(self) -> None:
        """Print the per-run gem line once. ``reset`` arms it again for the next run."""
        if self._gem_summary_logged:
            return
        self._gem_summary_logged = True
        self.log(format_gem_summary(self))

    def _kite_direction(self, pts: np.ndarray, px: float, py: float, dt: float) -> tuple[float, float]:

        phase = self.phase
        step = self.orbit * dt * 2.0 * math.pi / phase.orbit_period_s
        self._spun += step
        self.heading += step
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
        if self._escape_reason == "revive" and cover >= t.cover_trigger:
            self._revive_ring = True
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
                self._keep_orbit(self.breakout)
                self._end_breakout()
                self._flee_remaining = 0.0
                return
            if self._breakout_check >= 0.4:
                # Re-check the gap: switch only if the chosen corridor clearly closed.
                self._breakout_check = 0.0
                costs = corridor_costs(pts, px, py, t.corridor_len, t.corridor_width)
                cur = DIRS.index(_nearest_dir(self.breakout))
                best = _pick_min(self._no_reverse(costs, self._breakout_origin or self.breakout), self.breakout)
                if costs[cur] > 1.6 * costs[best] + 0.5:
                    self.breakout = tuple(_UNIT[best])
            return
        if self._flee_remaining > 0 and self.breakout is None:
            # Post-revive / HP-crash: commit to a corridor even if the ring is not closing yet.
            self._start_breakout(pts, px, py)
            return
        if not closing:
            return
        # A fast density rise into a real ring is a wall closing, including past
        # the flower window (run1 ~11:00: cells 105 -> 1746, cover 0.25 -> 0.88).
        # Commit the thin side the same way a flower ring does, so it is not
        # dropped after 1.1 s. cover just over cover_rising is not that wall.
        commit = cover >= t.ring_commit_cover and rising >= t.rise_trigger
        if (flower or commit) and t.ring_commit_s > 0:
            self._request_escape(t.ring_commit_s, "ring")
        self._start_breakout(pts, px, py)

    def _hold_revive_escape(self, t: Tuning, pts: np.ndarray, px: float, py: float) -> bool:
        """Keep a post-revive heading until the crowd is actually clear.

        A closed ring (cover at or above ``cover_trigger``) holds. A revive that
        started inside one also holds after the ring opens, while danger is
        still at or above ``safe_danger`` (test3: cover fell 0.94 -> 0.31 and
        the 12s hold never armed, then kite resumed at danger 0.7 and the ring
        shut again). A crowd that was never a ring still ends at
        ``revive_escape_s``. Open ground ends the hold. The cap is
        ``revive_escape_max_s``.
        """
        if self._escape_reason != "revive":
            return False
        danger = 0.0
        if pts is not None and len(pts):
            danger = float(_danger_at(pts, np.array([px]), np.array([py]), t.radius)[0])
        opened_but_close = self._revive_ring and danger >= t.safe_danger
        if self.cover < t.cover_trigger and not opened_but_close:
            return False
        room = t.revive_escape_max_s - self._escape_s
        if room <= 1e-3:
            return False
        self._escape_remaining = min(0.5, room)
        self._flee_remaining = self._escape_remaining
        if not self._revive_held:
            self._revive_held = True
            self.log(
                f"[nav] revive escape holding {_nearest_dir(self.breakout)} "
                f"(cover {self.cover:.2f} danger {danger:.2f}) "
                f"up to {t.revive_escape_max_s:.0f}s game~{self.game_s:.0f}s"
            )
        return True

    def _revive_hold_repick(self, costs: np.ndarray, cur: int) -> bool:
        """While a revive hold is running, aim at the current open corridor.

        The first ``revive_escape_s`` stays committed. After that, a closed ring
        keeps the escape going, but the heading follows the cheapest corridor,
        including a reversal when the old one has filled in.
        """
        t = self.tune
        if self._escape_reason != "revive" or self._escape_s <= t.revive_escape_s:
            return False
        if self.cover < t.cover_trigger:
            return False
        best = _pick_min(costs, self.breakout)
        if (
            best != cur
            and np.isfinite(costs[best])
            and costs[cur] > costs[best] * 1.1 + 0.15
        ):
            cur_d = DIRS[cur]
            self.breakout = tuple(float(v) for v in _UNIT[best])
            self._escape_origin = self.breakout
            self.log(
                f"[nav] revive escape re-aim {cur_d}->{DIRS[best]} at +{self._escape_s:.1f}s"
            )
        return True

    def _update_escape(self, pts: np.ndarray, px: float, py: float, dt: float) -> None:
        """Advance a committed post-REVIVE escape (no small-gap re-evaluation)."""
        t = self.tune
        self._breakout_s += dt
        self._escape_s += dt
        self._escape_remaining = max(0.0, self._escape_remaining - dt)
        self._flee_remaining = self._escape_remaining
        if self._escape_remaining <= 0:
            if not self._hold_revive_escape(t, pts, px, py):
                self._keep_orbit(self.breakout)
                self.log(f"[nav] {self._escape_reason} escape done {_nearest_dir(self.breakout)} game~{self.game_s:.0f}s")
                self._end_breakout()
                self._flee_remaining = 0.0
                return
        costs = escape_costs(pts, px, py, t)
        # Revive may lean toward a pending chest on a re-aim. A wall commit does not.
        if self._escape_reason != "ring":
            costs = self._nudge_leash(costs)
        cur_d = _nearest_dir(self.breakout)
        cur = DIRS.index(cur_d)
        if self._revive_hold_repick(costs, cur):
            return
        # The free early re-pick exists for stale frames right after REVIVE only;
        # ring (flower-window) escapes start from a live frame, so they are locked
        # from the first frame (no reversal, origin kept).
        early = self._escape_reason == "revive" and self._escape_s <= t.escape_repick_s
        if early:
            best = _pick_min(costs, self.breakout)
            switch = best != cur and costs[cur] > t.escape_repick_ratio * costs[best] + 0.3
        else:
            # Locked: only a clearly shut corridor, and never a reversal (the
            # replacement stays within 90 deg of the current heading).
            org = np.asarray(self._escape_origin or self.breakout, dtype=np.float32)
            ahead = ((_UNIT @ _UNIT[cur]) >= -1e-6) & ((_UNIT @ org) >= -1e-6)
            best = _pick_min(np.where(ahead, costs, np.inf), self.breakout)
            # Revive, before the hold: the picked corridor can fill without
            # reaching 3x. Same bar as a plain breakout re-check, and still no
            # reversal. Ring escapes, and a revive past its window, stay strict.
            if self._escape_reason == "revive" and self._escape_s <= t.revive_escape_s:
                switch = best != cur and costs[cur] > 1.6 * costs[best] + 0.5
            else:
                switch = best != cur and costs[cur] > t.escape_switch_ratio * costs[best] + 1.5
        if switch:
            self.breakout = tuple(float(v) for v in _UNIT[best])
            if early:
                self._escape_origin = self.breakout
            self.log(f"[nav] {self._escape_reason} escape re-aim {cur_d}->{DIRS[best]} at +{self._escape_s:.1f}s")

    def _note_timer_behind(self, reading: float) -> None:
        """Resync ``game_s`` when the HUD is far behind but keeps ticking.

        A stuck misread (the same wrong second repeated) does not grow the
        list, so it cannot pull the clock backwards. Three readings that each
        move forward by at least half a second are the real timer.
        """
        # A dropped minute digit (10:14 read as 0:14) ticks forward too, but it
        # is hundreds of seconds behind the estimate. Only a lag near the
        # current clock — a level-up pause — may resync.
        if self.game_s - reading > TIMER_BEHIND_MAX_S:
            self._timer_behind.clear()
            return
        prev = self._timer_behind
        if not prev or reading >= prev[-1] + 0.5:
            prev.append(reading)
        elif reading < prev[-1] - 0.5:
            self._timer_behind = [reading]
            return
        else:
            return
        if (
            len(self._timer_behind) >= TIMER_RESYNC_N
            and self._timer_behind[-1] - self._timer_behind[0] >= 1.0
        ):
            self.game_s = reading
            self._timer_behind.clear()

    def _chest_seen(
        self, result: VisionResult, escape: bool, dt: float,
    ) -> tuple[str, tuple[float, float]] | None:
        """Chest or arrow to chase, held through short detection misses.

        The first missed frame is always held, then up to ``chest_hold_s``.
        Flees and committed escapes return None so they keep their own heading.
        """
        if self.tune.chest_gain <= 0 or escape or self._flee_remaining > 0:
            self._chest_hold = None
            self._chest_miss_s = 0.0
            self._chest_miss_n = 0
            return None
        spot = result.chest or result.chest_arrow
        if spot is not None:
            kind = "chest" if result.chest is not None else "arrow"
            self._chest_hold = (kind, (float(spot[0]), float(spot[1])))
            self._chest_miss_s = 0.0
            self._chest_miss_n = 0
            return self._chest_hold
        if self._chest_hold is not None:
            self._chest_miss_s += dt
            self._chest_miss_n += 1
            # Live frames are ~0.2 s. Always keep the first miss, then keep
            # going until chest_hold_s (0.6 s). A second miss inside that
            # window used to clear when the limit was 0.10 s.
            if self._chest_miss_n <= 1 or self._chest_miss_s <= self.tune.chest_hold_s:
                return self._chest_hold
        self._chest_hold = None
        self._chest_miss_s = 0.0
        self._chest_miss_n = 0
        return None

    def _sync_timer(self, reading: float, now: float) -> None:
        """Plausibility-check an OCR'd HUD timer reading against the wall clock.

        ``self.game_s`` already holds the expected time (last accepted reading
        plus elapsed unpaused wall time). The minute digit is sometimes misread
        (e.g. 05:27 read as 06:27, 01:34 as 11:34), which would flip the phase, so:

        * a reading within ``TIMER_JUMP_TOL_S`` of the expectation is accepted;
        * a stuck misread behind the estimate is rejected (the clock does not
          jump backwards on a repeated wrong value);
        * a HUD that is behind but keeps ticking, and stays within
          ``TIMER_BEHIND_MAX_S`` of the estimate (a level-up pause), resyncs
          after ``TIMER_RESYNC_N`` increasing readings. A large backward jump
          (a dropped minute digit, 10:14 read as 0:14) is rejected;
        * a real drift ahead of the estimate (a stall of the loop) is resynced
          only when ``TIMER_RESYNC_N`` consecutive readings agree with each other.
        """
        expected = self.game_s
        if abs(reading - expected) <= TIMER_JUMP_TOL_S:
            self.game_s = max(expected, reading)
            self._timer_cands.clear()
            self._timer_behind.clear()
            return
        if reading < expected:
            # Far behind. A stuck misread stays rejected; a ticking HUD resyncs.
            self._timer_cands.clear()
            self._note_timer_behind(reading)
            return
        self._timer_behind.clear()
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

    def _advance_world(self, dt: float) -> None:
        """Dead-reckon the player from the chord just walked. The camera follows."""
        if dt <= 0.0 or self.current == (0, 0):
            return
        n = math.hypot(self.current[0], self.current[1])
        if n < 1e-6:
            return
        step = self.tune.player_speed * dt
        self._world = (
            self._world[0] + self.current[0] / n * step,
            self._world[1] + self.current[1] / n * step,
        )
        if self._leash is not None:
            self._leash_travel += step

    def _leash_offset(self) -> tuple[float, float]:
        return (self._leash[0] - self._world[0], self._leash[1] - self._world[1])

    def _leash_dist(self) -> float:
        if self._leash is None:
            return 0.0
        return math.hypot(*self._leash_offset())

    def _leash_enabled(self) -> bool:
        return self._leash is not None and self.tune.chest_gain > 0

    def set_build_state(self, evolution_ready: bool, all_maxed: bool) -> None:
        """Build progress from the level-up tracker. Logged only when it changes.

        ``evolution_ready`` means some weapon is max level and its partner
        passive is owned. ``all_maxed`` means every weapon is maxed or evolved.
        Both stay false until that tracker reports them.
        """
        evolution_ready = bool(evolution_ready)
        all_maxed = bool(all_maxed)
        if evolution_ready == self._evolution_ready and all_maxed == self._all_maxed:
            return
        self._evolution_ready = evolution_ready
        self._all_maxed = all_maxed
        self.log(f"[build] evolution_ready={evolution_ready} all_maxed={all_maxed}")

    def on_chest_opened(self) -> None:
        """Treasure Found screen. Count the open, then drop the anchor."""
        self._chests_opened += 1
        self._drop_leash("opened")

    def note_chest_opened(self) -> None:
        """Treasure screen: the chest is open, so the anchor is done."""
        self.on_chest_opened()

    def _open_reason(self) -> str | None:
        """Why the pending chest may be stepped on, or None while it is held."""
        if self._all_maxed:
            return "all-maxed"
        if self._chests_opened < self.tune.chest_free_opens:
            return "free"
        if self._evolution_ready and self.game_s >= self.tune.chest_evo_after_s:
            return "evolution"
        return None

    def _chest_open_allowed(self) -> bool:
        return self._open_reason() is not None

    def _note_open_policy(self) -> None:
        """Log the edge between holding a chest and being allowed to open it."""
        if self._leash is None:
            self._policy = ""
            return
        reason = self._open_reason()
        key = reason if reason is not None else "hold"
        if key == self._policy:
            return
        self._policy = key
        if reason is None:
            self.log(
                f"[chest] hold (opened={self._chests_opened} "
                f"evo_ready={self._evolution_ready} t={self.game_s:.0f})"
            )
        else:
            self.log(f"[chest] open-allowed ({reason})")

    def _keepout_costs(self) -> np.ndarray:
        """Cost for directions that would step into a held chest. Zeros otherwise."""
        costs = np.zeros(len(DIRS), dtype=np.float32)
        keep = self.tune.chest_hold_keepout
        if keep <= 0 or self._leash is None or self._chest_open_allowed():
            return costs
        dist = self._leash_dist()
        if dist < 1.0:
            return costs
        ox, oy = self._leash_offset()
        ix, iy = ox / dist, oy / dist
        step = max(float(self.tune.step_far), 1.0)
        for i, (ux, uy) in enumerate(_UNIT):
            inward = float(ux) * ix + float(uy) * iy
            radial = dist - inward * step
            if dist <= keep and inward > 0.05:
                costs[i] = 12.0
            elif inward > 0.0 and radial < keep:
                costs[i] = 12.0 * (keep - radial) / keep
        return costs

    def _bias_keepout(self, dec: Decision) -> None:
        pen = self._keepout_costs()
        if float(pen.max()) <= 0.0:
            return
        for i, d in enumerate(DIRS):
            if d in dec.scores:
                dec.scores[d] -= float(pen[i])
        dec.vector = _pick_best(dec.scores, list(DIRS), self._orbit_aim)

    def _drop_leash(self, reason: str) -> None:
        if self._leash is None:
            return
        if self._orbit_on:
            self._orbit_on = False
            self.log(f"[leash] orbit exit game~{self.game_s:.0f}s")
        self._step_in = False
        self._orbit_sign = 0
        self.log(f"[leash] drop {reason} {self._leash_kind} game~{self.game_s:.0f}s")
        self._leash = None
        self._leash_kind = ""
        self._leash_travel = 0.0
        self._policy = ""

    def _arrow_offset(self, arrow: tuple[float, float], px: float, py: float) -> tuple[float, float] | None:
        """Direction of the arrow, extended past the playfield edge."""
        dx, dy = float(arrow[0]) - px, float(arrow[1]) - py
        n = math.hypot(dx, dy)
        if n < 1e-3:
            return None
        ux, uy = dx / n, dy / n
        dist = _playfield_edge(px, py, ux, uy) + self.tune.chest_leash_beyond
        return (ux * dist, uy * dist)

    def _sight_leash(self, result: VisionResult, px: float, py: float) -> None:
        """Re-anchor from a raw chest or arrow. A chest in reach counts as arrived."""
        if result.chest is not None:
            off = (float(result.chest[0]) - px, float(result.chest[1]) - py)
            kind = "chest"
            if math.hypot(*off) <= _LEASH_REACH:
                self._drop_leash("reached")
                return
        elif result.chest_arrow is not None:
            off = self._arrow_offset(result.chest_arrow, px, py)
            if off is None:
                return
            kind = "arrow"
        else:
            return
        world = (self._world[0] + off[0], self._world[1] + off[1])
        first = self._leash is None
        self._leash = world
        self._leash_kind = kind
        self._leash_seen = self.game_s
        self._leash_travel = 0.0
        if first:
            self.log(
                f"[leash] {kind} ({world[0]:.0f},{world[1]:.0f}) "
                f"dist={math.hypot(*off):.0f} game~{self.game_s:.0f}s"
            )

    def _expire_leash(self) -> None:
        if self._leash is None:
            return
        if self._leash_dist() <= _LEASH_REACH:
            self._drop_leash("reached")
            return
        t = self.tune
        unseen = self.game_s - self._leash_seen
        if t.chest_leash_forget_s > 0 and unseen > t.chest_leash_forget_s:
            self._drop_leash("stale")
            return
        if t.chest_leash_slip > 0 and self._leash_travel > t.chest_leash_slip:
            self._drop_leash("slip")

    def _bias_heading(self, heading: tuple[float, float]) -> tuple[float, float]:
        """Lean ``heading`` toward the anchor. The lean grows with distance and stays < 1."""
        if (
            not self._leash_enabled()
            or self.tune.chest_leash_radius <= 0
            or self.tune.chest_leash_bias <= 0
        ):
            return heading
        ox, oy = self._leash_offset()
        dist = math.hypot(ox, oy)
        if dist < 1.0:
            return heading
        w = self.tune.chest_leash_bias * min(1.0, dist / self.tune.chest_leash_radius)
        hx, hy = float(heading[0]), float(heading[1])
        hn = math.hypot(hx, hy) or 1.0
        bx = hx / hn + w * ox / dist
        by = hy / hn + w * oy / dist
        bn = math.hypot(bx, by)
        if bn < 1e-6:
            return heading
        return (bx / bn, by / bn)

    def _hard_pull(self) -> bool:
        if not self._leash_enabled() or self.tune.chest_leash_radius <= 0 or self.tune.chest_gain <= 0:
            return False
        return self._leash_dist() > self.tune.chest_leash_radius

    def _leash_screen(self, px: float, py: float) -> tuple[float, float]:
        ox, oy = self._leash_offset()
        return (px + ox, py + oy)

    def _aim_leash(
        self,
        heading: tuple[float, float],
        live: tuple[float, float] | None,
        px: float,
        py: float,
        tune: Tuning,
        *,
        allow: bool,
    ) -> tuple[tuple[float, float], tuple[float, float] | None, Tuning, bool]:
        """Bias ``heading``. Past the radius, chase the dead-reckoned point.

        A live chest or arrow already in ``live`` wins, so this does not feed
        the off-screen point into the seek latch. The pull fades at
        ``chest_clear_danger`` instead of ``chest_danger``.
        """
        aim = self._bias_heading(heading) if allow else heading
        if live is not None or not allow or not self._hard_pull():
            return aim, live, tune, False
        far = replace(tune, chest_danger=max(tune.chest_danger, self.tune.chest_clear_danger))
        return aim, self._leash_screen(px, py), far, True

    def _nudge_leash(self, costs: np.ndarray) -> np.ndarray:
        """Shave a little cost off corridors facing the anchor. No leash: same array."""
        if not self._leash_enabled() or self.tune.chest_leash_tie <= 0:
            return costs
        ox, oy = self._leash_offset()
        dist = math.hypot(ox, oy)
        if dist < 1.0:
            return costs
        dots = _UNIT[:, 0] * (ox / dist) + _UNIT[:, 1] * (oy / dist)
        return costs - np.float32(self.tune.chest_leash_tie) * np.clip(dots, 0.0, None)

    def _orbit_radius(self, danger: float) -> float:
        t = self.tune
        extra = 0.0
        if t.chest_orbit_swell > 0 and t.chest_clear_danger > 0:
            extra = t.chest_orbit_swell * min(max(danger, 0.0), t.chest_clear_danger)
        radius = t.chest_orbit_radius + extra
        if t.chest_hold_keepout > 0 and not self._chest_open_allowed():
            radius = max(radius, t.chest_hold_keepout)
        return radius

    def _leash_chase(self, danger: float) -> str:
        """'orbit', 'step', or '' . Emergencies and a disabled orbit stay out of it."""
        if (
            not self._leash_enabled()
            or self.tune.chest_leash_radius <= 0
            or self.tune.chest_orbit_radius <= 0
            or self._critical_on
            or self.escaping
            or self._flee_remaining > 0
            or self.breakout is not None
        ):
            return ""
        if not self._chest_open_allowed():
            # Held: circle even when the corridor is clear. Danger at or past
            # chest_clear_danger falls through to the normal kite.
            if danger < self.tune.chest_clear_danger:
                return "orbit"
            return ""
        opening = danger < self.tune.chest_danger or (
            self._chest_latched and danger < self.tune.chest_clear_danger
        )
        if opening:
            return "step"
        if danger < self.tune.chest_clear_danger:
            return "orbit"
        return ""

    def _heading_cost(self, costs: np.ndarray, hx: float, hy: float) -> float:
        dots = _UNIT[:, 0] * hx + _UNIT[:, 1] * hy
        w = np.clip(dots, 0.0, None) ** 2
        total = float(w.sum())
        if total < 1e-6:
            return float(np.min(costs))
        return float((costs * w).sum() / total)

    def _set_orbit_sign(self, c_pos: float, c_neg: float, t_pos: tuple[float, float], t_neg: tuple[float, float]) -> int:
        """Pick the open tangent. A tie keeps the sign, or follows the current heading."""
        margin = self.tune.chest_orbit_switch if self.tune.chest_orbit_switch > 0 else 1.0
        if self._orbit_sign == 0:
            if c_neg < c_pos * margin:
                self._orbit_sign = -1
            elif c_pos < c_neg * margin:
                self._orbit_sign = 1
            else:
                hx, hy = math.cos(self.heading), math.sin(self.heading)
                dp = hx * t_pos[0] + hy * t_pos[1]
                dn = hx * t_neg[0] + hy * t_neg[1]
                self._orbit_sign = 1 if dp >= dn else -1
        else:
            cur = c_pos if self._orbit_sign > 0 else c_neg
            other = c_neg if self._orbit_sign > 0 else c_pos
            if other < cur * margin:
                self._orbit_sign = -self._orbit_sign
        return self._orbit_sign

    def _orbit_heading(self, pts: np.ndarray, px: float, py: float, danger: float) -> tuple[float, float]:
        """Tangent around the anchor plus a pull back to the orbit radius."""
        ox, oy = self._leash_offset()
        dist = math.hypot(ox, oy)
        if dist < 1.0:
            return (1.0, 0.0)
        ix, iy = ox / dist, oy / dist
        t_pos = (iy, -ix)
        t_neg = (-iy, ix)
        costs = corridor_costs(pts, px, py, self.tune.corridor_len, self.tune.corridor_width)
        if not self._chest_open_allowed():
            costs = costs + self._keepout_costs()
        sign = self._set_orbit_sign(
            self._heading_cost(costs, t_pos[0], t_pos[1]),
            self._heading_cost(costs, t_neg[0], t_neg[1]),
            t_pos, t_neg,
        )
        tx, ty = t_pos if sign > 0 else t_neg
        band = max(self.tune.chest_orbit_band, 1.0)
        err = max(-1.0, min(1.0, (dist - self._orbit_radius(danger)) / band))
        hx = tx * (1.0 - abs(err)) + ix * err
        hy = ty * (1.0 - abs(err)) + iy * err
        n = math.hypot(hx, hy)
        if n < 1e-6:
            return (tx, ty)
        return (hx / n, hy / n)

    def _note_leash_chase(self, danger: float) -> None:
        dist = self._leash_dist() if self._leash is not None else 0.0
        if self._chase == "orbit":
            if not self._orbit_on:
                self._orbit_on = True
                self.log(
                    f"[leash] orbit r={self._orbit_radius(danger):.0f} dist={dist:.0f} "
                    f"side={self._orbit_sign:+d} game~{self.game_s:.0f}s"
                )
            self._step_in = False
            return
        if self._orbit_on:
            self._orbit_on = False
            self.log(f"[leash] orbit exit game~{self.game_s:.0f}s")
        if self._chase == "step":
            if not self._step_in:
                self._step_in = True
                self.log(f"[leash] step-in dist={dist:.0f} game~{self.game_s:.0f}s")
            return
        self._step_in = False

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
        self._advance_world(dt)
        self._note_critical(hp)
        self._note_xp(result.xp)
        self._since_change += dt

        pts = result.threats
        if pts is None:
            pts = _points_from_detections(result)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        px, py = _ref_player(result)
        self._sight_leash(result, px, py)
        self._expire_leash()
        self._note_open_policy()
        phase = self.phase
        kite = self._kite_direction(pts, px, py, dt)
        spot = result.chest or result.chest_arrow
        self._aim_hint = (float(spot[0]), float(spot[1])) if spot is not None else None
        self._update_breakout(pts, px, py, dt, hp)
        escape = self.breakout is not None and self._escape_remaining > 0
        here = float(_danger_at(pts, np.array([px]), np.array([py]), self.tune.radius)[0])
        self._chase = self._leash_chase(here)
        self._orbit_aim = self._orbit_heading(pts, px, py, here) if self._chase == "orbit" else None
        self._note_leash_chase(here)
        # Denser early XP: boost gem pull before the flower without changing late knobs.
        score_tune = gem_tune_for(self.tune, self.game_s)
        seen = self._chest_seen(result, escape, dt)
        if seen is not None and not self._chest_open_allowed():
            # The orbit keeps the chest. A seek here would walk onto it.
            seen = None
        chest_kind = seen[0] if seen is not None else None
        chest_at = seen[1] if seen is not None else None
        cone: list[tuple[int, int]] | None = None
        gem_seek_on = False
        # Critical HP drops a latched chest. A committed escape (including a
        # fast-density wall) already made _chest_seen return None.
        if self._critical_on:
            chest_kind = None
            chest_at = None
            self._chest_latched = False
            self._chest_until = 0.0
            self._chest_hold = None
            self._chest_miss_s = 0.0
            self._chest_miss_n = 0
        dec: Decision | None = None
        leash_pull = False
        # A latched seek that scores into a real flee (danger >= chest_clear_danger)
        # is thrown away and rescored with no chest, so the commit cannot walk
        # into that crowd.
        for drop_chest in (False, True):
            gem_seek_on = False
            cone = None
            this_chest = None if drop_chest else chest_at
            pulled = False
            base = score_tune
            if self._chest_latched and this_chest is not None and self.tune.chest_commit_fade > base.chest_danger:
                base = replace(base, chest_danger=self.tune.chest_commit_fade)
            if self.breakout is not None:
                t = self.tune
                plain = not escape and self._flee_remaining <= 0
                in_early = plain and t.breakout_gem_until_s > 0 and self.game_s < t.breakout_gem_until_s
                in_late = (
                    plain
                    and t.late_gem_after_s > 0
                    and self.game_s >= t.late_gem_after_s
                    and self.game_s > t.flower_phase_hi
                )
                bgems = None if self._critical_on else (result.gems if in_early or in_late else None)
                btune = replace(base, engage_gain=0.0) if base.engage_gain else base
                if in_late and bgems and not in_early:
                    over: dict[str, float] = {}
                    if t.late_gem_danger > btune.gem_danger:
                        over["gem_danger"] = t.late_gem_danger
                    if this_chest is None and t.gem_seek_gain > btune.gem_target_gain:
                        over["gem_target_gain"] = t.gem_seek_gain
                    if over:
                        btune = replace(btune, **over)
                # Plain flee and breakout lean toward a pending chest. A committed
                # escape does not; its corridor pick already took the tie nudge.
                allow_leash = not drop_chest and not self._critical_on and not escape
                aim, score_chest, btune, pulled = self._aim_leash(
                    self.breakout, this_chest if plain else None, px, py, btune, allow=allow_leash,
                )
                dec = score_directions(
                    pts, (px, py), aim, self.tune.breakout_weight,
                    gems=bgems, current=self.current, hp=hp, tune=btune, idle_ok=False,
                    kite_floor=True, chest=score_chest,
                    gem_forward=bool(in_late and not in_early and bgems),
                )
                walked_chest = dec.mode == "chest"
                dec.mode = "breakout"
                kite = self.breakout
                if escape:
                    # Committed: best-scoring direction inside the ~45 deg cone only
                    # (the field can still side-step a body, never turn back).
                    # Ties follow the heading, not DIRS order (north is first).
                    bx, by = self.breakout
                    ox, oy = self._escape_origin or self.breakout
                    cone = [
                        d for i, d in enumerate(DIRS)
                        if _UNIT[i, 0] * bx + _UNIT[i, 1] * by >= self.tune.escape_cone_dot
                        and _UNIT[i, 0] * ox + _UNIT[i, 1] * oy >= -1e-6
                    ]
                    dec.vector = _pick_best(dec.scores, cone, self.breakout)
                    dec.mode = f"{self._escape_reason}-escape"
                elif self.tune.flower_noreverse_s > 0 and self._in_flower():
                    # Flower window: the field may side-step (<= 90 deg of the *start*
                    # heading), never gradually flip 180 via successive re-aims.
                    ox, oy = self._breakout_origin or self.breakout
                    cone = [d for i, d in enumerate(DIRS) if _UNIT[i, 0] * ox + _UNIT[i, 1] * oy >= -1e-6]
                    dec.vector = _pick_best(dec.scores, cone, self._breakout_origin or self.breakout)
                if walked_chest and not escape:
                    dec.mode = "chest"
            else:
                roam_tune = base
                if (
                    not self._critical_on
                    and self._chase == ""
                    and this_chest is None
                    and result.gems
                    and self.tune.gem_seek_gain > base.gem_target_gain
                ):
                    roam_tune = replace(base, gem_target_gain=self.tune.gem_seek_gain)
                    if (
                        self.tune.late_gem_after_s > 0
                        and self.game_s >= self.tune.late_gem_after_s
                        and self.game_s > self.tune.flower_phase_hi
                        and self.tune.late_gem_danger > roam_tune.gem_danger
                    ):
                        roam_tune = replace(roam_tune, gem_danger=self.tune.late_gem_danger)
                    gem_seek_on = True
                allow_leash = not drop_chest and not self._critical_on
                if self._chase == "orbit" and allow_leash and self._orbit_aim is not None:
                    # Circle the anchor. No straight chest pull; the heading carries it.
                    pulled = False
                    dec = score_directions(
                        pts, (px, py), self._orbit_aim, self.tune.chest_orbit_gain,
                        gems=None if self._critical_on else result.gems, current=self.current, hp=hp,
                        tune=roam_tune, idle_ok=False, kite_floor=True, chest=None,
                    )
                    self._bias_keepout(dec)
                elif (
                    self._chase == "step" and allow_leash and this_chest is None and self._leash is not None
                ):
                    pulled = True
                    dec = score_directions(
                        pts, (px, py), kite, phase.kite_weight,
                        gems=None if self._critical_on else result.gems, current=self.current, hp=hp,
                        tune=roam_tune, idle_ok=phase.idle_ok,
                        chest=self._leash_screen(px, py),
                    )
                else:
                    aim, score_chest, roam_tune, pulled = self._aim_leash(
                        kite, this_chest, px, py, roam_tune, allow=allow_leash,
                    )
                    dec = score_directions(
                        pts, (px, py), aim, phase.kite_weight,
                        gems=None if self._critical_on else result.gems, current=self.current, hp=hp,
                        tune=roam_tune, idle_ok=phase.idle_ok,
                        chest=score_chest,
                    )
            leash_pull = pulled
            if (
                drop_chest
                or this_chest is None
                or not self._chest_latched
                or dec.danger_here < self.tune.chest_clear_danger
            ):
                break
            self._chest_latched = False
        assert dec is not None
        if self._chase == "orbit" and dec.mode != "escape":
            dec.mode = "leash-orbit"
        elif self._chase == "step" and leash_pull and dec.mode == "chest":
            dec.mode = "leash-step"
        elif leash_pull and dec.mode == "chest":
            dec.mode = "leash"

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
        high = dec.danger_here >= self.tune.chest_clear_danger
        toward = (
            chest_at is not None
            and chest_kind is not None
            and _toward_point(choice, chest_at, px, py)
        )
        if chest_at is None or chest_kind is None or high:
            self._chest_latched = False
            self._chest_until = 0.0
        elif self._chest_latched or dec.danger_here < self.tune.chest_danger:
            self._chest_latched = True
            if dec.danger_here < self.tune.chest_danger or toward:
                self._chest_until = self.game_s + self.tune.chest_commit_s
        elif self.game_s >= self._chest_until:
            self._chest_latched = False
        if self._chest_latched and chest_at is not None and (toward or self.game_s < self._chest_until):
            self._note_chest(chest_kind, chest_at, dec.danger_here)
        else:
            self._note_chest(None, None, dec.danger_here)
        if gem_seek_on and not self._gem_seeking:
            self.gem_seeks += 1
        self._gem_seeking = gem_seek_on
        mode = f"{dec.mode}/{phase.name}"
        if self._critical_on:
            mode = f"{mode} critical"
        dec = Decision(choice, dec.scores, dec.danger_here, mode)
        self.last = dec
        return dec

    def _note_critical(self, hp: float | None) -> None:
        """Log the edges of the critical-HP band. The [nav] mode carries the tag."""
        on = hp is not None and self.tune.critical_hp > 0 and hp < self.tune.critical_hp
        if on == self._critical_on:
            return
        self._critical_on = on
        shown = "?" if hp is None else f"{hp:.2f}"
        self.log(f"[nav] critical HP {'on' if on else 'off'} hp={shown} game~{self.game_s:.0f}s")


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
