"""8-way repulsion pathfinding around nearby monsters."""

from __future__ import annotations

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


def snap_vector(fx: float, fy: float, deadzone: float = 0.18) -> tuple[int, int]:
    mag = (fx * fx + fy * fy) ** 0.5
    if mag < deadzone:
        return (0, 0)
    nx, ny = fx / mag, fy / mag
    sx = 0 if abs(nx) < 0.38 else (1 if nx > 0 else -1)
    sy = 0 if abs(ny) < 0.38 else (1 if ny > 0 else -1)
    return (sx, sy)


def choose_vector(
    result: VisionResult,
    cfg: Config,
    last: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """
    Pick an 8-way step that maximizes clearance from nearby blobs.

    Player is treated as the capture center (Vampire Survivors camera lock).
    """
    if result.menu or not result.monsters:
        return (0, 0)

    px, py = result.player
    radius = float(max(cfg.threat_radius, 8))

    best = (0, 0)
    best_score = float("-inf")
    for vec in VECTORS_8:
        score = _score_vector(px, py, vec, result, radius, last)
        if score > best_score:
            best_score = score
            best = vec

    # If every direction is roughly as bad as standing still, idle.
    idle = _score_vector(px, py, (0, 0), result, radius, last)
    if best != (0, 0) and best_score < idle + 0.05:
        return (0, 0)
    return best


def _score_vector(
    px: int,
    py: int,
    vec: tuple[int, int],
    result: VisionResult,
    radius: float,
    last: tuple[int, int],
) -> float:
    look = 36.0
    sx = px + vec[0] * look
    sy = py + vec[1] * look
    threat = 0.0
    nearest = 1e9
    for det in result.monsters:
        dx = sx - det.cx
        dy = sy - det.cy
        dist = (dx * dx + dy * dy) ** 0.5
        nearest = min(nearest, dist)
        # Extra penalty if the step walks toward the blob.
        toward = (det.cx - px) * vec[0] + (det.cy - py) * vec[1]
        weight = 1.0 + 0.55 * max(0.0, toward / (abs(toward) + 40.0))
        threat += weight / (dist * dist + radius)
    clearance = nearest
    # Prefer moving over standing when something is inside the threat radius.
    move_bonus = 0.15 if vec != (0, 0) and nearest < radius else 0.0
    inertia = 0.08 if vec == last and vec != (0, 0) else 0.0
    return clearance * 0.02 - threat + move_bonus + inertia


def keys_for_result(
    result: VisionResult,
    cfg: Config,
    last: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], tuple[str, ...]]:
    vector = choose_vector(result, cfg, last=last)
    return vector, cfg.keys_for_vector(vector)
