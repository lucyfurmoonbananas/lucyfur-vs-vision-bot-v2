"""8-way repulsion pathfinding around nearby monsters."""

from __future__ import annotations

import math

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


def choose_vector(
    result: VisionResult,
    last: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """
    Pick an 8-way step that maximizes clearance from nearby blobs.

    Player is treated as the capture center (Vampire Survivors camera lock).
    """
    if result.menu or not result.monsters:
        return (0, 0)

    px, py = result.player

    best = (0, 0)
    best_score = float("-inf")
    for vec in VECTORS_8:
        score = _score_vector(px, py, vec, result, last)
        if score > best_score:
            best_score = score
            best = vec

    # If every direction is roughly as bad as standing still, idle.
    idle = _score_vector(px, py, (0, 0), result, last)
    if best != (0, 0) and best_score < idle + 0.05:
        return (0, 0)
    return best


def _score_vector(
    px: int,
    py: int,
    vec: tuple[int, int],
    result: VisionResult,
    last: tuple[int, int],
) -> float:
    look = 36.0
    norm = math.hypot(vec[0], vec[1]) or 1.0
    sx = px + vec[0] / norm * look
    sy = py + vec[1] / norm * look
    nearest = 1e9
    for det in result.monsters:
        dx = sx - det.cx
        dy = sy - det.cy
        nearest = min(nearest, (dx * dx + dy * dy) ** 0.5)
    inertia = 0.08 if vec == last and vec != (0, 0) else 0.0
    return nearest * 0.02 + inertia


def keys_for_result(
    result: VisionResult,
    cfg: Config,
    last: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], tuple[str, ...]]:
    vector = choose_vector(result, last=last)
    return vector, cfg.keys_for_vector(vector)
