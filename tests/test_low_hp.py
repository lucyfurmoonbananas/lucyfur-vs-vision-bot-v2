"""Below ~20% HP the bot keeps a wider berth and stops chasing loot into enemies."""

from __future__ import annotations

import numpy as np

from vs_vision_bot.pathfind import Tuning, score_directions

from tests.test_movement import P, blob


def test_critical_hp_does_not_chase_chest_or_gems_into_enemies():
    enemies = np.array(blob(640, 256, n=6, spread=12), np.float32)
    gems = [(710.0, 250.0), (720.0, 260.0), (700.0, 270.0)]
    chest = (720.0, 256.0)
    low = score_directions(enemies, P, (1.0, 0.0), 1.0, gems=gems, hp=0.12, chest=chest)
    assert low.vector[0] <= 0
    assert low.mode != "chest"
    # Gems add nothing while HP is critical: the pull is already off under low_hp,
    # and it stays off here even with a chest in the same direction.
    bare = score_directions(enemies, P, (1.0, 0.0), 1.0, hp=0.12)
    with_gems = score_directions(enemies, P, (1.0, 0.0), 1.0, gems=gems, hp=0.12)
    assert with_gems.scores == bare.scores

    recovered = score_directions(enemies, P, (1.0, 0.0), 1.0, gems=gems, hp=0.85, chest=chest)
    assert recovered.vector[0] > 0
    assert Tuning().critical_hp == 0.2


def test_critical_hp_keeps_a_wider_berth_than_plain_low_hp():
    # One enemy 140 px ahead. At 45% the existing defensive band still walks toward
    # it; at 12% the berth should turn off.
    ahead = np.array([(P[0] + 140.0, P[1], 1.0)], np.float32)
    critical = score_directions(ahead, P, (1.0, 0.0), 1.0, hp=0.12)
    plain_low = score_directions(ahead, P, (1.0, 0.0), 1.0, hp=0.45)
    assert critical.vector[0] <= 0
    assert plain_low.vector[0] > 0
