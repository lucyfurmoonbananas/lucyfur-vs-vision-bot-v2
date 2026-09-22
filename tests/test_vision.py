import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.vision import detect_level_up_menu, detect_threat_blobs, draw_synthetic_arena


def test_synthetic_arena_finds_monsters_away_from_player():
    planted = [(700, 200), (180, 400)]
    frame = draw_synthetic_arena(960, 540, monsters=planted)
    cfg = Config()
    blobs = detect_threat_blobs(frame, (480, 270), cfg)
    assert {(blob.cx, blob.cy) for blob in blobs} == set(planted)


def test_level_up_cards_are_detected():
    frame = np.full((540, 960, 3), 18, dtype=np.uint8)
    # Three upgrade cards in the center — typical VS level-up layout.
    for x in (180, 400, 620):
        frame[140:400, x : x + 160] = (70, 90, 120)
    assert detect_level_up_menu(frame) is True


def test_open_playfield_is_not_a_menu():
    frame = draw_synthetic_arena(960, 540, monsters=[(600, 200)])
    assert detect_level_up_menu(frame) is False
