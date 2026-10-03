import cv2
import numpy as np

from vs_vision_bot.config import Config
from vs_vision_bot.vision import (
    Vision,
    detect_gems,
    detect_level_up_menu,
    detect_threat_blobs,
    draw_synthetic_arena,
    gem_mask,
    parse_timer_text,
)


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


def test_broken_model_latches_off_after_first_failure(capsys):
    vision = Vision(Config())
    calls = {"n": 0}

    def boom(_frame):
        calls["n"] += 1
        raise RuntimeError("broken graph")

    vision._model = object()
    vision._infer_model = boom
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first = vision.analyze(frame)
    second = vision.analyze(frame)

    assert calls["n"] == 1
    assert vision._model is None
    assert first.note == "heuristic"
    assert second.note == "heuristic"
    assert capsys.readouterr().err.count("model failed") == 1


def _paint_gem(frame, cx, cy):
    """Tiny saturated blue XP gem core (BGR ~228,180,38)."""
    for dy in range(-3, 4):
        for dx in range(-2, 3):
            if dx * dx + dy * dy <= 8:
                frame[cy + dy, cx + dx] = (228, 180, 38)


def _paint_glowing_bat(frame, cx, cy):
    """Dark bat body with a cyan-blue outline like the Glowing Bat boss."""
    cv2.ellipse(frame, (cx, cy), (14, 11), 0, 0, 360, (8, 8, 10), -1)
    cv2.ellipse(frame, (cx, cy), (16, 13), 0, 0, 360, (254, 202, 110), 2)


def test_parse_timer_text_accepts_common_forms():
    assert parse_timer_text("3:45") == 3 * 60 + 45
    assert parse_timer_text("12:00") == 12 * 60
    assert parse_timer_text("01:41") == 60 + 41
    assert parse_timer_text("0:20") == 20
    assert parse_timer_text("00:20") == 20
    assert parse_timer_text("  9:05\n") == 9 * 60 + 5
    assert parse_timer_text('"00:20') == 20  # leading OCR noise


def test_parse_timer_text_colon_read_as_a_digit():
    # Oct 3 run5 death frames (true 06:10-06:20, a sprite over the colon) and
    # session shots: tesseract's digit whitelist turns the colon into a digit.
    assert parse_timer_text("06313") == 6 * 60 + 13
    assert parse_timer_text("06210") == 6 * 60 + 10
    assert parse_timer_text("04056") == 4 * 60 + 56
    assert parse_timer_text("00205") == 5
    assert parse_timer_text("110205") is None


def test_parse_timer_text_rejects_garbage():
    assert parse_timer_text("") is None
    assert parse_timer_text("Sorat") is None
    assert parse_timer_text("3:99") is None
    assert parse_timer_text("99:01") is None
    assert parse_timer_text("0125") is None


def test_synthetic_gem_found_glowing_bat_rejected():
    frame = draw_synthetic_arena(960, 540, monsters=[])
    _paint_gem(frame, 600, 200)
    _paint_glowing_bat(frame, 700, 320)
    gems = detect_gems(frame)
    assert any(abs(x - 600) < 8 and abs(y - 200) < 8 for x, y in gems)
    assert not any(abs(x - 700) < 24 and abs(y - 320) < 24 for x, y in gems)
    # gem_mask itself must not light up the bat outline.
    m = gem_mask(frame)
    assert int(m[320 - 18 : 320 + 18, 700 - 18 : 700 + 18].sum()) == 0
