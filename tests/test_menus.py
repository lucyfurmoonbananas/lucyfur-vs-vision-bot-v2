"""Level-up / chest / revive handling on real 960x540 frames from the game."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot import menus
from vs_vision_bot.input_backend import NullBackend
from vs_vision_bot.menus import (
    Card,
    MenuPilot,
    choose_card,
    classify_chest_text,
    detect_level_up,
    find_word,
    match_name,
    read_hp_fraction,
)

FIX = Path(__file__).parent / "fixtures"
needs_tesseract = pytest.mark.skipif(not menus.tesseract_available(), reason="tesseract not installed")


def load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIX / name))
    assert frame is not None, name
    return frame


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_level_up_screen_geometry_and_selection():
    screen = detect_level_up(load("levelup_4cards_sel4.png"))
    assert screen is not None
    assert len(screen.cards) == 4
    tops = [c.y0 for c in screen.cards]
    assert tops == sorted(tops)
    # The white selection arrows sit on the 4th card in this frame.
    assert screen.selected == 3


@pytest.mark.parametrize(
    "name",
    ["main_menu.png", "gameplay_full_hp.png", "gameplay_hurt.png", "game_over_revive.png", "game_over_quit.png"],
)
def test_non_level_up_frames_are_not_level_up(name):
    assert detect_level_up(load(name)) is None


def test_synthetic_dark_frames_are_not_level_up():
    assert detect_level_up(np.zeros((540, 960, 3), np.uint8)) is None
    assert detect_level_up(np.full((540, 960, 3), 139, np.uint8)) is None


@needs_tesseract
def test_card_names_are_read_and_garlic_is_picked():
    frame = load("levelup_4cards_sel4.png")
    screen = menus.read_cards(frame, detect_level_up(frame))
    assert [c.name for c in screen.cards] == ["Garlic", "Knife", "Lightning Ring", "Pako Battiliar"]
    assert [c.new for c in screen.cards] == [False, True, False, True]
    assert choose_card(screen.cards) == 0


def test_match_name_snaps_ocr_noise():
    assert match_name("Enife") == "Knife"
    assert match_name("Lightning Reine") == "Lightning Ring"
    assert match_name("Pako Battlllar") == "Pako Battiliar"
    assert match_name("~ =") == ""


def _cards(*spec: tuple[str, bool]) -> list[Card]:
    return [Card(0, 1, 0, 1, name=n, new=new) for n, new in spec]


def test_priority_order():
    assert choose_card(_cards(("Knife", True), ("Magic Wand", True), ("Garlic", False))) == 2
    assert choose_card(_cards(("Axe", True), ("Pummarola", True))) == 1
    assert choose_card(_cards(("Lightning Ring", False), ("Silver Wind", True), ("Pummarola", False))) == 2


def test_owned_level_beats_new_unwanted_and_avoid_list_is_last():
    assert choose_card(_cards(("Whip", True), ("Bone", False))) == 1
    assert choose_card(_cards(("Axe", True), ("King Bible", True))) == 1
    assert choose_card(_cards(("Knife", True), ("Whip", False))) == 1


def test_maxed_build_chicken_when_low_hp_else_coins():
    offer = _cards(("Gold Coins", False), ("Floor Chicken", False))
    assert choose_card(offer, hp=0.3) == 1
    assert choose_card(offer, hp=0.9) == 0
    assert choose_card(offer, hp=None) == 0
    # Filler never beats a real item.
    assert choose_card(_cards(("Floor Chicken", False), ("Axe", True)), hp=0.1) == 1


def test_hp_bar_reading():
    assert read_hp_fraction(load("gameplay_full_hp.png")) == pytest.approx(1.0)
    assert read_hp_fraction(load("gameplay_hurt.png")) == pytest.approx(32 / 36, abs=0.03)
    assert read_hp_fraction(load("levelup_4cards_sel4.png")) is None


def test_chest_text_classification():
    assert classify_chest_text("Treasure Found! OPEN") == "chest"
    assert classify_chest_text("DONE Treasure") == "chest"
    assert classify_chest_text("Weapon Found! GET") == "chest"
    assert classify_chest_text("GAME OVER REVIVE") == "revive"
    assert classify_chest_text("GAME OVER QUIT") is None
    assert classify_chest_text("START ONLINE ADVENTURES POWER UP") is None
    assert classify_chest_text("RESUME OPTIONS QUIT") is None
    assert classify_chest_text("Results DONE") is None
    assert classify_chest_text("Revival +1") is None
    assert classify_chest_text("") is None


@needs_tesseract
def test_game_over_revive_button_located():
    words = menus.read_screen_words(load("game_over_revive.png"))
    assert classify_chest_text(" ".join(w[0] for w in words)) == "revive"
    x, y = find_word(words, "revive")
    assert 420 <= x <= 540 and 450 <= y <= 485


@needs_tesseract
def test_main_menu_scan_never_triggers():
    words = menus.read_screen_words(load("main_menu.png"))
    assert classify_chest_text(" ".join(w[0] for w in words)) is None


def _pilot(backend: NullBackend, clock: FakeClock, **kw) -> MenuPilot:
    clicks: list[tuple[int, int]] = []
    pilot = MenuPilot(
        tap=lambda k: backend.tap(k, hold_s=0),
        clock=clock,
        async_chest=False,
        click=lambda x, y: clicks.append((x, y)),
        log=lambda _m: None,
        **kw,
    )
    pilot.clicks = clicks  # type: ignore[attr-defined]
    return pilot


def _taps(backend: NullBackend) -> list[str]:
    return [k for action, k in backend.events if action == "down"]


@needs_tesseract
def test_pilot_navigates_up_to_garlic_then_confirms_once():
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    for _ in range(3):  # settle frames, reads names, first nav press
        pilot.step(frame, screen)
    assert _taps(backend) == ["up"]
    # Cooldown: same frame again right away does not double-press.
    pilot.step(frame, screen)
    assert _taps(backend) == ["up"]
    # Simulate the arrows moving up one card per press.
    for sel in (2, 1, 0):
        clock.t += 0.3
        screen.selected = sel
        pilot.step(frame, screen)
    assert _taps(backend) == ["up", "up", "up", "enter"]
    assert pilot.picks == ["Garlic"]
    # Still on screen right after Enter: cooldown blocks a second Enter.
    clock.t += 0.1
    for _ in range(3):
        pilot.step(frame, screen)
    assert _taps(backend).count("enter") == 1


def test_pilot_never_presses_on_main_menu(monkeypatch):
    monkeypatch.setattr(menus, "read_screen_words", lambda f: [("START", 0, 0, 1, 1), ("QUIT", 0, 0, 1, 1)])
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("main_menu.png")
    for _ in range(10):
        clock.t += 2
        assert pilot.step(frame) is None
    assert backend.events == []
    assert pilot.clicks == []


def test_pilot_chest_needs_two_scans_then_presses_enter_with_cooldown(monkeypatch):
    monkeypatch.setattr(menus, "tesseract_available", lambda: True)
    monkeypatch.setattr(menus, "read_screen_words", lambda f: [("Treasure", 0, 0, 1, 1), ("OPEN", 0, 0, 1, 1)])
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("gameplay_full_hp.png")
    assert pilot.step(frame) is None  # first sighting could be OCR noise
    clock.t += 1.6
    assert pilot.step(frame) == "chest"
    assert _taps(backend) == ["enter"]
    clock.t += 0.2
    pilot.step(frame)
    assert _taps(backend) == ["enter"]


@needs_tesseract
def test_pilot_clicks_revive_on_game_over():
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("game_over_revive.png")
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "revive"
    assert len(pilot.clicks) == 1
    x, y = pilot.clicks[0]
    assert 420 <= x <= 540 and 450 <= y <= 485
    assert pilot.revives == 1
    assert "enter" not in _taps(backend)


@needs_tesseract
def test_final_game_over_clicks_quit_then_done_then_stops(monkeypatch):
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("game_over_quit.png")
    assert detect_level_up(frame) is None
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "quit"
    assert len(pilot.clicks) == 1
    x, y = pilot.clicks[0]
    assert 440 <= x <= 520 and 355 <= y <= 385
    # Results screen (no real sample yet): DONE is clicked only after our QUIT.
    monkeypatch.setattr(
        menus, "read_screen_words", lambda f: [("Results", 400, 40, 90, 20), ("DONE", 800, 500, 60, 20)]
    )
    clock.t += 1.6
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "results"
    assert pilot.clicks[-1] == (830, 510)
    assert pilot.finished
    # Finished: nothing else is pressed until reset_run().
    for _ in range(4):
        clock.t += 2
        assert pilot.step(frame) is None
    assert len(pilot.clicks) == 2
    assert backend.events == []
    pilot.reset_run()
    assert not pilot.finished


def test_results_done_is_not_pressed_without_our_quit(monkeypatch):
    monkeypatch.setattr(menus, "tesseract_available", lambda: True)
    monkeypatch.setattr(menus, "read_screen_words", lambda f: [("Results", 0, 0, 1, 1), ("DONE", 5, 5, 4, 4)])
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("gameplay_full_hp.png")
    for _ in range(4):
        clock.t += 2
        assert pilot.step(frame) is None
    assert pilot.clicks == [] and backend.events == []


def test_pause_menu_quit_is_never_pressed(monkeypatch):
    monkeypatch.setattr(menus, "tesseract_available", lambda: True)
    monkeypatch.setattr(
        menus, "read_screen_words", lambda f: [("RESUME", 0, 0, 1, 1), ("OPTIONS", 0, 0, 1, 1), ("QUIT", 5, 5, 4, 4)]
    )
    monkeypatch.setattr(menus, "read_gold_title", lambda f: "GAME OVER")
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    frame = load("gameplay_full_hp.png")
    for _ in range(4):
        clock.t += 2
        assert pilot.step(frame) is None
    assert pilot.clicks == [] and backend.events == []


def test_game_over_title_matcher():
    assert menus.is_game_over_title("GAME OVER,")
    assert not menus.is_game_over_title("VAMPIRE SURVIVORS")
    assert not menus.is_game_over_title("")
