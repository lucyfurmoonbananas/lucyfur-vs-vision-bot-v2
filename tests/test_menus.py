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


def test_ocr_garbage_card_names_rank_below_avoid_and_real():
    """Short / unmatched OCR must not beat AVOID or real names via rank (15, idx)."""
    assert match_name("P") == ""
    assert match_name("wr") == ""
    assert match_name("") == ""
    # Pipeline leaves name="" for garbage; that must lose to real cards.
    assert choose_card(_cards(("", True), ("Garlic", False))) == 1
    assert choose_card(_cards(("Whip", True), ("", True), ("Garlic", False))) == 2
    # Known AVOID still beats unmatched garbage (junk tier 45 > AVOID-new 40).
    assert choose_card(_cards(("", True), ("Whip", True))) == 1
    junk_vs_owned = [
        Card(0, 1, 0, 1, name="", raw="P", new=True),
        Card(0, 1, 0, 1, name="Whip", raw="Whip", new=False),
    ]
    assert choose_card(junk_vs_owned) == 1
    junk_wr = [
        Card(0, 1, 0, 1, name="", raw="wr", new=True),
        Card(0, 1, 0, 1, name="Garlic", raw="Garlic", new=False),
    ]
    assert choose_card(junk_wr) == 1
    # All junk: still picks an index (unknown-select cycling stays elsewhere).
    assert choose_card(_cards(("", True), ("", True), ("", True))) in (0, 1, 2)


def _cards(*spec: tuple[str, bool]) -> list[Card]:
    return [Card(0, 1, 0, 1, name=n, new=new) for n, new in spec]


def test_match_name_snaps_glass_fandango_and_parm_aegis():
    """VOCAB must include Glass Fandango / Parm Aegis so OCR noise does not stay junk."""
    assert "Glass Fandango" in menus.OTHER
    assert "Parm Aegis" in menus.OTHER
    assert "Glass Fandango" in menus.WEAPONS
    assert "Parm Aegis" not in menus.WEAPONS  # passive invuln item
    assert match_name("Glass Fandango") == "Glass Fandango"
    assert match_name("Parm Aegis") == "Parm Aegis"
    # OCR-ish variants (difflib cutoff 0.6 once the real names are in VOCAB).
    assert match_name("Glass Fandanqo") == "Glass Fandango"
    assert match_name("Cless Fandango") == "Glass Fandango"
    assert match_name("Glass Fandang0") == "Glass Fandango"
    assert match_name("Parm Aegls") == "Parm Aegis"
    assert match_name("Parn Aegis") == "Parm Aegis"
    assert match_name("Parm Aeqis") == "Parm Aegis"
    # Matched names must beat AVOID / junk (no longer rely on rank-45 alone).
    assert choose_card(_cards(("Whip", True), ("Glass Fandango", True))) == 1
    assert choose_card(_cards(("", True), ("Parm Aegis", True))) == 1


def test_priority_order():
    assert choose_card(_cards(("Knife", True), ("Magic Wand", True), ("Garlic", False))) == 2
    assert choose_card(_cards(("Axe", True), ("Pummarola", True))) == 1
    assert choose_card(_cards(("Lightning Ring", False), ("Silver Wind", True), ("Pummarola", False))) == 2


def test_tirajisu_ranked_high():
    # Extra revival for low-HP Poe: both spellings sit with Pummarola / Armor.
    assert "Tirajisú" in menus.PRIORITY and "Tiragisú" in menus.PRIORITY
    assert menus.PRIORITY.index("Tirajisú") < menus.PRIORITY.index("Armor")
    assert menus.PRIORITY.index("Tiragisú") < menus.PRIORITY.index("Armor")
    # Healthy: Tirajisú beats a luxury passive and an avoid-list weapon.
    assert choose_card(_cards(("Duplicator", True), ("Tirajisú", True))) == 1
    assert choose_card(_cards(("Axe", True), ("Tiragisú", True))) == 1
    # Near Pummarola / Armor band (rank tier 5), above owned non-Garlic weapon levels.
    assert choose_card(_cards(("Cross", False), ("Tirajisú", True)), hp=0.9, owned={"Garlic", "Cross"}) == 1
    # Low HP: still in DEFENSE (after Floor Chicken / Armor / Hollow Heart / Pummarola).
    assert choose_card(_cards(("Duplicator", True), ("Tirajisú", True)), hp=0.2) == 1
    assert choose_card(_cards(("Tiragisú", True), ("Pummarola", True)), hp=0.2) == 1


def test_owned_level_beats_new_unwanted_and_avoid_list_is_last():
    assert choose_card(_cards(("Whip", True), ("Bone", False))) == 1
    assert choose_card(_cards(("Axe", True), ("King Bible", True))) == 1
    assert choose_card(_cards(("Knife", True), ("Whip", False))) == 1


def test_maxed_build_chicken_when_low_hp_else_coins():
    offer = _cards(("Gold Coins", False), ("Floor Chicken", False))
    assert choose_card(offer, hp=0.3) == 1
    assert choose_card(offer, hp=0.9) == 0
    assert choose_card(offer, hp=None) == 0
    # Filler never beats a real item while healthy (or HP unknown) ...
    assert choose_card(_cards(("Floor Chicken", False), ("Axe", True)), hp=0.9) == 1
    assert choose_card(_cards(("Floor Chicken", False), ("Axe", True)), hp=None) == 1
    # ... but at low HP the chicken's 30 HP heal comes first.
    assert choose_card(_cards(("Floor Chicken", False), ("Axe", True)), hp=0.1) == 0
    assert choose_card(_cards(("Garlic", False), ("Floor Chicken", False)), hp=0.2) == 1


def test_low_hp_prefers_healing_and_defense_over_duplicator():
    # Live run7: took Duplicator at 14 % / 6 % HP.
    offer = _cards(("Duplicator", True), ("Eight The Sparrow", True), ("Armor", True))
    assert choose_card(offer, hp=0.06) == 2
    offer = _cards(("Duplicator", False), ("Axe", True), ("Laurel", True))
    assert choose_card(offer, hp=0.14) == 2  # Laurel shield, not Duplicator
    offer = _cards(("Duplicator", False), ("Hollow Heart", True), ("Pummarola", True))
    assert choose_card(offer, hp=0.3) == 1
    offer = _cards(("Spellbinder", True), ("Pummarola", True), ("Hollow Heart", True))
    assert choose_card(offer, hp=0.3) == 2
    # Low HP with no defense offered: a weapon level beats a luxury passive.
    offer = _cards(("Duplicator", False), ("Cross", False))
    assert choose_card(offer, hp=0.2, owned={"Garlic", "Cross"}) == 1
    # Healthy: defense is not forced ahead of a crowd-clearing weapon.
    offer = _cards(("Armor", True), ("Santa Water", True))
    assert choose_card(offer, hp=0.9, owned={"Garlic"}) == 1


def test_new_crowd_weapons_first_while_slots_are_free():
    owned = {"Garlic"}
    assert choose_card(_cards(("Garlic", False), ("King Bible", True)), hp=1.0, owned=owned) == 1
    assert choose_card(_cards(("Magic Wand", True), ("Lightning Ring", True)), hp=1.0, owned=owned) == 1
    assert choose_card(_cards(("Santa Water", True), ("Lightning Ring", True), ("King Bible", True))) == 0
    # A new weapon beats a passive while slots are free.
    assert choose_card(_cards(("Duplicator", True), ("Peachone", True)), hp=1.0, owned=owned) == 1
    assert choose_card(_cards(("Empty Tome", True), ("Cross", True)), hp=1.0, owned=owned) == 1
    # Garlic levels still beat non-crowd new weapons.
    assert choose_card(_cards(("Magic Wand", True), ("Garlic", False)), owned=owned) == 1


def test_full_weapon_slots_stop_new_weapon_preference():
    six = {"Garlic", "Santa Water", "King Bible", "Lightning Ring", "Magic Wand", "Cross"}
    assert menus.owned_weapon_count(six) == 6
    offer = _cards(("Runetracer", True), ("Lightning Ring", False))
    assert choose_card(offer, hp=1.0, owned=six) == 1
    offer = _cards(("Peachone", True), ("Pummarola", True))
    assert choose_card(offer, hp=1.0, owned=six) == 1


def test_evolution_comes_first():
    assert choose_card(_cards(("Armor", True), ("Soul Eater", False)), hp=0.2) == 1


def test_hp_tracker_uses_median():
    t = menus.HpTracker(window=5)
    assert t.value is None
    for v in (0.9, 0.9, 0.1, 0.9, None, 0.85):
        t.add(v)
    assert t.value == pytest.approx(0.9)
    for v in (0.3, 0.3, 0.3):
        t.add(v)
    assert t.value == pytest.approx(0.3)


def _bar_frame(red_cols: int, occlude: tuple[int, int] | None = None) -> np.ndarray:
    frame = np.full((540, 960, 3), (40, 90, 50), np.uint8)
    x0, y0, x1, y1 = menus.HP_BAR_RECT_960
    frame[y0:y1, x0:x1] = (10, 10, 10)
    frame[y0:y1, x0 : x0 + red_cols] = (0, 0, 255)
    if occlude:
        frame[y0 - 3 : y1 + 3, x0 + occlude[0] : x0 + occlude[1]] = (180, 200, 220)
    return frame


def test_anchored_hp_bar_reader():
    assert menus.read_hp_bar(_bar_frame(36)) == pytest.approx(1.0)
    assert menus.read_hp_bar(_bar_frame(18)) == pytest.approx(0.5)
    assert menus.read_hp_bar(_bar_frame(4)) == pytest.approx(4 / 36)
    # A sprite over the dark part does not change the reading.
    assert menus.read_hp_bar(_bar_frame(18, occlude=(24, 30))) == pytest.approx(0.5)
    # Mostly hidden bar: no reading instead of a wrong one.
    assert menus.read_hp_bar(_bar_frame(18, occlude=(0, 30))) is None
    # Scaled capture (1280x720) reads the same.
    big = cv2.resize(_bar_frame(27), (1280, 720), interpolation=cv2.INTER_NEAREST)
    assert menus.read_hp_bar(big) == pytest.approx(0.75, abs=0.04)
    # No bar at all.
    assert menus.read_hp_bar(np.full((540, 960, 3), (40, 90, 50), np.uint8)) is None


def test_pilot_tracks_owned_weapons_and_resets_on_new_run():
    pilot = MenuPilot(tap=lambda k: None, async_chest=False, log=lambda m: None)
    assert pilot.owned == {"Garlic"}
    pilot.owned |= {"Cross", "Santa Water"}
    pilot.hp = 0.4
    assert pilot.hp == pytest.approx(0.4)
    pilot.new_run()
    assert pilot.owned == {"Garlic"} and pilot.hp is None


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


def test_chest_reward_screen_after_open_is_a_chest():
    # OCR of the post-OPEN reward screen (gold, item, DONE): no "Treasure Found!" title.
    assert classify_chest_text("01:17 - DONE Lever:3 Effect lasts 0.5 seconds longer. Base Damage up by 10.") == "chest"
    assert classify_chest_text("01:04 bv DONE level:2 [ALL] Base Speed up by 108.") == "chest"


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
    pilot = _pilot(backend, clock, levelup_click=False)
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
def test_revive_resets_hp_history():
    """Oct 1 runs 3 / 12: hp stayed 0.19 (pre-death median) after REVIVE refilled the bar."""
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    for v in (0.2, 0.19, 0.15, 0.1, 0.19):
        pilot._hp.add(v)
    assert pilot.hp == pytest.approx(0.19)
    frame = load("game_over_revive.png")
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "revive"
    assert pilot.revives == 1
    assert pilot.hp is None  # stale pre-death readings dropped
    # Readings during the hold-off (death screen fading) are ignored ...
    pilot.step(load("gameplay_hurt.png"))
    assert pilot.hp is None
    # ... then the revived bar is read fresh.
    clock.t += menus.REVIVE_HP_HOLDOFF_S + 0.1
    pilot.step(load("gameplay_full_hp.png"))
    assert pilot.hp == pytest.approx(1.0)


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


@needs_tesseract
def test_real_results_screen_done_button_after_quit():
    frame = load("results_screen.png")
    assert detect_level_up(frame) is None
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    for _ in range(3):
        clock.t += 1.6
        assert pilot.step(frame) is None  # not pressed without our own QUIT first
    pilot._quit_pressed = True
    clock.t += 1.6
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "results"
    x, y = pilot.clicks[-1]
    assert 440 <= x <= 520 and 490 <= y <= 520
    assert pilot.finished


def test_results_screenshot_saved_before_done(monkeypatch, tmp_path):
    monkeypatch.setattr(menus, "tesseract_available", lambda: True)
    monkeypatch.setattr(
        menus, "read_screen_words", lambda f: [("Results", 400, 40, 90, 20), ("DONE", 800, 500, 60, 20)]
    )
    backend = NullBackend()
    clock = FakeClock()
    order: list[str] = []
    pilot = _pilot(backend, clock, results_dir=str(tmp_path / "results"))
    real_click = pilot.click
    pilot.click = lambda x, y: (order.append("click:" + str(len(pilot.saved_results))), real_click(x, y))
    pilot._quit_pressed = True
    frame = load("results_screen.png")
    clock.t += 1.6
    pilot.step(frame)
    clock.t += 1.6
    assert pilot.step(frame) == "results"
    assert pilot.finished
    assert len(pilot.saved_results) == 1
    saved = cv2.imread(pilot.saved_results[0])
    assert saved is not None and saved.shape == frame.shape
    assert order == ["click:1"]  # screenshot written before DONE was clicked


def test_results_screenshot_disabled_without_dir(monkeypatch):
    monkeypatch.setattr(menus, "tesseract_available", lambda: True)
    monkeypatch.setattr(menus, "read_screen_words", lambda f: [("Results", 0, 0, 1, 1), ("DONE", 5, 5, 4, 4)])
    pilot = _pilot(NullBackend(), FakeClock())
    assert pilot._save_results(load("results_screen.png")) is None


@needs_tesseract
def test_pilot_clicks_the_target_card_instead_of_keyboard():
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    for _ in range(3):
        pilot.step(frame, screen)
    assert _taps(backend) == []  # no up/down/enter at all
    assert len(pilot.clicks) == 1
    x, y = pilot.clicks[0]
    card = screen.cards[0]  # Garlic
    assert card.x0 < x < card.x1 and card.y0 < y < card.y1
    assert pilot.picks == ["Garlic"]
    # Cooldown: no second click on the same screen right away.
    pilot.step(frame, screen)
    assert len(pilot.clicks) == 1


def test_keyboard_fallback_never_confirms_without_seeing_the_highlight(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    monkeypatch.setattr(menus, "read_cards", lambda f, s: s)
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock, levelup_click=False)
    for _ in range(200):
        clock.t += 0.3
        screen.selected = None  # highlight off the cards (e.g. on BANISH)
        pilot.step(frame, screen)
    taps = _taps(backend)
    assert "enter" not in taps
    assert len(taps) <= 12


def test_final_quit_button_found_over_a_busy_crowd():
    # Live run9: gold GAME OVER title unreadable over pink skeletons; QUIT must still be found.
    busy = load("game_over_quit_busy.png")
    words = menus.read_screen_words(busy) if menus.tesseract_available() else [("QUIT", 452, 362, 56, 20)]
    assert menus.is_final_quit_button(busy, words)
    assert menus.is_final_quit_button(load("game_over_quit.png"), [("QUIT", 452, 362, 56, 20)])
    # QUIT elsewhere (main menu top bar) or no red button: no.
    assert not menus.is_final_quit_button(load("main_menu.png"), [("QUIT", 330, 18, 50, 16)])
    assert not menus.is_final_quit_button(load("gameplay_full_hp.png"), [("QUIT", 452, 362, 56, 20)])


@needs_tesseract
def test_pilot_quits_busy_final_game_over():
    frame = load("game_over_quit_busy.png")
    backend = NullBackend()
    clock = FakeClock()
    pilot = _pilot(backend, clock)
    import unittest.mock as um
    with um.patch.object(menus, "read_screen_words", lambda f: [("-", 381, 370, 17, 6), ("QUIT", 452, 362, 56, 20)]), \
         um.patch.object(menus, "read_gold_title", lambda f: "c ora"):
        for _ in range(4):
            clock.t += 2.0
            pilot.step(frame, None)
    assert pilot._quit_pressed
    assert pilot.clicks and abs(pilot.clicks[0][0] - 480) < 40



def test_level_up_with_an_unread_card_saves_the_frame(monkeypatch, tmp_path):
    # Oct 4-6 logs: 63 of 485 level-ups had an unmatched card ('P', 'wr', 'or')
    # and no frame showed what it was. Keep one screenshot per such screen.
    frame = load("levelup_4cards_sel4.png")

    def fake_read(f, s, unread=True):
        for c in s.cards:
            c.name, c.raw = "Garlic", "Garlic"
        if unread:
            s.cards[1].name, s.cards[1].raw = "", "wr"
        return s

    for unread, expect in ((True, 1), (False, 0)):
        out = tmp_path / str(unread)
        monkeypatch.setattr(menus, "read_cards", lambda f, s: fake_read(f, s, unread))
        pilot = _pilot(NullBackend(), FakeClock(), results_dir=str(out))
        for _ in range(4):
            pilot.step(frame, detect_level_up(frame))
        assert len(list(out.glob("levelup_unread_*.png"))) == expect
        assert not list(out.glob("results_*.png"))
