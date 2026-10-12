"""The one Banish exception: a NEW Pentagram on Mad Forest (Lucyfur, Oct 10).

Banish must never fire for any other card, an owned Pentagram, another stage or
keyboard mode, and a failed / impossible banish falls back to a normal pick."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot import menus
from vs_vision_bot.menus import (
    BANISH_BUTTON_960,
    Card,
    MenuPilot,
    banish_allowed,
    banish_candidate,
    detect_level_up,
    normalize_stage,
    read_banish_button,
    read_cancel_button,
)

FIX = Path(__file__).parent / "fixtures"
needs_tesseract = pytest.mark.skipif(not menus.tesseract_available(), reason="tesseract not installed")
LEVELUPS = ("levelup_4cards_sel4.png", "levelup_armor_axe_misread.png", "levelup_candybox.png",
            "levelup_two_line_armor.png")
NOT_LEVELUPS = ("main_menu.png", "results_screen.png", "game_over_revive.png", "game_over_quit.png",
                "gameplay_full_hp.png", "gameplay_skeleton_swarm.png", "gameplay_flower_ring_menu_fp.png")
BTN_CENTER = (776, 315)
CANCEL_CENTER = (776, 347)


def load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIX / name))
    assert frame is not None, name
    return frame


def in_banish_button(x: int, y: int) -> bool:
    bx, by, bw, bh = BANISH_BUTTON_960
    return bx <= x < bx + bw and by <= y < by + bh


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def card(name: str, new: bool = True, score: float = 1.0) -> Card:
    return Card(0, 10, 0, 10, name=name, raw=name, new=new, score=score)


# ---------------------------------------------------------------- button on real frames


@needs_tesseract
@pytest.mark.parametrize("name", LEVELUPS)
def test_banish_button_and_uses_are_read_on_real_level_ups(name):
    # Her save shows "BANISH / 10 left" (stats panel: Banish +10) on every level-up.
    btn = read_banish_button(load(name))
    assert btn is not None
    assert btn.center == BTN_CENTER
    assert btn.left == 10


@pytest.mark.parametrize("name", NOT_LEVELUPS)
def test_no_banish_button_outside_level_ups(name):
    assert read_banish_button(load(name)) is None


def test_greyed_out_banish_button_is_not_a_button():
    frame = load("levelup_4cards_sel4.png").copy()
    bx, by, bw, bh = BANISH_BUTTON_960
    frame[by : by + bh, bx : bx + bw] = (90, 90, 90)  # e.g. no uses left
    assert read_banish_button(frame) is None


# ---------------------------------------------------------------- hard guard


def test_banish_allowed_only_for_a_confident_new_pentagram_on_mad_forest():
    assert banish_allowed(card("Pentagram"), "mad_forest")
    assert banish_allowed(card("Pentagram"), "Mad Forest")
    assert not banish_allowed(card("Pentagram", new=False), "mad_forest")  # owned: its level-up
    assert not banish_allowed(card("Pentagram", score=0.7), "mad_forest")  # weak OCR read
    for stage in ("inlaid_library", "dairy_plant", "gallo_tower", "capella_magna"):
        assert not banish_allowed(card("Pentagram"), stage)
    for name in menus.VOCAB + ["", "?"]:
        if name == "Pentagram":
            continue
        for new in (True, False):
            assert not banish_allowed(card(name, new=new), "mad_forest"), name


def test_banish_candidate_needs_another_card_to_pick():
    assert banish_candidate([card("Pentagram")], "mad_forest") is None
    assert banish_candidate([card("Knife"), card("Pentagram")], "mad_forest") == 1
    assert banish_candidate([card("Knife"), card("Axe"), card("Whip")], "mad_forest") is None


def test_stage_names_normalize():
    assert normalize_stage("Mad Forest") == "mad_forest"
    assert normalize_stage(None) == ""


def test_banish_needs_vs_stage_set_explicitly(monkeypatch):
    # No VS_STAGE: not Mad Forest for Banish purposes (normal avoid-rule pick).
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs, stage="")
    run(pilot, clock, frame, screen, n=4)
    assert not any(in_banish_button(*c) for c in clicks)
    assert pilot.picks == ["Lightning Ring"]


@needs_tesseract
def test_cancel_button_marks_banish_mode_on_real_frames():
    assert read_cancel_button(load("levelup_banish_armed.png")) == CANCEL_CENTER
    for name in ("levelup_banish_offer.png",) + LEVELUPS + NOT_LEVELUPS:
        assert read_cancel_button(load(name)) is None, name


# ---------------------------------------------------------------- pilot flow


class Game:
    """Fake OCR: the names on the fixture's 4 cards, editable between steps."""

    def __init__(self, monkeypatch, names, new=None, left=10, button=True, armed=True):
        self.names = list(names)
        self.armed = armed  # CANCEL shown (banish mode) whenever the pilot looks
        self.new = list(new) if new is not None else [True] * len(names)
        self.left = left
        self.button = button
        monkeypatch.setattr(menus, "read_cards", self.read_cards)
        monkeypatch.setattr(menus, "read_banish_button", self.read_button)
        monkeypatch.setattr(menus, "read_cancel_button", lambda f: CANCEL_CENTER if self.armed else None)

    def read_cards(self, frame, screen):
        for c, n, nw in zip(screen.cards, self.names, self.new):
            c.name, c.raw, c.new, c.score = n, n, nw, 1.0
        return screen

    def read_button(self, frame):
        return menus.BanishButton(center=BTN_CENTER, left=self.left) if self.button else None


def make_pilot(clock, logs, **kw):
    clicks: list[tuple[int, int]] = []
    taps: list[str] = []
    pilot = MenuPilot(
        tap=taps.append, clock=clock, async_chest=False, click=lambda x, y: clicks.append((x, y)),
        log=logs.append, game_clock=lambda: 120.0, **{"stage": "mad_forest", **kw},
    )
    return pilot, clicks, taps


def run(pilot, clock, frame, screen, n=8, dt=1.0):
    for _ in range(n):
        clock.t += dt
        pilot.step(frame, screen)


def center(c: Card) -> tuple[int, int]:
    return (c.x0 + c.x1) // 2, c.cy


def test_new_pentagram_on_mad_forest_is_banished_then_normal_pick(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    assert len(screen.cards) == 4
    game = Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    clock, logs = FakeClock(), []
    pilot, clicks, taps = make_pilot(clock, logs)
    run(pilot, clock, frame, screen, n=6)  # settle, BANISH, settle, card
    assert clicks[0] == BTN_CENTER
    assert clicks[1] == center(screen.cards[1])  # the Pentagram card, nothing else
    assert len(clicks) == 2
    # The game removes Pentagram and keeps the menu open (Jul 2026 update).
    game.names[1], game.left, game.armed = "Clover", 9, False
    run(pilot, clock, frame, screen, n=8)
    assert pilot.banished == {"Pentagram"}
    assert any(m.startswith("[levelup] BANISH Pentagram verified") and "10 -> 9" in m for m in logs)
    assert len(clicks) == 3 and clicks[2] == center(screen.cards[2])
    assert pilot.picks == ["Lightning Ring"]
    assert taps == []  # no keyboard, so never Reroll / Skip
    assert sum(in_banish_button(*c) for c in clicks) == 1


def test_menu_closing_after_the_banish_is_handled(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    monkeypatch.setattr(menus, "tesseract_available", lambda: False)
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    run(pilot, clock, frame, screen, n=6)
    assert len(clicks) == 2
    clock.t += 1
    pilot.step(load("gameplay_full_hp.png"), None)
    assert pilot._banish is None
    assert any("level-up closed after the Pentagram click" in m for m in logs)


@pytest.mark.parametrize("left,button", [(0, True), (None, True), (10, False)])
def test_no_uses_or_no_button_falls_back_to_a_normal_pick(monkeypatch, left, button):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"], left=left, button=button)
    clock, logs = FakeClock(), []
    pilot, clicks, taps = make_pilot(clock, logs)
    run(pilot, clock, frame, screen, n=4)
    assert not any(in_banish_button(*c) for c in clicks)
    assert pilot.picks == ["Lightning Ring"]  # early Pentagram avoid still applies
    assert any("BANISH Pentagram skipped" in m for m in logs)
    assert taps == []


def test_failed_banish_cancels_then_picks_normally(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    game = Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])  # never goes away
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    while len(clicks) < 4:
        run(pilot, clock, frame, screen, n=1)
    pent = center(screen.cards[1])
    assert clicks == [BTN_CENTER, pent, pent, CANCEL_CENTER]  # never a card while armed
    game.armed = False  # CANCEL worked: back on the normal level-up
    run(pilot, clock, frame, screen, n=12)
    assert pilot._banish_fails == 1 and not pilot.finished
    assert len(clicks) > 4 and set(clicks[4:]) == {center(screen.cards[2])}
    assert sum(in_banish_button(*c) for c in clicks) == 1


def test_armed_without_pentagram_cancels_and_never_picks(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    game = Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    while not clicks:
        run(pilot, clock, frame, screen, n=1)
    game.names[1] = ""  # armed screen, Pentagram not read
    while len(clicks) < 2:
        run(pilot, clock, frame, screen, n=1)
    assert clicks == [BTN_CENTER, CANCEL_CENTER]
    game.armed = False
    run(pilot, clock, frame, screen, n=12)
    assert set(clicks[2:]) == {center(screen.cards[2])} and not pilot.finished


def test_cancel_not_confirmed_stops_the_run(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    game = Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    while not clicks:
        run(pilot, clock, frame, screen, n=1)
    game.names[1] = ""
    run(pilot, clock, frame, screen, n=12)  # CANCEL never takes effect
    assert clicks == [BTN_CENTER, CANCEL_CENTER]
    assert pilot.finished and any("stopping the run" in m for m in logs)


def test_banish_click_that_did_not_arm_never_clicks_pentagram(monkeypatch):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"], armed=False)
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    run(pilot, clock, frame, screen, n=12)
    assert clicks[0] == BTN_CENTER
    assert center(screen.cards[1]) not in clicks
    assert set(clicks[1:]) == {center(screen.cards[2])}  # normal avoid-rule pick
    assert any("banish mode not seen" in m for m in logs)


@pytest.mark.parametrize("name", [n for n in menus.VOCAB if n != "Pentagram"])
def test_banish_never_fires_for_any_other_card(monkeypatch, name):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, [name, "Knife", "Axe", "Whip"])
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    run(pilot, clock, frame, screen, n=4)
    assert clicks and not any(in_banish_button(*c) for c in clicks)
    assert not any("BANISH" in m for m in logs)


@pytest.mark.parametrize(
    "case",
    ["owned_pentagram", "other_stage", "keyboard"],
)
def test_no_banish_for_owned_pentagram_other_stage_or_keyboard(monkeypatch, case):
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    new = [True, False, True, True] if case == "owned_pentagram" else None
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"], new=new)
    clock, logs = FakeClock(), []
    kw = {"stage": "inlaid_library"} if case == "other_stage" else {}
    if case == "keyboard":
        kw["levelup_click"] = False
    pilot, clicks, taps = make_pilot(clock, logs, **kw)
    run(pilot, clock, frame, screen, n=4)
    assert not any(in_banish_button(*c) for c in clicks)
    assert pilot._banish is None and not pilot.banished
    if case == "keyboard":
        assert clicks == []
        assert any("keyboard mode never banishes" in m for m in logs)


def test_new_run_forgets_banishes(monkeypatch):
    pilot = MenuPilot(tap=lambda k: None, log=lambda m: None)
    pilot.banished.add("Pentagram")
    pilot._banish_fails = 2
    pilot.new_run()
    assert not pilot.banished and pilot._banish_fails == 0


def test_armed_banish_still_clicks_pentagram_when_its_reread_is_less_confident(monkeypatch):
    # Oct 11 live read Pentagram at 0.89 (min 0.85). A dip on the armed re-read must not fall
    # back to a normal pick: with BANISH armed, that click banishes the other card.
    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    Game(monkeypatch, ["Knife", "Pentagram", "Lightning Ring", "Stone Mask"])
    clock, logs = FakeClock(), []
    pilot, clicks, _ = make_pilot(clock, logs)
    while not clicks:
        run(pilot, clock, frame, screen, n=1)
    assert clicks == [BTN_CENTER]
    read = menus.read_cards
    monkeypatch.setattr(menus, "read_cards", lambda f, s: [setattr(c, "score", 0.84) for c in read(f, s).cards] and s)
    run(pilot, clock, frame, screen, n=8)
    assert clicks[1] == center(screen.cards[1])
