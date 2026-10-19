"""Level-up picks should finish a weapon evolution instead of spreading levels.

Root cause locked in by the static fallback (no level tracker): ``_card_rank``
gives a new crowd weapon tier 2 and any other new weapon tier 4 while a slot
is free, a Garlic level tier 3, and every other owned weapon level tier 6.
Nothing records "this weapon is level N", so the bot never holds levels for
one weapon until 8. These offers are the live failure (Poe, Mad Forest):
Garlic stays at 2-5 while King Bible, Lightning Ring, Magic Wand, Santa
Water and Ebony Wings each pick up a level or two, and Pummarola is taken
but not as Garlic's evolution partner. With a certain inventory the same
offers max Garlic, take Pummarola as soon as it appears, then max one second
weapon. An uncertain read (level OCR far from the tracker, or a shaky name)
keeps the old spreading rank.
"""

from __future__ import annotations

import pytest

from vs_vision_bot.menus import (
    PASSIVE_MAX_LEVEL,
    WEAPONS,
    Card,
    MenuPilot,
    WEAPON_MAX_LEVEL,
    assess_inventory,
    choose_card,
    choose_pick,
    evolution_ready,
    parse_level_text,
)

from test_menus import needs_tesseract, load


def _card(name: str, new: bool, level: int | None = None, score: float = 1.0) -> Card:
    return Card(0, 1, 0, 1, name=name, raw=name, new=new, level=level, score=score)


def _offer(spec: list[tuple[str, bool]], levels: dict[str, int]) -> list[Card]:
    cards = []
    for name, new in spec:
        level = None if new else levels.get(name)
        cards.append(_card(name, new, level=level))
    return cards


def _play(offers: list[list[tuple[str, bool]]], levels: dict[str, int], hp: float = 1.0):
    picks: list[str] = []
    reasons: list[str] = []
    for spec in offers:
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), levels=dict(levels))
        name = cards[pick.index].name
        assert pick.reason in ("main", "partner", "second", "fallback")
        picks.append(name)
        reasons.append(pick.reason)
        cap = WEAPON_MAX_LEVEL if name in WEAPONS else PASSIVE_MAX_LEVEL
        levels[name] = min(cap, levels.get(name, 0) + 1)
    return picks, reasons


def test_parse_level_text_accepts_ocr_variants_and_rejects_garbage():
    assert parse_level_text("level:2") == 2
    assert parse_level_text("level :2") == 2
    assert parse_level_text("leval:2") == 2
    assert parse_level_text("Level: 8") == 8
    assert parse_level_text("OS") is None
    assert parse_level_text("") is None
    assert parse_level_text("level:9") is None
    assert parse_level_text("level:0") is None


def test_static_priority_still_spreads_when_levels_are_unknown():
    """No tracker: the old crowd-weapon rank is the uncertain-inventory fallback."""
    owned = {"Garlic"}
    cards = [_card("Garlic", False), _card("King Bible", True), _card("Lightning Ring", True)]
    assert choose_card(cards, hp=1.0, owned=owned) == 1
    assert choose_pick(cards, hp=1.0, owned=owned).reason == "fallback"


def test_scripted_offers_max_garlic_then_one_second_weapon():
    """The live spread, replayed. Fails while new crowd weapons outrank Garlic."""
    levels = {"Garlic": 1}
    garlic_offers = [
        [("King Bible", True), ("Lightning Ring", True), ("Garlic", False)],
        [("Santa Water", True), ("Magic Wand", True), ("Knife", True)],
        [("Pummarola", True), ("Garlic", False), ("Empty Tome", True)],
    ]
    # Six more Garlic levels (1 -> 2 on the first offer, then 2 -> 8).
    garlic_offers += [
        [("Armor", True), ("Garlic", False), ("King Bible", True)] for _ in range(6)
    ]
    picks, reasons = _play(garlic_offers, levels)
    assert picks[0] == "Garlic" and reasons[0] == "main"
    assert picks[1] == "Magic Wand" and reasons[1] == "fallback"  # nothing owned or partner
    assert picks[2] == "Pummarola" and reasons[2] == "partner"  # as soon as it is offered
    assert picks[3:] == ["Garlic"] * 6
    assert reasons[3:] == ["main"] * 6
    assert levels["Garlic"] == 8
    assert levels["Magic Wand"] == 1  # the one forced level, not a second project yet
    assert levels["Pummarola"] == 1
    assert evolution_ready(levels)

    # Main is done: level the weapon we already own, and take its partner early.
    later = [
        [("Magic Wand", False), ("Armor", True), ("King Bible", True)],
        [("Empty Tome", True), ("Magic Wand", False), ("Santa Water", True)],
        [("Santa Water", True), ("Armor", True), ("Magic Wand", False)],
        [("Hollow Heart", True), ("Axe", True), ("Spinach", True)],
    ]
    picks, reasons = _play(later, levels)
    assert picks == ["Magic Wand", "Empty Tome", "Magic Wand", "Hollow Heart"]
    assert reasons == ["second", "partner", "second", "fallback"]
    assert levels["Magic Wand"] == 3
    assert levels["Empty Tome"] == 1
    assert "King Bible" not in levels and "Santa Water" not in levels and "Axe" not in levels


def test_partner_beats_a_main_level_when_both_are_offered():
    levels = {"Garlic": 4}
    cards = _offer([("Garlic", False), ("Pummarola", True), ("King Bible", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Pummarola" and pick.reason == "partner"


def test_once_partner_is_owned_main_levels_beat_more_partner_levels():
    levels = {"Garlic": 4, "Pummarola": 1}
    cards = _offer([("Pummarola", False), ("Garlic", False), ("Santa Water", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"


def test_magic_wand_is_main_when_garlic_is_not_held():
    levels = {"Magic Wand": 2}
    cards = _offer([("Garlic", True), ("Magic Wand", False), ("Empty Tome", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Empty Tome" and pick.reason == "partner"
    cards = _offer([("Garlic", True), ("Magic Wand", False), ("King Bible", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Magic Wand" and pick.reason == "main"
    # Neither held: Garlic is the main to acquire, not a crowd weapon.
    cards = _offer([("King Bible", True), ("Magic Wand", True), ("Garlic", True)], {})
    pick = choose_pick(cards, owned=set(), levels={})
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"


def test_owned_level_beats_a_new_unwanted_item_when_the_plan_is_absent():
    levels = {"Garlic": 4, "Pummarola": 1, "Bone": 1}
    cards = _offer([("Whip", True), ("Bone", False), ("Knife", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Bone" and pick.reason == "fallback"


def test_survival_passives_after_the_partner_beat_a_new_weapon():
    levels = {"Garlic": 3, "Pummarola": 1}
    cards = _offer([("Santa Water", True), ("Armor", True), ("Hollow Heart", True)], levels)
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Armor" and pick.reason == "fallback"
    # Pummarola still leads the survival trio once it is already the partner.
    cards = _offer([("Hollow Heart", True), ("Pummarola", False), ("Armor", True)], levels)
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Pummarola"
    # Low HP still takes defense ahead of a main level (existing survive rule).
    cards = _offer([("Garlic", False), ("Armor", True)], levels)
    pick = choose_pick(cards, hp=0.2, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Armor"
    pick = choose_pick(cards, hp=0.9, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"


def test_level_ocr_far_from_the_tracker_falls_back_to_spreading():
    levels = {"Garlic": 2}
    cards = [_card("Garlic", False, level=2), _card("King Bible", True)]
    assert choose_pick(cards, owned=set(levels), levels=levels).index == 0
    # Off by one (current level vs the level the card will become) stays certain.
    cards = [_card("Garlic", False, level=3), _card("King Bible", True)]
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"
    # A wild OCR level is not an inventory. Old rank: new King Bible beats Garlic.
    cards = [_card("Garlic", False, level=6), _card("King Bible", True)]
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "King Bible" and pick.reason == "fallback"
    # A shaky name is the same fallback, and the tracker is not rewritten.
    cards = [_card("Garlic", False, level=2, score=0.7), _card("King Bible", True)]
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert pick.reason == "fallback" and cards[pick.index].name == "King Bible"
    eff, certain = assess_inventory(cards, levels)
    assert certain is False and eff.get("Garlic") == 2


def test_missing_level_text_on_a_tracked_item_stays_on_the_plan():
    levels = {"Garlic": 3}
    cards = [_card("Garlic", False, level=None), _card("Lightning Ring", True)]
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"


def test_early_pentagram_stays_last_when_the_inventory_is_known():
    levels = {"Garlic": 2}
    cards = _offer([("Pentagram", True), ("Whip", True), ("Spinach", True)], levels)
    pick = choose_pick(cards, game_s=120.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Spinach"
    # Only Pentagram: still a card. Reroll / Skip are never chosen.
    only = _offer([("Pentagram", True)], levels)
    assert choose_pick(only, game_s=60.0, owned=set(levels), levels=levels).index == 0


def test_birds_are_a_second_project_after_the_main_is_maxed():
    levels = {"Garlic": 8, "Pummarola": 1, "Peachone": 1}
    cards = _offer([("Ebony Wings", True), ("Peachone", False), ("King Bible", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Ebony Wings" and pick.reason == "partner"
    levels["Ebony Wings"] = 1
    cards = _offer([("Ebony Wings", False), ("Peachone", False), ("Santa Water", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Peachone" and pick.reason == "second"


def test_evolution_ready_is_main_at_8_plus_its_partner():
    assert not evolution_ready({"Garlic": 7, "Pummarola": 1})
    assert not evolution_ready({"Garlic": 8})
    assert evolution_ready({"Garlic": 8, "Pummarola": 1})
    assert not evolution_ready({"Magic Wand": 8})
    assert evolution_ready({"Magic Wand": 8, "Empty Tome": 2})
    assert not evolution_ready(None)
    assert not evolution_ready({})


def test_pilot_logs_the_pick_reason_and_raises_evolution_ready(monkeypatch):
    from test_menus import FakeClock, detect_level_up

    frame = load("levelup_4cards_sel4.png")
    screen = detect_level_up(frame)
    assert screen is not None and len(screen.cards) == 4

    def fake_read(frame, screen):
        spec = [("Garlic", False), ("King Bible", True), ("Whip", True), ("Axe", True)]
        for c, (name, new) in zip(screen.cards, spec):
            c.name, c.raw, c.new, c.score = name, name, new, 1.0
            c.level = 1 if name == "Garlic" else None
        return screen

    monkeypatch.setattr("vs_vision_bot.menus.read_cards", fake_read)
    logs: list[str] = []
    pilot = MenuPilot(
        tap=lambda k: None,
        clock=FakeClock(),
        async_chest=False,
        click=lambda x, y: None,
        log=logs.append,
        settle_frames=1,
    )
    assert pilot.levels == {"Garlic": 1} and pilot.evolution_ready is False
    for _ in range(2):
        pilot.step(frame, screen)
    assert pilot.picks == ["Garlic"]
    assert pilot.levels["Garlic"] == 2
    picked = [m for m in logs if m.startswith("[levelup] picked ")]
    assert len(picked) == 1
    assert "(reason: main)" in picked[0]
    assert "inv=Garlic:2" in picked[0]
    pilot.levels = {"Garlic": 8, "Pummarola": 1}
    pilot._log_evolution_ready()
    assert pilot.evolution_ready is True
    assert any(m.startswith("[levelup] evolution ready:") and "Garlic" in m and "Pummarola" in m for m in logs)
    pilot._log_evolution_ready()
    assert sum(1 for m in logs if m.startswith("[levelup] evolution ready:")) == 1
    pilot.new_run()
    assert pilot.levels == {"Garlic": 1} and pilot.evolution_ready is False and pilot.picks == []


@pytest.mark.parametrize(
    "raw, level",
    [("level:2", 2), ("level :2", 2), ("leval:2", 2)],
)
def test_level_patterns_cover_the_fixture_ocr(raw, level):
    assert parse_level_text(raw) == level


@needs_tesseract
def test_fixture_owned_cards_read_a_level_and_new_cards_do_not():
    from vs_vision_bot.menus import detect_level_up, read_cards

    frame = load("levelup_4cards_sel4.png")
    screen = read_cards(frame, detect_level_up(frame))
    assert screen.cards[0].name == "Garlic" and screen.cards[0].level == 2
    assert screen.cards[1].name == "Knife" and screen.cards[1].new and screen.cards[1].level is None
    assert screen.cards[2].name == "Lightning Ring" and screen.cards[2].level == 2
    assert screen.cards[3].new and screen.cards[3].level is None
