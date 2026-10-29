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
    MAX_WEAPONS,
    PASSIVE_MAX_LEVEL,
    WEAPONS,
    Card,
    HpTracker,
    MenuPilot,
    WEAPON_MAX_LEVEL,
    all_maxed,
    assess_inventory,
    chest_text_is_reward,
    choose_card,
    choose_pick,
    evolution_ready,
    evolutions_from_chest_text,
    hp_is_low,
    items_fully_maxed,
    levels_from_chest_text,
    low_hp_limit,
    match_name,
    merge_card_levels,
    owned_weapon_slots,
    parse_level_text,
    read_critical_flag,
)
from vs_vision_bot.pathfind import Tuning

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
        # Garlic is still 2 and the clock is early: do not take Santa Water yet.
        [("Santa Water", True), ("Garlic", False), ("Knife", True)],
        # Garlic just hit 3. Santa Water is now the pick on a weapon-only screen.
        [("Santa Water", True), ("Magic Wand", True), ("Knife", True)],
        [("Pummarola", True), ("Empty Tome", True), ("King Bible", True)],
    ]
    # Five more Garlic levels (offer 3 reaches 4, then 4 -> 8).
    garlic_offers += [
        [("Armor", True), ("Garlic", False), ("King Bible", True)] for _ in range(5)
    ]
    picks, reasons = _play(garlic_offers, levels)
    assert picks[0] == "Garlic" and reasons[0] == "main"
    assert picks[1] == "Garlic" and reasons[1] == "main"
    assert picks[2] == "Santa Water" and reasons[2] == "second"
    assert picks[3] == "Pummarola" and reasons[3] == "partner"
    assert picks[4:] == ["Garlic"] * 5
    assert reasons[4:] == ["main"] * 5
    assert levels["Garlic"] == 8
    assert levels["Santa Water"] == 1  # the one forced level, not a second project yet
    assert levels["Pummarola"] == 1
    assert evolution_ready(levels)

    # Main is done. Santa Water on the card is taken; Attractorb is taken
    # on the screen that does not also offer Santa Water.
    later = [
        [("Santa Water", False), ("Armor", True), ("King Bible", True)],
        [("Attractorb", True), ("Empty Tome", True), ("Magic Wand", True)],
        [("Magic Wand", True), ("Armor", True), ("Santa Water", False)],
        [("Hollow Heart", True), ("Axe", True), ("Spinach", True)],
    ]
    picks, reasons = _play(later, levels)
    assert picks == ["Santa Water", "Attractorb", "Santa Water", "Hollow Heart"]
    assert reasons == ["second", "partner", "second", "fallback"]
    assert levels["Santa Water"] == 3
    assert levels["Attractorb"] == 1
    assert "King Bible" not in levels and "Magic Wand" not in levels and "Axe" not in levels


def test_partner_beats_a_main_level_when_both_are_offered():
    """test13: a Garlic level comes before Pummarola. Pummarola still beats a weapon."""
    levels = {"Garlic": 4}
    cards = _offer([("Garlic", False), ("Pummarola", True), ("King Bible", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"
    cards = _offer([("Pummarola", True), ("King Bible", True), ("Peachone", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Pummarola" and pick.reason == "partner"


def test_once_partner_is_owned_main_levels_beat_more_partner_levels():
    levels = {"Garlic": 4, "Pummarola": 1}
    cards = _offer([("Pummarola", False), ("Garlic", False), ("Santa Water", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Santa Water" and pick.reason == "second"


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


def test_santa_water_is_the_first_second_weapon_and_armor_l2_yields():
    from vs_vision_bot.menus import CROWD_WEAPONS, OTHER, PRIORITY, SECOND_ORDER, _choose_static

    assert SECOND_ORDER[:4] == ("Garlic", "Santa Water", "King Bible", "Magic Wand")
    assert CROWD_WEAPONS[:2] == ("Santa Water", "King Bible")
    assert PRIORITY[:10] == [
        "Garlic", "Soul Eater", "Pummarola",
        "Santa Water", "La Borra", "Attractorb",
        "King Bible", "Unholy Vespers", "Spellbinder", "Candelabrador",
    ]
    assert "King Bible" not in OTHER and "Unholy Vespers" not in OTHER
    levels = {"Garlic": 8, "Pummarola": 1}
    cards = _offer([("Magic Wand", True), ("Santa Water", True), ("King Bible", True)], levels)
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels, game_s=700.0)
    assert cards[pick.index].name == "Santa Water"
    # Santa Water already maxed with its partner: King Bible is the third weapon.
    bible = {"Garlic": 8, "Pummarola": 1, "Santa Water": 8, "Attractorb": 1}
    cards = _offer([("Magic Wand", True), ("King Bible", True), ("Runetracer", True)], bible)
    pick = choose_pick(cards, hp=1.0, owned=set(bible), levels=bible, game_s=700.0)
    assert cards[pick.index].name == "King Bible"
    # Candelabrador is the AoE passive, ahead of Spinach and Duplicator.
    candle = {"Garlic": 4, "Pummarola": 1}
    cards = _offer([("Spinach", True), ("Candelabrador", True), ("Duplicator", True)], candle)
    pick = choose_pick(cards, hp=1.0, owned=set(candle), levels=candle)
    assert cards[pick.index].name == "Candelabrador"

    # test7 run 2, 155s, hp=0.33: Armor L2 must not beat Garlic.
    held = {"Garlic": 5, "Armor": 1}
    cards = _offer([("Garlic", False), ("Armor", False)], held)
    pick = choose_pick(cards, hp=0.33, owned=set(held), levels=held)
    assert cards[pick.index].name == "Garlic"
    # Still critical, and Garlic is already past 4: Armor L2 may lead.
    deep = {"Garlic": 6, "Armor": 1}
    cards = _offer([("Garlic", False), ("Armor", False)], deep)
    pick = choose_pick(cards, hp=0.20, owned=set(deep), levels=deep)
    assert cards[pick.index].name == "Garlic"

    # test7 run 3: Fire Wand as a third weapon while Garlic is 3.
    two = {"Garlic": 3, "Magic Wand": 1}
    cards = _offer([("Fire Wand", True), ("Attractorb", True), ("Spinach", True)], two)
    idx = _choose_static(cards, hp=0.9, owned=set(two), levels=two)
    assert cards[idx].name == "Attractorb"


def test_test9_does_not_spend_early_slots_and_takes_santa_water():
    """run1.log: Magic Wand at 9s, Runetracer at 20s, Silver Wind at 40s, then
    Pummarola over Santa Water at 296s with four weapons already owned.
    """
    garlic = {"Garlic": 1}

    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    # Weapon-only screens still click. The avoid list does not win that click
    # when a real weapon is also offered (test11 opened Axe because it was first).
    assert picked([("Axe", True), ("Magic Wand", True), ("Knife", True)], garlic, game_s=9) == "Magic Wand"
    assert picked(
        [("Pako Battiliar", True), ("Runetracer", True), ("Axe", True)], garlic, game_s=20,
    ) == "Runetracer"
    # Every card is on the avoid list. The click cannot be a keeper weapon.
    wind = picked([("Silver Wind", True), ("Whip", True), ("Axe", True)], garlic, game_s=40)
    assert wind == "Silver Wind"
    # A passive on that screen is taken instead.
    assert picked(
        [("Silver Wind", True), ("Whip", True), ("Candelabrador", True)], garlic, game_s=40,
    ) == "Candelabrador"

    # Garlic is already 6 and the clock is past 0:30, so Santa Water wins.
    late = {
        "Garlic": 6, "Attractorb": 2, "Magic Wand": 4, "Pummarola": 1,
        "Runetracer": 3, "Silver Wind": 1,
    }
    assert picked(
        [("Santa Water", True), ("Pummarola", False), ("Lightning Ring", True)],
        late, game_s=296,
    ) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Lightning Ring", True), ("Axe", True)],
        late, game_s=296,
    ) == "Santa Water"
    assert picked([("Santa Water", True), ("Armor", True)], {"Garlic": 3}, hp=0.25) == "Armor"
    assert picked([("Santa Water", True), ("Armor", True)], {"Garlic": 3}, hp=0.90) == "Santa Water"
    # The early-second rule does not take Magic Wand.
    assert picked(
        [("Magic Wand", True), ("King Bible", True), ("Bracer", True)], garlic, game_s=100,
    ) == "King Bible"


def test_test10_garlic_and_candle_beat_santa_water_and_no_third_weapon():
    """test10: Santa Water ate Hollow Heart and Candelabrador; Run 3 opened Knife
    as a third weapon while Garlic was still 1. Run 1 never rolled Santa Water.
    """
    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    # Above 25% HP Santa Water beats Hollow Heart. A Garlic level still beats Santa Water.
    assert picked(
        [("Hollow Heart", True), ("Ebony Wings", True), ("Santa Water", False)],
        {"Garlic": 2, "Santa Water": 1}, hp=1.0, game_s=100,
    ) == "Santa Water"
    # Santa Water beats the candle and a new Laurel once it is the offered weapon.
    assert picked(
        [("Santa Water", False), ("Laurel", True), ("Candelabrador", True), ("Parm Aegis", True)],
        {"Garlic": 3, "Santa Water": 2, "Hollow Heart": 2}, hp=0.36, game_s=138,
    ) == "Santa Water"
    # Garlic is already 3, so the gate is open and Santa Water wins above 25% HP.
    assert picked(
        [("Santa Water", False), ("Garlic", False), ("Whip", True)],
        {"Garlic": 3, "Santa Water": 1}, hp=1.0, game_s=66,
    ) == "Santa Water"
    # Run 3, 19s had no passive, so a click was forced. With a passive on the
    # same kind of screen the third weapon is not opened.
    two = {"Garlic": 1, "King Bible": 1}
    assert picked(
        [("Magic Wand", True), ("Knife", True), ("Hollow Heart", True)],
        two, hp=1.0, game_s=19,
    ) == "Hollow Heart"
    # Run 1, 7s. Santa Water was not on the card (it never appeared in run1.log).
    # Do not prefer Axe over the other weapons on that screen.
    assert picked(
        [("Heaven Sword", True), ("Axe", True), ("Peachone", True)],
        {"Garlic": 1}, hp=1.0, game_s=7,
    ) != "Axe"
    # Garlic is already 3, so Santa Water wins over the level and the partner.
    assert picked(
        [("Candelabrador", True), ("Garlic", False), ("Santa Water", False)],
        {"Garlic": 3, "Santa Water": 1}, hp=1.0, game_s=66,
    ) == "Santa Water"
    assert picked(
        [("Candelabrador", True), ("Pummarola", True), ("Santa Water", True)],
        {"Garlic": 3}, hp=1.0, game_s=50,
    ) == "Santa Water"


def test_test12_santa_water_waits_for_garlic_or_ninety_seconds():
    """test12 took Santa Water at 0:08 and Garlic was still 2-4 at 3:00.

    test11 Run 2 (30:02, Lv99) took Garlic first and Santa Water at 1:49,
    while Garlic was still 2. After that gate, Santa Water is not passed up
    above 25% HP. A Garlic level still wins when both are offered.
    """
    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    early = [("Ebony Wings", True), ("Magic Wand", True), ("Eight The Sparrow", True), ("Santa Water", True)]
    assert picked(early, {"Garlic": 1}, hp=1.0, game_s=8) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Garlic", False), ("Knife", True)],
        {"Garlic": 2}, hp=1.0, game_s=19,
    ) == "Garlic"
    # 1:49 on the Lv99 run, Garlic still 2. The clock opens the gate.
    assert picked(
        [("Empty Tome", False), ("Axe", True), ("Cross", True), ("Santa Water", True)],
        {"Garlic": 2, "Magic Wand": 1, "Attractorb": 2, "Empty Tome": 1},
        hp=1.0, game_s=109,
    ) == "Santa Water"
    # Garlic L3 opens the gate even before 1:30, and a Garlic level still leads.
    assert picked(
        [("Santa Water", True), ("Garlic", False), ("Whip", True)],
        {"Garlic": 3}, hp=1.0, game_s=40,
    ) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Armor", True), ("Hollow Heart", True)],
        {"Garlic": 3}, hp=0.9, game_s=40,
    ) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Armor", True)],
        {"Garlic": 3}, hp=0.25, game_s=40,
    ) == "Armor"


def test_test11_cap_and_santa_water_above_quarter_hp():
    """test11 run 1: Axe, Knife, Candelabrador, Whip, then Magic Wand, Garlic stuck at 7.

    The cap ranked every extra weapon the same, so the first card on a weapon-only
    screen was opened. King Bible's early return also skipped the cap once two
    damage weapons were held. Armor at 39% HP beat Santa Water.
    """
    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    garlic = {"Garlic": 1}
    # 9s. Clock Lancet does not add a damage weapon. Axe must not win by being first.
    assert picked(
        [("Axe", True), ("Magic Wand", True), ("Clock Lancet", True)],
        garlic, hp=1.0, game_s=9,
    ) == "Clock Lancet"
    # A passive on that screen is the cap. The weapon-only case above is the exception.
    assert picked(
        [("Axe", True), ("Magic Wand", True), ("Hollow Heart", True)],
        garlic, hp=1.0, game_s=9,
    ) == "Hollow Heart"
    # 20s, after Axe was already opened. Still weapon-only: Lightning Ring, not Knife.
    two = {"Garlic": 1, "Axe": 1}
    assert picked(
        [("Knife", True), ("Peachone", True), ("Lightning Ring", True), ("Ebony Wings", True)],
        two, hp=1.0, game_s=20,
    ) == "Lightning Ring"
    assert picked(
        [("Knife", True), ("Peachone", True), ("Hollow Heart", True)],
        two, hp=1.0, game_s=20,
    ) == "Hollow Heart"
    # 38s. Candelabrador was the fallback over both heals.
    assert picked(
        [("Hollow Heart", True), ("Tirajisú", True), ("Candelabrador", True)],
        two, hp=1.0, game_s=38,
    ) == "Hollow Heart"
    # 126s weapon-only, and the same screen with a passive.
    assert picked(
        [("Whip", True), ("Lightning Ring", True), ("Pako Battiliar", True)],
        {"Garlic": 2, "Axe": 1, "Knife": 1}, hp=0.36, game_s=126,
    ) == "Lightning Ring"
    assert picked(
        [("Whip", True), ("Lightning Ring", True), ("Stone Mask", True)],
        {"Garlic": 2, "Axe": 1, "Knife": 1}, hp=0.36, game_s=126,
    ) == "Stone Mask"
    # King Bible used to skip the cap and open a third damage weapon.
    assert picked(
        [("King Bible", True), ("Bracer", True)],
        {"Garlic": 3, "Santa Water": 1}, hp=1.0, game_s=100,
    ) == "Bracer"
    # 155s. Armor outranked a Garlic level at 36% HP.
    assert picked(
        [("Whip", False), ("Garlic", False), ("Armor", True)],
        {"Garlic": 3, "Axe": 1, "Knife": 1, "Whip": 1}, hp=0.36, game_s=155,
    ) == "Garlic"
    # 292s. Santa Water is taken above 25% HP, Armor does not outrank it.
    held = {
        "Garlic": 5, "Armor": 1, "Axe": 1, "Candelabrador": 1,
        "Hollow Heart": 2, "Knife": 1, "Pummarola": 1, "Whip": 1,
    }
    assert picked(
        [("Armor", False), ("Santa Water", True), ("Stone Mask", True)],
        held, hp=0.39, game_s=292,
    ) == "Santa Water"
    # Six weapons already owned. Santa Water is still allowed.
    full = dict(held)
    full.update({"Magic Wand": 1, "King Bible": 1, "Runetracer": 1, "Cross": 1})
    assert picked(
        [("Armor", False), ("Santa Water", True), ("Stone Mask", True)],
        full, hp=0.39, game_s=300,
    ) == "Santa Water"
    # 593s. Hollow Heart must not outrank a Garlic level above 25% HP.
    assert picked(
        [("Garlic", False), ("Bracer", False), ("Hollow Heart", False)],
        dict(held, Bracer=1, **{"Hollow Heart": 3}), hp=0.31, game_s=593,
    ) == "Garlic"


def test_test14_santa_water_is_the_second_weapon_and_garlic_survives_low_hp():
    """test14 run 2 took Ebony Wings over Santa Water at 0:08.

    At 2:24 Pummarola beat a Garlic level at hp=0.25. At 4:33 Tirajisú beat
    one at hp=0.28. Run 1 did not see Santa Water until 7:12, and took it.
    """
    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    assert picked(
        [("Santa Water", True), ("Ebony Wings", True)],
        {"Garlic": 1}, hp=1.0, game_s=8,
    ) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Garlic", False), ("Knife", True)],
        {"Garlic": 2}, hp=1.0, game_s=8,
    ) == "Garlic"
    assert picked(
        [("Pummarola", False), ("Garlic", False), ("Runetracer", True)],
        {"Garlic": 2, "Pummarola": 1}, hp=0.25, game_s=144,
    ) == "Garlic"
    assert picked(
        [("Garlic", False), ("Tirajisú", True), ("Duplicator", True), ("Axe", True)],
        {"Garlic": 3, "Pummarola": 3}, hp=0.28, game_s=273,
    ) == "Garlic"
    assert picked(
        [("Garlic", False), ("Floor Chicken", True), ("Tirajisú", True)],
        {"Garlic": 3}, hp=0.2, game_s=200,
    ) == "Floor Chicken"


def test_owned_level_beats_a_new_unwanted_item_when_the_plan_is_absent():
    levels = {"Garlic": 4, "Pummarola": 1, "Bone": 1}
    cards = _offer([("Whip", True), ("Bone", False), ("Knife", True)], levels)
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Bone" and pick.reason == "fallback"


def test_survival_passives_after_the_partner_beat_a_new_weapon():
    levels = {"Garlic": 3, "Pummarola": 1}
    cards = _offer([("Santa Water", True), ("Armor", True), ("Hollow Heart", True)], levels)
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Santa Water" and pick.reason == "second"
    pick = choose_pick(cards, hp=0.25, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Armor"
    # Pummarola still leads the survival trio once it is already the partner.
    cards = _offer([("Hollow Heart", True), ("Pummarola", False), ("Armor", True)], levels)
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Pummarola"
    # Floor Chicken is the low-HP exception. Armor is not.
    cards = _offer([("Garlic", False), ("Armor", True)], levels)
    pick = choose_pick(cards, hp=0.2, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"
    cards = _offer([("Garlic", False), ("Floor Chicken", True)], levels)
    pick = choose_pick(cards, hp=0.2, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Floor Chicken"
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
    assert cards[pick.index].name == "Santa Water" and pick.reason == "second"


def test_evolution_ready_is_main_at_8_plus_its_partner():
    """Main-weapon cases. A side weapon or the birds are covered by the broader flag test."""
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


def test_evolution_ready_is_any_settled_weapon_not_already_evolved():
    """Broader than the main weapon: a side weapon at 8 with its passive counts."""
    bible = {"Garlic": 2, "King Bible": 8, "Spellbinder": 1}
    assert evolution_ready(bible)
    assert not all_maxed(bible)  # Garlic is still short of 8
    assert not evolution_ready({"Garlic": 8, "Pummarola": 1}, {"Garlic"})
    assert evolution_ready({"Peachone": 8, "Ebony Wings": 8})
    assert not evolution_ready({"Peachone": 8, "Ebony Wings": 7})
    assert not evolution_ready({"Peachone": 8, "Ebony Wings": 8}, {"Peachone", "Ebony Wings"})
    # One label for the union, not one per bird.
    from vs_vision_bot.menus import ready_evolution_labels

    assert ready_evolution_labels({"Peachone": 8, "Ebony Wings": 8}) == [
        "Peachone L8 + Ebony Wings L8"
    ]


def test_all_maxed_waits_for_a_partner_unless_the_slots_are_full():
    """A maxed weapon with no partner is not done while a slot is free.

    Otherwise the chest gate would open every chest before the evolution
    exists. Six full slots are the exception the flag was defined with.
    """
    assert not all_maxed({"Garlic": 8})
    assert not all_maxed({"Garlic": 8, "Pummarola": 1})  # still wants a second weapon
    assert not evolution_ready({"Garlic": 8})
    finished = {"Garlic": 8, "Pummarola": 1, "Magic Wand": 8, "Empty Tome": 1}
    assert evolution_ready(finished) and all_maxed(finished)
    assert all_maxed(finished, {"Garlic", "Magic Wand"})
    assert not evolution_ready(finished, {"Garlic", "Magic Wand"})
    full = {
        name: 8
        for name in (
            "Garlic",
            "Magic Wand",
            "King Bible",
            "Santa Water",
            "Runetracer",
            "Lightning Ring",
        )
    }
    assert len(full) == MAX_WEAPONS
    assert all_maxed(full) and not evolution_ready(full)
    birds = {"Garlic": 8, "Pummarola": 1, "Peachone": 8, "Ebony Wings": 8}
    assert len(owned_weapon_slots(birds, {"Peachone", "Ebony Wings"})) == 2
    assert evolution_ready(birds, {"Peachone", "Ebony Wings"})


def test_chest_text_marks_the_evolved_base_and_ignores_a_stat_chest():
    assert evolutions_from_chest_text("Treasure Found Soul Eater DONE") == {"Garlic"}
    assert evolutions_from_chest_text("You got Holy Wand") == {"Magic Wand"}
    assert evolutions_from_chest_text("Vandalier") == {"Peachone", "Ebony Wings"}
    assert evolutions_from_chest_text("01:04 bv DONE level:2 [ALL] Base Speed up by 108.") == set()
    assert evolutions_from_chest_text("01:17 - DONE Lever:3") == set()
    assert evolutions_from_chest_text("") == set()
    assert levels_from_chest_text("01:04 bv DONE level:2 [ALL] Base Speed up by 108.") == {}
    assert levels_from_chest_text("01:17 - DONE Lever:3") == {}
    assert levels_from_chest_text("Treasure Found Pummarola level:1 DONE") == {"Pummarola": 1}
    assert levels_from_chest_text("Garlic Armor level:3 DONE") == {}
    # A wild jump is decided by the pilot; the parser only reports the one item.
    assert levels_from_chest_text("Garlic level:2 DONE") == {"Garlic": 2}


def test_build_flags_and_log_across_a_simulated_run():
    """Garlic ready, then both weapons maxed, then each chest evolution clears one."""
    logs: list[str] = []
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append, settle_frames=1)

    def grant(name: str, times: int = 1) -> None:
        pilot._names = [name]
        pilot._chosen = name
        pilot._pick_reason = "main"
        pilot._inventory_certain = True
        for _ in range(times):
            pilot._record_pick(0)

    assert pilot.evolution_ready is False and pilot.all_maxed is False
    grant("Pummarola")
    grant("Garlic", 7)  # 1 -> 8
    assert [m for m in logs if m.startswith("[build]")] == [
        "[build] evo_ready=1 all_maxed=0 inv=Garlic:8,Pummarola:1"
    ]
    assert pilot.evolution_ready and not pilot.all_maxed

    grant("Magic Wand", 8)
    grant("Empty Tome")
    assert [m for m in logs if m.startswith("[build]")] == [
        "[build] evo_ready=1 all_maxed=0 inv=Garlic:8,Pummarola:1",
        "[build] evo_ready=1 all_maxed=1 inv=Garlic:8,Empty Tome:1,Magic Wand:8,Pummarola:1",
    ]

    pilot._note_chest_reward("Treasure Found Soul Eater DONE")
    assert pilot.evolved == {"Garlic"}
    assert pilot.levels["Garlic"] == 8
    assert pilot.evolution_ready and pilot.all_maxed  # the wand can still evolve
    assert len([m for m in logs if m.startswith("[build]")]) == 2

    pilot._note_chest_reward("You got Holy Wand")
    assert pilot.evolved == {"Garlic", "Magic Wand"}
    assert not pilot.evolution_ready and pilot.all_maxed
    assert [m for m in logs if m.startswith("[build]")][-1] == (
        "[build] evo_ready=0 all_maxed=1 inv=Garlic:8*,Empty Tome:1,Magic Wand:8*,Pummarola:1"
    )

    pilot.new_run()
    assert pilot.evolved == set() and pilot.levels == {"Garlic": 1}
    assert not pilot.evolution_ready and not pilot.all_maxed
    assert [m for m in logs if m.startswith("[build]")][-1] == (
        "[build] evo_ready=0 all_maxed=0 inv=Garlic:1"
    )
    # The same flags do not log again.
    pilot._publish_build()
    assert sum(m.startswith("[build] evo_ready=0 all_maxed=0") for m in logs) == 1


@needs_tesseract
def test_chest_confirm_parses_an_evolution_past_the_log_snippet():
    import numpy as np

    logs: list[str] = []
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append, async_chest=True, settle_frames=1)
    pilot.levels = {"Garlic": 8, "Pummarola": 1}
    pilot._publish_build()
    now = pilot.clock()
    pilot._last_scan = now
    pilot._next_action = 0
    text = ("pad " * 30) + "Soul Eater treasure open"
    assert len(text) > 80
    pilot._chest_hit = (now, text, "chest", None)
    assert pilot._handle_chest(np.zeros((8, 8, 3), np.uint8), now) == "chest"
    assert "Garlic" in pilot.evolved
    assert pilot.chest_presses == 1
    chest_logs = [m for m in logs if m.startswith("[chest]")]
    assert len(chest_logs) == 1 and "Soul Eater" not in chest_logs[0]
    assert any(m.startswith("[build] evo_ready=0 all_maxed=0") for m in logs)


def test_chest_level_adopts_an_unknown_item_and_ignores_a_wild_jump():
    logs: list[str] = []
    pilot = MenuPilot(tap=lambda _k: None, log=logs.append)
    pilot._note_chest_reward("01:17 - DONE Lever:3")
    pilot._note_chest_reward("01:04 bv DONE level:2 [ALL] Base Speed up by 108.")
    assert pilot.levels == {"Garlic": 1} and pilot.evolved == set()
    pilot._note_chest_reward("Treasure Found Pummarola level:1 DONE")
    assert pilot.levels["Pummarola"] == 1
    pilot.levels["Garlic"] = 8
    pilot._note_chest_reward("Garlic level:2 DONE treasure")
    assert pilot.levels["Garlic"] == 8  # not lowered
    pilot._note_chest_reward("Pummarola level:2 DONE")
    assert pilot.levels["Pummarola"] == 2


def _capped() -> dict[str, int]:
    """Main and second weapon at 8, both passives at 5. Nothing left to level."""
    return {"Garlic": 8, "Pummarola": 5, "Magic Wand": 8, "Empty Tome": 5}


def test_filler_names_match_the_level_up_cards():
    """The capped screen offers Big Coin Bag and Floor Chicken."""
    assert match_name("Big Coin Bag") == "Big Coin Bag"
    assert match_name("Floor Chicken") == "Floor Chicken"
    assert match_name("Gold Coin Bag") == "Gold Coin Bag"
    assert match_name("Gold Coin") == "Gold Coin"
    assert low_hp_limit() == Tuning().low_hp


def test_capped_build_takes_gold_unless_hp_is_low():
    levels = _capped()
    assert items_fully_maxed(levels)
    cards = [
        _card("Big Coin Bag", True),
        _card("Floor Chicken", True),
        _card("King Bible", True),
    ]
    high = choose_pick(cards, hp=0.9, owned=set(levels), levels=levels)
    assert cards[high.index].name == "Big Coin Bag"
    low = choose_pick(cards, hp=low_hp_limit() - 0.01, owned=set(levels), levels=levels)
    assert cards[low.index].name == "Floor Chicken"
    # The band is exclusive: exactly Tuning.low_hp is not low.
    edge = choose_pick(cards, hp=low_hp_limit(), owned=set(levels), levels=levels)
    assert cards[edge.index].name == "Big Coin Bag"
    # Lucyfur's names are the same gold pick.
    alias = [_card("Gold Coin Bag", True), _card("Floor Chicken", True)]
    pick = choose_pick(alias, hp=0.8, owned=set(levels), levels=levels)
    assert alias[pick.index].name == "Gold Coin Bag"
    # Blind with no prior low is not low. Blind after a low reading is.
    blind = choose_pick(cards, hp=None, owned=set(levels), levels=levels)
    assert cards[blind.index].name == "Big Coin Bag"
    after = choose_pick(cards, hp=None, blind_low=True, owned=set(levels), levels=levels)
    assert cards[after.index].name == "Floor Chicken"
    # Navigator critical flag forces the chicken even when the median looks healthy.
    flagged = choose_pick(cards, hp=0.9, critical=True, owned=set(levels), levels=levels)
    assert cards[flagged.index].name == "Floor Chicken"
    assert not read_critical_flag(object())

    class _Nav:
        _critical_on = True

    assert read_critical_flag(_Nav())
    assert hp_is_low(0.9, critical=True)
    assert not hp_is_low(None)
    assert hp_is_low(None, blind_low=True)


def test_filler_rule_stays_off_until_weapons_and_passives_are_capped():
    """Gold does not replace a weapon or a passive the build still needs."""
    opening = {"Garlic": 8, "Pummarola": 5}
    assert not items_fully_maxed(opening)
    cards = [
        _card("Big Coin Bag", True),
        _card("Floor Chicken", True),
        _card("Magic Wand", True),
    ]
    pick = choose_pick(cards, hp=0.95, owned=set(opening), levels=opening)
    assert cards[pick.index].name == "Magic Wand" and pick.reason == "second"
    # Passive still under 5: take the level, not the coin bag.
    levels = _capped()
    levels["Pummarola"] = 4
    assert not items_fully_maxed(levels)
    cards = [
        _card("Big Coin Bag", True),
        _card("Floor Chicken", True),
        _card("Pummarola", False, level=4),
    ]
    pick = choose_pick(cards, hp=0.95, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Pummarola"


def test_hp_tracker_blind_after_a_low_reading_counts_as_low():
    limit = low_hp_limit()
    low = HpTracker(miss_limit=3)
    low.add(0.4)
    low.add(0.4)
    low.add(0.4)
    assert low.value == 0.4 and not low.blind
    for _ in range(3):
        low.add(None)
    assert low.blind and low.value is None and low.blind_after_low(limit)

    healthy = HpTracker(miss_limit=3)
    healthy.add(0.8)
    healthy.add(0.8)
    healthy.add(0.8)
    for _ in range(3):
        healthy.add(None)
    assert healthy.blind and healthy.value is None and not healthy.blind_after_low(limit)

    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None, critical_hp=lambda: False)
    for _ in range(3):
        pilot._hp.add(0.8)
    assert pilot.hp == 0.8 and not pilot.hp_blind and not pilot.pick_is_low()
    for _ in range(3):
        pilot._hp.add(None)
    assert pilot.hp is None and pilot.hp_blind and not pilot.pick_is_low()

    crashed = MenuPilot(tap=lambda _k: None, log=lambda _m: None)
    for _ in range(3):
        crashed._hp.add(0.4)
    for _ in range(3):
        crashed._hp.add(None)
    assert crashed.hp is None and crashed.pick_is_low()

    sticky = MenuPilot(
        tap=lambda _k: None,
        log=lambda _m: None,
        critical_hp=lambda: True,
    )
    for _ in range(3):
        sticky._hp.add(0.9)
    assert sticky.hp == 0.9 and sticky.pick_is_low()


def test_sync_build_state_skips_a_navigator_without_the_method():
    from vs_vision_bot.main import sync_build_state

    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None)
    pilot.levels = {"Garlic": 8, "Pummarola": 1}
    sync_build_state(object(), pilot)  # no set_build_state: the PR still runs

    seen: list[tuple[bool, bool]] = []

    class _Nav:
        def set_build_state(self, evolution_ready: bool, all_maxed: bool) -> None:
            seen.append((evolution_ready, all_maxed))

    sync_build_state(_Nav(), pilot)
    assert seen == [(True, False)]
    pilot.levels["Magic Wand"] = 8
    pilot.levels["Empty Tome"] = 1
    sync_build_state(_Nav(), pilot)
    assert seen[-1] == (True, True)


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


def test_a_card_two_levels_up_corrects_the_tracker_and_stays_on_garlic():
    """Run 1 ended Garlic 8 on the results screen and 6 in the tracker.

    A chest grants +1 with no item name. The next card's 'level N' is how the
    tracker catches up. Two above is still the same weapon; four above is the
    old wild misread and stays on the spreading rank.
    """
    levels = {"Garlic": 5, "Pummarola": 1}
    cards = [_card("Garlic", False, level=7), _card("King Bible", True)]
    eff, certain = assess_inventory(cards, levels)
    assert certain is True and eff["Garlic"] == 7
    pick = choose_pick(cards, owned=set(levels), levels=dict(levels))
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"
    # The results screen was two higher, at the cap. That read is the inventory,
    # so the chest gate can see an evolution.
    capped = {"Garlic": 6, "Pummarola": 1}
    at_cap = [_card("Garlic", False, level=8), _card("Whip", True)]
    eff, certain = assess_inventory(at_cap, capped)
    assert certain is True and eff["Garlic"] == 8
    merge_card_levels(capped, eff)
    assert capped["Garlic"] == 8 and evolution_ready(capped)


def test_hurt_bar_still_takes_defense_when_it_is_actually_offered():
    """1 AM run 1 took Garlic at hp=0.44. The other cards were Candelabrador
    and Cross, not Armor. Under half HP, defense still comes first when it
    is on the card. That screen never happened in the five runs.
    """
    levels = {"Garlic": 3, "Pummarola": 1}
    offered = [
        _card("Garlic", False, level=3),
        _card("Candelabrador", True),
        _card("Cross", True),
    ]
    pick = choose_pick(offered, hp=0.44, owned=set(levels), levels=levels)
    assert offered[pick.index].name == "Garlic" and pick.reason == "main"
    defense = _offer([("Garlic", False), ("Armor", True), ("Hollow Heart", True)], levels)
    pick = choose_pick(defense, hp=0.4, owned=set(levels), levels=levels)
    assert defense[pick.index].name == "Garlic" and pick.reason == "main"


def test_uncertain_screen_levels_an_owned_weapon_before_a_new_crowd_weapon():
    """A shaky extra card must not throw the plan onto Santa Water.

    Static rank puts a new crowd weapon at tier 2 and a Garlic level at tier 3.
    That is what took Fire Wand, Santa Water and King Bible in run 1.
    """
    levels = {"Garlic": 4, "Pummarola": 1}
    cards = [
        _card("Garlic", False, level=4),
        _card("Santa Water", True),
        _card("Candelabrador", True, score=0.7),
    ]
    pick = choose_pick(cards, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Santa Water" and pick.reason == "second"


def test_owned_weapon_level_beats_extra_armor_and_a_new_passive():
    """Run 1 at 10:00 had Armor 3 beside Runetracer 4. Another Armor level,
    and a new Hollow Heart / Tirajisú, lose to the weapon already being leveled.
    """
    levels = {"Garlic": 5, "Runetracer": 4, "Armor": 2, "Pummarola": 1}
    cards = _offer(
        [("Armor", False), ("Runetracer", False), ("Hollow Heart", True), ("Tirajisú", True)],
        levels,
    )
    pick = choose_pick(cards, hp=1.0, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Runetracer" and pick.reason == "fallback"


def test_chest_armor_line_corrects_an_owned_level_and_a_generic_line_does_not():
    text = "DONE -. level:3 Reduces incoming Denage by 1."
    assert levels_from_chest_text(text) == {"Armor": 3}
    generic = "DONE -- level:5 Cooldown reduced by 0.1 seconds. Base Damage up by 2."
    assert levels_from_chest_text(generic) == {}
    assert chest_text_is_reward("ad Treasure Found! ome OPEN") is False
    assert chest_text_is_reward(generic) is True
    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None)
    pilot.levels = {"Garlic": 4, "Armor": 1, "Pummarola": 1}
    pilot._note_chest_reward(generic)
    assert pilot.levels["Garlic"] == 4 and pilot.levels["Armor"] == 1
    pilot._note_chest_reward(text)
    assert pilot.levels["Armor"] == 3


def test_next_level_tag_does_not_double_count_a_certain_pick():
    """Run 3 had one Garlic pick and no chest. Results said Garlic 2.

    The log said Garlic:3. The owned card's level tag is the level this pick
    becomes (tracker 1, tag 2). Copying that tag and then adding 1 records 3.
    A tag sitting on a card we do not take must not move the tracker either
    (run 1 Clover 1 -> 2 on a Pummarola pick). Two above still replaces it.
    """
    levels = {"Garlic": 1}
    cards = [_card("Garlic", False, level=2), _card("Lightning Ring", True)]
    eff, certain = assess_inventory(cards, levels)
    assert certain is True and eff["Garlic"] == 1
    merge_card_levels(levels, eff)
    pick = choose_pick(cards, owned=set(levels), levels=dict(levels))
    assert cards[pick.index].name == "Garlic" and pick.reason == "main"
    levels["Garlic"] = min(WEAPON_MAX_LEVEL, levels["Garlic"] + 1)
    assert levels["Garlic"] == 2

    side = {"Garlic": 3, "Clover": 1, "Pummarola": 1}
    screen = [
        _card("Pummarola", False, level=2),
        _card("Clover", False, level=2),
        _card("Duplicator", True),
    ]
    eff, certain = assess_inventory(screen, side)
    assert certain is True and eff["Clover"] == 1 and eff["Pummarola"] == 1


def test_unfinished_main_takes_a_passive_instead_of_opening_a_weapon():
    """Run 1 pick 8, hp=0.44: Glass Fandango, Fire Wand, Wings. Fire Wand opened.

    Wings is not on the priority list, so a paired weapon outranked it. While
    Garlic is under 8, that weapon loses to Wings. A screen of only weapons
    still takes Magic Wand. A priority passive such as Spinach still loses to
    a late Pentagram.
    """
    levels = {"Garlic": 4, "Pummarola": 1}
    cards = _offer([("Glass Fandango", True), ("Fire Wand", True), ("Wings", True)], levels)
    pick = choose_pick(cards, hp=0.44, owned=set(levels), levels=levels)
    assert cards[pick.index].name == "Wings"
    only_weapons = _offer([("Santa Water", True), ("Magic Wand", True), ("Knife", True)], levels)
    pick = choose_pick(only_weapons, hp=1.0, owned=set(levels), levels=levels)
    assert only_weapons[pick.index].name == "Santa Water" and pick.reason == "second"


def test_test13_garlic_then_pummarola_and_santa_after_thirty_seconds():
    """test13: Pummarola beat a Garlic level, and Santa Water lost at 0:31.

    Laurel, Hollow Heart, Lightning Ring, and Peachone also filled early picks.
    A Garlic level wins over all of those. After 0:30, or once Garlic is L3,
    Santa Water wins above 25% HP even against a Garlic level.
    """
    def picked(spec, levels, hp=1.0, game_s=0.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    early = {"Garlic": 2, "Laurel": 2, "Hollow Heart": 1, "Peachone": 1}
    assert picked(
        [("Garlic", False), ("Pummarola", True), ("Eight The Sparrow", True)],
        {"Garlic": 1, "Laurel": 1, "Peachone": 2}, hp=1.0, game_s=71,
    ) == "Garlic"
    assert picked(
        [("Laurel", False), ("Hollow Heart", False), ("Garlic", False)],
        early, hp=0.44, game_s=20,
    ) == "Garlic"
    assert picked(
        [("Lightning Ring", True), ("Peachone", True), ("Garlic", False)],
        {"Garlic": 2}, hp=1.0, game_s=20,
    ) == "Garlic"
    # 0:31, Garlic still 1. The shortened gate takes Santa Water.
    assert picked(
        [("King Bible", True), ("Santa Water", True), ("Magic Wand", True)],
        {"Garlic": 1, "Clock Lancet": 2}, hp=1.0, game_s=31,
    ) == "Santa Water"
    # Still before 0:30 and Garlic under 3: the Garlic level wins.
    assert picked(
        [("Santa Water", True), ("Garlic", False), ("Knife", True)],
        {"Garlic": 2}, hp=1.0, game_s=19,
    ) == "Garlic"
    # Past the gate, Santa Water beats the Garlic level above 25% HP.
    assert picked(
        [("Garlic", False), ("Santa Water", False), ("Whip", True)],
        {"Garlic": 4, "Santa Water": 1}, hp=0.75, game_s=200,
    ) == "Santa Water"
    assert picked(
        [("Santa Water", True), ("Armor", True)],
        {"Garlic": 4}, hp=0.25, game_s=200,
    ) == "Armor"


def test_done_page_is_marked_as_the_reward_followup(monkeypatch):
    import numpy as np

    monkeypatch.setattr("vs_vision_bot.menus.tesseract_available", lambda: True)
    pilot = MenuPilot(tap=lambda _k: None, log=lambda _m: None, async_chest=True, settle_frames=1)
    now = 1000.0
    pilot.clock = lambda: now
    frame = np.zeros((8, 8, 3), np.uint8)
    pilot._last_scan = now
    pilot._next_action = 0
    pilot._chest_hit = (now, "ad Treasure Found! ome OPEN", "chest", None)
    assert pilot._handle_chest(frame, now) == "chest"
    assert pilot.chest_is_reward is False
    pilot._next_action = 0
    pilot._chest_hit = (now, "DONE -- level:5 Cooldown reduced by 0.1 seconds.", "chest", None)
    assert pilot._handle_chest(frame, now) == "chest"
    assert pilot.chest_is_reward is True
