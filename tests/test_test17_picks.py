"""test17 level-up screens: filler passives, forced weapons, and build cards."""

from __future__ import annotations

from vs_vision_bot.menus import Card, choose_pick


def _card(name: str, new: bool, level: int | None = None, score: float = 1.0) -> Card:
    return Card(0, 1, 0, 1, name=name, raw=name, new=new, level=level, score=score)


def _offer(spec: list[tuple[str, bool]], levels: dict[str, int]) -> list[Card]:
    cards = []
    for name, new in spec:
        level = None if new else levels.get(name)
        cards.append(_card(name, new, level=level))
    return cards


def test_test17_filler_loses_to_a_weapon_when_no_build_card_is_up():
    """test17 screens with no build card. Duplicator, Bracer, and full-HP Armor lost.

    A useful passive still beats opening a weapon. Armor at or below 25% HP,
    and a screen that is only Armor and Stone Mask, keep the old rank.
    Weapon-only screens stay a forced click.
    """
    def picked(spec, levels, hp=1.0, game_s=100.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name, pick.reason

    garlic = {"Garlic": 2, "Santa Water": 1}
    # Run 2 at 106s, hp unread. Axe before Knife on the avoid list.
    assert picked(
        [("Axe", True), ("Knife", True), ("Duplicator", True)], garlic, hp=None,
    ) == ("Axe", "fallback")
    # Run 2 at 153s. Magic Wand is ahead of Lightning Ring.
    assert picked(
        [("Lightning Ring", True), ("Bracer", True), ("Magic Wand", True)],
        garlic, hp=0.75,
    ) == ("Magic Wand", "fallback")
    # Run 3 at 47s, full HP. Magic Wand ahead of Fire Wand.
    assert picked(
        [("Magic Wand", True), ("Fire Wand", True), ("Armor", True)],
        {"Garlic": 1, "Santa Water": 1}, hp=1.0,
    ) == ("Magic Wand", "fallback")
    # No build card, but Spinach is useful and Duplicator is not.
    assert picked(
        [("Axe", True), ("Duplicator", True), ("Spinach", True)], garlic, hp=1.0,
    )[0] == "Spinach"
    # At or below 25% HP, with no build card, Armor still beats a new weapon.
    assert picked(
        [("Magic Wand", True), ("Armor", True)], garlic, hp=0.20,
    )[0] == "Armor"
    # At or below 25% HP Armor may beat a build card. That path is unchanged.
    assert picked(
        [("Santa Water", True), ("Armor", True)], {"Garlic": 3}, hp=0.25,
    )[0] == "Armor"
    # Only junk passives. Armor stays.
    assert picked(
        [("Armor", True), ("Stone Mask", True)], {"Garlic": 3}, hp=0.89,
    )[0] == "Armor"


def test_test17_weapon_only_screens_stay_a_forced_click():
    """No build card and no useful passive. The click is still the best weapon."""
    def picked(spec, levels, hp=1.0, game_s=100.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    assert picked(
        [("Phiera Der Tuphello", True), ("Fire Wand", True), ("Cross", True), ("Knife", True)],
        {"Garlic": 1},
    ) == "Fire Wand"
    assert picked(
        [("Knife", True), ("Axe", True), ("Ebony Wings", True)],
        {"Garlic": 1, "Santa Water": 1},
    ) == "Ebony Wings"
    assert picked(
        [("Runetracer", True), ("Knife", True), ("Glass Fandango", True)],
        {"Garlic": 1},
    ) == "Runetracer"
    assert picked(
        [("Fire Wand", True), ("Silver Wind", True), ("Magic Wand", True)],
        {"Garlic": 2},
    ) == "Magic Wand"


def test_test17_build_card_beats_a_partner_passive_and_full_hp_armor():
    """A held Santa Water beat Spinach. King Bible beats Armor at full HP.

    Attractorb and Candelabrador are build cards, so they beat a new weapon
    and Duplicator. The local held-partner rank used to put Armor above King Bible.
    """
    def picked(spec, levels, hp=1.0, game_s=100.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name, pick.reason

    assert picked(
        [("Santa Water", False), ("Spinach", True), ("Clock Lancet", True)],
        {"Garlic": 3, "Santa Water": 2, "Fire Wand": 1}, hp=0.36, game_s=142,
    ) == ("Santa Water", "second")
    assert picked(
        [("Magic Wand", False), ("Armor", True), ("King Bible", True)],
        {"Garlic": 8, "Pummarola": 2, "Magic Wand": 5, "Empty Tome": 1, "Runetracer": 1},
        hp=1.0, game_s=300,
    ) == ("King Bible", "second")
    assert picked(
        [("Clover", True), ("Attractorb", True), ("Spinach", True)],
        {"Garlic": 2}, hp=1.0,
    )[0] == "Attractorb"
    assert picked(
        [("Duplicator", True), ("Candelabrador", True)],
        {"Garlic": 2}, hp=1.0,
    )[0] == "Candelabrador"


def test_test18_filler_screens():
    """test18 on #51 plus the local patch. Same rule as test17.

    A build card beats Clover. Armor at half HP loses to a real weapon when
    no build card is up. A useful passive beats Duplicator. Weapon-only
    screens with no build card stay a forced click. Armor at 25% HP may
    still beat a Garlic level.
    """
    def picked(spec, levels, hp=1.0, game_s=100.0):
        cards = _offer(spec, levels)
        pick = choose_pick(cards, hp=hp, owned=set(levels), game_s=game_s, levels=dict(levels))
        return cards[pick.index].name

    # Run 1 at 212s. Spellbinder and Candelabrador were on the screen.
    # The local partner hook took Clover (Cross was held).
    assert picked(
        [("Spellbinder", False), ("Candelabrador", True), ("Clover", True)],
        {
            "Garlic": 4, "Cross": 1, "Empty Tome": 1, "King Bible": 1,
            "Magic Wand": 1, "Pummarola": 1, "Santa Water": 1, "Spellbinder": 2,
        },
        hp=1.0, game_s=212,
    ) == "Spellbinder"
    # Run 1 at 398s, hp=0.50. Magic Wand level, Clover level, new Duplicator.
    assert picked(
        [("Magic Wand", False), ("Clover", False), ("Duplicator", True)],
        {
            "Garlic": 7, "Clover": 1, "Cross": 1, "Empty Tome": 1,
            "King Bible": 1, "Magic Wand": 2, "Pummarola": 1,
            "Santa Water": 1, "Spellbinder": 3,
        },
        hp=0.5, game_s=398,
    ) == "Clover"
    # Run 3 at 102s, hp=0.53. No build card. Armor must not beat Fire Wand.
    assert picked(
        [("Knife", True), ("Armor", True), ("Fire Wand", True)],
        {"Garlic": 2, "King Bible": 2, "Clover": 2},
        hp=0.53, game_s=102,
    ) == "Fire Wand"
    # Run 3 at 236s, hp=0.25. Armor may beat the Garlic level.
    assert picked(
        [("Garlic", False), ("Armor", False), ("Duplicator", True), ("Knife", True)],
        {"Garlic": 4, "King Bible": 4, "Armor": 1, "Attractorb": 1},
        hp=0.25, game_s=236,
    ) == "Armor"
    # No build card and no useful passive: the click stays the best weapon.
    assert picked(
        [("Knife", True), ("Cross", True), ("Glass Fandango", True), ("Magic Wand", True)],
        {"Garlic": 1}, hp=1.0, game_s=9,
    ) == "Magic Wand"
    assert picked(
        [("Cross", True), ("Silver Wind", True), ("Whip", True)],
        {"Garlic": 1, "Magic Wand": 1}, hp=1.0, game_s=24,
    ) == "Cross"
    assert picked(
        [("Phiera Der Tuphello", True), ("Runetracer", True), ("Whip", True)],
        {"Garlic": 1}, hp=1.0, game_s=7,
    ) == "Runetracer"
    assert picked(
        [("Whip", True), ("Clock Lancet", True), ("Eight The Sparrow", True)],
        {"Garlic": 1, "Runetracer": 1}, hp=1.0, game_s=19,
    ) == "Clock Lancet"
    assert picked(
        [("Runetracer", False), ("Clock Lancet", False), ("Fire Wand", True)],
        {"Garlic": 7, "Clock Lancet": 1, "King Bible": 3, "Runetracer": 1, "Santa Water": 3},
        hp=1.0, game_s=269,
    ) == "Runetracer"
