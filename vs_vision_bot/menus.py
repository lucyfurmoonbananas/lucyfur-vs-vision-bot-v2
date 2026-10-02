"""Level-up card picking and treasure-chest screens.

Only screens that are positively recognized are acted on:

* Level Up: the purple-blue center panel holding 1-4 gray option cards
  (1 card additionally needs the "Level Up" title via OCR).
* Treasure / item found: OCR must read chest words (OPEN / DONE / Treasure /
  Found) and none of the main-menu, pause, game-over or results words.

Anything else (main menu, pause menu, results, game over) is never pressed.
Reroll / Banish / Skip are never selected: menu navigation only moves up and
down through the card column and confirms with Enter.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# BGR colours measured on the real 960x540 Level Up screen.
PANEL_BGR = np.array([116, 79, 75], dtype=np.int16)
CARD_GRAY = 139

# Pick priority (user's play-a-vampire-survivors-run skill, Level-up section).
PRIORITY: list[str] = [
    "Garlic",
    "Soul Eater",
    "Pummarola",
    "Tirajisú",
    "Tiragisú",
    "Runetracer",
    "NO FUTURE",
    "Magic Wand",
    "Holy Wand",
    "Empty Tome",
    "Vandalier",
    "Peachone",
    "Ebony Wings",
    "Lightning Ring",
    "Thunder Loop",
    "Duplicator",
    "Cross",
    "Heaven Sword",
    "Armor",
    "Clover",
    "Spinach",
    "Santa Water",
    "La Borra",
    "Attractorb",
    "Pentagram",
    "Gorgeous Moon",
    "Crown",
    "Candelabrador",
    "Spellbinder",
    "Stone Mask",
    "Hollow Heart",
]
# Skipped when anything else is on offer (skill: "Skip Whip, Axe, Knife ...").
AVOID: list[str] = ["Whip", "Axe", "Knife", "Silver Wind", "Phiera Der Tuphello"]
# Other known names so OCR noise snaps to a real item instead of a priority one.
OTHER: list[str] = [
    "King Bible",
    "Fire Wand",
    "Bone",
    "Cherry Bomb",
    "Carréllo",
    "Celestial Dusting",
    "Laurel",
    "Song of Mana",
    "Gatti Amari",
    "Shadow Pinion",
    "Clock Lancet",
    "Eight The Sparrow",
    "Gentle Wings",
    "Bracer",
    "Wings",
    "Skull O'Maniac",
    "Metaglio Left",
    "Metaglio Right",
    "Torrona's Box",
    "Pako Battiliar",
    "Floor Chicken",
    "Gold Coins",
    "Coin Bag",
    "Big Coin Bag",
    "Bloody Tear",
    "Unholy Vespers",
    "Hellfire",
    "Death Spiral",
    "Mannajja",
    "Thousand Edge",
    "Heart of Fire",
]
VOCAB: list[str] = PRIORITY + AVOID + OTHER

# Words that mark screens we must never press Enter on.
BANNED_WORDS = (
    "quit",
    "revive",
    "result",
    "game over",
    "resume",
    "options",
    "start",
    "collection",
    "unlocks",
    "power up",
    "exit",
    "credits",
    "reroll",
    "banish",
    "skip",
)
CHEST_WORDS = ("treasure", "open", "done", "found")


@dataclass
class Card:
    y0: int
    y1: int
    x0: int
    x1: int
    name: str = ""
    raw: str = ""
    new: bool = False

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2


@dataclass
class LevelUpScreen:
    panel: tuple[int, int, int, int]  # x, y, w, h
    cards: list[Card] = field(default_factory=list)
    selected: int | None = None


# ---------------------------------------------------------------- detection


def _panel_mask(frame: np.ndarray) -> np.ndarray:
    diff = np.abs(frame.astype(np.int16) - PANEL_BGR).max(axis=2)
    return (diff <= 14).astype(np.uint8) * 255


def _gray_mask(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.int16)
    spread = f.max(axis=2) - f.min(axis=2)
    lo = np.abs(f[:, :, 1] - CARD_GRAY) <= 10
    return ((spread <= 6) & lo).astype(np.uint8)


def find_level_up_panel(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    mask = _panel_mask(frame)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.2 * w or bw > 0.55 * w or bh < 0.5 * h:
            continue
        cx = x + bw / 2
        if abs(cx - w / 2) > 0.12 * w:
            continue
        if area < 0.08 * bw * bh:
            continue
        if best is None or area > best[4]:
            best = (x, y, bw, bh, area)
    if best is None:
        return None
    x, y, bw, bh, _ = best
    return int(x), int(y), int(bw), int(bh)


def find_cards(frame: np.ndarray, panel: tuple[int, int, int, int]) -> list[Card]:
    h = frame.shape[0]
    px, py, pw, ph = panel
    sub = _gray_mask(frame[py : py + ph, px : px + pw])
    row_frac = sub.mean(axis=1)
    on = row_frac > 0.3
    cards: list[Card] = []
    start = None
    min_h = max(12, int(0.06 * h))
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_h:
                cols = sub[start:i].mean(axis=0) > 0.3
                xs = np.flatnonzero(cols)
                if xs.size:
                    cards.append(
                        Card(y0=py + start, y1=py + i, x0=px + int(xs[0]), x1=px + int(xs[-1]) + 1)
                    )
            start = None
    return cards[:4]


def find_selected(frame: np.ndarray, panel: tuple[int, int, int, int], cards: list[Card]) -> int | None:
    """The selection is marked by two pale arrows just outside the panel edges."""
    if not cards:
        return None
    h, w = frame.shape[:2]
    px, py, pw, ph = panel
    band = max(6, int(0.03 * w))
    min_run = max(4, int(0.012 * h))
    band_runs: list[list[tuple[int, int]]] = []
    for x0, x1 in ((px - band, px + 2), (px + pw - 2, px + pw + band)):
        x0, x1 = max(0, x0), min(w, x1)
        region = frame[py : py + ph, x0:x1].astype(np.int16)
        bright = (region[:, :, 2] > 200) & (region[:, :, 1] > 190) & (region[:, :, 0] > 80)
        on = bright.sum(axis=1) >= 2
        runs: list[tuple[int, int]] = []
        start = None
        for i, v in enumerate(list(on) + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_run:
                    runs.append((py + start, py + i))
                start = None
        band_runs.append(runs)
    left, right = band_runs
    # The two arrows sit on the same row: prefer runs present on both sides.
    paired = [
        (max(a0, b0), min(a1, b1))
        for a0, a1 in left
        for b0, b1 in right
        if min(a1, b1) - max(a0, b0) >= min_run // 2
    ]
    pool = paired or left + right
    if not pool:
        return None
    y0, y1 = max(pool, key=lambda r: r[1] - r[0])
    y = (y0 + y1) / 2.0
    dists = [abs(y - c.cy) for c in cards]
    idx = int(np.argmin(dists))
    if dists[idx] > (cards[idx].y1 - cards[idx].y0):
        return None
    return idx


def detect_level_up(frame: np.ndarray) -> LevelUpScreen | None:
    panel = find_level_up_panel(frame)
    if panel is None:
        return None
    cards = find_cards(frame, panel)
    if not cards:
        return None
    heights = [c.y1 - c.y0 for c in cards]
    if max(heights) > 2.0 * min(heights):
        return None
    screen = LevelUpScreen(panel=panel, cards=cards)
    if len(cards) == 1:
        # One card alone is too weak; require the title text.
        title = ocr_image(_white_text(frame[panel[1] : cards[0].y0, panel[0] : panel[0] + panel[2]]))
        if "level" not in title.lower():
            return None
    screen.selected = find_selected(frame, panel, cards)
    return screen


HP_BAR_W_960 = 36  # full player HP bar width on a 960-wide frame


# Measured on real 960x540 frames: the bar spans x 462..497, y 272..276.
HP_BAR_RECT_960 = (462, 272, 498, 277)


def read_hp_bar(frame: np.ndarray) -> float | None:
    """HP from the fixed-position bar under the centered character.

    Each bar column is red (health), dark (missing health) or occluded (a sprite or
    projectile on top). HP = end of the red part / bar width. None unless most of
    the bar is visible and it really looks like the bar (red starting at the left edge).
    """
    h, w = frame.shape[:2]
    sx, sy = w / 960.0, h / 540.0
    x0, y0, x1, y1 = HP_BAR_RECT_960
    x0, x1 = int(round(x0 * sx)), max(int(round(x1 * sx)), int(round(x0 * sx)) + 2)
    y0, y1 = int(round(y0 * sy)), max(int(round(y1 * sy)), int(round(y0 * sy)) + 1)
    roi = frame[y0:y1, x0:x1].astype(np.int16)
    if roi.size == 0:
        return None
    red = ((roi[:, :, 2] > 200) & (roi[:, :, 1] < 50) & (roi[:, :, 0] < 50)).mean(axis=0) >= 0.5
    dark = (roi.max(axis=2) < 45).mean(axis=0) >= 0.5
    n = roi.shape[1]
    seen = red | dark
    if seen.sum() < 0.6 * n:
        return None
    reds = np.flatnonzero(red)
    if reds.size == 0:
        # Empty-looking bar: only trust it if the left end is visible and dark.
        return 0.0 if dark[: max(2, n // 6)].all() else None
    if reds[0] > max(1, n // 12):
        # Red must start at the left edge unless that part is occluded.
        if dark[: reds[0]].any():
            return None
    end = int(reds[-1]) + 1
    if red[end:].sum() == 0 and dark[end:].sum() + red[:end].sum() < 0.5 * n:
        return None
    return min(1.0, end / n)


def read_hp_fraction(frame: np.ndarray) -> float | None:
    """Player HP from the red bar just under the centered character (None if not seen)."""
    anchored = read_hp_bar(frame)
    if anchored is not None:
        return anchored
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.42), int(w * 0.58)
    y0, y1 = int(h * 0.47), int(h * 0.56)
    roi = frame[y0:y1, x0:x1].astype(np.int16)
    red = (roi[:, :, 2] > 200) & (roi[:, :, 1] < 50) & (roi[:, :, 0] < 50)
    dark = roi.max(axis=2) < 45
    nominal = HP_BAR_W_960 * w / 960.0
    best = None
    for r in range(roi.shape[0]):
        xs = np.flatnonzero(red[r])
        if xs.size < 2:
            continue
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = max(runs, key=len)
        if len(run) < 2 or len(run) > nominal * 1.3:
            continue
        # Missing HP shows as the dark rest of the bar to the right.
        end = run[-1] + 1
        missing = 0
        while end + missing < roi.shape[1] and dark[r, end + missing]:
            missing += 1
        total = len(run) + missing
        if total < nominal * 0.7:
            total = nominal
        frac = min(1.0, len(run) / total)
        if best is None or len(run) > best[0]:
            best = (len(run), frac)
    return None if best is None else best[1]


# ---------------------------------------------------------------- OCR


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image(img: np.ndarray, psm: int = 7) -> str:
    if img is None or img.size == 0 or not tesseract_available():
        return ""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    try:
        out = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", str(psm)],
            input=buf.tobytes(),
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[menus] tesseract failed: {exc}", file=sys.stderr)
        return ""
    return out.stdout.decode("utf-8", "replace").strip()


def ocr_words(img: np.ndarray, psm: int = 11) -> list[tuple[str, int, int, int, int]]:
    """Words with boxes (text, x, y, w, h) in the coordinates of ``img``."""
    if img is None or img.size == 0 or not tesseract_available():
        return []
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return []
    try:
        out = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", str(psm), "tsv"],
            input=buf.tobytes(),
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[menus] tesseract failed: {exc}", file=sys.stderr)
        return []
    words = []
    for line in out.stdout.decode("utf-8", "replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12 or not parts[11].strip():
            continue
        try:
            x, y, w, h = (int(v) for v in parts[6:10])
        except ValueError:
            continue
        words.append((parts[11].strip(), x, y, w, h))
    return words


def _white_text(crop: np.ndarray, thr: int = 190, scale: int = 2) -> np.ndarray:
    m = crop.min(axis=2)
    b = np.where(m > thr, 0, 255).astype(np.uint8)
    b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    return cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def match_name(raw: str) -> str:
    text = re.sub(r"[^A-Za-z' ]", "", raw).strip()
    if len(text) < 3:
        return ""
    lower = {v.lower(): v for v in VOCAB}
    best = difflib.get_close_matches(text.lower(), list(lower), n=1, cutoff=0.6)
    if best:
        return lower[best[0]]
    return ""


def read_card(frame: np.ndarray, card: Card) -> Card:
    cw = card.x1 - card.x0
    chh = card.y1 - card.y0
    ny0 = card.y0 + 1
    ny1 = card.y0 + max(12, int(0.3 * chh))
    name_crop = frame[ny0:ny1, card.x0 + int(0.17 * cw) : card.x0 + int(0.63 * cw)]
    card.raw = ocr_image(_white_text(name_crop))
    card.name = match_name(card.raw)
    # "New!" is yellow text in the right half of the title row; owned items say level:N in white.
    tag = frame[ny0:ny1, card.x0 + int(0.6 * cw) : card.x1].astype(np.int16)
    yellow = (tag[:, :, 2] > 180) & (tag[:, :, 1] > 180) & (tag[:, :, 0] < 90)
    card.new = int(yellow.sum()) >= 6
    return card


def read_cards(frame: np.ndarray, screen: LevelUpScreen) -> LevelUpScreen:
    threads = [threading.Thread(target=read_card, args=(frame, c)) for c in screen.cards]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return screen


CHICKEN = "Floor Chicken"
COINS = ("Gold Coins", "Coin Bag", "Big Coin Bag")
LOW_HP = 0.6        # maxed build: chicken below this, otherwise coins
DEFENSIVE_HP = 0.5  # below this, healing / defense items come first
MAX_WEAPONS = 6

# Every weapon (base + evolved) so owned-weapon slots can be counted.
WEAPONS: frozenset[str] = frozenset(
    {
        "Whip", "Magic Wand", "Knife", "Axe", "Cross", "King Bible", "Fire Wand", "Garlic",
        "Santa Water", "Runetracer", "Lightning Ring", "Pentagram", "Peachone", "Ebony Wings",
        "Phiera Der Tuphello", "Eight The Sparrow", "Gatti Amari", "Song of Mana", "Shadow Pinion",
        "Clock Lancet", "Laurel", "Bone", "Cherry Bomb", "Carréllo", "Celestial Dusting",
        "Silver Wind", "Pako Battiliar",
        "Bloody Tear", "Holy Wand", "Thousand Edge", "Death Spiral", "Heaven Sword",
        "Unholy Vespers", "Hellfire", "Soul Eater", "La Borra", "NO FUTURE", "Thunder Loop",
        "Gorgeous Moon", "Vandalier", "Mannajja",
    }
)
EVOLUTIONS: tuple[str, ...] = (
    "Soul Eater", "Holy Wand", "NO FUTURE", "Thunder Loop", "Vandalier", "La Borra",
    "Unholy Vespers", "Heaven Sword", "Gorgeous Moon", "Hellfire", "Bloody Tear",
    "Death Spiral", "Thousand Edge", "Mannajja",
)
# Crowd-clearers that need no aiming: wanted early so the build has 5-6 weapons by 5:00.
CROWD_WEAPONS: tuple[str, ...] = ("Santa Water", "King Bible", "Lightning Ring")
OTHER_WEAPONS: tuple[str, ...] = (
    "Magic Wand", "Runetracer", "Peachone", "Ebony Wings", "Cross", "Pentagram", "Fire Wand", "Laurel",
)
# Healing / defense, best first. Tirajisú is an extra revival; Laurel is a hit-blocking shield.
DEFENSE: tuple[str, ...] = (CHICKEN, "Armor", "Hollow Heart", "Pummarola", "Tirajisú", "Tiragisú", "Laurel")
# Passives that do nothing for survival right now: never taken while HP is low if anything better is offered.
LUXURY: frozenset[str] = frozenset(
    {"Duplicator", "Spellbinder", "Stone Mask", "Crown", "Candelabrador", "Clover", "Empty Tome",
     "Attractorb", "Skull O'Maniac", "Bracer", "Wings", "Spinach"}
)


def owned_weapon_count(owned: set[str] | frozenset[str]) -> int:
    return sum(1 for n in owned if n in WEAPONS)


def _card_rank(c: Card, idx: int, hp: float | None, weapons: int) -> tuple[int, int]:
    name = c.name
    # OCR garbage (empty / unmatched after match_name): worse than AVOID-new (40)
    # so a real named card always wins. If every card is junk, choose_card still
    # picks one via this shared tier (existing unknown-select cycling stays).
    if not name:
        return (45, idx)
    low = hp is not None and hp < DEFENSIVE_HP
    slots_free = weapons < MAX_WEAPONS
    if name in EVOLUTIONS and not c.new:
        return (0, EVOLUTIONS.index(name))
    if low and name in DEFENSE:
        return (1, DEFENSE.index(name))
    if name in AVOID:
        return (40 if c.new else 30, idx)
    if name == CHICKEN or name in COINS:
        return (50, idx)
    if c.new and name in WEAPONS:
        if not slots_free:
            return (35, idx)
        if name in CROWD_WEAPONS:
            return (2, CROWD_WEAPONS.index(name))
        if name in OTHER_WEAPONS:
            return (4, OTHER_WEAPONS.index(name))
        return (20, idx)  # unknown / unranked new weapon
    if name == "Garlic":
        return (3, 0)
    if name in ("Pummarola", "Tirajisú", "Tiragisú", "Armor"):
        order = ("Pummarola", "Tirajisú", "Tiragisú", "Armor")
        return (5, order.index(name))
    if not c.new and name in WEAPONS:
        return (6, PRIORITY.index(name) if name in PRIORITY else len(PRIORITY) + idx)
    if name == "Hollow Heart":
        return (7, 0)
    if name in PRIORITY:
        # Passives by the skill's order; luxury ones sink below everything useful at low HP.
        return (25 if low and name in LUXURY else 8, PRIORITY.index(name))
    if not c.new:
        return (25 if low and name in LUXURY else 10, idx)  # a level on anything owned
    return (15, idx)


def choose_card(cards: list[Card], hp: float | None = None, owned: set[str] | None = None) -> int:
    """Index of the card to take.

    * Maxed-out build (only Floor Chicken / coin bag offered): chicken when HP is
      below ~60 %, otherwise the coin bag.
    * Evolutions first. HP below 50 %: Floor Chicken, Armor, Hollow Heart,
      Pummarola (then Tirajisú / Laurel) before anything else, and luxury
      passives such as Duplicator sink below weapon levels.
    * While weapon slots are free: new crowd-clearers (Santa Water, King Bible,
      Lightning Ring), then Garlic levels, then other good new weapons, then
      Pummarola / Armor, owned weapon levels, passives by priority.
    * The skill's avoid list (Whip, Axe, Knife, ...) and filler come last.
    Reroll / Banish / Skip are not cards and are never selected.
    """
    if not cards:
        return 0
    names = [c.name for c in cards]
    filler = {CHICKEN, *COINS}
    if any(n in filler for n in names) and all(n in filler or not n for n in names):
        chicken = [i for i, n in enumerate(names) if n == CHICKEN]
        coins = [i for i, n in enumerate(names) if n in COINS]
        if chicken and hp is not None and hp < LOW_HP:
            return chicken[0]
        if coins:
            return coins[0]
        return chicken[0]
    known = set(owned or ())
    known |= {c.name for c in cards if c.name and not c.new}
    weapons = owned_weapon_count(known)
    return min(range(len(cards)), key=lambda i: _card_rank(cards[i], i, hp, weapons))


class HpTracker:
    """Median of the last few HP-bar readings: one frame with a sprite over the bar
    (or a red hit flash) must not swing the level-up choice."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.values: list[float] = []

    def add(self, v: float | None) -> None:
        if v is None:
            return
        self.values.append(float(v))
        del self.values[: -self.window]

    def reset(self) -> None:
        self.values.clear()

    @property
    def value(self) -> float | None:
        if not self.values:
            return None
        return float(np.median(self.values))


# ---------------------------------------------------------------- chest OCR


def classify_chest_text(text: str) -> str | None:
    """'revive' when a REVIVE button is offered, 'chest' for treasure / item-found screens, else None."""
    t = " ".join(text.lower().split())
    if not t:
        return None
    if re.search(r"\brevive\b", t):
        return "revive"  # Lucyfur: always take REVIVE when it is offered.
    if any(word in t for word in BANNED_WORDS):
        return None
    words = set(re.findall(r"[a-z]+", t))
    if {"treasure", "open", "done"} & words or {"found", "get"} <= words:
        return "chest"
    return None


def read_screen_words(frame: np.ndarray) -> list[tuple[str, int, int, int, int]]:
    """OCR bright text in the center of the frame; boxes are in frame coordinates."""
    h, w = frame.shape[:2]
    ox, oy = int(0.2 * w), int(0.05 * h)
    center = frame[oy : int(0.97 * h), ox : int(0.8 * w)]
    img = _white_text(center, thr=200, scale=1)
    pad = 12  # _white_text border
    return [(t, x - pad + ox, y - pad + oy, bw, bh) for t, x, y, bw, bh in ocr_words(img, psm=11)]


def read_gold_title(frame: np.ndarray) -> str:
    """OCR the gold serif title (GAME OVER) in the upper middle of the frame."""
    h, w = frame.shape[:2]
    r = frame[int(0.15 * h) : int(0.5 * h), int(0.2 * w) : int(0.8 * w)].astype(np.int16)
    gold = (r[:, :, 2] > 190) & (r[:, :, 1] > 110) & (r[:, :, 0] < 120)
    img = np.where(gold, 0, 255).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    return ocr_image(img, psm=6)


def is_game_over_title(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return any(difflib.SequenceMatcher(None, a, "game").ratio() >= 0.75 for a in words) and any(
        difflib.SequenceMatcher(None, a, "over").ratio() >= 0.75 for a in words
    )


QUIT_BUTTON_960 = (480, 372)  # center of the red QUIT button on the final GAME OVER


def is_final_quit_button(frame: np.ndarray, words: list[tuple[str, int, int, int, int]]) -> bool:
    """The final GAME OVER's big red QUIT button: the OCR'd word QUIT sits on a solid
    red button at the fixed spot. Works when the gold GAME OVER title is unreadable
    over a busy crowd (live run9 sat on this screen because the title OCR failed)."""
    h, w = frame.shape[:2]
    sx, sy = w / 960.0, h / 540.0
    cx, cy = QUIT_BUTTON_960[0] * sx, QUIT_BUTTON_960[1] * sy
    hit = find_word(words, "quit")
    if hit is None or abs(hit[0] - cx) > 50 * sx or abs(hit[1] - cy) > 25 * sy:
        return False
    box = frame[int(355 * sy) : int(390 * sy), int(410 * sx) : int(550 * sx)].astype(np.int16)
    if box.size == 0:
        return False
    red = (box[:, :, 2] > 150) & (box[:, :, 1] < 90) & (box[:, :, 0] < 70)
    return float(red.mean()) > 0.6


def read_chest_text(frame: np.ndarray) -> str:
    return " ".join(t for t, *_ in read_screen_words(frame))


def find_word(words: list[tuple[str, int, int, int, int]], target: str) -> tuple[int, int] | None:
    """Center of the first OCR word matching ``target`` (case-insensitive, fuzzy)."""
    for t, x, y, w, h in words:
        clean = re.sub(r"[^a-z]", "", t.lower())
        if clean and difflib.SequenceMatcher(None, clean, target).ratio() >= 0.8:
            return x + w // 2, y + h // 2
    return None


# ---------------------------------------------------------------- pilot


class MenuPilot:
    """Drives level-up and chest screens through the bot's own input backend."""

    def __init__(
        self,
        tap: Callable[[str], None],
        refocus: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        nav_cooldown_s: float = 0.25,
        confirm_cooldown_s: float = 1.2,
        chest_cooldown_s: float = 1.5,
        chest_scan_s: float = 1.5,
        async_chest: bool = True,
        click: Callable[[int, int], None] | None = None,
        settle_frames: int = 2,
        log: Callable[[str], None] = print,
        results_dir: str | None = None,
        start_items: tuple[str, ...] = ("Garlic",),
        levelup_click: bool = True,
    ) -> None:
        self.levelup_click = levelup_click
        self.results_dir = results_dir
        self.start_items = tuple(start_items)
        self.owned: set[str] = set(self.start_items)
        self._hp = HpTracker()
        self.saved_results: list[str] = []
        self.tap = tap
        self.refocus = refocus or (lambda: True)
        self.clock = clock
        self.nav_cooldown_s = nav_cooldown_s
        self.confirm_cooldown_s = confirm_cooldown_s
        self.chest_cooldown_s = chest_cooldown_s
        self.chest_scan_s = chest_scan_s
        self.async_chest = async_chest
        self.log = log
        self.click = click
        self.revives = 0
        self._quit_pressed = False
        self.finished = False
        self.settle_frames = settle_frames
        self._stable = 0
        self._next_action = 0.0
        self._sig: bytes | None = None
        self._target: int | None = None
        self._names: list[str] = []
        self._chosen = ""
        self._presses = 0
        self._unknown_sel = 0
        self._last_scan = -1e9
        self._scan_thread: threading.Thread | None = None
        self._chest_hit: tuple | None = None
        self._chest_streak = 0
        self.picks: list[str] = []
        self.chest_presses = 0
        self.status = ""

    @property
    def hp(self) -> float | None:
        """Smoothed HP fraction (median of recent bar readings), None until seen."""
        return self._hp.value

    @hp.setter
    def hp(self, v: float | None) -> None:
        self._hp.reset()
        self._hp.add(v)

    def new_run(self) -> None:
        """A fresh run: forget owned items and HP history."""
        self.reset_run()
        self.revives = 0
        self.owned = set(self.start_items)
        self._hp.reset()
        self.picks = []

    # -- helpers
    def _press(self, key: str, cooldown: float) -> None:
        self.refocus()
        self.tap(key)
        self._next_action = self.clock() + cooldown

    @staticmethod
    def _signature(frame: np.ndarray, screen: LevelUpScreen) -> bytes:
        x, y, w, h = screen.panel
        c0, c1 = screen.cards[0], screen.cards[-1]
        crop = frame[c0.y0 : c1.y1, c0.x0 : c0.x1]
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (24, 16), interpolation=cv2.INTER_AREA)
        return (small // 32).tobytes() + bytes([len(screen.cards)])

    def reset_run(self) -> None:
        """Forget GAME OVER / results state (new run, bot unpaused again)."""
        self._quit_pressed = False
        self.finished = False
        self._chest_hit = None
        self._chest_streak = 0
        self._target = None
        self._sig = None

    # -- main entry
    def step(self, frame: np.ndarray, screen: LevelUpScreen | None = None) -> str | None:
        """Handle one frame. Returns 'levelup', 'chest', 'revive', 'quit', 'results' or None."""
        if self.finished:
            return None
        if screen is None:
            screen = detect_level_up(frame)
        now = self.clock()
        if screen is not None:
            self._chest_hit = None
            self._handle_level_up(frame, screen, now)
            return "levelup"
        self._sig = None
        self._target = None
        self._hp.add(read_hp_fraction(frame))
        return self._handle_chest(frame, now)

    def _handle_level_up(self, frame: np.ndarray, screen: LevelUpScreen, now: float) -> None:
        sig = self._signature(frame, screen)
        if self._target is None:
            # Cards slide in: wait until the card column is stable before reading names.
            if sig != self._sig:
                self._sig = sig
                self._stable = 0
                self.status = "LEVEL UP (settling)"
                return
            self._stable += 1
            if self._stable < self.settle_frames:
                return
            read_cards(frame, screen)
            self._names = [c.name or c.raw or "?" for c in screen.cards]
            self.owned |= {c.name for c in screen.cards if c.name and not c.new}
            self._target = choose_card(screen.cards, hp=self.hp, owned=self.owned)
            self._chosen = screen.cards[self._target].name
            self._presses = 0
            self._unknown_sel = 0
            self.log(
                f"[levelup] cards={self._names} new={[c.new for c in screen.cards]} "
                f"-> pick #{self._target + 1} {self._names[self._target]} "
                f"(hp={self.hp if self.hp is None else round(self.hp, 2)}, weapons={owned_weapon_count(self.owned)})"
            )
        if now < self._next_action:
            self.status = f"LEVEL UP -> {self._names[self._target]}"
            return
        target = min(self._target or 0, len(screen.cards) - 1)
        if self.click is not None and self.levelup_click:
            # Click the card itself: one click picks it. Keyboard navigation could
            # wander off the card column onto REROLL / BANISH (live run8), so the
            # mouse is the default; the click point is the card's own center.
            card = screen.cards[target]
            x = (card.x0 + card.x1) // 2
            y = card.cy
            self.refocus()
            self.click(x, y)
            self._next_action = self.clock() + self.confirm_cooldown_s
            self._record_pick(target)
            self.status = f"LEVEL UP -> {self._names[target] if target < len(self._names) else '?'}"
            return
        sel = screen.selected
        if sel is None:
            self._unknown_sel += 1
            if self._unknown_sel < 4 or self._presses >= 12:
                # Wait: arrows animate. After many presses stop pressing at all rather
                # than confirm something that might be REROLL / BANISH.
                return
            self._press("down", self.nav_cooldown_s)
            self._presses += 1
            self._unknown_sel = 0
            return
        if sel < target:
            self._press("down", self.nav_cooldown_s)
            self._presses += 1
        elif sel > target:
            self._press("up", self.nav_cooldown_s)
            self._presses += 1
        else:
            # Enter only when the highlight is positively on the target card.
            self._press("enter", self.confirm_cooldown_s)
            self._record_pick(target)
        self.status = f"LEVEL UP -> {self._names[target] if target < len(self._names) else '?'}"

    def _record_pick(self, target: int) -> None:
        name = self._names[target] if target < len(self._names) else "?"
        self.picks.append(name)
        if self._chosen:
            self.owned.add(self._chosen)
        self.log(f"[levelup] picked #{target + 1} {name} (total picks {len(self.picks)})")
        self._target = None
        self._sig = None

    def _save_results(self, frame: np.ndarray) -> str | None:
        """Keep a screenshot of the Results screen (time / level / kills) before DONE."""
        if not self.results_dir:
            return None
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            path = os.path.join(self.results_dir, time.strftime("results_%Y%m%d_%H%M%S.png"))
            if cv2.imwrite(path, frame):
                self.saved_results.append(path)
                self.log(f"[results] saved {path}")
                return path
        except Exception as exc:  # noqa: BLE001
            self.log(f"[results] could not save screenshot: {exc}")
        return None

    def _scan(self, frame: np.ndarray) -> None:
        words = read_screen_words(frame)
        text = " ".join(t for t, *_ in words)
        kind = classify_chest_text(text)
        lower = set(re.findall(r"[a-z]+", text.lower()))
        if kind is None and "quit" in lower and not ({"resume", "options", "start"} & lower):
            # Final GAME OVER (no revive left): QUIT, then DONE on Results, then stop.
            if is_final_quit_button(frame, words) or is_game_over_title(read_gold_title(frame)):
                kind = "quit"
        if kind is None and self._quit_pressed and "done" in lower:
            kind = "results"
        if kind:
            self._chest_streak += 1
            if self._chest_streak >= 2:  # two scans in a row: not OCR noise
                target = {"revive": "revive", "quit": "quit", "results": "done"}.get(kind)
                where = find_word(words, target) if target else None
                self._chest_hit = (self.clock(), text[:80], kind, where)
        else:
            self._chest_streak = 0
            self._chest_hit = None

    def _handle_chest(self, frame: np.ndarray, now: float) -> str | None:
        if not tesseract_available():
            return None
        busy = self._scan_thread is not None and self._scan_thread.is_alive()
        if not busy and now - self._last_scan >= self.chest_scan_s:
            self._last_scan = now
            if self.async_chest:
                self._scan_thread = threading.Thread(target=self._scan, args=(frame.copy(),), daemon=True)
                self._scan_thread.start()
            else:
                self._scan(frame)
        hit = self._chest_hit
        if hit is None or now - hit[0] > 2.0 * self.chest_scan_s:
            return None
        kind, where = hit[2], hit[3]
        if now >= self._next_action:
            if kind == "results":
                self._save_results(frame)
            if kind in ("revive", "quit", "results"):
                label = {"revive": "REVIVE", "quit": "QUIT", "results": "DONE"}[kind]
                if where is not None and self.click is not None:
                    self.refocus()
                    self.click(*where)
                    self._next_action = self.clock() + self.chest_cooldown_s
                    self.log(f"[{kind}] clicked {label} at {where} ({hit[1]!r})")
                else:
                    # The only / highlighted button on these screens.
                    self._press("enter", self.chest_cooldown_s)
                    self.log(f"[{kind}] Enter on {label} ({hit[1]!r})")
                if kind == "revive":
                    self.revives += 1
                elif kind == "quit":
                    self._quit_pressed = True
                else:
                    self.finished = True
                    self.log("[menus] run finished (QUIT + DONE); the bot stops acting until p is pressed")
            else:
                self._press("enter", self.chest_cooldown_s)
                self.chest_presses += 1
                self.log(f"[chest] Enter on chest screen ({hit[1]!r})")
            self._chest_hit = None
            self._chest_streak = 0  # the next screen must be confirmed by two fresh scans
        self.status = kind.upper()
        return kind
