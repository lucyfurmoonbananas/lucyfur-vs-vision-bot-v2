"""Level-up card picking and treasure-chest screens.

Only screens that are positively recognized are acted on:

* Level Up: the purple-blue center panel holding 1-4 gray option cards
  (1 card additionally needs the "Level Up" title via OCR).
* Treasure / item found: OCR must read chest words (OPEN / DONE / Treasure /
  Found) and none of the main-menu, pause, game-over or results words.

Anything else (main menu, pause menu, results, game over) is never pressed.
Reroll and Skip are never selected: menu navigation only moves up and down
through the card column and confirms with Enter. Banish has exactly one
exception (Lucyfur, Oct 10): on Mad Forest a NEW Pentagram card is banished by
clicking the BANISH button and then that Pentagram card (see ``BANISH_ALLOWED``,
``banish_candidate`` and ``MenuPilot._banish_step``). Banish is never used for
any other card or stage.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

# BGR colours measured on the real 960x540 Level Up screen.
PANEL_BGR = np.array([116, 79, 75], dtype=np.int16)
CARD_GRAY = 139
CARD_ROW_FRAC = 0.12  # min gray fraction of a panel row that belongs to a card

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
    "Glass Fandango",
    "Parm Aegis",
    "Candybox",
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
    score: float = 0.0  # similarity of ``raw`` to ``name`` (match_name_score)

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
    # Rows between cards are ~0 gray; a two-line description drops a card row to ~0.27,
    # which used to split the card (short top half, icon-shifted x0, name crop cut "Armor"
    # to "or"). A low threshold keeps the card whole; the gap rows stay well below it.
    on = row_frac > CARD_ROW_FRAC
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


# One budget for a whole level-up read. Each call used to have its own 5s
# timeout, and a miss is a weak match so every card retried twice; running those
# at once also made a healthy read slower than the budget. subprocess.run then
# wait()ed with no deadline after killing only the direct child.
MENU_OCR_DEADLINE_S = 3.0
_ocr_deadline_at: float | None = None


def _arm_ocr_deadline() -> bool:
    global _ocr_deadline_at
    now = time.monotonic()
    if _ocr_deadline_at is not None and now < _ocr_deadline_at:
        return False
    _ocr_deadline_at = now + MENU_OCR_DEADLINE_S
    return True


def _disarm_ocr_deadline(armed: bool) -> None:
    global _ocr_deadline_at
    if armed:
        _ocr_deadline_at = None


@contextmanager
def _menu_ocr_budget() -> Iterator[None]:
    armed = _arm_ocr_deadline()
    try:
        yield
    finally:
        _disarm_ocr_deadline(armed)


def _ocr_timeout() -> float | None:
    if _ocr_deadline_at is None:
        return MENU_OCR_DEADLINE_S
    left = _ocr_deadline_at - time.monotonic()
    if left < 0.05:
        return None
    return left


def _kill_tesseract(proc: subprocess.Popen) -> None:
    """SIGKILL the tesseract process group. Do not wait without a deadline."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass


def _drop_pipes(proc: subprocess.Popen) -> None:
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is None:
            continue
        try:
            os.close(stream.fileno())
        except (OSError, ValueError):
            pass
        try:
            stream.close()
        except OSError:
            pass


def _run_tesseract(args: list[str], data: bytes) -> bytes | None:
    """Run one tesseract. None when it misses the menu-OCR deadline."""
    timeout = _ocr_timeout()
    if timeout is None:
        return None
    argv = ["tesseract", *args]
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"[menus] tesseract failed: {exc}", file=sys.stderr)
        return None
    try:
        try:
            out, _err = proc.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(
                f"[menus] tesseract failed: Command {argv!r} timed out after {timeout:.0f}s",
                file=sys.stderr,
            )
            return None
        except OSError as exc:
            print(f"[menus] tesseract failed: {exc}", file=sys.stderr)
            return None
        return out
    finally:
        _kill_tesseract(proc)
        _drop_pipes(proc)


def ocr_image(img: np.ndarray, psm: int = 7) -> str:
    if img is None or img.size == 0 or not tesseract_available():
        return ""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    out = _run_tesseract(["stdin", "stdout", "--psm", str(psm)], buf.tobytes())
    if out is None:
        return ""
    return out.decode("utf-8", "replace").strip()


def ocr_words(img: np.ndarray, psm: int = 11) -> list[tuple[str, int, int, int, int]]:
    """Words with boxes (text, x, y, w, h) in the coordinates of ``img``."""
    if img is None or img.size == 0 or not tesseract_available():
        return []
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return []
    out = _run_tesseract(["stdin", "stdout", "--psm", str(psm), "tsv"], buf.tobytes())
    if out is None:
        return []
    words = []
    for line in out.decode("utf-8", "replace").splitlines()[1:]:
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


def match_name_score(raw: str) -> tuple[str, float]:
    """Best VOCAB name for an OCR reading and its similarity (0 when nothing >= 0.6)."""
    text = re.sub(r"[^A-Za-z' ]", "", raw).strip()
    if len(text) < 3:
        return "", 0.0
    lower = {v.lower(): v for v in VOCAB}
    best = difflib.get_close_matches(text.lower(), list(lower), n=1, cutoff=0.6)
    if best:
        return lower[best[0]], difflib.SequenceMatcher(None, text.lower(), best[0]).ratio()
    return "", 0.0


def match_name(raw: str) -> str:
    return match_name_score(raw)[0]


# A name read below this similarity is re-OCR'd at other white thresholds (Oct 6:
# "Armor" read as "Axrer" at 190 -> snapped to Axe, the avoid list; 200/210 read "Armor").
NAME_RETRY_BELOW = 0.85
NAME_RETRY_THRS = (210, 170)


def read_card(frame: np.ndarray, card: Card) -> Card:
    cw = card.x1 - card.x0
    chh = card.y1 - card.y0
    ny0 = card.y0 + 1
    ny1 = card.y0 + max(12, int(0.3 * chh))
    name_crop = frame[ny0:ny1, card.x0 + int(0.17 * cw) : card.x0 + int(0.63 * cw)]
    card.raw = ocr_image(_white_text(name_crop))
    card.name, score = match_name_score(card.raw)
    if score < NAME_RETRY_BELOW:
        for thr in NAME_RETRY_THRS:
            raw = ocr_image(_white_text(name_crop, thr=thr))
            name, sc = match_name_score(raw)
            if sc > score:
                card.raw, card.name, score = raw, name, sc
            if score >= NAME_RETRY_BELOW:
                break
    card.score = score
    # "New!" is yellow text in the right half of the title row; owned items say level:N in white.
    tag = frame[ny0:ny1, card.x0 + int(0.6 * cw) : card.x1].astype(np.int16)
    yellow = (tag[:, :, 2] > 180) & (tag[:, :, 1] > 180) & (tag[:, :, 0] < 90)
    card.new = int(yellow.sum()) >= 6
    return card


def read_cards(frame: np.ndarray, screen: LevelUpScreen) -> LevelUpScreen:
    # One tesseract at a time. Four at once (OpenMP) took longer on a 4-core
    # machine than the same crops in series, and a shared deadline would then
    # cut off a read that is fine on its own.
    with _menu_ocr_budget():
        for card in screen.cards:
            read_card(frame, card)
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
        "Silver Wind", "Pako Battiliar", "Glass Fandango",
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


# New weapons that hurt early in a run. Pentagram at levels 1-2 erases every gem and
# pickup on screen when it fires (VS wiki); Oct 9 runs that took it early died at
# level 6-8 while the run that skipped it reached level 11. Lucyfur: avoid it early.
# Ranked below every real weapon / passive (avoid list included) as a NEW pick, but a
# card is still always chosen: Reroll / Skip are never pressed. Leveling an
# already-owned Pentagram is unaffected, and after the cutoff it keeps its normal rank.
# On Mad Forest the pilot first tries to BANISH a new Pentagram (below); this rank is
# the fallback when that is not possible (no uses left, button not seen, failed).
EARLY_AVOID_NEW: frozenset[str] = frozenset({"Pentagram"})
EARLY_AVOID_UNTIL_S = 600.0     # game time (nav game~, s) before which the rule applies
EARLY_AVOID_MIN_WEAPONS = 4     # fallback when game time is unknown: early = fewer weapons
EARLY_AVOID_RANK = 42           # below AVOID-new (40), above OCR garbage (45) / filler (50)


def is_early_run(game_s: float | None, weapons: int) -> bool:
    """Early in a run: game time before EARLY_AVOID_UNTIL_S, or (no game time)
    fewer than EARLY_AVOID_MIN_WEAPONS weapons owned."""
    if game_s is not None:
        return game_s < EARLY_AVOID_UNTIL_S
    return weapons < EARLY_AVOID_MIN_WEAPONS


def owned_weapon_count(owned: set[str] | frozenset[str]) -> int:
    return sum(1 for n in owned if n in WEAPONS)


def _card_rank(c: Card, idx: int, hp: float | None, weapons: int, early: bool = False) -> tuple[int, int]:
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
    if early and c.new and name in EARLY_AVOID_NEW:
        return (EARLY_AVOID_RANK, idx)
    if name in AVOID:
        return (40 if c.new else 30, idx)
    if name == CHICKEN or name in COINS:
        return (50, idx)
    if c.new and name in WEAPONS:
        if not slots_free:
            return (35, idx)
        if name in CROWD_WEAPONS:
            return (2, CROWD_WEAPONS.index(name))
        # A/B (VS_LAUREL_DAMAGE=0): a new Laurel is no kill weapon; low HP still takes it via DEFENSE.
        other = tuple(w for w in OTHER_WEAPONS if w != "Laurel") if os.environ.get("VS_LAUREL_DAMAGE") == "0" else OTHER_WEAPONS
        if name in other:
            return (4, other.index(name))
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


def choose_card(
    cards: list[Card],
    hp: float | None = None,
    owned: set[str] | None = None,
    game_s: float | None = None,
) -> int:
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
    * Early in a run (``game_s`` < 10:00, or fewer than 4 weapons when the game
      time is unknown) a NEW Pentagram ranks below every other real card, the
      avoid list included: it wipes on-screen gems at low level. Owned Pentagram
      levels and later runs are unaffected.
    Reroll / Banish / Skip are not cards and are never selected here; the one
    Banish exception (new Pentagram on Mad Forest) is handled by MenuPilot first.
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
    early = is_early_run(game_s, weapons)
    return min(range(len(cards)), key=lambda i: _card_rank(cards[i], i, hp, weapons, early))


# ---------------------------------------------------------------- banish (one exception)

# Lucyfur (Oct 10): on Mad Forest press BANISH on a NEW Pentagram card. This is the
# only Banish use: every other card, an owned Pentagram's level and every other
# stage keep the old rule (never Banish). Hard guard: both clicks of a banish go
# through banish_allowed(), and nothing else ever clicks the BANISH button.
BANISH_ALLOWED: frozenset[str] = frozenset({"Pentagram"})
BANISH_STAGES: frozenset[str] = frozenset({"mad_forest"})  # only when VS_STAGE says so
# Red BANISH button ("BANISH / 10 left") right of the panel on every saved level-up
# frame (45 Oct 6-7 frames + 4 fixtures): x 707..845, y 299..330 on 960x540.
BANISH_BUTTON_960 = (707, 299, 139, 32)
# Red CANCEL button that replaces it while BANISH is armed (both Oct 11 levelup_banish_armed_*
# frames: red 0.81 of the rect; <= 0.005 on every other saved level-up / fixture).
CANCEL_BUTTON_960 = (707, 331, 139, 32)
BANISH_RED_FRAC = 0.6        # measured 0.73-0.75 of the rect is button red
BANISH_MIN_NAME_SCORE = 0.85  # the Pentagram card name must be a confident read
BANISH_MAX_CARD_CLICKS = 2    # clicks on the Pentagram card before giving up
BANISH_MAX_FAILS_PER_RUN = 2  # failed banishes before the run stops trying
BANISH_COOLDOWN_S = 0.8


def normalize_stage(stage: str | None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (stage or "").strip().lower()).strip("_")
    return s


def banish_allowed(card: Card, stage: str) -> bool:
    """The ONLY cards Banish may ever be used on: a confidently read NEW Pentagram on Mad Forest."""
    return (
        normalize_stage(stage) in BANISH_STAGES
        and card.name in BANISH_ALLOWED
        and card.new
        and card.score >= BANISH_MIN_NAME_SCORE
    )


def banish_candidate(cards: list[Card], stage: str) -> int | None:
    """Index of the card to banish, or None. Needs at least one other card to pick after."""
    if len(cards) < 2:
        return None
    for i, c in enumerate(cards):
        if banish_allowed(c, stage):
            return i
    return None


@dataclass
class BanishButton:
    center: tuple[int, int]
    left: int | None  # remaining uses read from "N left"; None when unreadable


def _red_button(frame: np.ndarray, rect_960: tuple[int, int, int, int], label: str):
    """(x0, y0, x1, y1, ym) of a red button whose title OCRs as ``label``, else None."""
    h, w = frame.shape[:2]
    sx, sy = w / 960.0, h / 540.0
    bx, by, bw, bh = rect_960
    x0, y0 = int(round(bx * sx)), int(round(by * sy))
    x1, y1 = int(round((bx + bw) * sx)), int(round((by + bh) * sy))
    box = frame[y0:y1, x0:x1].astype(np.int16)
    if box.size == 0:
        return None
    red = (box[:, :, 2] > 180) & (box[:, :, 1] < 90) & (box[:, :, 0] < 70)
    if float(red.mean()) < BANISH_RED_FRAC:
        return None
    ym = y0 + int(round(17 * sy))
    title = ocr_image(_white_text(frame[y0:ym, x0:x1], thr=190))
    word = re.sub(r"[^a-z]", "", title.lower())
    if difflib.SequenceMatcher(None, word, label).ratio() < 0.8:
        return None
    return x0, y0, x1, y1, ym


def read_cancel_button(frame: np.ndarray) -> tuple[int, int] | None:
    """Center of the CANCEL button, i.e. BANISH is armed; None otherwise."""
    b = _red_button(frame, CANCEL_BUTTON_960, "cancel")
    return None if b is None else ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def read_banish_button(frame: np.ndarray) -> BanishButton | None:
    """The red BANISH button on a level-up screen, with its remaining-uses count.

    None unless the rect is mostly button red and OCR reads BANISH on it."""
    b = _red_button(frame, BANISH_BUTTON_960, "banish")
    if b is None:
        return None
    x0, y0, x1, y1, ym = b
    # The pale-pink "10 left" line only binarizes cleanly at 150 / 3x (45 of 45 frames).
    sub = ocr_image(_white_text(frame[ym:y1, x0:x1], thr=150, scale=3))
    m = re.search(r"(\d+)\s*[l1I|]", sub)
    left = int(m.group(1)) if m else None
    return BanishButton(center=((x0 + x1) // 2, (y0 + y1) // 2), left=left)


# After REVIVE the bar refills while the death screen fades; readings in this window
# (and the pre-death history) are stale. Oct 1 runs 3 / 12 logged hp=0.19 after a
# revive while the bar read 0.50, which kept the HP-aware picks and flee in "low HP".
REVIVE_HP_HOLDOFF_S = 1.0


# Consecutive missed bar reads that drop a stale value. One occlusion must not
# swing a level-up; a swarm sitting on the bar (run1: none for ~4.6s, med stuck
# at 0.92) must not keep that high reading.
HP_BLIND_MISSES = 3
# A red sprite or damage flash in the fallback search window reads as ~1.0 while
# the anchored bar is hidden (run6 quit: 0.00 → 0.80 → 0.53 → 1.00 → 0.06).
# An upward jump this large is ignored until several frames agree.
HP_JUMP_UP = 0.30
HP_JUMP_CONFIRM = 3
HP_JUMP_AGREE = 0.15


class HpTracker:
    """Median of the last few HP-bar readings: one frame with a sprite over the bar
    (or a red hit flash) must not swing the level-up choice."""

    def __init__(self, window: int = 5, miss_limit: int = HP_BLIND_MISSES) -> None:
        self.window = window
        self.miss_limit = miss_limit
        self.values: list[float] = []
        self.misses = 0
        self.blind = False
        self._anchor: float | None = None
        self._pending: list[float] = []

    def add(self, v: float | None) -> None:
        if v is None:
            self._pending.clear()
            self.misses += 1
            if self.values and self.misses >= self.miss_limit:
                self._anchor = self.value
                self.values.clear()
                self.blind = True
            return
        v = float(v)
        base = self.value if self.values else self._anchor
        if base is not None and v > base + HP_JUMP_UP:
            if self._pending and abs(v - self._pending[-1]) > HP_JUMP_AGREE:
                self._pending.clear()
            self._pending.append(v)
            if (
                len(self._pending) >= HP_JUMP_CONFIRM
                and max(self._pending) - min(self._pending) <= HP_JUMP_AGREE
            ):
                self._accept_window(self._pending[-HP_JUMP_CONFIRM :])
            return
        self._accept(v)

    def _accept(self, v: float) -> None:
        self.misses = 0
        self.blind = False
        self._pending.clear()
        self.values.append(float(v))
        del self.values[: -self.window]
        self._anchor = self.value

    def _accept_window(self, samples: list[float]) -> None:
        self.misses = 0
        self.blind = False
        self._pending.clear()
        self.values = [float(s) for s in samples[-self.window :]]
        self._anchor = self.value

    def reset(self) -> None:
        self.values.clear()
        self.misses = 0
        self.blind = False
        self._anchor = None
        self._pending.clear()

    @property
    def value(self) -> float | None:
        if not self.values:
            return None
        return float(np.median(self.values))


def hp_for_nav(hp: float | None, blind: bool) -> float | None:
    """HP nav should act on. A bar lost under a swarm counts as empty."""
    if blind:
        return 0.0
    return hp


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
        game_clock: Callable[[], float | None] | None = None,
        stage: str = "",
    ) -> None:
        # Stage being played (session sets VS_STAGE; play_run.sh only starts Mad Forest).
        # Banish is only ever used on BANISH_STAGES.
        self.stage = normalize_stage(stage)
        self.banished: set[str] = set()
        self._banish: dict | None = None  # in-progress banish of the current level-up
        self._banish_fails = 0
        self._banish_skip_screen = False  # gave up on this level-up screen: pick normally
        # Current game time (s) for time-dependent picks (early Pentagram avoid); None = unknown.
        self.game_clock = game_clock
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
        self._hp_ignore_until = 0.0
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
        """Smoothed HP fraction (median of recent bar readings), None until seen or when the bar is lost."""
        return self._hp.value

    @hp.setter
    def hp(self, v: float | None) -> None:
        self._hp.reset()
        self._hp.add(v)

    @property
    def hp_blind(self) -> bool:
        """True after the bar stays unreadable: the last high reading has been dropped."""
        return self._hp.blind

    def new_run(self) -> None:
        """A fresh run: forget owned items and HP history."""
        self.reset_run()
        self.revives = 0
        self._hp_ignore_until = 0.0
        self.owned = set(self.start_items)
        self._hp.reset()
        self.picks = []
        self.banished = set()
        self._banish = None
        self._banish_fails = 0
        self._banish_skip_screen = False

    # -- helpers
    def _game_s(self) -> float | None:
        if self.game_clock is None:
            return None
        try:
            v = self.game_clock()
        except Exception:
            return None
        return None if v is None else float(v)

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
        self._banish_skip_screen = False
        if self._banish is not None:
            phase = self._banish["phase"]
            self.log(
                "[levelup] BANISH Pentagram: level-up closed after "
                + ("the BANISH click" if phase == "armed" else "the Pentagram click")
                + " (game closes the menu on a banish in older versions; if BANISH was not armed it was a pick)"
            )
            self._banish = None
        if now >= self._hp_ignore_until:
            self._hp.add(read_hp_fraction(frame))
        return self._handle_chest(frame, now)

    def _handle_level_up(self, frame: np.ndarray, screen: LevelUpScreen, now: float) -> None:
        sig = self._signature(frame, screen)
        if self._banish is not None:
            self._banish_step(frame, screen, now, sig)
            return
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
            with _menu_ocr_budget():
                read_cards(frame, screen)
                self._names = [c.name or c.raw or "?" for c in screen.cards]
                self.owned |= {c.name for c in screen.cards if c.name and not c.new}
                if self._try_start_banish(frame, screen):
                    return
                game_s = self._game_s()
                self._target = choose_card(screen.cards, hp=self.hp, owned=self.owned, game_s=game_s)
                self._chosen = screen.cards[self._target].name
                self._presses = 0
                self._unknown_sel = 0
                self.log(
                    f"[levelup] cards={self._names} new={[c.new for c in screen.cards]} "
                    f"-> pick #{self._target + 1} {self._names[self._target]} "
                    f"(hp={self.hp if self.hp is None else round(self.hp, 2)}, weapons={owned_weapon_count(self.owned)}, "
                    f"game~={'?' if game_s is None else f'{game_s:.0f}s'})"
                )
                if any(not c.name for c in screen.cards):
                    # Unmatched OCR ('P', 'wr', 'or'): keep the frame to learn what the card is.
                    path = self._save_frame(frame, "levelup_unread")
                    if path:
                        self.log(f"[levelup] unread card, saved {path}")
        if now < self._next_action:
            self.status = f"LEVEL UP -> {self._names[self._target]}"
            return
        target = min(self._target or 0, len(screen.cards) - 1)
        if self.click is not None and self.levelup_click:
            # Click the card itself: one click picks it. Keyboard navigation could
            # wander off the card column onto REROLL / BANISH (live run8), so the
            # mouse is the default; the click point is the card's own center.
            # (The only deliberate BANISH click is _try_start_banish, above.)
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
                # than confirm something that might be REROLL / BANISH (keyboard mode
                # never banishes, not even Pentagram).
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

    # -- banish (Mad Forest, NEW Pentagram only)
    def _try_start_banish(self, frame: np.ndarray, screen: LevelUpScreen) -> bool:
        """Click BANISH when this settled level-up offers a NEW Pentagram on Mad Forest.

        False (normal pick) when it does not apply or cannot be done safely."""
        if self._banish_skip_screen:
            return False
        idx = banish_candidate(screen.cards, self.stage)
        if idx is None:
            return False
        names = [c.name or c.raw or "?" for c in screen.cards]
        if self.click is None or not self.levelup_click:
            self.log("[levelup] BANISH Pentagram skipped: keyboard mode never banishes; normal pick")
            return False
        if self._banish_fails >= BANISH_MAX_FAILS_PER_RUN:
            self.log(f"[levelup] BANISH Pentagram skipped: {self._banish_fails} failed tries this run; normal pick")
            return False
        btn = read_banish_button(frame)
        if btn is None:
            self.log("[levelup] BANISH Pentagram skipped: no BANISH button seen; normal pick")
            return False
        if btn.left is None or btn.left < 1:
            self.log(f"[levelup] BANISH Pentagram skipped: uses left={btn.left}; normal pick")
            return False
        card = screen.cards[idx]
        if not banish_allowed(card, self.stage):  # hard guard (never reached otherwise)
            return False
        self._save_banish_frame(frame, "offer")
        self.refocus()
        self.click(*btn.center)
        self._next_action = self.clock() + BANISH_COOLDOWN_S
        self._banish = {"phase": "armed", "name": card.name, "left_before": btn.left, "card_clicks": 0}
        self._sig = None
        self._stable = 0
        self.status = "LEVEL UP -> BANISH Pentagram"
        self.log(
            f"[levelup] BANISH Pentagram: cards={names} -> clicked BANISH at {btn.center} "
            f"({btn.left} left), next: click card #{idx + 1}"
        )
        return True

    def _banish_step(self, frame: np.ndarray, screen: LevelUpScreen, now: float, sig: bytes) -> None:
        """After the BANISH click: click the Pentagram card (and nothing else), then verify it is gone.

        While a banish may be armed the ONLY card ever clicked is the Pentagram one, so a
        mis-detected state can never banish another card."""
        state = self._banish
        assert state is not None
        if now < self._next_action:
            return
        if sig != self._sig:
            self._sig = sig
            self._stable = 0
            return
        self._stable += 1
        if self._stable < self.settle_frames:
            return
        with _menu_ocr_budget():
            read_cards(frame, screen)
            names = [c.name or c.raw or "?" for c in screen.cards]
            idx = next((i for i, c in enumerate(screen.cards) if c.name in BANISH_ALLOWED), None)
            # Keep what the screen looks like with BANISH armed / after the banish (no
            # frames of either exist yet), for checking the flow after the first live run.
            self._save_banish_frame(frame, "armed" if state["phase"] == "armed" else f"after{state['card_clicks']}")
            cancel = read_cancel_button(frame)
            if state["phase"] == "cancel":
                if cancel is not None or read_banish_button(frame) is None:
                    self.log(f"[levelup] BANISH Pentagram: still armed after CANCEL, cards={names}; stopping the run")
                    self._banish = None
                    self.finished = True
                    return
                self.log("[levelup] BANISH Pentagram: backed out with CANCEL; normal pick")
                self._end_banish(failed=True)
                return
            if state["phase"] == "armed" or idx is not None or cancel is not None:
                if cancel is None:
                    self.log(f"[levelup] BANISH Pentagram: banish mode not seen (no CANCEL), cards={names}; normal pick")
                    self._end_banish(failed=True)
                    return
                if idx is None or state["card_clicks"] >= BANISH_MAX_CARD_CLICKS:
                    self.log(f"[levelup] BANISH Pentagram failed after {state['card_clicks']} clicks, cards={names}; CANCEL")
                    self.refocus()
                    self.click(*cancel)
                    state["phase"] = "cancel"
                    self._next_action = self.clock() + BANISH_COOLDOWN_S
                    self._sig = None
                    return
                card = screen.cards[idx]
                self.refocus()
                self.click((card.x0 + card.x1) // 2, card.cy)
                state["phase"] = "clicked"
                state["card_clicks"] += 1
                self._next_action = self.clock() + BANISH_COOLDOWN_S
                self._sig = None
                self.log(f"[levelup] BANISH Pentagram: clicked the Pentagram card #{idx + 1} (click {state['card_clicks']})")
                return
            # Clicked, the level-up is still open and Pentagram is gone: banished (the
            # Jul 2026 game keeps the menu open after a banish). Pick from the rest.
            btn = read_banish_button(frame)
            left = None if btn is None else btn.left
            self.banished.add(state["name"])
            self.log(
                f"[levelup] BANISH Pentagram verified: card gone, cards={names}, "
                f"BANISH {state['left_before']} -> {left if left is not None else '?'} left; picking from the rest"
            )
            if left is not None and left != state["left_before"] - 1:
                self.log("[levelup] BANISH Pentagram note: uses count did not drop by exactly 1")
            self._end_banish(failed=False)

    def _save_banish_frame(self, frame: np.ndarray, tag: str) -> None:
        path = self._save_frame(frame, f"levelup_banish_{tag}")
        if path:
            self.log(f"[levelup] BANISH Pentagram: saved {path}")

    def _end_banish(self, failed: bool) -> None:
        if failed:
            self._banish_fails += 1
            self._banish_skip_screen = True
        self._banish = None
        self._target = None
        self._sig = None
        self._stable = 0

    def _record_pick(self, target: int) -> None:
        name = self._names[target] if target < len(self._names) else "?"
        self.picks.append(name)
        if self._chosen:
            self.owned.add(self._chosen)
        self.log(f"[levelup] picked #{target + 1} {name} (total picks {len(self.picks)})")
        self._target = None
        self._sig = None

    def _save_frame(self, frame: np.ndarray, prefix: str) -> str | None:
        if not self.results_dir:
            return None
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            path = os.path.join(self.results_dir, time.strftime(f"{prefix}_%Y%m%d_%H%M%S.png"))
            if cv2.imwrite(path, frame):
                return path
        except Exception as exc:  # noqa: BLE001
            self.log(f"[{prefix}] could not save screenshot: {exc}")
        return None

    def _save_results(self, frame: np.ndarray) -> str | None:
        """Keep a screenshot of the Results screen (time / level / kills) before DONE."""
        path = self._save_frame(frame, "results")
        if path:
            self.saved_results.append(path)
            self.log(f"[results] saved {path}")
        return path

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
                    # Pre-death readings are stale; start HP over from the revived bar.
                    self._hp.reset()
                    self._hp_ignore_until = self.clock() + REVIVE_HP_HOLDOFF_S
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
