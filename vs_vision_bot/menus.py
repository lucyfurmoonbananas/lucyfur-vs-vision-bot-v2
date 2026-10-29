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
    "Santa Water",
    "La Borra",
    "Attractorb",
    "King Bible",
    "Unholy Vespers",
    "Spellbinder",
    "Candelabrador",
    "Empty Tome",
    "Spinach",
    "Duplicator",
    "Magic Wand",
    "Holy Wand",
    "Tirajisú",
    "Tiragisú",
    "Runetracer",
    "NO FUTURE",
    "Vandalier",
    "Peachone",
    "Ebony Wings",
    "Lightning Ring",
    "Thunder Loop",
    "Cross",
    "Heaven Sword",
    "Armor",
    "Clover",
    "Pentagram",
    "Gorgeous Moon",
    "Crown",
    "Stone Mask",
    "Hollow Heart",
]
# Skipped when anything else is on offer (skill: "Skip Whip, Axe, Knife ...").
AVOID: list[str] = ["Whip", "Axe", "Knife", "Silver Wind", "Phiera Der Tuphello"]
# Other known names so OCR noise snaps to a real item instead of a priority one.
OTHER: list[str] = [
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
    "Big Coin Bag",
    "Gold Coin Bag",
    "Gold Coins",
    "Gold Coin",
    "Coin Bag",
    "Bloody Tear",
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
    level: int | None = None  # parsed "level N" for an owned card; None if New! or unread

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


# A real bar is this many rows of the same columns. A damage number is a
# shifting slash (test3 quit: 2 rows read as 1.0) or a 2px speck (0.06).
_FALLBACK_MIN_ROWS = 4
_FALLBACK_MIN_RUN = 4
_FALLBACK_ALIGN = 2


def _fallback_hp(frame: np.ndarray) -> float | None:
    """HP from a bar-shaped red run when the fixed rect is occluded.

    The search window also contains damage numbers and red sprites. Those are
    not a bar unless the same run repeats for several rows. A tall solid sprite
    still matches; one such frame stays a spike for the jump filter.
    """
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.42), int(w * 0.58)
    y0, y1 = int(h * 0.47), int(h * 0.56)
    roi = frame[y0:y1, x0:x1].astype(np.int16)
    if roi.size == 0:
        return None
    red = (roi[:, :, 2] > 200) & (roi[:, :, 1] < 50) & (roi[:, :, 0] < 50)
    dark = roi.max(axis=2) < 45
    nominal = HP_BAR_W_960 * w / 960.0
    runs: list[tuple[int, int] | None] = []
    for r in range(roi.shape[0]):
        xs = np.flatnonzero(red[r])
        if xs.size < _FALLBACK_MIN_RUN:
            runs.append(None)
            continue
        parts = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = max(parts, key=len)
        if len(run) < _FALLBACK_MIN_RUN or len(run) > nominal * 1.3:
            runs.append(None)
            continue
        runs.append((int(run[0]), int(run[-1])))
    best: tuple[int, float] | None = None
    i = 0
    n = len(runs)
    while i < n:
        if runs[i] is None:
            i += 1
            continue
        j = i
        a0, b0 = runs[i]
        while j + 1 < n and runs[j + 1] is not None:
            a1, b1 = runs[j + 1]
            if abs(a1 - a0) > _FALLBACK_ALIGN or abs(b1 - b0) > _FALLBACK_ALIGN:
                break
            j += 1
        if j - i + 1 >= _FALLBACK_MIN_ROWS:
            seg = [runs[k] for k in range(i, j + 1) if runs[k] is not None]
            start = int(np.median([s[0] for s in seg]))
            end = int(np.median([s[1] for s in seg])) + 1
            length = end - start
            mid = (i + j) // 2
            missing = 0
            while end + missing < roi.shape[1] and dark[mid, end + missing]:
                missing += 1
            total = length + missing
            if total < nominal * 0.7:
                total = nominal
            frac = min(1.0, length / total) if total else 0.0
            if best is None or length > best[0]:
                best = (length, frac)
        i = j + 1
    return None if best is None else best[1]


def read_hp_fraction(frame: np.ndarray) -> float | None:
    """Player HP from the red bar just under the centered character (None if not seen)."""
    anchored = read_hp_bar(frame)
    if anchored is not None:
        return anchored
    return _fallback_hp(frame)


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


# Weapons stop at 8. Passives stop at 5. The level tag on a card is the level
# the item is at now (OCR sometimes reads the next level, one higher).
WEAPON_MAX_LEVEL = 8
PASSIVE_MAX_LEVEL = 5
# A chest grants +1, and the tag is occasionally one high. Adopt an upward gap
# inside this bound so the tracker catches the chest. Wider than this (the live
# misread of 2 vs 6) is not an inventory.
LEVEL_RECONCILE = 2
_LEVEL_RE = re.compile(r"l[e3]v[ae]l\s*[:;.]?\s*(\d+)", re.IGNORECASE)


def parse_level_text(raw: str) -> int | None:
    """Pull a weapon/passive level out of an OCR tag. None when it is not a level."""
    m = _LEVEL_RE.search(raw or "")
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= WEAPON_MAX_LEVEL:
        return n
    return None


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
    # Owned cards print "level:N" in white on that same title row. New! is yellow
    # and has no level. A failed read leaves level None; the tracker still stands.
    card.level = None
    if not card.new:
        lvl = frame[ny0:ny1, card.x0 + int(0.55 * cw) : card.x1]
        if lvl.size:
            card.level = parse_level_text(ocr_image(_white_text(lvl)))
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
# Level-up filler once every owned weapon and passive is capped. The card the
# game shows is Big Coin Bag (wiki COINBAG2). Gold Coin / Gold Coin Bag are the
# names Lucyfur uses and the strings OCR may return; the ground pickups stay
# too so a misread still counts as gold.
COINS = ("Big Coin Bag", "Gold Coin Bag", "Coin Bag", "Gold Coins", "Gold Coin")
FILLER: frozenset[str] = frozenset((CHICKEN, *COINS))
LOW_HP = 0.6        # only-filler screen before the build is capped: chicken below this
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
     "Skull O'Maniac", "Bracer", "Wings", "Spinach"}
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


# Base evolution pairs (Mad Forest). The passive can be any level. Peachone and
# Ebony Wings are the exception: both weapons have to reach 8, no passive.
EVOLUTION_PAIRS: dict[str, tuple[str, str]] = {
    "Garlic": ("Pummarola", "Soul Eater"),
    "Magic Wand": ("Empty Tome", "Holy Wand"),
    "King Bible": ("Spellbinder", "Unholy Vespers"),
    "Santa Water": ("Attractorb", "La Borra"),
    "Runetracer": ("Armor", "NO FUTURE"),
    "Lightning Ring": ("Duplicator", "Thunder Loop"),
    "Fire Wand": ("Spinach", "Hellfire"),
    "Axe": ("Candelabrador", "Death Spiral"),
    "Cross": ("Clover", "Heaven Sword"),
    "Knife": ("Bracer", "Thousand Edge"),
    "Whip": ("Hollow Heart", "Bloody Tear"),
    "Pentagram": ("Crown", "Gorgeous Moon"),
    "Peachone": ("Ebony Wings", "Vandalier"),
    "Ebony Wings": ("Peachone", "Vandalier"),
}
BIRD_WEAPONS: frozenset[str] = frozenset({"Peachone", "Ebony Wings"})


def _evolution_bases() -> dict[str, frozenset[str]]:
    """Evolution name -> base weapon(s). Vandalier is both birds."""
    out: dict[str, set[str]] = {}
    for base, (_partner, evo) in EVOLUTION_PAIRS.items():
        out.setdefault(evo, set()).add(base)
    return {evo: frozenset(bases) for evo, bases in out.items()}


EVOLUTION_BASES: dict[str, frozenset[str]] = _evolution_bases()
# Next weapon to max once the main is done. Avoid-list weapons stay last.
SECOND_ORDER: tuple[str, ...] = (
    "Garlic",
    "Santa Water",
    "King Bible",
    "Magic Wand",
    "Runetracer",
    "Lightning Ring",
    "Fire Wand",
    "Cross",
    "Peachone",
    "Ebony Wings",
    "Axe",
    "Knife",
    "Whip",
    "Pentagram",
)
# After the evolution partner is owned: healing / armor / max-HP, then the extra revival.
SURVIVAL: tuple[str, ...] = ("Pummarola", "Armor", "Hollow Heart", "Tirajisú", "Tiragisú")
PICK_REASONS: tuple[str, ...] = ("main", "partner", "second", "fallback")


def partner_of(weapon: str | None) -> str | None:
    if not weapon:
        return None
    pair = EVOLUTION_PAIRS.get(weapon)
    return None if pair is None else pair[0]


def main_weapon_name(levels: dict[str, int] | None) -> str | None:
    """Garlic if it is held, otherwise Magic Wand. None when neither is held."""
    if not levels:
        return None
    if levels.get("Garlic", 0) > 0:
        return "Garlic"
    if levels.get("Magic Wand", 0) > 0:
        return "Magic Wand"
    return None


def _partner_ready(weapon: str, levels: dict[str, int]) -> bool:
    """Passive owned at any level, or the other bird already at 8."""
    partner = partner_of(weapon)
    if partner is None:
        return False
    if weapon in BIRD_WEAPONS:
        return levels.get(partner, 0) >= WEAPON_MAX_LEVEL
    return levels.get(partner, 0) > 0


def ready_evolution_labels(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> list[str]:
    """Weapons a chest could evolve right now, main first.

    A passive pair needs the weapon at 8 and the partner owned. Peachone and
    Ebony Wings need both at 8. Anything already evolved is skipped. The two
    birds produce one label.
    """
    if not levels:
        return []
    done = evolved or set()
    main = main_weapon_name(levels)
    names = [n for n, lv in levels.items() if lv > 0 and n in EVOLUTION_PAIRS]

    def sort_key(name: str) -> tuple[int, int, str]:
        if name == main:
            return (0, 0, name)
        if name in SECOND_ORDER:
            return (1, SECOND_ORDER.index(name), name)
        return (2, 99, name)

    labels: list[str] = []
    birds_logged = False
    for name in sorted(names, key=sort_key):
        if name in done or levels.get(name, 0) < WEAPON_MAX_LEVEL:
            continue
        if name in BIRD_WEAPONS:
            if birds_logged:
                continue
            other = partner_of(name)
            if (
                other
                and other not in done
                and levels.get(other, 0) >= WEAPON_MAX_LEVEL
            ):
                birds_logged = True
                labels.append(f"{name} L{levels[name]} + {other} L{levels[other]}")
            continue
        partner = partner_of(name)
        if partner and levels.get(partner, 0) > 0:
            labels.append(f"{name} L{levels[name]} + {partner}")
    return labels


def evolution_ready(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> bool:
    """Some owned weapon can evolve, and has not evolved yet.

    That is a weapon at 8 whose passive is owned, or both birds at 8. Boss
    chests (and the 1:00 Glowing Bat chest) are what actually evolve it.
    Pathing is unchanged; the navigator reads this flag when it has one.
    """
    return bool(ready_evolution_labels(levels, evolved))


def owned_weapon_slots(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> list[str]:
    """Owned weapon slots. Evolution results are not a second slot.

    Peachone and Ebony Wings become one Vandalier slot once both are evolved.
    Until that chest, they are still two weapons.
    """
    if not levels:
        return []
    done = evolved or set()
    names = [n for n, lv in levels.items() if int(lv) > 0 and n in WEAPONS and n not in EVOLUTIONS]
    if "Peachone" in names and "Ebony Wings" in names and "Peachone" in done and "Ebony Wings" in done:
        names = [n for n in names if n != "Ebony Wings"]
    return names


def _plan_settled(
    name: str, levels: dict[str, int], evolved: set[str] | frozenset[str]
) -> bool:
    """Evolved, or at 8 with its partner. A maxed weapon still missing the partner is open."""
    if name in evolved:
        return True
    if levels.get(name, 0) < WEAPON_MAX_LEVEL:
        return False
    if name not in EVOLUTION_PAIRS:
        return True
    return _partner_ready(name, levels)


def wants_new_weapon(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> bool:
    """True while the plan still wants another weapon slot filled.

    The plan is one main and one second weapon. A free slot does not by itself
    mean a new weapon is wanted: finish the weapon already in hand first, and
    stop once the main and every other held weapon are settled. Six full slots
    never want another weapon.
    """
    done = evolved or set()
    slots = owned_weapon_slots(levels, done)
    if len(slots) >= MAX_WEAPONS:
        return False
    if not levels:
        return True
    main = main_weapon_name(levels)
    if main is None or not _plan_settled(main, levels, done):
        # No main yet: we do want Garlic (or the wand). Main unfinished: do not
        # open another weapon until this one is settled.
        return main is None
    others = [n for n in slots if n != main]
    if not others:
        return True
    # A second weapon is already in hand. Finish it, or stop once it is settled.
    # Either way another new weapon is not the plan.
    return False


def all_maxed(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> bool:
    """Every owned weapon is maxed or evolved, and we are not still filling slots.

    A weapon at 8 whose partner is missing is not finished while a slot is
    free. Calling that all-maxed would open every chest before the evolution
    exists. Once all six slots are full, a maxed loadout counts even if a
    partner never showed up.
    """
    if not levels:
        return False
    done = evolved or set()
    slots = owned_weapon_slots(levels, done)
    if not slots:
        return False
    for name in slots:
        if name in done:
            continue
        if levels.get(name, 0) < WEAPON_MAX_LEVEL:
            return False
    if len(slots) >= MAX_WEAPONS:
        return True
    for name in slots:
        if name in done or name not in EVOLUTION_PAIRS:
            continue
        if not _partner_ready(name, levels):
            return False
    return not wants_new_weapon(levels, done)


def items_fully_maxed(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> bool:
    """Every owned weapon and passive is at its cap, and no new weapon is wanted.

    That is when the level-up screen offers Floor Chicken and a coin bag. A
    weapon still under 8, a passive under 5, or an open second weapon is not
    this, so the gold/chicken rule stays off.
    """
    if not levels:
        return False
    done = evolved or set()
    weapons: list[str] = []
    passives: list[str] = []
    for name, lv in levels.items():
        if int(lv) <= 0 or name in EVOLUTIONS or name in FILLER:
            continue
        if name in WEAPONS:
            weapons.append(name)
        else:
            passives.append(name)
    if not weapons or not passives:
        return False
    for name in weapons:
        if name in done:
            continue
        if levels.get(name, 0) < WEAPON_MAX_LEVEL:
            return False
    for name in passives:
        if levels.get(name, 0) < PASSIVE_MAX_LEVEL:
            return False
    return not wants_new_weapon(levels, done)


def low_hp_limit() -> float:
    """``Tuning.low_hp`` (0.5). Cached so a pick does not rebuild the navigator."""
    cached = getattr(low_hp_limit, "_value", None)
    if cached is None:
        from vs_vision_bot.pathfind import Tuning

        cached = float(Tuning().low_hp)
        setattr(low_hp_limit, "_value", cached)
    return float(cached)


def read_critical_flag(nav: object) -> bool:
    """Sticky critical on the navigator, if that build has the flag.

    PR #26 holds critical HP across a bad bar read. The attribute is not on
    every branch, so a missing flag is just false. ``_critical_on`` is the
    flag this tree already keeps.
    """
    for name in ("critical_held", "hp_critical", "sticky_critical", "_critical_on"):
        flag = getattr(nav, name, None)
        if flag is True:
            return True
    return False


def hp_is_low(
    hp: float | None,
    *,
    blind_low: bool = False,
    critical: bool = False,
    low_held: bool = False,
) -> bool:
    """Low for the filler pick.

    A number under ``Tuning.low_hp`` is low. So is the navigator's critical
    flag, and the tracker's held critical. Blind with no prior low is not low;
    blind right after a low reading is.
    """
    if critical or low_held:
        return True
    if hp is None:
        return bool(blind_low)
    return hp < low_hp_limit()


def format_inventory(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None
) -> str:
    if not levels:
        return "-"
    done = evolved or set()
    main = main_weapon_name(levels)
    names = sorted((n for n, lv in levels.items() if lv > 0), key=lambda n: (n != main, n))
    return ",".join(f"{n}:{levels[n]}{'*' if n in done else ''}" for n in names)


def _vocab_hits(text: str, min_score: float = NAME_RETRY_BELOW) -> dict[str, float]:
    """Best vocabulary match for each 1-3 word window, at or above ``min_score``."""
    words = re.findall(r"[A-Za-z']+", text or "")
    best: dict[str, float] = {}
    for width in (1, 2, 3):
        if len(words) < width:
            continue
        for i in range(len(words) - width + 1):
            name, score = match_name_score(" ".join(words[i : i + width]))
            if name and score >= min_score and score > best.get(name, 0.0):
                best[name] = score
    return best


def evolutions_from_chest_text(text: str) -> set[str]:
    """Base weapons this chest evolved, when the evolution name is readable.

    Matches the evolution's letters in a row (6 or more) or a vocabulary hit
    at the card-name threshold. Vandalier returns both birds. An empty or
    speed-up chest returns nothing.
    """
    if not text or not text.strip():
        return set()
    found: set[str] = set()
    letters = re.sub(r"[^a-z]", "", text.lower())
    for evo, bases in EVOLUTION_BASES.items():
        key = re.sub(r"[^a-z]", "", evo.lower())
        if len(key) >= 6 and key in letters:
            found.update(bases)
    for name in _vocab_hits(text):
        found.update(EVOLUTION_BASES.get(name, ()))
    return found


# Armor's chest line is "Reduces incoming damage by N". OCR often reads "Denage".
# Cooldown / base-damage text is shared by every weapon, so it names nothing.
_ARMOR_CHEST_RE = re.compile(r"incoming\s+d[a-z]*ge", re.IGNORECASE)


def chest_text_is_reward(text: str) -> bool:
    """True for the DONE page after Treasure Found, false for the OPEN prompt.

    One chest shows both. The prompt is the open that spends a free chest.
    The reward page is the same chest.
    """
    low = (text or "").lower()
    if "treasure" in low or re.search(r"\bopen\b", low):
        return False
    return "done" in low or parse_level_text(text or "") is not None


def levels_from_chest_text(text: str) -> dict[str, int]:
    """One non-evolution item and one level tag. Empty when the screen is ambiguous."""
    level = parse_level_text(text or "")
    if level is None:
        return {}
    if _ARMOR_CHEST_RE.search(text or ""):
        # Named before the vocab pass: "damage" must not become a second item.
        return {"Armor": level}
    hits = {name: score for name, score in _vocab_hits(text).items() if name not in EVOLUTIONS}
    names = list(hits)
    kept: list[str] = []
    for name in names:
        words = set(name.lower().split())
        dominated = False
        for other in names:
            if other == name:
                continue
            other_words = set(other.lower().split())
            if words < other_words and hits[other] >= hits[name]:
                dominated = True
                break
        if not dominated:
            kept.append(name)
    if len(kept) != 1:
        return {}
    return {kept[0]: level}


def level_cap(name: str) -> int:
    return WEAPON_MAX_LEVEL if name in WEAPONS else PASSIVE_MAX_LEVEL


def assess_inventory(
    cards: list[Card], levels: dict[str, int] | None
) -> tuple[dict[str, int], bool]:
    """Copy ``levels`` and fold in confident card reads.

    Certain when every confident card agrees with the tracker. A missing level
    tag on a tracked item is fine. A tag one higher is the level this pick will
    become, which is how owned cards read, so the tracker stays put and the
    pick adds one. A tag two higher replaces the tracker (a missed chest). A
    tag one lower stays on the plan without lowering it. A shaky name, a New!
    on something we own, an owned item we have never seen a level for, or a
    wider gap is not an inventory: the caller uses the static priority only
    when no card still agrees.
    """
    if levels is None:
        return {}, False
    eff = {k: int(v) for k, v in levels.items() if int(v) > 0}
    for c in cards:
        if not c.name:
            continue
        if c.score and c.score < NAME_RETRY_BELOW:
            return eff, False
        if c.name in FILLER:
            continue
        held = eff.get(c.name, 0)
        if c.new:
            if held > 0:
                return eff, False
            continue
        if c.level is None:
            if held <= 0:
                return eff, False
            continue
        if held <= 0:
            eff[c.name] = c.level
            continue
        if c.level > held + LEVEL_RECONCILE or held - c.level > 1:
            return eff, False
        # Gap of 1 is the next-level tag. Copying it and then incrementing the
        # pick records two levels for one offer (run 3: results Garlic 2, log
        # Garlic:3, no chest). Gap of 2 is a real missed level.
        if c.level > held + 1:
            eff[c.name] = c.level
    return eff, True


def merge_card_levels(levels: dict[str, int], eff: dict[str, int]) -> None:
    """Write a certain read into the tracker. Higher card levels replace stale ones."""
    for name, lv in eff.items():
        if name in FILLER:
            continue
        held = levels.get(name, 0)
        if held <= 0 or lv > held:
            levels[name] = lv


def _card_tracks(card: Card, levels: dict[str, int]) -> bool:
    """This card agrees with the tracker closely enough to pick from it."""
    if not card.name or card.name in FILLER:
        return False
    if card.score and card.score < NAME_RETRY_BELOW:
        return False
    held = levels.get(card.name, 0)
    if card.new:
        return held <= 0
    if card.level is None:
        return held > 0
    if held <= 0:
        return True
    if card.level > held + LEVEL_RECONCILE or held - card.level > 1:
        return False
    return True


def _missing_partner_name(name: str, levels: dict[str, int]) -> bool:
    """``name`` is the evolution partner of a weapon we already own, and we lack it."""
    if levels.get(name, 0) > 0:
        return False
    return any(lv > 0 and partner_of(weapon) == name for weapon, lv in levels.items())


def _second_weapon(levels: dict[str, int], main: str) -> str | None:
    """Unfinished weapon we already own, after ``main``. Highest level, then SECOND_ORDER."""
    best: str | None = None
    best_key: tuple[int, int, int] | None = None
    for name, lv in levels.items():
        if name == main or lv <= 0 or name not in EVOLUTION_PAIRS or name in EVOLUTIONS:
            continue
        other = partner_of(name)
        unfinished = lv < WEAPON_MAX_LEVEL
        partner_open = bool(other) and levels.get(other, 0) <= 0
        bird_open = name in BIRD_WEAPONS and bool(other) and levels.get(other, 0) < WEAPON_MAX_LEVEL
        if not unfinished and not partner_open and not bird_open:
            continue
        has_partner = 0 if other and levels.get(other, 0) > 0 else 1
        order = SECOND_ORDER.index(name) if name in SECOND_ORDER else 99
        key = (-lv, has_partner, order)
        if best_key is None or key < best_key:
            best, best_key = name, key
    return best


def _level_targets(levels: dict[str, int], second: str | None) -> list[str]:
    if not second:
        return []
    out: list[str] = []
    if levels.get(second, 0) < WEAPON_MAX_LEVEL:
        out.append(second)
    if second in BIRD_WEAPONS:
        other = partner_of(second)
        if (
            other
            and levels.get(second, 0) >= WEAPON_MAX_LEVEL
            and 0 < levels.get(other, 0) < WEAPON_MAX_LEVEL
        ):
            out.append(other)
    return out


# The only new weapons the early fallback may spend a slot on.
CORE_SECOND_WEAPONS: frozenset[str] = frozenset({"Santa Water", "King Bible"})
NO_DAMAGE_WEAPONS: frozenset[str] = frozenset({"Laurel", "Clock Lancet"})


def second_dmg_until_s() -> float:
    """``Tuning.levelup_second_dmg_until_s`` with VS_TUNE applied (0 = off)."""
    raw = os.environ.get("VS_TUNE") or ""
    cache = getattr(second_dmg_until_s, "_cache", None)
    if cache is None or cache[0] != raw:
        from vs_vision_bot.pathfind import tuning_from_env

        tune = tuning_from_env(raw, log=lambda *_a, **_k: None)
        cache = (raw, float(tune.levelup_second_dmg_until_s))
        setattr(second_dmg_until_s, "_cache", cache)
    return float(cache[1])


def damage_weapon_count(
    levels: dict[str, int] | None, evolved: set[str] | frozenset[str] | None = None,
) -> int:
    """Held weapons that deal damage (Laurel / Clock Lancet excluded)."""
    names = {n for n, lv in (levels or {}).items() if lv > 0} | set(evolved or ())
    return sum(1 for n in names if n in WEAPONS and n not in NO_DAMAGE_WEAPONS)


def wants_second_damage_weapon(
    levels: dict[str, int] | None,
    game_s: float | None,
    evolved: set[str] | frozenset[str] | None = None,
    until_s: float | None = None,
) -> bool:
    """Before 6:00 with only Garlic dealing damage. Unknown time leaves the rule off."""
    limit = second_dmg_until_s() if until_s is None else until_s
    if limit <= 0 or game_s is None or game_s >= limit:
        return False
    return damage_weapon_count(levels, evolved) < 2


def _core_slot_open(levels: dict[str, int] | None) -> bool:
    """Santa Water or King Bible is not owned yet, so other new weapons wait.

    Unknown inventory (no level map) keeps the old static order.
    """
    if not levels:
        return False
    return levels.get("Santa Water", 0) <= 0 or levels.get("King Bible", 0) <= 0


def _santa_water_now(hp: float | None) -> bool:
    """Take Santa Water unless HP is at or under 25% and a heal can win instead."""
    return hp is None or hp > 0.25


def _armor_upgrade_yields_to_garlic(
    name: str | None, new: bool, hp: float | None, garlic_lv: int,
) -> bool:
    """Armor L2+ does not outrank an unfinished Garlic above 25% HP, or while Garlic < 4.

    test7 run 2 at hp=0.33 took Armor L2 over a Garlic level. A first point of
    Armor (a new card) is unchanged.
    """
    if name != "Armor" or new:
        return False
    if garlic_lv <= 0 or garlic_lv >= WEAPON_MAX_LEVEL:
        return False
    if hp is not None and hp > 0.25:
        return True
    return garlic_lv < 4


def _holds_two_damage_weapons(levels: dict[str, int] | None) -> bool:
    """Two damage weapons in hand and the main still under 8: do not open a third."""
    if not levels:
        return False
    main = main_weapon_name(levels)
    if main is None or levels.get(main, 0) <= 0 or levels.get(main, 0) >= WEAPON_MAX_LEVEL:
        return False
    quiet = {"Laurel", "Clock Lancet"}
    n = sum(1 for name, lv in levels.items() if lv > 0 and name in WEAPONS and name not in quiet)
    return n >= 2


def _extra_weapon_capped(name: str | None, new: bool, levels: dict[str, int] | None) -> bool:
    """No extra weapon while a core slot is open, and no third while Garlic is under 8.

    Applies to the avoid list as well. Laurel and Clock Lancet are weapons, so
    the third-weapon cap includes them. Santa Water and King Bible stay eligible
    while one of those two slots is still empty and Garlic is the only damage weapon.
    """
    if not new or not name or name not in WEAPONS:
        return False
    if name in CORE_SECOND_WEAPONS or name == "Garlic" or name == "Pentagram":
        return False
    return _core_slot_open(levels) or _holds_two_damage_weapons(levels)


def _card_rank(
    c: Card,
    idx: int,
    hp: float | None,
    weapons: int,
    early: bool = False,
    levels: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Static priority. This is what spreads levels: a new crowd weapon is tier 2
    and any other new weapon is tier 4, while a Garlic level is tier 3 and every
    other owned weapon level is tier 6. Used only when the inventory is unknown.
    """
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
        garlic_lv = (levels or {}).get("Garlic", 0)
        # A new Laurel is a third weapon once two damage weapons are held.
        # The low-HP shield shortcut must not spend that slot.
        third = c.new and name in WEAPONS and _holds_two_damage_weapons(levels)
        if not third and not _armor_upgrade_yields_to_garlic(name, c.new, hp, garlic_lv):
            return (1, DEFENSE.index(name))
    if early and c.new and name in EARLY_AVOID_NEW:
        return (EARLY_AVOID_RANK, idx)
    # Before the avoid list, so Knife / Axe do not beat a demoted keeper.
    if _extra_weapon_capped(name, c.new, levels):
        return (41, idx)
    if name in AVOID:
        return (40 if c.new else 30, idx)
    if name == CHICKEN or name in COINS:
        return (50, idx)
    if name == "Candelabrador" and c.new and (levels or {}).get(name, 0) <= 0:
        if levels and 0 < levels.get("Garlic", 0) < 8 and _santa_water_now(hp):
            return (4, 0)
    if name == "Santa Water" and _santa_water_now(hp):
        gl = (levels or {}).get("Garlic", 0)
        if levels and 0 < gl < 8:
            return (6, 0)
        return (2, 0)
    if c.new and name in WEAPONS:
        if not slots_free:
            return (35, idx)
        # test7 run 3 opened Fire Wand while Garlic was still 3 and a second
        # weapon was already held. Passives and the two held weapons come first.
        # A card we already hold is a level, even if OCR still says "new".
        if _holds_two_damage_weapons(levels) and (levels or {}).get(name, 0) <= 0:
            return (22, SECOND_ORDER.index(name) if name in SECOND_ORDER else 50)
        if _core_slot_open(levels) and name not in CORE_SECOND_WEAPONS and name != "Garlic":
            return (41, idx)
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


def _only_filler_index(cards: list[Card], hp: float | None) -> int | None:
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
    return None


def _choose_static(
    cards: list[Card],
    hp: float | None = None,
    owned: set[str] | None = None,
    game_s: float | None = None,
    levels: dict[str, int] | None = None,
) -> int:
    """Old priority list. Still the pick when ``levels`` is missing or the read disagrees.

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
    filler = _only_filler_index(cards, hp)
    if filler is not None:
        return filler
    known = set(owned or ())
    known |= {c.name for c in cards if c.name and not c.new}
    weapons = owned_weapon_count(known)
    early = is_early_run(game_s, weapons)
    return min(
        range(len(cards)),
        key=lambda i: _card_rank(cards[i], i, hp, weapons, early, levels),
    )


def _defer_new_weapons(cards: list[Card], levels: dict[str, int]) -> bool:
    """Main weapon is under 8, and a non-priority passive is on this screen.

    A new weapon joins the offer pool, so later level-ups show it instead of
    Garlic. Wings is that passive (run 1 took Fire Wand beside it). Priority
    passives stay under a paired weapon: late Pentagram still beats Spinach.
    Weapon-only screens stay on the old fallback.
    """
    main = main_weapon_name(levels)
    if main is None or levels.get(main, 0) >= WEAPON_MAX_LEVEL:
        return False
    for card in cards:
        name = card.name
        if not name or name in FILLER or name in WEAPONS or name in PRIORITY:
            continue
        return True
    return False


def _owned_weapon_level_offered(cards: list[Card], levels: dict[str, int]) -> bool:
    """An owned weapon that is not yet at 8 is on this screen.

    A Santa Water level does not count while Garlic is under 8. That level used
    to mark the screen as "protect the weapon", which sank Hollow Heart under
    Santa Water (test10 run2 at 37s).
    """
    garlic_open = 0 < levels.get("Garlic", 0) < WEAPON_MAX_LEVEL
    for card in cards:
        if card.new or not card.name or card.name not in WEAPONS:
            continue
        if garlic_open and card.name == "Santa Water":
            continue
        held = levels.get(card.name, 0)
        if 0 < held < WEAPON_MAX_LEVEL:
            return True
    return False


def _focused_rank(
    c: Card,
    idx: int,
    hp: float | None,
    weapons: int,
    early: bool,
    levels: dict[str, int],
    protect_weapons: bool = False,
    defer_new_weapons: bool = False,
) -> tuple[int, int, str]:
    """Lower tier wins. Reasons are main / partner / second / fallback.

    The main weapon takes every level until 8. Its partner is taken the moment
    it is offered, including on a screen that also has a main level (a partner
    may not come back; the owned weapon will). A second paired weapon is opened
    only after the main is maxed, and is maxed the same way. While the main is
    under 8, a new weapon loses to a non-weapon on the same screen. On a
    weapon-only screen it stays a last resort. Pummarola / Armor / Hollow Heart
    come next, and they beat a main level while HP is under half.
    """
    name = c.name
    low = hp is not None and hp < DEFENSIVE_HP
    main = main_weapon_name(levels)
    main_lv = levels.get(main, 0) if main else 0
    main_maxed = main is not None and main_lv >= WEAPON_MAX_LEVEL
    second = _second_weapon(levels, main) if main_maxed and main else None
    if not name:
        return (45, idx, "fallback")
    if name in EVOLUTIONS and not c.new:
        evo = EVOLUTION_PAIRS.get(main, (None, ""))[1] if main else ""
        return (0, EVOLUTIONS.index(name), "main" if name == evo else "second")
    # Santa Water is second, behind a Garlic level, its partner, Candelabrador,
    # and the survival passives while Garlic is under 8. test10 run2 took it
    # over Hollow Heart and the candle and stalled at Garlic 6.
    if name == "Santa Water" and _santa_water_now(hp):
        if main == "Garlic" and 0 < main_lv < WEAPON_MAX_LEVEL:
            return (8, 0, "second")
        return (6, 2, "second")
    if (
        name == "Candelabrador"
        and c.new
        and levels.get(name, 0) <= 0
        and not (hp is not None and hp <= 0.25)
    ):
        return (6, 0, "fallback")
    if low and name in DEFENSE:
        third = c.new and name in WEAPONS and _holds_two_damage_weapons(levels)
        if not third and not _armor_upgrade_yields_to_garlic(name, c.new, hp, main_lv if main == "Garlic" else 0):
            missing = main is not None and name == partner_of(main) and levels.get(name, 0) <= 0
            return (1, DEFENSE.index(name), "partner" if missing else "fallback")
    if early and c.new and name in EARLY_AVOID_NEW:
        return (EARLY_AVOID_RANK, idx, "fallback")
    if main is None and name in ("Garlic", "Magic Wand"):
        return (2, 0 if name == "Garlic" else 1, "main")
    if main and name == partner_of(main) and levels.get(name, 0) <= 0:
        return (3, 0, "partner")
    if main and name == main and not c.new and main_lv < WEAPON_MAX_LEVEL:
        return (4, 0, "main")
    if main_maxed and second:
        other = partner_of(second)
        if (
            other
            and name == other
            and levels.get(other, 0) <= 0
            and not (c.new and name in WEAPONS and weapons >= MAX_WEAPONS)
        ):
            return (3, 1, "partner")
        if name in _level_targets(levels, second) and not c.new:
            return (5, 0, "second")
    if (
        main_maxed
        and second is None
        and c.new
        and name in EVOLUTION_PAIRS
        and name != main
        and weapons < MAX_WEAPONS
        and not (early and name in EARLY_AVOID_NEW)
    ):
        held = 0 if levels.get(partner_of(name) or "", 0) > 0 else 1
        order = SECOND_ORDER.index(name) if name in SECOND_ORDER else 50
        return (6, held * 100 + order, "second")
    # A weapon level is on the screen and the main is not finished. Do not spend
    # it on another passive level, or on a new passive that is not some owned
    # weapon's partner. New Armor still beats a new weapon when no weapon level
    # is offered (that screen has nothing owned to protect).
    if protect_weapons and name not in WEAPONS and name not in EVOLUTIONS:
        if not _missing_partner_name(name, levels):
            if c.new:
                return (27, idx, "fallback")
            if levels.get(name, 0) > 0:
                return (12, idx, "fallback")
    if name in SURVIVAL:
        return (7, SURVIVAL.index(name), "fallback")
    if not c.new:
        pri = PRIORITY.index(name) if name in PRIORITY else 100 + idx
        if low and name in LUXURY:
            return (28, pri, "fallback")
        return (10, pri, "fallback")
    if c.new and name in WEAPONS and weapons >= MAX_WEAPONS:
        return (36, idx, "fallback")
    # King Bible is the third weapon. It waits for a Garlic level and for Santa
    # Water, and it is the only other weapon the early fallback may open.
    if (
        c.new
        and name == "King Bible"
        and levels.get(name, 0) <= 0
        and weapons < MAX_WEAPONS
        and _santa_water_now(hp)
        and not (early and name in EARLY_AVOID_NEW)
    ):
        # Behind a Garlic level (4), Candelabrador (6, 0) and survival (7).
        # Ahead of Santa Water only once Garlic is maxed; until then Santa
        # Water is (8, 0) and this is (8, 1).
        if main == "Garlic" and 0 < main_lv < WEAPON_MAX_LEVEL:
            return (8, 1, "second")
        return (6, 3, "second")
    if _extra_weapon_capped(name, c.new, levels):
        # Same tier for keepers and the avoid list, so Knife does not beat
        # Magic Wand, and either loses to a passive. A weapon-only screen
        # still clicks the first card. Late Pentagram keeps its own rank.
        return (41, idx, "fallback")
    if (
        c.new
        and name in EVOLUTION_PAIRS
        and name not in AVOID
        and not (early and name in EARLY_AVOID_NEW)
    ):
        order = SECOND_ORDER.index(name) if name in SECOND_ORDER else 50
        # Below a generic new passive (34), above an avoid weapon (40).
        if defer_new_weapons:
            return (35, order, "fallback")
        return (20, order, "fallback")
    if name in AVOID and c.new:
        return (40, idx, "fallback")
    if c.new and name in PRIORITY:
        pri = PRIORITY.index(name)
        if low and name in LUXURY:
            return (32, pri, "fallback")
        return (25, pri, "fallback")
    if name == CHICKEN or name in COINS:
        return (50, idx, "fallback")
    if c.new:
        if defer_new_weapons and name in WEAPONS:
            return (35, idx, "fallback")
        return (34, idx, "fallback")
    return (15, idx, "fallback")


def _coin_index(cards: list[Card]) -> int | None:
    """Best gold card on the screen. Big Coin Bag beats a smaller bag."""
    best_i: int | None = None
    best_rank: int | None = None
    for i, card in enumerate(cards):
        if card.name not in COINS:
            continue
        rank = COINS.index(card.name)
        if best_rank is None or rank < best_rank:
            best_i, best_rank = i, rank
    return best_i


def _filler_choice(cards: list[Card], low: bool) -> int | None:
    """Gold, unless HP is low and Floor Chicken is offered."""
    coins = _coin_index(cards)
    chicken = next((i for i, card in enumerate(cards) if card.name == CHICKEN), None)
    if coins is None and chicken is None:
        return None
    if low and chicken is not None:
        return chicken
    if coins is not None:
        return coins
    return chicken


def _choose_focused(
    cards: list[Card],
    hp: float | None,
    levels: dict[str, int],
    game_s: float | None,
    *,
    low: bool = False,
    evolved: set[str] | frozenset[str] | None = None,
) -> tuple[int, str]:
    # Only after every owned weapon and passive is capped. Before that, gold
    # must not beat a level the build still wants.
    if items_fully_maxed(levels, evolved):
        filler = _filler_choice(cards, low)
        if filler is not None:
            return filler, "fallback"
    filler = _only_filler_index(cards, hp)
    if filler is not None:
        return filler, "fallback"
    weapons = owned_weapon_count(set(levels))
    early = is_early_run(game_s, weapons)
    protect = _owned_weapon_level_offered(cards, levels)
    defer = _defer_new_weapons(cards, levels)
    ranked = [
        _focused_rank(cards[i], i, hp, weapons, early, levels, protect, defer)
        for i in range(len(cards))
    ]
    idx = min(range(len(cards)), key=lambda i: ranked[i][:2])
    return idx, ranked[idx][2]


@dataclass(frozen=True)
class CardPick:
    index: int
    reason: str  # main, partner, second, or fallback


def choose_pick(
    cards: list[Card],
    hp: float | None = None,
    owned: set[str] | None = None,
    game_s: float | None = None,
    levels: dict[str, int] | None = None,
    evolved: set[str] | frozenset[str] | None = None,
    blind_low: bool = False,
    critical: bool = False,
    low_held: bool = False,
) -> CardPick:
    """Card to take, and why.

    A certain inventory follows the evolution plan. A contradictory read still
    uses that plan for every card that agrees with the tracker (owned levels
    and missing partners). The static priority, which prefers a new crowd
    weapon, runs only when nothing owned agrees.

    ``hp`` is the HpTracker median, or None when the bar is blind. Filler
    (coin bag vs Floor Chicken) uses that number only once every owned weapon
    and passive is capped.
    """
    if not cards:
        return CardPick(0, "fallback")
    eff, certain = assess_inventory(cards, levels)
    if not certain:
        if levels is not None:
            usable = [c for c in cards if _card_tracks(c, levels)]
            preferred = [
                c
                for c in usable
                if (not c.new and levels.get(c.name, 0) > 0) or _missing_partner_name(c.name, levels)
            ]
            if preferred:
                low = hp_is_low(hp, blind_low=blind_low, critical=critical, low_held=low_held)
                idx, reason = _choose_focused(
                    usable, hp, dict(levels), game_s, low=low, evolved=evolved
                )
                reason = reason if reason in PICK_REASONS else "fallback"
                return CardPick(cards.index(usable[idx]), reason)
        return CardPick(_choose_static(cards, hp, owned, game_s, levels), "fallback")
    low = hp_is_low(hp, blind_low=blind_low, critical=critical, low_held=low_held)
    idx, reason = _choose_focused(cards, hp, eff, game_s, low=low, evolved=evolved)
    return CardPick(idx, reason if reason in PICK_REASONS else "fallback")


def choose_card(
    cards: list[Card],
    hp: float | None = None,
    owned: set[str] | None = None,
    game_s: float | None = None,
    levels: dict[str, int] | None = None,
    evolved: set[str] | frozenset[str] | None = None,
    blind_low: bool = False,
    critical: bool = False,
    low_held: bool = False,
) -> int:
    """Index of the card to take. See ``choose_pick`` for the evolution policy."""
    return choose_pick(
        cards,
        hp=hp,
        owned=owned,
        game_s=game_s,
        levels=levels,
        evolved=evolved,
        blind_low=blind_low,
        critical=critical,
        low_held=low_held,
    ).index


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
# Same band as Tuning.critical_hp. A read below it sticks across misses until the
# accepted window itself recovers (run1 death 1: 0.06 then none, median still 0.92).
HP_CRITICAL = 0.20
# test3: critical turned off at a 0.25 median (10:17) and after three frames of
# 0.95 / 0.96 / 1.00 (10:50), which was a damage flash, not a heal. Leave only
# once several accepted readings sit at or above the exit, and a near-full flash
# needs a longer streak than a normal heal (three 0.55s still count as one).
HP_CRITICAL_EXIT = 0.40
HP_RECOVER_N = 3
HP_FULL_FLASH = 0.90
HP_FULL_FLASH_N = 6
# A chest or level-up darkens the playfield and the anchored bar reads 0.00
# for a frame (test6: 21 critical-on lines, 14 HP-crash flees, previous frame
# still full). One such drop from a healthy bar is the overlay, not a hit.
# A second 0.00, or any read that is merely low (0.06), is real.
HP_OVERLAY_ZERO = 0.02
HP_OVERLAY_HEALTHY = 0.35


class HpTracker:
    """Median of the last few HP-bar readings: one frame with a sprite over the bar
    (or a red hit flash) must not swing the level-up choice."""

    def __init__(self, window: int = 5, miss_limit: int = HP_BLIND_MISSES) -> None:
        self.window = window
        self.miss_limit = miss_limit
        self.values: list[float] = []
        self.misses = 0
        self.blind = False
        self.low_held = False
        self._anchor: float | None = None
        self._pending: list[float] = []
        self._recover: list[float] = []
        self._zero_pending = 0
        self._overlay = False
        self._overlay_base: float | None = None
        self._healthy: float | None = None

    def add(self, v: float | None) -> None:
        if v is None:
            self._pending.clear()
            self._zero_pending = 0
            self._recover.clear()
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
        if self._overlay and v <= HP_OVERLAY_ZERO:
            return
        # One exact empty-bar read from a healthy window is the chest / level-up
        # frame that occludes the bar. The next different read, or a second
        # 0.00, is accepted. A drop to 0.06 is not this case.
        if (
            not self.low_held
            and base is not None
            and base >= HP_OVERLAY_HEALTHY
            and v <= HP_OVERLAY_ZERO
        ):
            self._zero_pending += 1
            if self._zero_pending < 2:
                return
        self._zero_pending = 0
        self._accept(v)

    def _accept(self, v: float) -> None:
        self.misses = 0
        self.blind = False
        self._pending.clear()
        self.values.append(float(v))
        del self.values[: -self.window]
        self._anchor = self.value
        self._sync_low([float(v)])
        if not self.low_held and self.value is not None and self.value >= HP_OVERLAY_HEALTHY:
            self._healthy = float(self.value)

    def _accept_window(self, samples: list[float]) -> None:
        self.misses = 0
        self.blind = False
        self._pending.clear()
        self.values = [float(s) for s in samples[-self.window :]]
        self._anchor = self.value
        self._sync_low([float(s) for s in self.values])

    def _sync_low(self, samples: list[float]) -> None:
        """Hold critical until a heal is above the exit and several frames agree.

        One accepted sample under ``HP_CRITICAL`` sticks across misses and
        rejected spikes. A later median in the gap between enter and exit
        (test3: 0.25) does not release it. Release takes ``HP_RECOVER_N``
        accepted readings at or above ``HP_CRITICAL_EXIT`` that agree with each
        other. A jump from critical into a near-full bar is the damage-flash
        misread (0.95, 0.96, 1.00) and needs the longer streak.
        """
        for v in samples:
            if v < HP_CRITICAL:
                self.low_held = True
                self._recover.clear()
                continue
            if not self.low_held:
                continue
            if v < HP_CRITICAL_EXIT:
                self._recover.clear()
                continue
            self._recover.append(v)
            del self._recover[: -HP_FULL_FLASH_N]
            if self._recovered():
                self.low_held = False
                self._recover.clear()

    def _recovered(self) -> bool:
        if self.value is None or self.value < HP_CRITICAL_EXIT or not self._recover:
            return False
        if len(self._recover) >= HP_FULL_FLASH_N:
            tail = self._recover[-HP_FULL_FLASH_N :]
            if min(tail) >= HP_FULL_FLASH and max(tail) - min(tail) <= HP_JUMP_AGREE:
                return True
        if len(self._recover) < HP_RECOVER_N:
            return False
        tail = self._recover[-HP_RECOVER_N :]
        if min(tail) >= HP_FULL_FLASH:
            return False
        return min(tail) >= HP_CRITICAL_EXIT and max(tail) - min(tail) <= HP_JUMP_AGREE

    def reset(self) -> None:
        self.values.clear()
        self.misses = 0
        self.blind = False
        self.low_held = False
        self._anchor = None
        self._pending.clear()
        self._recover.clear()
        self._zero_pending = 0
        self._overlay = False
        self._overlay_base = None
        self._healthy = None

    def suspend_for_overlay(self) -> None:
        """Chest or level-up is up. Remember the last healthy bar and ignore 0.00."""
        if self._overlay_base is None and self._healthy is not None:
            self._overlay_base = self._healthy
        self._overlay = True
        self._zero_pending = 0

    def resume_after_overlay(self) -> None:
        """The menu closed. Drop zeros that the dark frame wrote over a healthy bar.

        A real low sample (0.06, not 0.00) in the window keeps the hold. A window
        that never had a healthy reading is left alone.
        """
        self._overlay = False
        self._zero_pending = 0
        base = self._overlay_base
        self._overlay_base = None
        if base is None:
            return
        real_low = [v for v in self.values if HP_OVERLAY_ZERO < v < HP_OVERLAY_HEALTHY]
        if real_low:
            return
        if not self.low_held and not any(v <= HP_OVERLAY_ZERO for v in self.values):
            return
        kept = [v for v in self.values if v > HP_OVERLAY_ZERO]
        self.values = (kept or [base])[-self.window :]
        self.low_held = False
        self._recover.clear()
        self.blind = False
        self.misses = 0
        self._anchor = self.value
        if self.value is not None and self.value >= HP_OVERLAY_HEALTHY:
            self._healthy = float(self.value)

    def forget_for_revive(self) -> None:
        """Drop the death-screen bar, and stay critical until a new one agrees.

        ``reset`` would clear ``low_held`` too. The next frame is then hp=?
        with nothing held, and nav treats that as healthy. The revived bar
        (often ~0.50, sometimes a one-frame flash) has to agree before it counts.
        """
        self.reset()
        self.low_held = True

    def blind_after_low(self, limit: float) -> bool:
        """Blind, and the last reading was under ``limit`` or still held critical.

        Blind with no prior low reading is false.
        """
        if not self.blind:
            return False
        if self.low_held:
            return True
        return self._anchor is not None and self._anchor < limit

    @property
    def value(self) -> float | None:
        if not self.values:
            return None
        return float(np.median(self.values))


def hp_for_nav(hp: float | None, blind: bool, low_held: bool = False) -> float | None:
    """HP nav should act on. A bar lost under a swarm counts as empty.

    ``low_held`` keeps that empty reading when the last accepted sample was
    critical and the median has not recovered yet (the window can still be the
    pre-crash high while the bar is already unreadable).
    """
    if blind or (low_held and (hp is None or hp >= HP_CRITICAL)):
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


def _lowest_word(words: list[tuple[str, int, int, int, int]], target: str) -> tuple[int, int] | None:
    """Center of the lowest OCR word matching ``target``. The DONE button sits under the card."""
    best: tuple[int, int] | None = None
    for t, x, y, w, h in words:
        clean = re.sub(r"[^a-z]", "", t.lower())
        if not clean or difflib.SequenceMatcher(None, clean, target).ratio() < 0.8:
            continue
        cy = y + h // 2
        if best is None or cy > best[1]:
            best = (x + w // 2, cy)
    return best


def read_chest_card_text(frame: np.ndarray, words: list[tuple[str, int, int, int, int]] | None = None) -> str:
    """OCR the reward card above DONE.

    The full darkened page (timer, kill counter, vignette) is what Tesseract
    returns as ``'Venda Feet...'`` / ``'18304 oF] ~ . oe DONE'``. The item name
    and ``Level: N`` are the lighter text on the card, so a tight crop above
    the button is read at a lower threshold and a larger scale.
    """
    if frame is None or frame.size == 0 or not tesseract_available():
        return ""
    h, w = frame.shape[:2]
    if words is None:
        words = read_screen_words(frame)
    done = _lowest_word(words, "done")
    if done is not None:
        cx, cy = done
        x0 = max(0, int(cx - 0.22 * w))
        x1 = min(w, int(cx + 0.22 * w))
        y1 = max(0, int(cy - 0.04 * h))
        y0 = max(0, int(cy - 0.56 * h))
    else:
        x0, x1 = int(0.27 * w), int(0.73 * w)
        y0, y1 = int(0.24 * h), int(0.64 * h)
    if y1 <= y0 + 8 or x1 <= x0 + 8:
        return ""
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    best = ""
    for thr in (170, 150, 200):
        text = ocr_image(_white_text(crop, thr=thr, scale=3), psm=6).replace("\n", " ")
        text = " ".join(text.split())
        if not text:
            continue
        if evolutions_from_chest_text(text) or levels_from_chest_text(text):
            return text
        if len(text) > len(best):
            best = text
    return best


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
        critical_hp: Callable[[], bool] | None = None,
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
        # Navigator critical flag, if that build has one. Missing means not low.
        self.critical_hp = critical_hp
        self.levelup_click = levelup_click
        self.results_dir = results_dir
        self.start_items = tuple(start_items)
        self.owned: set[str] = set(self.start_items)
        # Levels come from our own picks, checked against "level N" on the card.
        # Poe starts with Garlic at 1. A contradictory OCR read does not overwrite this.
        self.levels: dict[str, int] = {n: 1 for n in self.start_items if n}
        # Base weapons a chest has already evolved. Soul Eater marks Garlic;
        # Vandalier marks both birds. Not a second weapon slot.
        self.evolved: set[str] = set()
        self._pick_reason = "fallback"
        self._inventory_certain = True
        self._evol_logged = False
        self._build_state = (False, False)
        self.chest_is_reward = False
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
    def evolution_ready(self) -> bool:
        """Some owned weapon is at 8 with its partner and has not evolved yet."""
        return evolution_ready(self.levels, self.evolved)

    @property
    def all_maxed(self) -> bool:
        """Every owned weapon is maxed or evolved, and no new weapon is wanted."""
        return all_maxed(self.levels, self.evolved)

    def _log_evolution_ready(self) -> None:
        if self._evol_logged or not self.evolution_ready:
            return
        labels = ready_evolution_labels(self.levels, self.evolved)
        if not labels:
            return
        self._evol_logged = True
        self.log("[levelup] evolution ready: " + ", ".join(labels))

    def _publish_build(self) -> None:
        """Log the chest-gate flags when they change. Inventory is the snapshot."""
        flags = (bool(self.evolution_ready), bool(self.all_maxed))
        if flags == self._build_state:
            return
        self._build_state = flags
        evo, mx = flags
        inv = format_inventory(self.levels, self.evolved)
        self.log(f"[build] evo_ready={int(evo)} all_maxed={int(mx)} inv={inv}")

    def _note_chest_reward(self, text: str) -> None:
        """Mark evolutions, and adopt a single confident level, from chest OCR."""
        evolved_now = evolutions_from_chest_text(text)
        for base in evolved_now:
            self.evolved.add(base)
            self.owned.add(base)
            self.levels[base] = max(WEAPON_MAX_LEVEL, self.levels.get(base, 0))
        for name, lv in levels_from_chest_text(text).items():
            if name in evolved_now or name in self.evolved:
                continue
            held = self.levels.get(name, 0)
            if held <= 0:
                self.levels[name] = lv
                self.owned.add(name)
            elif lv > held and lv - held <= LEVEL_RECONCILE:
                self.levels[name] = min(level_cap(name), lv)
        self._publish_build()

    def _note_shown_cards(self, screen: LevelUpScreen) -> None:
        """An evolution named on a level-up card has already happened.

        Soul Eater on the card marks Garlic evolved and at 8. A level tag one
        above the tracker stays the next-level offer (it is not copied here).
        """
        changed = False
        for card in screen.cards:
            if not card.name or card.new:
                continue
            for base in EVOLUTION_BASES.get(card.name, ()):
                if base in self.evolved:
                    continue
                self.evolved.add(base)
                self.owned.add(base)
                self.levels[base] = max(WEAPON_MAX_LEVEL, self.levels.get(base, 0))
                changed = True
        if changed:
            self._publish_build()

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

    @property
    def hp_low_held(self) -> bool:
        """True after a critical read until a trusted recovery, including missed frames."""
        return self._hp.low_held

    def new_run(self) -> None:
        """A fresh run: forget owned items and HP history."""
        self.reset_run()
        self.revives = 0
        self._hp_ignore_until = 0.0
        self.owned = set(self.start_items)
        self.levels = {n: 1 for n in self.start_items if n}
        self.evolved = set()
        self._pick_reason = "fallback"
        self._inventory_certain = True
        self._evol_logged = False
        self._hp.reset()
        self.picks = []
        self.banished = set()
        self._banish = None
        self._banish_fails = 0
        self._banish_skip_screen = False
        self._publish_build()

    # -- helpers
    def _game_s(self) -> float | None:
        if self.game_clock is None:
            return None
        try:
            v = self.game_clock()
        except Exception:
            return None
        return None if v is None else float(v)

    def _critical_now(self) -> bool:
        fn = self.critical_hp
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def pick_is_low(self) -> bool:
        """Low HP for a capped build's Floor Chicken vs coin bag choice.

        The number passed into the pick is the median, or None while blind.
        """
        blind = self.hp_blind
        return hp_is_low(
            None if blind else self.hp,
            blind_low=self._hp.blind_after_low(low_hp_limit()) if blind else False,
            critical=self._critical_now(),
            low_held=self.hp_low_held,
        )

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
                self._note_shown_cards(screen)
                self._names = [c.name or c.raw or "?" for c in screen.cards]
                self.owned |= {c.name for c in screen.cards if c.name and not c.new}
                if self._try_start_banish(frame, screen):
                    return
                game_s = self._game_s()
                eff, certain = assess_inventory(screen.cards, self.levels)
                self._inventory_certain = certain
                if certain:
                    merge_card_levels(self.levels, eff)
                    self.owned |= {n for n in eff if n not in FILLER and self.levels.get(n, 0) > 0}
                pick = choose_pick(
                    screen.cards,
                    hp=None if self.hp_blind else self.hp,
                    owned=self.owned,
                    game_s=game_s,
                    levels=self.levels,
                    evolved=self.evolved,
                    blind_low=self.hp_blind and self._hp.blind_after_low(low_hp_limit()),
                    critical=self._critical_now(),
                    low_held=self.hp_low_held,
                )
                self._target = pick.index
                self._pick_reason = pick.reason
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
        taken = self._chosen or (name if name in VOCAB else "")
        # Coin bags and Floor Chicken are not a weapon or a passive slot.
        if taken in EVOLUTION_BASES:
            for base in EVOLUTION_BASES[taken]:
                self.evolved.add(base)
                self.owned.add(base)
                self.levels[base] = max(WEAPON_MAX_LEVEL, self.levels.get(base, 0))
        elif taken and taken not in FILLER:
            self.owned.add(taken)
            self.levels[taken] = min(level_cap(taken), self.levels.get(taken, 0) + 1)
        shown = taken or name
        inv = format_inventory(self.levels, self.evolved)
        if not self._inventory_certain:
            inv = f"{inv} (uncertain)"
        reason = self._pick_reason if self._pick_reason in PICK_REASONS else "fallback"
        self.log(
            f"[levelup] picked {shown} (reason: {reason}) inv={inv} "
            f"(total picks {len(self.picks)})"
        )
        if self._inventory_certain:
            self._log_evolution_ready()
        self._publish_build()
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
                card = read_chest_card_text(frame, words) if kind == "chest" else ""
                self._chest_hit = (self.clock(), text, kind, where, card)
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
        shown = (hit[1] or "")[:80]
        if kind == "chest":
            self.chest_is_reward = chest_text_is_reward(hit[1] or "")
        if now >= self._next_action:
            if kind == "results":
                self._save_results(frame)
            if kind in ("revive", "quit", "results"):
                label = {"revive": "REVIVE", "quit": "QUIT", "results": "DONE"}[kind]
                if where is not None and self.click is not None:
                    self.refocus()
                    self.click(*where)
                    self._next_action = self.clock() + self.chest_cooldown_s
                    self.log(f"[{kind}] clicked {label} at {where} ({shown!r})")
                else:
                    # The only / highlighted button on these screens.
                    self._press("enter", self.chest_cooldown_s)
                    self.log(f"[{kind}] Enter on {label} ({shown!r})")
                if kind == "revive":
                    self.revives += 1
                    # Pre-death readings are stale. Stay critical until the new bar agrees.
                    self._hp.forget_for_revive()
                    self._hp_ignore_until = self.clock() + REVIVE_HP_HOLDOFF_S
                elif kind == "quit":
                    self._quit_pressed = True
                else:
                    self.finished = True
                    self.log("[menus] run finished (QUIT + DONE); the bot stops acting until p is pressed")
            else:
                # Treasure Found grants the item. It does not ask the bot to
                # choose Floor Chicken vs a coin bag, so the filler rule stays
                # on the level-up screen.
                self._press("enter", self.chest_cooldown_s)
                self.chest_presses += 1
                self.log(f"[chest] Enter on chest screen ({shown!r})")
                full = hit[1] or ""
                card = hit[4] if len(hit) > 4 else ""
                reward = card if card and (evolutions_from_chest_text(card) or levels_from_chest_text(card)) else full
                self._note_chest_reward(reward)
            self._chest_hit = None
            self._chest_streak = 0  # the next screen must be confirmed by two fresh scans
        self.status = kind.upper()
        return kind
