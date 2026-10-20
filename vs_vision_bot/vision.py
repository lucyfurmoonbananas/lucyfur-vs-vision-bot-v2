"""Monster / menu vision. Heuristic by default; optional TorchScript model path."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from vs_vision_bot.config import Config


REF_W_DEFAULT, REF_H_DEFAULT = 960, 540


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


@dataclass
class VisionResult:
    player: tuple[int, int]
    monsters: list[Detection] = field(default_factory=list)
    menu: bool = False
    note: str = ""
    # Threat grid in reference 960x540 pixels: rows of (x, y, weight).
    threats: np.ndarray | None = None
    # XP gem centers in reference pixels. ``gem_colors`` is parallel
    # ("blue", "green", or "red"); empty means every gem is blue.
    gems: list[tuple[float, float]] = field(default_factory=list)
    gem_colors: list[str] = field(default_factory=list)
    # On-screen treasure chest center in reference pixels, else None.
    chest: tuple[float, float] | None = None
    # Yellow off-screen-chest arrow centroid in reference pixels, else None.
    chest_arrow: tuple[float, float] | None = None
    # On-screen stage timer in seconds, when OCR is confident; else None.
    timer_s: float | None = None
    # Filled fraction of the top XP bar, 0..1. None when the bar is not visible.
    # Rises are the gem-pickup proxy (a level-up resets it).
    xp: float | None = None
    # Frame size the detections were measured in.
    size: tuple[int, int] = (REF_W_DEFAULT, REF_H_DEFAULT)


class Vision:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        self._timer_at = 0.0
        if cfg.model_path:
            self._model = self._load_torch_model(cfg.model_path)

    def analyze(self, frame: np.ndarray) -> VisionResult:
        h, w = frame.shape[:2]
        player = player_position(w, h)
        timer_s = self._maybe_read_timer(frame)
        xp = read_xp_fraction(frame)
        if self._model is not None:
            try:
                monsters = self._infer_model(frame)
                menu = detect_level_up_menu(frame)
                return VisionResult(
                    player=player, monsters=monsters, menu=menu, note="torch", timer_s=timer_s, xp=xp,
                )
            except Exception as exc:  # noqa: BLE001
                self._model = None
                print(f"[vision] model failed, falling back to heuristic: {exc}", file=sys.stderr)
        mask = enemy_mask(frame)
        sx, sy = REF_W / w, REF_H / h
        colored = detect_colored_gems(frame, mask)
        gems = [(gx * sx, gy * sy) for gx, gy, _color in colored]
        gem_colors = [color for _gx, _gy, color in colored]
        monsters = detect_threat_blobs(frame, player, self.cfg, mask=mask)
        threats = threat_points(mask)
        menu = detect_level_up_menu(frame)
        chest_xy = detect_chest(frame)
        arrow_xy = detect_chest_arrow(frame)
        chest = (chest_xy[0] * sx, chest_xy[1] * sy) if chest_xy else None
        chest_arrow = (arrow_xy[0] * sx, arrow_xy[1] * sy) if arrow_xy else None
        return VisionResult(
            player=player, monsters=monsters, menu=menu, note="heuristic",
            threats=threats, gems=gems, gem_colors=gem_colors, timer_s=timer_s, size=(w, h), xp=xp,
            chest=chest, chest_arrow=chest_arrow,
        )

    def _maybe_read_timer(self, frame: np.ndarray) -> float | None:
        """OCR the HUD timer at most every TIMER_OCR_PERIOD_S (avoids hitching)."""
        now = time.monotonic()
        if self._timer_at and now - self._timer_at < TIMER_OCR_PERIOD_S:
            return None
        self._timer_at = now
        return read_timer_seconds(frame)

    def _load_torch_model(self, path: str):
        try:
            import torch
        except ImportError as exc:
            print(
                f"[vision] VS_MODEL_PATH is set ({path}) but torch is not installed: {exc}",
                file=sys.stderr,
            )
            return None
        try:
            model = torch.jit.load(path, map_location="cpu")
            model.eval()
            print(f"[vision] loaded TorchScript model from {path}")
            return model
        except Exception as exc:  # noqa: BLE001
            print(f"[vision] could not load {path}: {exc}", file=sys.stderr)
            return None

    def _infer_model(self, frame: np.ndarray) -> list[Detection]:
        import torch

        # float RGB NCHW scaled 0–1. Rows are [x, y, w, h, score, ...].
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            out = self._model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        boxes = out.detach().cpu().numpy()
        if boxes.ndim == 3:
            boxes = boxes[0]
        detections: list[Detection] = []
        for row in np.atleast_2d(boxes):
            if row.size < 5:
                continue
            x, y, bw, bh, score = (float(v) for v in row[:5])
            if score < 0.35:
                continue
            detections.append(Detection(int(x), int(y), int(bw), int(bh)))
        return detections


# Reference layout (960x540 capture of the windowed game). Everything below is
# scaled to the real frame size.
REF_W, REF_H = 960, 540
# The game renders a 16:9 playfield with dimmed pillarbox strips on the sides.
PLAYFIELD_X0, PLAYFIELD_X1 = 48, 912
# HUD rectangles (x0, y0, x1, y1): XP bar, weapon/passive icons, timer, kill
# count + gold, right-hand widgets. Their black outlines look like enemies.
HUD_RECTS = (
    (0, 0, 960, 18),
    (48, 18, 196, 70),
    (400, 16, 565, 54),
    (715, 16, 912, 46),
    (858, 16, 912, 96),
)
# Tight crop of the stage timer digits inside HUD_RECTS[2] (ref 960x540).
TIMER_RECT = (445, 20, 520, 48)
# Top XP bar at 960x540: blue fill starts at x=51, the dark track ends at x=868.
XP_BAR_X0, XP_BAR_X1 = 51, 869
XP_BAR_Y0, XP_BAR_Y1 = 4, 14
TIMER_OCR_PERIOD_S = 2.0
# Player sprite + HP bar (camera keeps the character here).
PLAYER_RECT = (452, 220, 508, 282)
PLAYER_POS = (480, 256)

DARK_MAX = 40        # enemies (bats, ghouls, skeleton outlines) are near-black
SIDE_DARK_MAX = 12   # the side strips are dimmed to ~1/3 brightness
MIN_DARK_AREA = 30   # smaller dark specks are bush outlines / gem outlines
CELL = 12            # threat grid cell size (reference pixels)
CELL_WEIGHT = 0.25   # max weight of one cell


def player_position(w: int, h: int) -> tuple[int, int]:
    return (int(round(PLAYER_POS[0] * w / REF_W)), int(round(PLAYER_POS[1] * h / REF_H)))


def _scaled_rect(rect: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    sx, sy = w / REF_W, h / REF_H
    x0, y0, x1, y1 = rect
    return int(x0 * sx), int(y0 * sy), int(round(x1 * sx)), int(round(y1 * sy))


def ignore_mask(w: int, h: int) -> np.ndarray:
    """True where detections must be ignored (HUD + player)."""
    mask = np.zeros((h, w), dtype=bool)
    for rect in HUD_RECTS + (PLAYER_RECT,):
        x0, y0, x1, y1 = _scaled_rect(rect, w, h)
        mask[y0:y1, x0:x1] = True
    return mask


_MASK_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _masks(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    key = (w, h)
    if key not in _MASK_CACHE:
        side = np.zeros((h, w), dtype=bool)
        x0 = int(PLAYFIELD_X0 * w / REF_W)
        x1 = int(round(PLAYFIELD_X1 * w / REF_W))
        side[:, :x0] = True
        side[:, x1:] = True
        _MASK_CACHE[key] = (ignore_mask(w, h), side)
    return _MASK_CACHE[key]


# Padding (px) around each dark component's bounding box for the bush band
# check in enemy_mask; also the crop margin for its per-component work.
BAND_PAD = 4


def enemy_mask(frame: np.ndarray) -> np.ndarray:
    """
    Binary mask of enemy pixels.

    Vampire Survivors sprites (bats, ghouls, mudmen outlines, skeletons,
    mantises) carry near-black pixels, while stage ground (grass, dirt) never
    goes that dark. Mad Forest bushes / clovers / trees also draw near-black
    outlines; those are stripped below by shape + vegetation cues so they are
    not counted as threats.
    """
    h, w = frame.shape[:2]
    ignore, side = _masks(w, h)
    mx = frame.max(axis=2)
    dark = mx < DARK_MAX
    dark[side] = mx[side] < SIDE_DARK_MAX
    dark[ignore] = False
    mask = dark.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale = (w * h) / (REF_W * REF_H)
    min_area = max(4, int(MIN_DARK_AREA * scale))
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    # Bright leaf pixels and purple berries / flowers that sit inside bush rings.
    lime = (g > 90) & (g > r + 20) & (g > b + 20)
    purple = (r > 70) & (b > 70) & (g < np.minimum(r, b) - 15)
    keep = np.zeros(n, dtype=bool)
    max_flat = max(4, int(round(8 * h / REF_H)))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        fill = area / float(bw * bh)
        aspect = bw / float(max(bh, 1))
        # Flat decoration strokes (bush rims, ground cracks).
        if bh <= max_flat and aspect >= 2.6:
            continue
        if bw <= max_flat and aspect <= (1.0 / 2.6):
            continue
        # Work on the component's bounding box padded by BAND_PAD (clipped to
        # the frame) instead of the whole frame. The pad covers the lime /
        # purple band below, and its >= 1 px ring of non-component pixels
        # gives the distance transform the same nearest zeros it sees on the
        # full frame, so results are identical.
        y0, y1 = max(0, y - BAND_PAD), min(h, y + bh + BAND_PAD)
        x0, x1 = max(0, x - BAND_PAD), min(w, x + bw + BAND_PAD)
        comp = labels[y0:y1, x0:x1] == i
        pix = frame[y0:y1, x0:x1][comp]
        # Bush outlines are green/brown tinted; bat / ghoul bodies are gray-black.
        chroma = float((pix.max(axis=1) - pix.min(axis=1)).mean()) if len(pix) else 0.0
        if chroma >= 8.0 and area < int(200 * scale):
            continue
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        max_t = float(dist.max())
        interior = float((dist >= 2.0).sum()) / float(area)
        # Compact but hollow bramble clumps: gray like bats, no solid body.
        if (
            area < int(160 * scale)
            and max_t < 2.6
            and interior < 0.05
            and fill > 0.30
        ):
            continue
        # Small thin rings hugging lime leaves or purple berries / flowers:
        # the bbox padded by BAND_PAD px (the crop above), minus the component.
        if area < int(100 * scale) and max_t <= 2.0:
            band = ~comp
            if (
                int(lime[y0:y1, x0:x1][band].sum()) >= 40
                or int(purple[y0:y1, x0:x1][band].sum()) >= 8
            ):
                continue
        keep[i] = True
    return keep[labels]


def gem_mask(frame: np.ndarray) -> np.ndarray:
    """Blue XP gems: saturated cyan-blue cores (grass is never bluer than green).

    The Glowing Bat's outline is also cyan-blue but carries much more red
    (r ~ 110 vs gem cores r ~ 38); requiring ``b - r > 150`` and ``r < 90``
    drops that outline before geometry runs.
    """
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    m = (b > 170) & (b - r > 150) & (g > 110) & (b > g + 25) & (r < 90)
    h, w = frame.shape[:2]
    ignore, side = _masks(w, h)
    m[ignore | side] = False
    return m


def threat_points(mask: np.ndarray, cell: int = CELL) -> np.ndarray:
    """Collapse an enemy mask into grid cells: rows of (x, y, weight 0..CELL_WEIGHT)."""
    h, w = mask.shape[:2]
    c = max(4, int(round(cell * w / REF_W)))
    gh, gw = h // c, w // c
    if gh == 0 or gw == 0:
        return np.zeros((0, 3), dtype=np.float32)
    grid = mask[: gh * c, : gw * c].reshape(gh, c, gw, c).mean(axis=(1, 3))
    ys, xs = np.nonzero(grid > 0.04)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # A typical enemy sprite covers ~4 cells: scale so one enemy ~= 1 unit of
    # danger (the unit pathfind.Tuning is expressed in).
    weights = np.minimum(grid[ys, xs] / 0.35, 1.0) * CELL_WEIGHT
    pts = np.stack([(xs + 0.5) * c, (ys + 0.5) * c, weights], axis=1).astype(np.float32)
    # Report in reference (960x540) pixels so movement tuning is resolution-free.
    pts[:, 0] *= REF_W / w
    pts[:, 1] *= REF_H / h
    return pts


def _boxes(mask: np.ndarray, min_area: int, limit: int = 64) -> list[Detection]:
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        out.append(Detection(int(x), int(y), int(bw), int(bh)))
    return out[:limit] if len(out) > limit else out


def detect_threat_blobs(
    frame: np.ndarray, player: tuple[int, int], cfg: Config, mask: np.ndarray | None = None
) -> list[Detection]:
    """Enemy sprites (dark-pixel components) sorted nearest-first, HUD/player masked."""
    if mask is None:
        mask = enemy_mask(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merged = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    dets = _boxes(merged, min_area=cfg.min_blob_area, limit=10_000)
    px, py = player
    dets.sort(key=lambda d: (d.cx - px) ** 2 + (d.cy - py) ** 2)
    return dets[:64]


def detect_gems(frame: np.ndarray, enemies: np.ndarray | None = None) -> list[tuple[int, int]]:
    """
    Centers of blue XP gems, in frame pixels.

    Real gems are tiny saturated cores (area ~2–20 at 960x540). The Glowing
    Bat's blue outline is larger, redder, and rings a dark body — rejected by
    color (``gem_mask``), max area, aspect, and dark-fill checks. When
    ``enemies`` (from enemy_mask) is given, a leftover candidate sitting in a
    lot of dark pixels is also dropped, and real gems get their thin dark
    outline erased from ``enemies`` in place.
    """
    m = gem_mask(frame)
    h, w = frame.shape[:2]
    scale = (w * h) / (REF_W * REF_H)
    max_area = max(8, int(round(40 * scale)))
    n, _l, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    rx = max(4, int(round(12 * w / REF_W)))
    ry = max(4, int(round(14 * h / REF_H)))
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not 2 <= area <= max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = bw / float(max(bh, 1))
        if aspect < 0.4 or aspect > 2.2:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        # Hollow ring around a dark bat body (works even without enemies mask).
        roi_mx = frame[by : by + bh, bx : bx + bw].max(axis=2)
        dark_frac = float((roi_mx < DARK_MAX).sum()) / float(max(1, bw * bh))
        if dark_frac > 0.12:
            continue
        cx, cy = int(cents[i][0]), int(cents[i][1])
        if enemies is not None:
            y0, y1, x0, x1 = max(0, cy - ry), cy + ry, max(0, cx - rx), cx + rx
            dark = int(np.count_nonzero(enemies[y0:y1, x0:x1]))
            if dark > 70 * scale:
                continue
            enemies[y0:y1, x0:x1] = False
        out.append((cx, cy))
    return out


def _hue_gem_mask(frame: np.ndarray, color: str) -> np.ndarray:
    """Bright rim of a green or red XP gem.

    A real blue core is exactly BGR (228, 180, 38): one channel high, one
    mid, one low. The red crystal is that sprite with the hue moved by about
    +63 (OpenCV), BGR (156, 38, 225): red high, blue still the mid channel,
    green low. The green crystal is the same sprite the other way, BGR
    (168, 228, 38), HSV (81, 212, 228): green high, blue still mid, red low.
    A pure red or pure green pixel (both other channels near 0) is an enemy,
    a damage streak, or a spark, not a gem. Mantichana greens are dimmer and
    muddier, about (87, 196, 64), and fail the gaps. Orange (the winged
    heart, red flowers, Floor Chicken) has green well above blue.
    """
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    if color == "red":
        # Real rim BGR (156, 38, 225). Blue stays the mid channel (130-210).
        m = (r >= 200) & (b >= 130) & (b <= 210) & (g <= 90) & (r > b) & (r - g >= 140) & (b - g >= 70)
    else:
        # Real rim BGR (168, 228, 38). Blue stays the mid channel, as on red.
        m = (g >= 200) & (b >= 130) & (b <= 210) & (r <= 90) & (g > b) & (g - r >= 140) & (b - r >= 70)
    h, w = frame.shape[:2]
    ignore, side = _masks(w, h)
    m[ignore | side] = False
    return m


def _loose_body(frame: np.ndarray, color: str) -> np.ndarray:
    """Wider red/green than the rim, so a heart, flower, or splatter stays one blob."""
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    if color == "red":
        # Pure-red sprite body, and the crystal rim itself. The rim's blue
        # channel is ~156, so r - b is only ~70 and the pure-red test misses it.
        # Including it makes a large magenta sprite one blob.
        crystal = (r >= 200) & (b >= 130) & (g <= 90) & (r > b)
        m = ((r >= 160) & (r >= g + 80) & (r >= b + 80)) | crystal
    else:
        # The crystal's blue is ~168, so g - b is only ~60 and a pure-green
        # test misses it. Including it keeps a large green sprite one blob.
        crystal = (g >= 200) & (b >= 130) & (r <= 90) & (g > b)
        m = ((g >= 160) & (g >= r + 80) & (g >= b + 80)) | crystal
    h, w = frame.shape[:2]
    ignore, side = _masks(w, h)
    m[ignore | side] = False
    return m


def _hue_gems(frame: np.ndarray, color: str, enemies: np.ndarray | None) -> list[tuple[int, int]]:
    """Green or red gem centers. Same idea as ``detect_gems``, stricter shape.

    The rim has to be gem-sized (about 6x8, area >= 12 at 960x540), not a
    hollow dark body, and not a piece of a larger red/green sprite. A dark
    enemy under the candidate is rejected the same way a blue gem is.
    """
    m = _hue_gem_mask(frame, color)
    h, w = frame.shape[:2]
    scale = (w * h) / (REF_W * REF_H)
    sx, sy = w / float(REF_W), h / float(REF_H)
    max_area = max(8, int(round(40 * scale)))
    bw_min = max(3, int(round(4 * sx)))
    bw_max = max(bw_min, int(round(10 * sx)))
    bh_max = max(5, int(round(12 * sy)))
    if color == "red":
        # Real red rims are often the short half of the crystal (about 5x4,
        # area 8-18). Green stays on the taller gate: a 4x4 leaf speck is not a gem.
        min_area = max(6, int(round(8 * scale)))
        bh_min = max(3, int(round(4 * sy)))
        dark_limit = 0.30
    else:
        # Real greens are the full 6x8 crystal (area 14-17, dark fill about
        # 0.04-0.13), not the short red rim. A 4x4 leaf still fails the colour.
        min_area = max(8, int(round(12 * scale)))
        bh_min = max(5, int(round(7 * sy)))
        dark_limit = 0.16
    slack = max(4, int(round(8 * scale)))
    loose = _loose_body(frame, color).astype(np.uint8)
    _ln, llabels, lstats, _lc = cv2.connectedComponentsWithStats(loose, connectivity=8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    rx = max(4, int(round(12 * w / REF_W)))
    ry = max(4, int(round(14 * h / REF_H)))
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < bw_min or bw > bw_max or bh < bh_min or bh > bh_max:
            continue
        aspect = bw / float(max(bh, 1))
        if aspect < 0.4 or aspect > 2.2:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        roi_mx = frame[by : by + bh, bx : bx + bw].max(axis=2)
        dark_frac = float((roi_mx < DARK_MAX).sum()) / float(max(1, bw * bh))
        # A red rim is a thin arc, so the bbox holds more of the dark outline
        # than a blue core does (measured about 0.15-0.22 on the real crystals).
        if dark_frac > dark_limit:
            continue
        touched = np.unique(llabels[labels == i])
        touched = touched[touched > 0]
        body = int(sum(int(lstats[j, cv2.CC_STAT_AREA]) for j in touched))
        if body > area + slack:
            continue
        cx, cy = int(cents[i][0]), int(cents[i][1])
        if enemies is not None:
            y0, y1, x0, x1 = max(0, cy - ry), cy + ry, max(0, cx - rx), cx + rx
            dark = int(np.count_nonzero(enemies[y0:y1, x0:x1]))
            # Red crystals drop inside the pack that just died. Their own dark
            # outline plus that pack is well over the blue-gem limit (measured
            # 74-180 vs 70), and this palette is not the Red Death wing.
            # Red and green crystals drop inside the pack that just died.
            # Their outline plus that pack is over the blue-gem limit of 70
            # (red measured 74-180, green up to about 100).
            limit = 400 * scale if color in ("red", "green") else 70 * scale
            if dark > limit:
                continue
            enemies[y0:y1, x0:x1] = False
        out.append((cx, cy))
    return out


def detect_colored_gems(
    frame: np.ndarray, enemies: np.ndarray | None = None
) -> list[tuple[int, int, str]]:
    """XP gem centers and colour. Blue uses ``detect_gems`` unchanged.

    Green and red have to pass the gem shape gate above. A pixel that is not
    one of those rims is not given a colour here; callers treat a missing
    colour as blue.
    """
    found = [(x, y, "blue") for x, y in detect_gems(frame, enemies)]
    found.extend((x, y, "red") for x, y in _hue_gems(frame, "red", enemies))
    found.extend((x, y, "green") for x, y in _hue_gems(frame, "green", enemies))
    return found


def _yellow_core(frame: np.ndarray) -> np.ndarray:
    """Saturated yellow/gold (chest arrow, coin, chest lock). Grass and the XP bar fail.

    Coins on real Mad Forest frames peak near BGR (48, 198, 254); the XP bar's
    brightest pixels are bluer (b ~ 102). ``b < 80`` keeps the coins and a
    chest arrow and drops the bar. Coins are tiny; the arrow is not.
    """
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    return (r > 210) & (g > 165) & (b < 80) & (g > b + 90) & (r > b + 120) & (r + 15 > g) & (r < g + 100)


def _chest_wood(frame: np.ndarray) -> np.ndarray:
    """Brown wooden chest body: redder than the dirt path (dirt has r ≈ g)."""
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    return (r > 85) & (r < 200) & (r > g + 22) & (g > b + 8) & (b < 120) & (g > 50) & (g < 170)


def _pale_yellow(frame: np.ndarray) -> np.ndarray:
    """Pale UI yellow of the real chest chevron (edge arrow and on-screen marker).

    Live markers are BGR about (140-172, 254, 230-254). ``_yellow_core`` only
    keeps saturated gold (b < 80), which matches coins and the painted fixture
    arrow and misses every real one.
    """
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    return (r > 200) & (g > 220) & (b > 110) & (b < 200) & (g + 5 >= r) & (r + 40 > g) & (g > b + 40)


def _playfield_edge_distance(cx: float, cy: float, w: int, h: int) -> float:
    """Pixels from ``(cx, cy)`` in to the nearest playfield border.

    Negative when the point sits in the side bars, outside the playfield.
    """
    x0 = PLAYFIELD_X0 * w / float(REF_W)
    x1 = PLAYFIELD_X1 * w / float(REF_W)
    return min(cx - x0, x1 - cx, cy, (h - 1) - cy)


def _chest_under_marker(wood: np.ndarray, cx: float, cy: float, s: float) -> tuple[int, int] | None:
    """Wooden chest body just below a pale chevron, or None."""
    h, w = wood.shape[:2]
    y0 = min(h, int(cy) + 2)
    y1 = min(h, int(cy) + int(round(42 * s)))
    x0 = max(0, int(cx) - int(round(22 * s)))
    x1 = min(w, int(cx) + int(round(22 * s)))
    if y1 <= y0 or x1 <= x0:
        return None
    win = wood[y0:y1, x0:x1]
    if int(win.sum()) < 50 * s * s:
        return None
    ys, xs = np.nonzero(win)
    width = int(xs.max() - xs.min() + 1)
    if width < 12 * s or width > 30 * s:
        return None
    return (int(round(x0 + float(xs.mean()))), int(round(y0 + float(ys.mean()))))


def _pale_markers(frame: np.ndarray) -> list[tuple[int, int, float, float, tuple[int, int] | None]]:
    """Pale chevrons: (cx, cy, solidity, edge distance, chest center or None).

    A chevron with a wooden body under it is an on-screen chest. The same shape
    on the playfield edge, with no body, is the off-screen arrow.
    """
    h, w = frame.shape[:2]
    pale = _pale_yellow(frame)
    ignore, _side = _masks(w, h)
    pale[ignore] = False
    s = h / float(REF_H)
    wood = _chest_wood(frame)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(pale.astype(np.uint8), connectivity=8)
    out: list[tuple[int, int, float, float, tuple[int, int] | None]] = []
    min_area = 100 * s * s
    max_area = 320 * s * s
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short, long = min(bw, bh), max(bw, bh)
        if short < 8 * s or long > 42 * s or long / float(short) > 3.4:
            continue
        if area / float(bw * bh) < 0.28:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _hier = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        solidity = area / float(cv2.contourArea(hull) or 1.0)
        if solidity < 0.65:
            continue
        pix = frame[labels == i]
        med_b = float(np.median(pix[:, 0]))
        med_g = float(np.median(pix[:, 1]))
        if med_g < 250 or not (120.0 <= med_b <= 190.0):
            continue
        cx = float(cents[i][0])
        cy = float(cents[i][1])
        dist = _playfield_edge_distance(cx, cy, w, h)
        chest = _chest_under_marker(wood, cx, cy, s)
        out.append((int(round(cx)), int(round(cy)), solidity, dist, chest))
    return out


def _chest_icon_pixels(frame: np.ndarray, cx: int, cy: int, s: float) -> int:
    """Near-white chest glyph in the chevron. The yellow outline itself is darker blue.

    Real chest arrows carry that glyph (about 55–120 px at 960x540). A bare pale
    streak, or a red heart pickup beside a chevron, does not.
    """
    h, w = frame.shape[:2]
    rad = max(4, int(round(22 * s)))
    x0, y0 = max(0, int(cx) - rad), max(0, int(cy) - rad)
    x1, y1 = min(w, int(cx) + rad + 1), min(h, int(cy) + rad + 1)
    win = frame[y0:y1, x0:x1]
    b = win[:, :, 0].astype(np.int16)
    g = win[:, :, 1].astype(np.int16)
    r = win[:, :, 2].astype(np.int16)
    # Pale yellow tops out under b=200. The chest icon sits at b~224, g~254, r~254.
    return int(((r > 190) & (g > 200) & (b >= 200)).sum())


def detect_chest_arrow(frame: np.ndarray) -> tuple[int, int] | None:
    """Centroid of the yellow off-screen chest arrow, in frame pixels.

    The arrow sits inside the playfield, near an edge, and points at a treasure
    chest that is still off-screen. Painted fixtures are saturated gold in a
    tight frame-edge band. Real arrows are pale yellow and sit 30-110 px inside
    the playfield (the side bars push left arrows past a 56 px frame band).
    None when no arrow is visible.
    """
    h, w = frame.shape[:2]
    yellow = _yellow_core(frame)
    ignore, _side = _masks(w, h)
    yellow[ignore] = False
    s = h / float(REF_H)
    band = max(8, int(round(56 * s)))
    edge = np.zeros((h, w), dtype=bool)
    edge[:band, :] = True
    edge[-band:, :] = True
    edge[:, :band] = True
    edge[:, -band:] = True
    yellow &= edge
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(yellow.astype(np.uint8), connectivity=8)
    best: tuple[int, int] | None = None
    best_area = 0
    min_area = 70 * s * s
    max_area = 1400 * s * s
    min_side = 11 * s
    max_side = 72 * s
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short, long = min(bw, bh), max(bw, bh)
        if short < min_side or long > max_side or long / float(short) > 2.8:
            continue
        if area / float(bw * bh) < 0.34:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        if min(cx, cy, (w - 1) - cx, (h - 1) - cy) > 42 * s:
            continue
        if area > best_area:
            best_area = area
            best = (int(round(cx)), int(round(cy)))
    if best is not None:
        return best
    # Real chevron: pale yellow, within ~120 px of the playfield border, and no
    # wooden chest under it (that marker belongs to detect_chest). The most
    # solid blob wins, so a nearby projectile smear loses to the UI arrow.
    band = 120 * s
    pale_sol = -1.0
    # Other off-screen pickups use a pale edge chevron too. The boss chest is
    # the one with a near-white chest glyph inside the chevron.
    icon_min = 40 * s * s
    for cx, cy, sol, dist, chest in _pale_markers(frame):
        if chest is not None or dist < 0 or dist > band:
            continue
        if _chest_icon_pixels(frame, cx, cy, s) < icon_min:
            continue
        if sol > pale_sol:
            pale_sol = sol
            best = (cx, cy)
    return best


def detect_chest(frame: np.ndarray) -> tuple[int, int] | None:
    """Center of an on-screen treasure chest, in frame pixels.

    A chest is a compact brown wooden body with a bright gold lock/band.
    Gold coins are the same yellow but sit on grass, so the wood test drops
    them. None when nothing on screen looks like a chest. The nearest wins.
    """
    h, w = frame.shape[:2]
    wood = _chest_wood(frame)
    gold = _yellow_core(frame)
    ignore, side = _masks(w, h)
    wood[ignore | side] = False
    gold[ignore | side] = False
    s = h / float(REF_H)
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(gold.astype(np.uint8), connectivity=8)
    px, py = player_position(w, h)
    best: tuple[int, int] | None = None
    best_d = 1e18
    pad = int(round(16 * s))
    # Coins on the Mad Forest fixtures top out near 32 yellow pixels. A chest
    # lock+band is several times that; the edge arrow is larger still and is
    # not a chest (it has no wooden body).
    min_gold = 56 * s * s
    max_gold = 400 * s * s
    min_wood = 140 * s * s
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_gold or area > max_gold:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short = min(bw, bh)
        if short < 3 * s or max(bw, bh) / float(max(short, 1)) > 3.2:
            continue
        gx, gy = int(round(cents[i][0])), int(round(cents[i][1]))
        y0, y1 = max(0, gy - pad), min(h, gy + pad)
        x0, x1 = max(0, gx - pad), min(w, gx + pad)
        win = wood[y0:y1, x0:x1]
        wood_n = int(win.sum())
        if wood_n < min_wood:
            continue
        ys, xs = np.nonzero(win)
        wx0, wx1 = int(xs.min()), int(xs.max())
        wy0, wy1 = int(ys.min()), int(ys.max())
        ww, hh = wx1 - wx0 + 1, wy1 - wy0 + 1
        if not (16 * s <= ww <= 56 * s and 14 * s <= hh <= 52 * s):
            continue
        if wood_n / float(ww * hh) < 0.28:
            continue
        # The lock sits on the body, not beside a tree the window happened to clip.
        if not (wx0 - 4 <= gx - x0 <= wx1 + 4 and wy0 - 4 <= gy - y0 <= wy1 + 4):
            continue
        # A chest's gold is a horizontal band across the wooden body. A Fire Wand
        # fireball is a yellow core inside an orange halo: same colors, but the
        # gold does not span the halo the way a lock spans a chest.
        if bw < bh * 1.6 or bw < 0.65 * ww or bh > 0.65 * hh:
            continue
        cx = x0 + (wx0 + wx1) // 2
        cy = y0 + (wy0 + wy1) // 2
        if float(np.hypot(cx - gx, cy - gy)) > 14 * s:
            continue
        d = (cx - px) ** 2 + (cy - py) ** 2
        if d < best_d:
            best_d = d
            best = (int(cx), int(cy))
    if best is not None:
        return best
    # Real chest: the pale chevron floats ~28 px above a small wooden body.
    # The body center is the seek point. Saturated gold-on-wood still wins above
    # when a painted fixture matches, so coins and the Fire Wand stay rejected.
    for _cx, _cy, _sol, _dist, chest in _pale_markers(frame):
        if chest is None:
            continue
        d = (chest[0] - px) ** 2 + (chest[1] - py) ** 2
        if d < best_d:
            best_d = d
            best = chest
    return best


def parse_timer_text(text: str) -> float | None:
    """Parse a stage-timer string like ``3:45`` / ``12:00`` / ``01:41`` into seconds."""
    if not text:
        return None
    # The digit whitelist can turn the colon into a digit ("06313" = 06:13).
    m = re.search(r"(\d{1,2}):(\d{2})", text.strip()) or re.fullmatch(r"(\d{2})\d(\d{2})", text.strip())
    if not m:
        return None
    minutes, seconds = int(m.group(1)), int(m.group(2))
    if seconds > 59 or minutes > 59:
        return None
    return float(minutes * 60 + seconds)


def _timer_ocr_image(img: np.ndarray) -> str:
    """Tesseract digit+colon whitelist over a white-text crop (menus-style)."""
    if img is None or img.size == 0 or shutil.which("tesseract") is None:
        return ""
    # Bright HUD glyphs -> black-on-white, upscaled (same idea as menus._white_text).
    bright = img.min(axis=2) if img.ndim == 3 else img
    binary = np.where(bright > 190, 0, 255).astype(np.uint8)
    binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    binary = cv2.GaussianBlur(binary, (3, 3), 0)
    binary = cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", binary)
    if not ok:
        return ""
    try:
        out = subprocess.run(
            [
                "tesseract", "stdin", "stdout",
                "--psm", "7",
                "-c", "tessedit_char_whitelist=0123456789:",
            ],
            input=buf.tobytes(),
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[vision] timer OCR failed: {exc}", file=sys.stderr)
        return ""
    return out.stdout.decode("utf-8", "replace").strip()


def read_xp_fraction(frame: np.ndarray) -> float | None:
    """Filled fraction of the top XP bar, 0..1.

    The bar is a blue run starting at the left of the track (x=51 at 960-wide).
    None when that run is not there (menus, results). A level-up resets the fill,
    so a later drop is not a pickup.
    """
    h, w = frame.shape[:2]
    sx, sy = w / float(REF_W), h / float(REF_H)
    x0, x1 = int(round(XP_BAR_X0 * sx)), int(round(XP_BAR_X1 * sx))
    y0, y1 = int(round(XP_BAR_Y0 * sy)), int(round(XP_BAR_Y1 * sy))
    if x1 <= x0 + 8 or y1 <= y0:
        return None
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    blue = (b > 160) & (b > g + 40) & (b > r + 40)
    col = blue.mean(axis=0) >= 0.5
    if int(col.sum()) < 8 or not bool(col[:6].any()):
        return None
    xs = np.flatnonzero(col)
    if int(xs[0]) > 4:
        return None
    return float(int(xs[-1]) + 1) / float(x1 - x0)


def read_timer_seconds(frame: np.ndarray) -> float | None:
    """OCR the on-screen Mad Forest timer; None when the reading is not confident."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _scaled_rect(TIMER_RECT, w, h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return parse_timer_text(_timer_ocr_image(crop))


# Real level-up / pause / game-over overlays leave at most ~40% of the center
# showing grass (level-up fixture 0.35, GAME OVER ~0.04); Mad Forest gameplay
# is 0.78-0.99 even inside a flower ring.
MENU_GRASS_MAX = 0.6


def playfield_grass_fraction(region: np.ndarray) -> float:
    """Fraction of pixels in ``region`` (BGR) that look like Mad Forest grass."""
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hue >= 35) & (hue <= 85) & (sat >= 60) & (val >= 45)
    return float(grass.mean())


def detect_level_up_menu(frame: np.ndarray) -> bool:
    """
    Level-up / pause cards sit as a few large rounded rectangles on a dim overlay.
    Movement must stay off while those screens are up.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    center = gray[h // 5 : 4 * h // 5, w // 8 : 7 * w // 8]
    if center.size == 0:
        return False

    mean = float(np.mean(center))
    if mean > 145:
        return False
    if playfield_grass_fraction(frame[h // 5 : 4 * h // 5, w // 8 : 7 * w // 8]) >= MENU_GRASS_MAX:
        # Open Mad Forest grass in the middle: this is gameplay, not a card
        # overlay. Dense Man-eater flower rings (~5:00) otherwise produce big
        # dilated edge blobs that count as "cards", and the bot froze in place
        # for seconds right before most deaths (Oct 4 blackboxes: ~60% of
        # pre-death gameplay frames were flagged as a menu).
        return False

    blur = cv2.GaussianBlur(center, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = 0
    ch, cw = center.shape[:2]
    min_area = (ch * cw) * 0.035
    max_area = (ch * cw) * 0.35
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 50 or bh < 70:
            continue
        ratio = bw / float(bh)
        if 0.35 <= ratio <= 1.8:
            cards += 1
    return cards >= 2


def draw_synthetic_arena(
    width: int = 960,
    height: int = 540,
    monsters: list[tuple[int, int]] | None = None,
    t: float = 0.0,
) -> np.ndarray:
    """Offline playfield for --demo so the loop can be exercised without Steam."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (34, 72, 44)  # Mad Forest grass (BGR)
    for gy in range(0, height, 48):
        cv2.line(frame, (0, gy), (width, gy), (40, 84, 52), 1)
    for gx in range(0, width, 48):
        cv2.line(frame, (gx, 0), (gx, height), (40, 84, 52), 1)

    px, py = player_position(width, height)
    cv2.circle(frame, (px, py), 10, (80, 220, 255), -1)
    cv2.circle(frame, (px, py), 14, (20, 80, 200), 2)

    if monsters is None:
        monsters = []
        for i, angle in enumerate((0.3, 1.1, 2.4, 3.7, 5.0)):
            r = 90 + 40 * (i % 3)
            mx = int(px + r * np.cos(angle + t * 0.4))
            my = int(py + r * np.sin(angle + t * 0.4))
            monsters.append((mx, my))

    for mx, my in monsters:
        # Dark bat-like sprite with a red eye, like the real Pipistrello.
        cv2.circle(frame, (mx, my), 12, (12, 12, 16), -1)
        cv2.circle(frame, (mx, my), 16, (5, 5, 5), 2)
        cv2.circle(frame, (mx, my), 2, (30, 30, 200), -1)
    return frame
