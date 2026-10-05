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
    # Blue XP gem centers in reference pixels.
    gems: list[tuple[float, float]] = field(default_factory=list)
    # On-screen stage timer in seconds, when OCR is confident; else None.
    timer_s: float | None = None
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
        if self._model is not None:
            try:
                monsters = self._infer_model(frame)
                menu = detect_level_up_menu(frame)
                return VisionResult(
                    player=player, monsters=monsters, menu=menu, note="torch", timer_s=timer_s,
                )
            except Exception as exc:  # noqa: BLE001
                self._model = None
                print(f"[vision] model failed, falling back to heuristic: {exc}", file=sys.stderr)
        mask = enemy_mask(frame)
        sx, sy = REF_W / w, REF_H / h
        gems = [(gx * sx, gy * sy) for gx, gy in detect_gems(frame, mask)]
        monsters = detect_threat_blobs(frame, player, self.cfg, mask=mask)
        threats = threat_points(mask)
        menu = detect_level_up_menu(frame)
        return VisionResult(
            player=player, monsters=monsters, menu=menu, note="heuristic",
            threats=threats, gems=gems, timer_s=timer_s, size=(w, h),
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
