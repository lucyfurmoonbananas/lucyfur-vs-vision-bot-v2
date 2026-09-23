"""Monster / menu vision. Heuristic by default; optional TorchScript model path."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from vs_vision_bot.config import Config


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


class Vision:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        if cfg.model_path:
            self._model = self._load_torch_model(cfg.model_path)

    def analyze(self, frame: np.ndarray) -> VisionResult:
        h, w = frame.shape[:2]
        player = (w // 2, h // 2)
        if self._model is not None:
            try:
                monsters = self._infer_model(frame)
                menu = detect_level_up_menu(frame)
                return VisionResult(player=player, monsters=monsters, menu=menu, note="torch")
            except Exception as exc:  # noqa: BLE001
                self._model = None
                print(f"[vision] model failed, falling back to heuristic: {exc}", file=sys.stderr)
        monsters = detect_threat_blobs(frame, player, self.cfg)
        menu = detect_level_up_menu(frame)
        return VisionResult(player=player, monsters=monsters, menu=menu, note="heuristic")

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


def detect_threat_blobs(
    frame: np.ndarray, player: tuple[int, int], cfg: Config
) -> list[Detection]:
    """
    Find saturated / high-contrast sprites that are not the centered player.

    Vampire Survivors keeps the player near the camera center. Ground tiles are
    relatively muted; enemies and pickups pop in saturation.
    """
    h, w = frame.shape[:2]
    inset_x = max(8, w // 18)
    inset_y = max(8, h // 12)
    roi = frame[inset_y : h - inset_y, inset_x : w - inset_x]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Sprites pop against local tiles; a global sat cut lights up whole stages.
    blur = cv2.GaussianBlur(roi, (31, 31), 0)
    contrast = cv2.cvtColor(cv2.absdiff(roi, blur), cv2.COLOR_BGR2GRAY)
    contrast_mask = cv2.threshold(contrast, 26, 255, cv2.THRESH_BINARY)[1]

    sat_floor = max(95, int(np.median(sat) + 40))
    sat_mask = cv2.threshold(sat, sat_floor, 255, cv2.THRESH_BINARY)[1]
    val_floor = max(165, int(np.median(val) + 55))
    val_mask = cv2.threshold(val, val_floor, 255, cv2.THRESH_BINARY)[1]

    color_mask = cv2.bitwise_or(sat_mask, val_mask)
    combined = cv2.bitwise_and(contrast_mask, color_mask)
    if np.count_nonzero(combined) < 80:
        combined = contrast_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    px, py = player
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_blob_area or area > cfg.max_blob_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        x += inset_x
        y += inset_y
        cx, cy = x + bw // 2, y + bh // 2
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist < cfg.player_radius:
            continue
        # Ignore huge UI bars that hug the frame edge.
        if bh > h * 0.28 and bw > w * 0.35:
            continue
        detections.append(Detection(x, y, bw, bh))

    detections.sort(key=lambda d: (d.cx - px) ** 2 + (d.cy - py) ** 2)
    return detections[:48]


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
    frame[:, :] = (40, 40, 40)
    for gy in range(0, height, 48):
        cv2.line(frame, (0, gy), (width, gy), (50, 50, 50), 1)
    for gx in range(0, width, 48):
        cv2.line(frame, (gx, 0), (gx, height), (50, 50, 50), 1)

    px, py = width // 2, height // 2
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
        cv2.circle(frame, (mx, my), 12, (40, 40, 220), -1)
        cv2.circle(frame, (mx, my), 16, (20, 20, 160), 2)
    return frame
