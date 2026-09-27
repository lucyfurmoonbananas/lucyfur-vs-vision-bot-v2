"""enemy_mask speed-up regression: the bbox-cropped per-component filter must
match the straightforward whole-frame implementation pixel for pixel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vs_vision_bot.vision import (
    DARK_MAX,
    MIN_DARK_AREA,
    REF_H,
    REF_W,
    SIDE_DARK_MAX,
    _masks,
    enemy_mask,
)

FIX = Path(__file__).parent / "fixtures"


def _reference_enemy_mask(frame: np.ndarray) -> np.ndarray:
    """Whole-frame per-component version of enemy_mask (8785d4e), kept as the oracle."""
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
        comp = labels == i
        pix = frame[comp]
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
        # Small thin rings hugging lime leaves or purple berries / flowers.
        if area < int(100 * scale) and max_t <= 2.0:
            pad = 4
            y0, y1 = max(0, y - pad), min(h, y + bh + pad)
            x0, x1 = max(0, x - pad), min(w, x + bw + pad)
            band = np.zeros((h, w), dtype=bool)
            band[y0:y1, x0:x1] = True
            band[comp] = False
            if int(lime[band].sum()) >= 40 or int(purple[band].sum()) >= 8:
                continue
        keep[i] = True
    return keep[labels]



def _load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIX / name))
    assert frame is not None, name
    return frame


def _variants(frame: np.ndarray):
    yield "native", frame
    yield "flip", np.ascontiguousarray(frame[::-1, ::-1])  # comps on other edges
    yield "1280", cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
    yield "640", cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize("name", sorted(p.name for p in FIX.glob("*.png")))
def test_enemy_mask_matches_reference(name):
    for tag, frame in _variants(_load(name)):
        got = enemy_mask(frame)
        want = _reference_enemy_mask(frame)
        assert got.dtype == want.dtype and got.shape == want.shape, (name, tag)
        assert np.array_equal(got, want), (name, tag, int((got != want).sum()))


@pytest.mark.parametrize("name", ["gameplay_skeleton_swarm.png", "gameplay_hurt.png"])
def test_enemy_mask_nonempty_on_busy_frames(name):
    # Guard against a vacuous comparison: these frames must yield enemies.
    assert int(enemy_mask(_load(name)).sum()) > 1000
