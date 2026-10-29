#!/usr/bin/env python3
import cv2, subprocess, re, sys

import numpy as np

# (y0, y1, x0, x1). Survival / level / kills stay on the gray>170 crops.
STAT_CROPS = ((30, 250, 20, 380), (30, 250, 20, 500), (40, 280, 300, 700))
# Gold earned is right-aligned near x=440-490. A crop that ends at x=380
# keeps only a sliver of the first digit ("3", "0", "35").
GOLD_CROP = (70, 220, 20, 560)
# Antialiased gold strokes sit under gray 170. The max channel still clears this.
GOLD_THRESH = 145


def _gray_mask(im, y0, y1, x0, x1, thresh, scale):
    c = im[y0:y1, x0:x1]
    g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
    _, th = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    return th


def _max_mask(im, y0, y1, x0, x1, thresh, scale):
    c = im[y0:y1, x0:x1]
    mx = np.maximum(np.maximum(c[:, :, 0], c[:, :, 1]), c[:, :, 2])
    mx = cv2.resize(mx, None, fx=scale, fy=scale)
    _, th = cv2.threshold(mx, thresh, 255, cv2.THRESH_BINARY)
    return th


def ocr_text(im) -> str:
    h, w = im.shape[:2]
    crops = []
    for y0, y1, x0, x1 in STAT_CROPS:
        crops.append(_gray_mask(im, y0, y1, min(x0, w), min(x1, w), 170, 3))
    full = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
    _, fullth = cv2.threshold(full, 170, 255, cv2.THRESH_BINARY)
    crops.append(fullth)
    gy0, gy1, gx0, gx1 = GOLD_CROP
    crops.append(_max_mask(im, gy0, min(gy1, h), gx0, min(gx1, w), GOLD_THRESH, 3))
    texts = []
    for i, c in enumerate(crops):
        cv2.imwrite(f"/tmp/ocr_c{i}.png", c)
        r = subprocess.run(["tesseract", f"/tmp/ocr_c{i}.png", "stdout", "--psm", "6"], capture_output=True, text=True)
        texts.append(r.stdout)
    txt = "\n".join(texts)
    return txt


def parse_stats(txt: str) -> dict:
    surv = None
    for m in re.finditer(r"(?:Survived|Survived:)\s*(\d{1,2})[:.](\d{2})", txt, re.I):
        surv = f"{int(m.group(1)):02d}:{m.group(2)}"
    if not surv:
        for m in re.finditer(r"\b(\d{1,2})[:.](\d{2})\b", txt):
            # prefer first mm:ss under 30 min
            a,b = int(m.group(1)), m.group(2)
            if a <= 30:
                surv = f"{a:02d}:{b}"; break
    lvl = None
    for m in re.finditer(r"Level\s*reached:\s*(\d{1,2})", txt, re.I):
        lvl = m.group(1); break
    if not lvl:
        for m in re.finditer(r"(?:Level|reached|Neve|Teve)\D{0,12}(\d{1,2})\b", txt, re.I):
            lvl = m.group(1)
    kills = None
    # First "Enemies defeated: N" on the same line wins (like level); "$27" misreads never match.
    for m in re.finditer(r"Enemies\s+defeated:?[ \t]*(\d{1,6})\b", txt, re.I):
        kills = m.group(1); break
    if not kills:
        for m in re.finditer(r"(?:defeated|Enemies|Frem)[^\d\n]{0,12}(\d{1,5})\b", txt, re.I):
            kills = m.group(1); break
    # "Gold x 1" is the stage multiplier, not earned gold. A clipped crop
    # reads "Gold earned: 3" beside the full "Gold earned: 170"; the longer
    # digit run is the value column. A real 0 stays 0.
    earned = re.findall(r"Gold\s+earned:?[ \t]*(\d+)\b", txt, re.I)
    gold = max(earned, key=lambda s: (len(s), int(s))) if earned else None
    return {"survival": surv, "level": lvl, "kills": kills, "gold": gold}


if __name__ == "__main__":
    txt = ocr_text(cv2.imread(sys.argv[1]))
    r = parse_stats(txt)
    out = (
        f"survival={r['survival'] or '?'}\nlevel={r['level'] or '?'}\n"
        f"kills={r['kills'] or '?'}\ngold={r['gold'] if r['gold'] is not None else '?'}\n"
        f"raw={txt[:400].replace(chr(10),' | ')}"
    )
    open(sys.argv[2], "w").write(out)
    print(out)
