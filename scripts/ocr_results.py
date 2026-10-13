#!/usr/bin/env python3
import cv2, subprocess, re, sys


def ocr_text(im) -> str:
    h, w = im.shape[:2]
    # Focus on left stats panel
    crops = []
    for y0,y1,x0,x1 in [(30,250,20,380),(30,250,20,500),(40,280,300,700)]:
        c = im[y0:y1, x0:x1]
        g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3)
        _, th = cv2.threshold(g, 170, 255, cv2.THRESH_BINARY)
        crops.append(th)
    full = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
    _, fullth = cv2.threshold(full, 170, 255, cv2.THRESH_BINARY)
    crops.append(fullth)
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
    return {"survival": surv, "level": lvl, "kills": kills}


if __name__ == "__main__":
    txt = ocr_text(cv2.imread(sys.argv[1]))
    r = parse_stats(txt)
    out = f"survival={r['survival'] or '?'}\nlevel={r['level'] or '?'}\nkills={r['kills'] or '?'}\nraw={txt[:400].replace(chr(10),' | ')}"
    open(sys.argv[2], "w").write(out)
    print(out)
