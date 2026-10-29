"""scripts/ocr_results.py: results-screen stats parse (Oct 12 day run1 read 27 instead of 527)."""
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ocr_results", ROOT / "scripts" / "ocr_results.py")
ocr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ocr)

RAW = ("Survived: | Level reached: | Enemies defeated: \n"
       "Survived: 06:51\nGold earned: 170\nLevel reached: 7\nEnemies defeated: 527\n"
       "Enemies defeated: $27\nGarlic 2 2.7k 6:51 6.4\n")


def test_first_kill_match_wins_over_dollar_misread():
    assert ocr.parse_stats(RAW) == {"survival": "06:51", "level": "7", "kills": "527", "gold": "170"}


def test_gold_prefers_the_full_earned_number_over_a_clipped_fragment():
    """test10 read Gold earned as 3, 0 and 35. The value column is the long number.

    "Gold x 1" is the stage coin multiplier and is not earned gold.
    """
    clipped = RAW + "Gold earned: 3\nGold earned: 0\nGold earned: 35\nGold x 1\n"
    assert ocr.parse_stats(clipped)["gold"] == "170"
    stage = "Survived: 02:20\nLevel reached: 5\nEnemies defeated: 271\nGold x 1\n"
    assert ocr.parse_stats(stage)["gold"] is None
    zero = "Survived: 02:20\nGold earned: 0\nLevel reached: 4\nEnemies defeated: 271\nGold x 1\n"
    assert ocr.parse_stats(zero)["gold"] == "0"


def test_test13_gold_earned_is_the_digit_six():
    """results_20261029_134442.png gold earned is 6, one glyph against the divider.

    The OCR line "Gold earned: 6" matches that glyph. A longer earned number
    still beats a one-digit fragment.
    """
    im = cv2.imread(str(ROOT / "tests" / "fixtures" / "results_test13_0733.png"))
    assert im is not None
    band = im[126:148, 300:490].max(axis=2)
    cols = np.where((band > 140).sum(axis=0) > 2)[0]
    assert cols.size and int(cols.max() - cols.min()) < 28
    assert ocr.parse_stats("Survived: 07:33\nGold earned: 6\nLevel reached: 15\nEnemies defeated: 1208\nGold x 1\n")["gold"] == "6"
    assert ocr.parse_stats("Gold earned: 3\nGold earned: 6\nGold x 1\n")["gold"] == "6"


def test_gold_crop_covers_the_value_column():
    """results_run1_527.png gold is 170, right-aligned. x=380 clips it; gray>170 drops the 0."""
    im = cv2.imread(str(ROOT / "tests" / "fixtures" / "results_run1_527.png"))
    assert im is not None and im.shape[1] == 960
    _y0, _y1, _x0, x1 = ocr.GOLD_CROP
    assert x1 >= 520
    # Right edge of the "0" (x=476-484, y=108-160): gray 170 sees nothing, max-channel does.
    col = im[108:160, 476:484]
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    mx = col.max(axis=2)
    assert int((gray > 170).sum()) == 0
    assert int((mx > ocr.GOLD_THRESH).sum()) > 0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_results_screen_reads_527():
    im = cv2.imread(str(ROOT / "tests" / "fixtures" / "results_run1_527.png"))
    assert ocr.parse_stats(ocr.ocr_text(im))["kills"] == "527"
