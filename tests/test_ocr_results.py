"""scripts/ocr_results.py: results-screen stats parse (Oct 12 day run1 read 27 instead of 527)."""
import importlib.util
import shutil
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ocr_results", ROOT / "scripts" / "ocr_results.py")
ocr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ocr)

RAW = ("Survived: | Level reached: | Enemies defeated: \n"
       "Survived: 06:51\nGold earned: 170\nLevel reached: 7\nEnemies defeated: 527\n"
       "Enemies defeated: $27\nGarlic 2 2.7k 6:51 6.4\n")


def test_first_kill_match_wins_over_dollar_misread():
    assert ocr.parse_stats(RAW) == {"survival": "06:51", "level": "7", "kills": "527"}


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_results_screen_reads_527():
    im = cv2.imread(str(ROOT / "tests" / "fixtures" / "results_run1_527.png"))
    assert ocr.parse_stats(ocr.ocr_text(im))["kills"] == "527"
