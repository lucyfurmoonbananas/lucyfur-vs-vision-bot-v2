"""scripts/run_metrics.py: early-levelling metrics for the early-engage A/B."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_metrics import format_metrics, parse_metrics  # noqa: E402

LOG = """\
[nav] game~    0s kite/wide-circles v=(1, 0)
[levelup] cards=['Cross', 'Whip', 'Knife'] new=[True, True, True] -> pick #1 Cross (hp=1.0, weapons=1, game~=78s)
[levelup] picked #1 Cross (total picks 1)
[levelup] cards=['Whip'] new=[True] -> pick #1 Whip (hp=1.0, weapons=2, game~=130s)
[levelup] picked #1 Whip (total picks 2)
[levelup] cards=['Garlic'] new=[False] -> pick #1 Garlic (hp=1.0, weapons=2, game~=240s)
[levelup] picked #1 Garlic (total picks 3)
[levelup] cards=['Armor'] new=[True] -> pick #1 Armor (hp=1.0, weapons=2, game~=250s)
[nav] game~  321s ring-escape/wide-circles v=(0, 1)
[revive] clicked REVIVE at (479, 467) ('~ REVIVE -')
[nav] game~  400s kite
[quit] clicked QUIT at (479, 372) ('QUIT .')
"""


def test_levels_count_only_picked_cards_at_or_before_each_mark():
    m = parse_metrics(LOG.splitlines())
    assert m["first_levelup_s"] == 78
    assert m["picks"] == 3  # the Armor offer at 250s was never picked
    assert (m["lv_at_120"], m["lv_at_240"], m["lv_at_300"]) == (2, 4, 4)


def test_first_death_is_last_nav_before_first_revive():
    assert parse_metrics(LOG.splitlines())["first_death_s"] == 321


def test_empty_log_gives_na_and_level_1():
    m = parse_metrics([])
    assert m["first_levelup_s"] is None and m["first_death_s"] is None
    assert m["lv_at_300"] == 1
    assert "first_levelup_s=NA" in format_metrics(m)


def test_cli_prints_key_values(tmp_path):
    p = tmp_path / "run1.log"
    p.write_text(LOG)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_metrics.py"), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "first_levelup_s=78" in out and "lv_at_240=4" in out
