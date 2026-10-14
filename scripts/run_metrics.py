#!/usr/bin/env python3
"""run_metrics.py RUN_LOG — early-levelling metrics from one bot run log (Bob's Oct 13 test plan).

Prints key=value lines:
  first_levelup_s  game~ seconds of the first picked level-up card (target <= 55)
  lv_at_120 / lv_at_240 / lv_at_300  1 + level-up picks made at or before that game~ second
  first_death_s    game~ of the last [nav] line before the first REVIVE/QUIT click
  picks            total level-up picks
Levels come from '[levelup] cards=... game~=NNs' lines followed by a '[levelup] picked' line,
so chest upgrades are not counted (they do not raise the player level). game~ runs a few
seconds ahead of the HUD clock, so these are slightly early, the same for both variants.
"""
from __future__ import annotations

import re
import sys

_CARDS = re.compile(r"^\[levelup\] cards=.*game~=(\d+)s")
_PICKED = re.compile(r"^\[levelup\] picked ")
_NAV = re.compile(r"^\[nav\] game~\s*(\d+)s")
_DEATH = re.compile(r"^\[(revive|quit)\] clicked (REVIVE|QUIT)")


def parse_metrics(lines) -> dict:
    picks: list[int] = []
    pending: int | None = None
    last_nav: int | None = None
    first_death: int | None = None
    for raw in lines:
        line = raw.strip()
        m = _CARDS.match(line)
        if m:
            pending = int(m.group(1))
            continue
        if _PICKED.match(line) and pending is not None:
            picks.append(pending)
            pending = None
            continue
        m = _NAV.match(line)
        if m:
            last_nav = int(m.group(1))
            continue
        if first_death is None and _DEATH.match(line):
            first_death = last_nav
    out: dict = {
        "first_levelup_s": picks[0] if picks else None,
        "picks": len(picks),
        "first_death_s": first_death,
    }
    for t in (120, 240, 300):
        out[f"lv_at_{t}"] = 1 + sum(1 for p in picks if p <= t)
    return out


def format_metrics(m: dict) -> str:
    return "\n".join(f"{k}={'NA' if v is None else v}" for k, v in m.items())


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: run_metrics.py RUN_LOG", file=sys.stderr)
        return 2
    with open(argv[1], errors="replace") as fh:
        print(format_metrics(parse_metrics(fh)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
