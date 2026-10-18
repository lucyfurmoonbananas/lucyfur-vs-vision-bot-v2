"""Death-trace: on REVIVE / QUIT, log the last ~5 s of HP, danger and mode and save frames."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from vs_vision_bot import main as m
from vs_vision_bot.main import DeathTrace

from test_main_watchdog import FakeCapture, FakeDebug, FakeHotkeys, FakeVision


def test_dump_on_revive_and_quit_logs_window_and_saves_frames(tmp_path):
    frames = [np.full((6, 6, 3), 20 * (i + 1), np.uint8) for i in range(4)]
    logs: list[str] = []
    tr = DeathTrace(str(tmp_path), log=logs.append)
    tr.observe(0.0, 0.99, 0.99, 0.01, "old/wide-circles", frames[0])
    tr.observe(2.0, 0.84, 0.90, 0.04, "kite/wide-circles", frames[1])
    tr.observe(5.0, 0.50, 0.84, 0.20, "kite/wide-circles", frames[2])
    tr.observe(6.2, None, 0.84, None, "revive", frames[3])
    paths = tr.dump("revive", frames[3], 6.2, game_s=72.0)
    text = "\n".join(logs)
    assert "[death-trace] revive game~72s n=3" in text
    assert "hp=0.99" not in text
    assert "hp=0.84" in text and "med=0.90" in text
    assert "danger=0.04" in text and "mode=kite/wide-circles" in text
    assert "hp=none" in text and "danger=none" in text and "mode=revive" in text
    assert len(paths) == 3
    assert [p.split("_")[-1] for p in paths] == ["00.png", "01.png", "02.png"]
    got = [int(cv2.imread(p)[0, 0, 0]) for p in paths]
    assert got == [frames[3][0, 0, 0], frames[2][0, 0, 0], frames[1][0, 0, 0]]

    logs.clear()
    tr.observe(7.0, 0.10, 0.10, 1.50, "quit", frames[0])
    qpaths = tr.dump("quit", frames[0], 7.0, game_s=90.4)
    qtext = "\n".join(logs)
    assert "[death-trace] quit game~90s" in qtext
    assert "hp=0.10" in qtext and "danger=1.50" in qtext and "mode=quit" in qtext
    assert qpaths and all("death_quit_" in p and p.endswith(".png") for p in qpaths)
    assert all(os.path.isfile(p) for p in qpaths)


class _DyingPilot:
    """Level-up and chest must not dump; REVIVE then QUIT must."""

    def __init__(self, **kw) -> None:
        self.hp = 0.84
        self.revives = 0
        self._quit_pressed = False
        self.finished = False
        self.status = ""
        self.calls = 0

    def step(self, frame, screen=None):
        self.calls += 1
        if self.calls == 1:
            return "levelup"
        if self.calls == 2:
            return "chest"
        if self.calls == 4:
            self.revives = 1
            return "revive"
        if self.calls == 7:
            self._quit_pressed = True
            return "quit"
        return None

    def reset_run(self):
        pass

    def new_run(self):
        pass


class _ThenFrozen(FakeCapture):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.n = 0

    def grab(self):
        self.n += 1
        if self.n < 12:
            return np.full((54, 96, 3), (self.n * 17) % 255, np.uint8)
        return self.frame


@pytest.fixture
def dying_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("VS_DRY_RUN", "1")
    monkeypatch.setenv("VS_AUTO_MENUS", "1")
    monkeypatch.setenv("VS_WATCHDOG_S", "0.3")
    monkeypatch.setenv("VS_FPS", "60")
    monkeypatch.setenv("VS_HOLD_MS", "1")
    monkeypatch.setenv("VS_SESSION", str(tmp_path))
    monkeypatch.setattr(m, "DebugWindow", FakeDebug)
    monkeypatch.setattr(m, "HotkeyService", FakeHotkeys)
    monkeypatch.setattr(m, "Vision", FakeVision)
    monkeypatch.setattr(m, "ScreenCapture", _ThenFrozen)
    monkeypatch.setattr(m, "MenuPilot", _DyingPilot)
    monkeypatch.setattr(m, "detect_level_up", lambda frame: None)
    monkeypatch.setattr(m, "tesseract_available", lambda: True)
    return tmp_path


def test_main_dumps_trace_on_revive_and_quit_not_on_menus(dying_loop, capsys):
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    lines = capsys.readouterr().out.splitlines()
    revives = [ln for ln in lines if ln.startswith("[death-trace] revive ")]
    quits = [ln for ln in lines if ln.startswith("[death-trace] quit ")]
    assert len(revives) == 1 and len(quits) == 1
    assert lines.index(revives[0]) < lines.index(quits[0])
    samples = [ln for ln in lines if ln.startswith("[death-trace] ") and "hp=" in ln]
    assert samples
    assert all("danger=" in ln and "mode=" in ln for ln in samples)
    revive = sorted((dying_loop / "deaths").glob("death_revive_*.png"))
    quit_frames = sorted((dying_loop / "deaths").glob("death_quit_*.png"))
    assert len(revive) == 3 and len(quit_frames) == 3
