"""main(): the no-progress watchdog ends a run with exit status 3 (no game, no X needed)."""

import numpy as np
import pytest

from vs_vision_bot import main as m
from vs_vision_bot.capture import CaptureRegion
from vs_vision_bot.vision import VisionResult
from vs_vision_bot.watchdog import Stall


class FakeCapture:
    def __init__(self, cfg, fail: bool = False) -> None:
        self.region = CaptureRegion(0, 0, 96, 54)
        self.fail = fail
        self.frame = np.full((54, 96, 3), 80, np.uint8)  # a frozen game frame

    def grab(self):
        if self.fail:
            raise RuntimeError("XGetImage failed")
        return self.frame

    def align_to_mouse(self):
        return None

    def display_size(self):
        return (1920, 1080)


class FakeDebug:
    def __init__(self, cfg) -> None:
        pass

    def ensure(self, *a):
        pass

    def repark(self, *a):
        pass

    def render(self, *a, **k):
        pass

    def poll_key(self) -> int:
        return -1

    def close(self):
        pass


class FakeHotkeys:
    """Unpauses immediately, like play_run.sh pressing p."""

    def __init__(self, bus, on_change) -> None:
        self.bus, self.on_change = bus, on_change

    def start(self):
        self.bus.toggle_pause()
        self.on_change(False)

    def stop(self):
        pass

    def handle_waitkey(self, key):
        pass


class FakeVision:
    def __init__(self, cfg) -> None:
        pass

    def analyze(self, frame):
        return VisionResult(player=(48, 27), timer_s=None, size=(96, 54))


@pytest.fixture
def frozen_game(monkeypatch):
    monkeypatch.setenv("VS_DRY_RUN", "1")
    monkeypatch.setenv("VS_AUTO_MENUS", "0")
    monkeypatch.setenv("VS_WATCHDOG_S", "0.3")
    monkeypatch.setenv("VS_FPS", "60")
    monkeypatch.setattr(m, "DebugWindow", FakeDebug)
    monkeypatch.setattr(m, "HotkeyService", FakeHotkeys)
    monkeypatch.setattr(m, "Vision", FakeVision)
    return monkeypatch


def test_frozen_screen_ends_the_run_with_exit_3(frozen_game, capsys):
    frozen_game.setattr(m, "ScreenCapture", FakeCapture)
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    out = capsys.readouterr().out
    assert "[watchdog] STALL reason=frozen_frame" in out
    assert "keys released, goodbye" in out


def test_capture_that_keeps_failing_ends_the_run(frozen_game, capsys):
    frozen_game.setattr(m, "ScreenCapture", lambda cfg: FakeCapture(cfg, fail=True))
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    assert "[watchdog] STALL reason=no_frames" in capsys.readouterr().out


def test_blocked_loop_releases_keys_then_hard_exits(monkeypatch, capsys):
    released, exits = [], []
    monkeypatch.setattr(m.os, "_exit", exits.append)
    m._abort_blocked_loop(Stall("loop_blocked", 91.0), lambda: released.append(True))
    assert released == [True]
    assert exits == [m.EXIT_WATCHDOG]
    assert "reason=loop_blocked" in capsys.readouterr().out


def test_watchdog_env_knobs(monkeypatch):
    from vs_vision_bot.config import Config

    monkeypatch.setenv("VS_WATCHDOG_S", "45")
    monkeypatch.setenv("VS_LOOP_STALL_S", "0")
    cfg = Config.from_env()
    assert cfg.watchdog_s == 45.0 and cfg.loop_stall_s == 0.0
    monkeypatch.delenv("VS_WATCHDOG_S")
    monkeypatch.delenv("VS_LOOP_STALL_S")
    cfg = Config.from_env()
    assert cfg.watchdog_s == 60.0 and cfg.loop_stall_s == 90.0
