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


class RevivingPilot:
    """Clicks REVIVE on the first frame, then reports plain gameplay."""

    def __init__(self, **kw) -> None:
        self.hp = 0.5
        self.revives = 0
        self._quit_pressed = False
        self.finished = False
        self.status = ""
        self.calls = 0

    def step(self, frame, screen=None):
        self.calls += 1
        if self.calls == 1:
            self.revives = 1
            return "revive"
        return None

    def reset_run(self):
        pass

    def new_run(self):
        pass


class LiveThenFrozenCapture(FakeCapture):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.n = 0

    def grab(self):
        self.n += 1
        if self.n < 60:
            rng = np.random.default_rng(self.n)
            return rng.integers(0, 255, size=(54, 96, 3), dtype=np.uint8)
        return self.frame


def test_revive_traces_vector_and_held_keys(frozen_game, capsys):
    frozen_game.setenv("VS_AUTO_MENUS", "1")
    frozen_game.setattr(m, "tesseract_available", lambda: True)
    frozen_game.setattr(m, "detect_level_up", lambda frame: None)
    frozen_game.setattr(m, "MenuPilot", RevivingPilot)
    frozen_game.setattr(m, "ScreenCapture", LiveThenFrozenCapture)
    frozen_game.setattr(m, "REVIVE_TRACE_S", 0.4)
    frozen_game.setattr(m, "REVIVE_TRACE_EVERY_S", 0.05)
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[revive+")]
    assert len(lines) >= 3
    assert "menu=revive" in lines[0]
    assert all(" v=(" in ln and " held=" in ln and " keys=" in ln for ln in lines)
    assert any("menu=-" in ln for ln in lines[1:])


class HeuristicMenuVision(FakeVision):
    """Vision fallback that flags every frame as a menu (heuristic only, no pilot)."""

    def analyze(self, frame):
        return VisionResult(player=(48, 27), timer_s=None, size=(96, 54), menu=True, note="heuristic")


def test_revive_escape_ignores_heuristic_menu_until_frames_agree(frozen_game, capsys):
    frozen_game.setenv("VS_AUTO_MENUS", "1")
    frozen_game.setattr(m, "tesseract_available", lambda: True)
    frozen_game.setattr(m, "detect_level_up", lambda frame: None)
    frozen_game.setattr(m, "MenuPilot", RevivingPilot)
    frozen_game.setattr(m, "ScreenCapture", LiveThenFrozenCapture)
    frozen_game.setattr(m, "Vision", HeuristicMenuVision)
    frozen_game.setattr(m, "REVIVE_TRACE_S", 0.4)
    frozen_game.setattr(m, "REVIVE_TRACE_EVERY_S", 0.0)
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[revive+")]
    assert "menu=revive" in lines[0]
    # The first heuristic-only frames during the escape keep moving ...
    assert "menu hint ignored x1" in lines[1] and "revive-escape" in lines[1]
    assert "menu hint ignored x2" in lines[2]
    assert " v=(0, 0)" not in lines[1]
    # ... the third agreeing frame freezes like before.
    assert " v=(0, 0)" in lines[3] and "ignored" not in lines[3]
    assert "[nav] revive escape" in out


class _PickingPilot:
    """One level-up pick, then plain gameplay. Reports a weapon ready to evolve."""

    def __init__(self, **_kw) -> None:
        self.hp = 1.0
        self.revives = 0
        self._quit_pressed = False
        self.finished = False
        self.status = ""
        self.picks: list[str] = []
        self.chest_presses = 0
        self.evolution_ready = False
        self.all_maxed = False
        self.calls = 0

    def step(self, _frame, screen=None):
        self.calls += 1
        if self.calls == 1:
            self.picks.append("Garlic")
            self.evolution_ready = True
            return "levelup"
        return None

    def reset_run(self):
        pass

    def new_run(self):
        self.picks.clear()
        self.evolution_ready = False
        self.all_maxed = False


def test_main_pushes_build_state_after_a_pick_when_the_navigator_accepts_it(frozen_game):
    seen: list[tuple[bool, bool]] = []

    class _RecordingNav(m.Navigator):
        def set_build_state(self, evolution_ready: bool, all_maxed: bool) -> None:
            seen.append((bool(evolution_ready), bool(all_maxed)))

    frozen_game.setenv("VS_AUTO_MENUS", "1")
    frozen_game.setattr(m, "tesseract_available", lambda: True)
    frozen_game.setattr(m, "detect_level_up", lambda frame: None)
    frozen_game.setattr(m, "MenuPilot", _PickingPilot)
    frozen_game.setattr(m, "Navigator", _RecordingNav)
    frozen_game.setattr(m, "ScreenCapture", FakeCapture)
    assert m.main(["--dry-run"]) == m.EXIT_WATCHDOG
    assert seen[0] == (False, False)
    assert (True, False) in seen
    assert seen.count((True, False)) == 1
