from vs_vision_bot.config import ALL_MOVE_KEYS, XDOTOOL_KEY_NAMES, Config
from vs_vision_bot import input_backend as ib
from vs_vision_bot.input_backend import NullBackend, XdotoolBackend, make_backend


class _XdotoolResult:
    """Duck-typed stand-in for subprocess.CompletedProcess from _run_xdotool."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_xdotool(monkeypatch, *, stdout: str = "12345678\n", active_stdout: str | None = None):
    """Record _run_xdotool argv and return a successful CompletedProcess-shaped result."""
    calls: list[list[str]] = []

    def fake_run(args):
        recorded = list(args)
        calls.append(recorded)
        if active_stdout is not None and recorded[:1] == ["getactivewindow"]:
            return _XdotoolResult(stdout=active_stdout)
        return _XdotoolResult(stdout=stdout)

    monkeypatch.setattr(ib, "_run_xdotool", fake_run)
    return calls


def test_demo_and_dry_run_use_null_backend():
    demo = Config(demo=True, dry_run=True)
    assert isinstance(make_backend(demo), NullBackend)
    dry = Config(dry_run=True)
    assert isinstance(make_backend(dry), NullBackend)


def test_live_default_backend_is_xdotool(monkeypatch):
    monkeypatch.delenv("VS_DRY_RUN", raising=False)
    monkeypatch.delenv("VS_INPUT_BACKEND", raising=False)
    monkeypatch.setattr(ib.shutil, "which", lambda _name: "/usr/bin/xdotool")
    cfg = Config.from_env()
    assert cfg.dry_run is False
    assert cfg.demo is False
    assert cfg.input_backend == "xdotool"
    backend = make_backend(cfg)
    assert isinstance(backend, XdotoolBackend)
    assert backend.name == "xdotool"


def test_xdotool_sends_wasd_keydown_keyup(monkeypatch):
    calls = _stub_xdotool(monkeypatch)
    backend = XdotoolBackend("Vampire Survivors")
    assert backend.activate_game_once() is True
    for key in ("w", "a", "s", "d"):
        backend.keydown(key)
        backend.keyup(key)
        assert ["keydown", key] in calls
        assert ["keyup", key] in calls
    assert ["search", "--name", "Vampire Survivors"] in calls
    assert ["windowactivate", "--sync", "12345678"] in calls
    # Activate once — a second call must not repeat windowactivate.
    activate_calls = [c for c in calls if c[:1] == ["windowactivate"]]
    backend.activate_game_once()
    activate_calls_after = [c for c in calls if c[:1] == ["windowactivate"]]
    assert activate_calls == activate_calls_after


def test_xdotool_sends_arrow_keydown_keyup(monkeypatch):
    calls = _stub_xdotool(monkeypatch)
    backend = XdotoolBackend("Vampire Survivors")
    assert backend.activate_game_once() is True
    expected = {
        "up": "Up",
        "down": "Down",
        "left": "Left",
        "right": "Right",
    }
    for key, xname in expected.items():
        backend.keydown(key)
        backend.keyup(key)
        assert ["keydown", xname] in calls
        assert ["keyup", xname] in calls


def test_xdotool_release_refocuses_then_lifts_both_schemes(monkeypatch):
    calls = _stub_xdotool(monkeypatch, stdout="999\n", active_stdout="1\n")
    backend = XdotoolBackend("Vampire Survivors")
    backend.window_id = "999"
    backend._activated = True
    backend.release_all_move_keys()

    assert ["getactivewindow"] in calls
    assert ["windowactivate", "--sync", "999"] in calls
    activate_at = calls.index(["windowactivate", "--sync", "999"])
    first_up = next(i for i, call in enumerate(calls) if call[:1] == ["keyup"])
    assert activate_at < first_up
    for key in ALL_MOVE_KEYS:
        assert ["keyup", XDOTOOL_KEY_NAMES[key]] in calls


def test_xdotool_key_name_table_covers_both_schemes():
    assert set(XDOTOOL_KEY_NAMES) == set(ALL_MOVE_KEYS)
    assert XDOTOOL_KEY_NAMES["w"] == "w"
    assert XDOTOOL_KEY_NAMES["left"] == "Left"


def test_xdotool_key_events_are_global_xtest_not_window_targeted(monkeypatch):
    calls = _stub_xdotool(monkeypatch)
    backend = XdotoolBackend("Vampire Survivors")
    backend.activate_game_once()
    backend.keydown("d")
    backend.keydown("right")
    key_calls = [c for c in calls if c and c[0] in {"keydown", "keyup"}]
    assert ["keydown", "d"] in key_calls
    assert ["keydown", "Right"] in key_calls
    assert all("--window" not in c for c in key_calls)
