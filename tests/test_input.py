from vs_vision_bot.config import ALL_MOVE_KEYS, XDOTOOL_KEY_NAMES, Config
from vs_vision_bot import input_backend as ib
from vs_vision_bot.input_backend import NullBackend, XdotoolBackend, make_backend


def test_demo_and_dry_run_use_null_backend():
    demo = Config(demo=True, dry_run=True)
    assert isinstance(make_backend(demo), NullBackend)
    dry = Config(dry_run=True)
    assert isinstance(make_backend(dry), NullBackend)


def test_default_config_names_xdotool(monkeypatch):
    monkeypatch.delenv("VS_MOVE_KEYS", raising=False)
    monkeypatch.delenv("VS_INPUT_BACKEND", raising=False)
    monkeypatch.delenv("VS_DRY_RUN", raising=False)
    cfg = Config.from_env()
    assert cfg.input_backend == "xdotool"
    assert cfg.move_scheme == "wasd"


def test_live_default_backend_is_xdotool(monkeypatch):
    monkeypatch.delenv("VS_DRY_RUN", raising=False)
    cfg = Config.from_env()
    assert cfg.dry_run is False
    assert cfg.demo is False
    backend = make_backend(cfg)
    assert isinstance(backend, XdotoolBackend)
    assert backend.name == "xdotool"


def test_xdotool_sends_wasd_keydown_keyup(monkeypatch):
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "12345678\n"
        stderr = ""

    def fake_run(args):
        calls.append(args)
        return Result()

    monkeypatch.setattr(ib, "_run_xdotool", fake_run)
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
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "12345678\n"
        stderr = ""

    def fake_run(args):
        calls.append(args)
        return Result()

    monkeypatch.setattr(ib, "_run_xdotool", fake_run)
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
    calls: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str = "999\n", returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args):
        calls.append(list(args))
        if args[:1] == ["getactivewindow"]:
            return Result(stdout="1\n")
        return Result()

    monkeypatch.setattr(ib, "_run_xdotool", fake_run)
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
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "12345678\n"
        stderr = ""

    def fake_run(args):
        calls.append(list(args))
        return Result()

    monkeypatch.setattr(ib, "_run_xdotool", fake_run)
    backend = XdotoolBackend("Vampire Survivors")
    backend.activate_game_once()
    backend.keydown("d")
    backend.keydown("right")
    key_calls = [c for c in calls if c and c[0] in {"keydown", "keyup"}]
    assert ["keydown", "d"] in key_calls
    assert ["keydown", "Right"] in key_calls
    assert all("--window" not in c for c in key_calls)
