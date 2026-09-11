from vs_vision_bot.config import Config
from vs_vision_bot import input_backend as ib
from vs_vision_bot.input_backend import NullBackend, XdotoolBackend, make_backend


def test_demo_and_dry_run_use_null_backend():
    demo = Config(demo=True, dry_run=True)
    assert isinstance(make_backend(demo), NullBackend)
    dry = Config(dry_run=True)
    assert isinstance(make_backend(dry), NullBackend)


def test_default_config_names_xdotool():
    cfg = Config.from_env()
    assert cfg.input_backend == "xdotool"
    assert cfg.move_scheme == "wasd"


def test_live_default_backend_is_xdotool():
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
    backend.keydown("w")
    backend.keyup("w")
    assert ["search", "--name", "Vampire Survivors"] in calls
    assert ["windowactivate", "--sync", "12345678"] in calls
    assert ["keydown", "w"] in calls
    assert ["keyup", "w"] in calls
    # Activate once — a second call must not repeat windowactivate.
    activate_calls = [c for c in calls if c[:1] == ["windowactivate"]]
    backend.activate_game_once()
    activate_calls_after = [c for c in calls if c[:1] == ["windowactivate"]]
    assert activate_calls == activate_calls_after
