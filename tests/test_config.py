from vs_vision_bot.config import ALL_MOVE_KEYS, Config


def test_defaults_are_wasd_and_xdotool(monkeypatch):
    monkeypatch.delenv("VS_MOVE_KEYS", raising=False)
    monkeypatch.delenv("VS_INPUT_BACKEND", raising=False)
    cfg = Config.from_env()
    assert cfg.move_scheme == "wasd"
    assert cfg.input_backend == "xdotool"
    assert cfg.keys_for_vector((0, -1)) == ("w",)
    assert cfg.keys_for_vector((-1, 0)) == ("a",)
    assert cfg.keys_for_vector((0, 1)) == ("s",)
    assert cfg.keys_for_vector((1, 0)) == ("d",)


def test_arrows_only_via_explicit_env(monkeypatch):
    monkeypatch.setenv("VS_MOVE_KEYS", "arrows")
    cfg = Config.from_env()
    assert cfg.move_scheme == "arrows"
    assert cfg.keys_for_vector((0, -1)) == ("up",)


def test_cleanup_key_set_covers_wasd_and_arrows():
    assert set(ALL_MOVE_KEYS) == {"w", "a", "s", "d", "up", "down", "left", "right"}


def test_pynput_is_opt_in(monkeypatch):
    monkeypatch.setenv("VS_INPUT_BACKEND", "pynput")
    cfg = Config.from_env()
    assert cfg.input_backend == "pynput"
