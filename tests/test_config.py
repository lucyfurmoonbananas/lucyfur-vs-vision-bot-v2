from vs_vision_bot.config import ALL_MOVE_KEYS, ARROW_VECTORS, WASD_VECTORS, Config
from vs_vision_bot.main import apply_cli, build_parser


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


def test_both_schemes_map_all_eight_directions():
    wasd = Config(move_scheme="wasd")
    arrows = Config(move_scheme="arrows")
    assert wasd.vectors is WASD_VECTORS
    assert arrows.vectors is ARROW_VECTORS
    expected_wasd = {
        (0, 0): (),
        (0, -1): ("w",),
        (0, 1): ("s",),
        (-1, 0): ("a",),
        (1, 0): ("d",),
        (-1, -1): ("w", "a"),
        (1, -1): ("w", "d"),
        (-1, 1): ("s", "a"),
        (1, 1): ("s", "d"),
    }
    expected_arrows = {
        (0, 0): (),
        (0, -1): ("up",),
        (0, 1): ("down",),
        (-1, 0): ("left",),
        (1, 0): ("right",),
        (-1, -1): ("up", "left"),
        (1, -1): ("up", "right"),
        (-1, 1): ("down", "left"),
        (1, 1): ("down", "right"),
    }
    for vector, keys in expected_wasd.items():
        assert wasd.keys_for_vector(vector) == keys
    for vector, keys in expected_arrows.items():
        assert arrows.keys_for_vector(vector) == keys


def test_cli_keys_override_env(monkeypatch):
    monkeypatch.setenv("VS_MOVE_KEYS", "arrows")
    cfg = apply_cli(Config.from_env(), build_parser().parse_args(["--keys", "wasd"]))
    assert cfg.move_scheme == "wasd"
    cfg = apply_cli(Config.from_env(), build_parser().parse_args(["--keys", "arrows"]))
    assert cfg.move_scheme == "arrows"


def test_cleanup_key_set_covers_wasd_and_arrows():
    assert set(ALL_MOVE_KEYS) == {"w", "a", "s", "d", "up", "down", "left", "right"}


def test_pynput_is_opt_in(monkeypatch):
    monkeypatch.setenv("VS_INPUT_BACKEND", "pynput")
    cfg = Config.from_env()
    assert cfg.input_backend == "pynput"
