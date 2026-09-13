from vs_vision_bot.config import Config
from vs_vision_bot.pathfind import choose_vector, keys_for_result
from vs_vision_bot.vision import Detection, VisionResult


def test_flees_left_when_monster_is_on_the_right():
    cfg = Config()
    result = VisionResult(
        player=(200, 200),
        monsters=[Detection(x=280, y=190, w=20, h=20)],
    )
    vec = choose_vector(result, cfg)
    assert vec[0] <= 0
    assert vec != (1, 0)
    assert vec != (1, -1)
    assert vec != (1, 1)


def test_idle_without_monsters_or_on_menu():
    cfg = Config()
    empty = VisionResult(player=(200, 200), monsters=[])
    assert choose_vector(empty, cfg) == (0, 0)
    menu = VisionResult(
        player=(200, 200),
        monsters=[Detection(10, 10, 20, 20)],
        menu=True,
    )
    assert choose_vector(menu, cfg) == (0, 0)


def test_keys_for_result_follows_wasd_and_arrows_schemes():
    result = VisionResult(
        player=(200, 200),
        monsters=[Detection(x=280, y=190, w=20, h=20)],
    )
    wasd = Config(move_scheme="wasd")
    arrows = Config(move_scheme="arrows")
    vec_w, keys_w = keys_for_result(result, wasd)
    vec_a, keys_a = keys_for_result(result, arrows)
    assert vec_w == vec_a
    assert keys_w == wasd.keys_for_vector(vec_w)
    assert keys_a == arrows.keys_for_vector(vec_a)
    assert not ({"w", "a", "s", "d"} & set(keys_a))
    assert not ({"up", "down", "left", "right"} & set(keys_w))
