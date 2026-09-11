from vs_vision_bot.config import Config
from vs_vision_bot.pathfind import choose_vector
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
