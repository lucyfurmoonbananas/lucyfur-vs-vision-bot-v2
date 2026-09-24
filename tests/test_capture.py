from vs_vision_bot.capture import ScreenCapture
from vs_vision_bot.config import Config


def test_align_to_mouse_clamps_near_edge_origin(monkeypatch):
    """A cursor near the right or bottom edge must not hang the grab off-screen."""
    screen_w, screen_h = 1920, 1080
    capture_w, capture_h = 960, 540
    monkeypatch.setattr(
        "vs_vision_bot.capture.get_mouse_position",
        lambda: (screen_w - 8, screen_h - 8),
    )
    cfg = Config(capture_x=0, capture_y=0, capture_w=capture_w, capture_h=capture_h)
    capture = ScreenCapture(cfg)
    monkeypatch.setattr(capture, "display_size", lambda: (screen_w, screen_h))

    region = capture.align_to_mouse()

    assert region is not None
    assert region.x == screen_w - capture_w
    assert region.y == screen_h - capture_h
    assert 0 <= region.x <= max(0, screen_w - capture_w)
    assert 0 <= region.y <= max(0, screen_h - capture_h)
    assert region.x + region.w <= screen_w
    assert region.y + region.h <= screen_h
    assert cfg.capture_x == region.x
    assert cfg.capture_y == region.y


def test_align_to_mouse_keeps_origin_already_inside(monkeypatch):
    monkeypatch.setattr("vs_vision_bot.capture.get_mouse_position", lambda: (120, 40))
    cfg = Config(capture_x=0, capture_y=0, capture_w=960, capture_h=540)
    capture = ScreenCapture(cfg)
    monkeypatch.setattr(capture, "display_size", lambda: (1920, 1080))

    region = capture.align_to_mouse()

    assert region is not None
    assert (region.x, region.y) == (120, 40)
    assert (cfg.capture_x, cfg.capture_y) == (120, 40)
