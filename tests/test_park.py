from vs_vision_bot.capture import CaptureRegion, park_debug_origin


def test_debug_window_parks_off_the_playfield():
    capture = CaptureRegion(x=200, y=80, w=960, h=540)
    x, y = park_debug_origin(1920, 1080, capture, 400, 225)
    debug = CaptureRegion(x, y, 400, 225)
    assert not debug.overlaps(capture)


def test_explicit_debug_origin_is_respected():
    capture = CaptureRegion(x=0, y=0, w=800, h=600)
    x, y = park_debug_origin(1920, 1080, capture, 400, 225, preferred=(12, 34))
    assert (x, y) == (12, 34)
