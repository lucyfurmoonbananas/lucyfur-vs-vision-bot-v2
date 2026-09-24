import cv2
import numpy as np
import pytest

from vs_vision_bot.config import Config
from vs_vision_bot.debug_window import WINDOW_NAME, DebugWindow
from vs_vision_bot.vision import Detection, VisionResult


class _Mover:
    def status_line(self) -> str:
        return "backend=test scheme=wasd held=none queue=0"


def _axis(value: float, src: int, dst: int) -> int:
    return int(round(value * dst / src))


def _point(x: float, y: float, src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    return _axis(x, src_w, dst_w), _axis(y, src_h, dst_h)


def _spy_render(monkeypatch):
    """Record overlay calls and the buffer passed to imshow. Drawing still runs."""
    log: list[tuple] = []
    shown: dict = {}
    originals = {
        "resize": cv2.resize,
        "circle": cv2.circle,
        "rectangle": cv2.rectangle,
        "arrowedLine": cv2.arrowedLine,
        "putText": cv2.putText,
    }

    def spy_resize(src, dsize, *args, **kwargs):
        log.append(("resize", tuple(src.shape), tuple(dsize), src.copy()))
        return originals["resize"](src, dsize, *args, **kwargs)

    def spy_circle(img, center, radius, color, thickness, *args, **kwargs):
        log.append(("circle", id(img), tuple(img.shape), center, radius, thickness))
        return originals["circle"](img, center, radius, color, thickness, *args, **kwargs)

    def spy_rectangle(img, pt1, pt2, color, thickness, *args, **kwargs):
        log.append(("rectangle", id(img), tuple(img.shape), pt1, pt2, thickness))
        return originals["rectangle"](img, pt1, pt2, color, thickness, *args, **kwargs)

    def spy_arrow(img, pt1, pt2, color, thickness, *args, **kwargs):
        log.append(("arrowedLine", id(img), tuple(img.shape), pt1, pt2, thickness))
        return originals["arrowedLine"](img, pt1, pt2, color, thickness, *args, **kwargs)

    def spy_text(img, text, org, font, scale, color, thickness, *args, **kwargs):
        log.append(("putText", id(img), tuple(img.shape), text, org, scale, thickness))
        return originals["putText"](img, text, org, font, scale, color, thickness, *args, **kwargs)

    def spy_imshow(name, image):
        shown["name"] = name
        shown["id"] = id(image)
        shown["shape"] = tuple(image.shape)

    monkeypatch.setattr(cv2, "resize", spy_resize)
    monkeypatch.setattr(cv2, "circle", spy_circle)
    monkeypatch.setattr(cv2, "rectangle", spy_rectangle)
    monkeypatch.setattr(cv2, "arrowedLine", spy_arrow)
    monkeypatch.setattr(cv2, "putText", spy_text)
    monkeypatch.setattr(cv2, "imshow", spy_imshow)
    return log, shown


def test_module_docstring_describes_a_parked_window_without_xdotool():
    import vs_vision_bot.debug_window as debug_window

    doc = debug_window.__doc__ or ""
    assert "xdotool" not in doc.lower()
    assert "small" in doc.lower()
    assert "playfield" in doc.lower()


@pytest.mark.parametrize("debug_w,debug_h", [(400, 225), (400, 180)])
def test_overlays_are_drawn_on_the_debug_sized_frame(monkeypatch, debug_w, debug_h):
    capture_w, capture_h = 960, 540
    frame = np.zeros((capture_h, capture_w, 3), dtype=np.uint8)
    frame[10, 20] = (7, 8, 9)
    player = (480, 270)
    det = Detection(120, 80, 60, 40)
    result = VisionResult(player=player, monsters=[det], note="heuristic")
    vector = (1, -1)
    log, shown = _spy_render(monkeypatch)

    window = DebugWindow(Config(debug_w=debug_w, debug_h=debug_h))
    window.origin = (12, 34)
    window.render(frame, result, paused=False, mover=_Mover(), vector=vector)

    assert frame.shape == (capture_h, capture_w, 3)
    assert tuple(int(v) for v in frame[10, 20]) == (7, 8, 9)

    kind, src_shape, dsize, src = log[0]
    assert kind == "resize"
    assert src_shape == (capture_h, capture_w, 3)
    assert dsize == (debug_w, debug_h)
    # Banner and footer must not be painted onto the capture before the resize.
    assert int(src[0, 0, 0]) == 0
    assert int(src[-1, 0, 0]) == 0

    draws = [entry for entry in log[1:] if entry[0] != "resize"]
    assert draws
    assert all(entry[2] == (debug_h, debug_w, 3) for entry in draws)
    assert len({entry[1] for entry in draws}) == 1

    circles = [entry for entry in draws if entry[0] == "circle"]
    rects = [entry for entry in draws if entry[0] == "rectangle"]
    arrows = [entry for entry in draws if entry[0] == "arrowedLine"]
    texts = [entry for entry in draws if entry[0] == "putText"]

    player_pt = _point(player[0], player[1], capture_w, capture_h, debug_w, debug_h)
    assert circles[0][3] == player_pt
    assert circles[0][4] == 8
    assert circles[0][5] == 2
    assert rects[0][3] == _point(det.x, det.y, capture_w, capture_h, debug_w, debug_h)
    assert rects[0][4] == _point(det.x + det.w, det.y + det.h, capture_w, capture_h, debug_w, debug_h)
    assert rects[0][5] == 1
    assert circles[1][3] == _point(det.cx, det.cy, capture_w, capture_h, debug_w, debug_h)
    assert circles[1][4] == 3
    assert arrows[0][3] == player_pt
    assert arrows[0][4] == _point(
        player[0] + vector[0] * 48,
        player[1] + vector[1] * 48,
        capture_w,
        capture_h,
        debug_w,
        debug_h,
    )
    assert arrows[0][5] == 2

    assert rects[1][3] == (0, 0)
    assert rects[1][4] == (debug_w, 28)
    assert rects[2][3] == (0, debug_h - 24)
    assert rects[2][4] == (debug_w, debug_h)
    assert texts[0][3].startswith("RUNNING")
    assert texts[0][4] == (8, 20)
    assert texts[0][5] == 0.5
    assert texts[0][6] == 1
    assert texts[1][4] == (8, debug_h - 8)
    assert texts[1][5] == 0.42
    assert "park=(12, 34)" in texts[1][3]

    assert shown["name"] == WINDOW_NAME
    assert shown["shape"] == (debug_h, debug_w, 3)
    assert shown["id"] == draws[0][1]

    if (debug_w, debug_h) == (400, 225):
        assert player_pt == (200, 112)
        assert circles[1][3] == (62, 42)
        assert arrows[0][4] == (220, 92)


def test_zero_vector_does_not_draw_an_arrow(monkeypatch):
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    result = VisionResult(player=(480, 270), monsters=[])
    log, shown = _spy_render(monkeypatch)
    DebugWindow(Config(debug_w=400, debug_h=225)).render(
        frame, result, paused=True, mover=_Mover(), vector=(0, 0)
    )
    assert not any(entry[0] == "arrowedLine" for entry in log)
    texts = [entry for entry in log if entry[0] == "putText"]
    assert texts[0][3].startswith("PAUSED")
    assert shown["shape"] == (225, 400, 3)


def test_matching_frame_size_keeps_capture_coordinates(monkeypatch):
    debug_w, debug_h = 320, 180
    frame = np.zeros((debug_h, debug_w, 3), dtype=np.uint8)
    player = (100, 80)
    det = Detection(40, 30, 20, 10)
    result = VisionResult(player=player, monsters=[det])
    log, _shown = _spy_render(monkeypatch)
    DebugWindow(Config(debug_w=debug_w, debug_h=debug_h)).render(
        frame, result, paused=False, mover=_Mover(), vector=(1, 0)
    )
    circles = [entry for entry in log if entry[0] == "circle"]
    arrows = [entry for entry in log if entry[0] == "arrowedLine"]
    assert circles[0][3] == player
    assert circles[1][3] == (det.cx, det.cy)
    assert arrows[0][4] == (player[0] + 48, player[1])
