"""debug_window.close must not raise when OpenCV's GUI receiver is already gone."""

from __future__ import annotations

import cv2

from vs_vision_bot.config import Config
from vs_vision_bot.debug_window import DebugWindow


def test_close_swallows_destroy_window_without_a_gui(monkeypatch):
    win = DebugWindow(Config())
    win._created = True

    def boom(_name):
        raise cv2.error(
            "OpenCV(4.14.0) window_QT.cpp:578: error: (-27:Null pointer) "
            "NULL guiReceiver (please create a window) in function 'cvDestroyWindow'"
        )

    monkeypatch.setattr(cv2, "destroyWindow", boom)
    win.close()
    assert win._created is False
