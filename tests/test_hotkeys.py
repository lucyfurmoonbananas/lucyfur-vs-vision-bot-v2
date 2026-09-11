from vs_vision_bot.hotkeys import CommandBus


def test_starts_paused():
    bus = CommandBus()
    assert bus.paused is True
    assert bus.pause_event.is_set()


def test_toggle_and_align_and_quit():
    bus = CommandBus()
    assert bus.toggle_pause() is False
    assert bus.paused is False
    assert not bus.pause_event.is_set()
    assert bus.toggle_pause() is True
    assert bus.pause_event.is_set()

    bus.request_align()
    assert bus.consume_align() is True
    assert bus.consume_align() is False

    bus.request_quit()
    assert bus.quit is True
    assert bus.paused is True
