from vs_vision_bot.hotkeys import CommandBus, HotkeyService


def test_starts_paused():
    bus = CommandBus()
    assert bus.paused is True


def test_toggle_and_align_and_quit():
    bus = CommandBus(toggle_debounce_s=0)
    assert bus.toggle_pause() is False
    assert bus.paused is False
    assert bus.toggle_pause() is True
    assert bus.paused is True

    bus.request_align()
    assert bus.consume_align() is True
    assert bus.consume_align() is False

    bus.request_quit()
    assert bus.quit is True
    assert bus.paused is True


def test_pause_toggle_is_debounced():
    class Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

    clock = Clock()
    bus = CommandBus(clock=clock, toggle_debounce_s=0.2)
    assert bus.toggle_pause() is False
    clock.t += 0.05
    assert bus.toggle_pause() is False
    assert bus.paused is False
    clock.t += 0.2
    assert bus.toggle_pause() is True
    assert bus.paused is True


def test_waitkey_p_unpauses_once_when_delivered_twice():
    bus = CommandBus()
    seen: list[bool] = []
    service = HotkeyService(bus, seen.append)
    service.handle_waitkey(ord("p"))
    service.handle_waitkey(ord("p"))
    assert bus.paused is False
    assert seen[0] is False


def test_waitkey_esc_requests_quit_and_forces_pause_release():
    bus = CommandBus()
    bus.toggle_pause()
    assert bus.paused is False
    seen: list[bool] = []
    service = HotkeyService(bus, seen.append)
    service.handle_waitkey(27)
    assert bus.quit is True
    assert bus.paused is True
    assert seen == [True]


def test_waitkey_and_global_p_share_one_toggle():
    bus = CommandBus()
    seen: list[bool] = []
    service = HotkeyService(bus, seen.append)
    service.handle_waitkey(ord("p"))
    # Same keypress arriving on the other path must not flip back to paused.
    assert bus.toggle_pause() is False
    assert bus.paused is False
    assert seen == [False]


def test_waitkey_q_requests_align():
    bus = CommandBus()
    service = HotkeyService(bus, lambda _paused: None)
    service.handle_waitkey(ord("q"))
    assert bus.consume_align() is True
