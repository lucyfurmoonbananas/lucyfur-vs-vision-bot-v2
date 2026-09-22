import time

import pytest

from vs_vision_bot.config import ALL_MOVE_KEYS, Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.input_backend import NullBackend


class RecordingBackend(NullBackend):
    def __init__(self) -> None:
        super().__init__()
        self.refocus_calls = 0
        self.activate_calls = 0

    def refocus_if_needed(self) -> bool:
        self.refocus_calls += 1
        return True

    def activate_game_once(self) -> bool:
        self.activate_calls += 1
        return True


def _controller(scheme: str = "wasd") -> tuple[MoveController, NullBackend]:
    backend = NullBackend()
    cfg = Config(dry_run=True, move_scheme=scheme)
    mover = MoveController(backend, cfg)
    mover.set_paused(False)
    return mover, backend


def test_controller_starts_paused_and_ignores_intent():
    backend = NullBackend()
    mover = MoveController(backend, Config(dry_run=True))
    assert mover.paused is True
    mover.set_intent(("w", "d"))
    assert mover.queue_size == 0
    mover.tick()
    assert mover.held == frozenset()
    assert all(action != "down" for action, _ in backend.events)


@pytest.mark.parametrize(
    ("scheme", "chord", "next_key"),
    [
        ("wasd", ("w", "a"), "d"),
        ("arrows", ("up", "left"), "right"),
    ],
)
def test_tick_presses_and_releases_chord(scheme, chord, next_key):
    mover, backend = _controller(scheme)
    mover.set_intent(chord)
    mover.tick()
    assert mover.held == frozenset(chord)
    for key in chord:
        assert ("down", key) in backend.events

    backend.events.clear()
    mover.set_intent((next_key,))
    mover.tick()
    assert ("down", next_key) in backend.events
    for key in chord:
        assert ("up", key) in backend.events
        assert backend.events.index(("up", key)) < backend.events.index(("down", next_key))
    assert mover.held == frozenset({next_key})


@pytest.mark.parametrize(
    ("scheme", "from_key", "to_key"),
    [
        ("wasd", "a", "d"),
        ("arrows", "left", "right"),
    ],
)
def test_left_to_right_does_not_hold_opposites(scheme, from_key, to_key):
    mover, backend = _controller(scheme)
    mover.set_intent((from_key,))
    mover.tick()
    backend.events.clear()
    mover.set_intent((to_key,))
    mover.tick()
    assert from_key not in mover.held
    assert mover.held == frozenset({to_key})
    assert backend.events.index(("up", from_key)) < backend.events.index(("down", to_key))


def test_pause_clears_queue_and_releases_wasd_and_arrows():
    mover, backend = _controller()
    mover.set_intent(("w", "d"))
    mover.tick()
    assert "w" in mover.held
    mover.set_paused(True)
    assert mover.queue_size == 0
    assert mover.held == frozenset()
    released = {key for action, key in backend.events if action == "up"}
    assert set(ALL_MOVE_KEYS) <= released


def test_hold_is_interrupted_by_pause():
    mover, backend = _controller()
    mover.set_intent(("a",))
    mover.tick()
    assert "a" in mover.held

    started = time.monotonic()
    mover.set_paused(True)
    interrupted = mover.hold_current(2.0)
    elapsed = time.monotonic() - started
    assert interrupted is True
    assert elapsed < 0.5
    assert mover.held == frozenset()
    released = {key for action, key in backend.events if action == "up"}
    assert set(ALL_MOVE_KEYS) <= released


def test_shutdown_releases_everything():
    mover, backend = _controller()
    mover.sync_keys(("s",))
    mover.shutdown()
    released = {key for action, key in backend.events if action == "up"}
    assert set(ALL_MOVE_KEYS) <= released
    assert mover.queue_size == 0


def test_pause_during_keydown_does_not_leave_a_hold():
    holder: list[MoveController | None] = [None]

    class RaceBackend(NullBackend):
        def keydown(self, key: str) -> None:
            super().keydown(key)
            mover = holder[0]
            if mover is not None:
                mover.set_paused(True)

    backend = RaceBackend()
    mover = MoveController(backend, Config(dry_run=True))
    holder[0] = mover
    mover.set_paused(False)
    mover.sync_keys(("d",))
    assert mover.paused is True
    assert mover.held == frozenset()


def test_idle_tick_refocuses_before_keyup():
    backend = RecordingBackend()
    mover = MoveController(backend, Config(dry_run=False, demo=False))
    mover.set_paused(False)
    mover.set_intent(("a",))
    mover.tick()
    assert backend.activate_calls == 1
    backend.refocus_calls = 0
    mover.set_intent(())
    mover.tick()
    assert backend.refocus_calls >= 1
    assert ("up", "a") in backend.events
    assert mover.held == frozenset()


def test_release_all_move_keys_refocuses():
    backend = RecordingBackend()
    mover = MoveController(backend, Config(dry_run=True))
    mover.set_paused(False)
    mover.sync_keys(("w",))
    backend.refocus_calls = 0
    mover.set_paused(True)
    assert backend.refocus_calls >= 1
    released = {key for action, key in backend.events if action == "up"}
    assert set(ALL_MOVE_KEYS) <= released


def test_failed_activate_is_retried_and_not_marked_ready():
    class FlakyActivate(RecordingBackend):
        def activate_game_once(self) -> bool:
            self.activate_calls += 1
            return self.activate_calls >= 2

    backend = FlakyActivate()
    mover = MoveController(backend, Config(dry_run=False, demo=False))
    mover.set_paused(False)
    mover.set_intent(("w",))
    mover.tick()
    assert mover._activated is False
    assert backend.activate_calls == 1
    # Same chord must still retry activate — do not skip because keys match.
    mover.set_intent(("w",))
    mover.tick()
    assert backend.activate_calls == 2
    assert mover._activated is True
