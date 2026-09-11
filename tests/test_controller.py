import time

from vs_vision_bot.config import ALL_MOVE_KEYS, Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.input_backend import NullBackend


def _controller() -> tuple[MoveController, NullBackend]:
    backend = NullBackend()
    cfg = Config(dry_run=True)
    mover = MoveController(backend, cfg)
    mover.set_paused(False)
    return mover, backend


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
    # Pause from another "thread" after a short delay by setting it now,
    # then waiting a long hold — interrupt must return immediately.
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
