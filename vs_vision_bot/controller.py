"""Pause-safe movement: queued intents, interruptible holds, full key release."""

from __future__ import annotations

import atexit
import threading
from collections import deque

from vs_vision_bot.config import ALL_MOVE_KEYS, Config
from vs_vision_bot.input_backend import InputBackend


class MoveController:
    """
    Holds the current WASD/arrow chord and a short intent queue.

    Pause, stop, and process exit all:
    - clear the queue
    - release WASD and arrow keys
    - wake any in-flight hold wait
    """

    def __init__(self, backend: InputBackend, cfg: Config) -> None:
        self.backend = backend
        self.cfg = cfg
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._queue: deque[tuple[str, ...]] = deque()
        self._paused = True
        self._hold_interrupt = threading.Event()
        self._hold_interrupt.set()
        self._shutdown = False
        self._activated = False
        atexit.register(self.release_and_clear)

    @property
    def held(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._held)

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            was = self._paused
            self._paused = paused
        if paused and not was:
            self._hold_interrupt.set()
            self.release_and_clear()
        elif not paused and was:
            self._hold_interrupt.clear()

    def set_intent(self, keys: tuple[str, ...]) -> None:
        """Replace the move queue with the latest vision decision (no lag stack)."""
        with self._lock:
            if self._paused or self._shutdown:
                self._queue.clear()
                return
            self._queue.clear()
            self._queue.append(tuple(keys))

    def clear_queue(self) -> None:
        with self._lock:
            self._queue.clear()

    def release_and_clear(self) -> None:
        """Clear intents and lift every movement key. Safe to call from any thread."""
        with self._lock:
            self._queue.clear()
        self.backend.release_all_move_keys()
        with self._lock:
            self._held.clear()

    def sync_keys(self, keys: tuple[str, ...]) -> None:
        """Press/release the delta between the current chord and ``keys``."""
        wanted = set(keys)
        with self._lock:
            if self._paused or self._shutdown:
                wanted = set()
            current = set(self._held)
        for key in sorted(current - wanted):
            self.backend.keyup(key)
        for key in sorted(wanted - current):
            self.backend.keydown(key)
        with self._lock:
            self._held = wanted

    def tick(self) -> None:
        """Apply the next queued chord, or idle if paused/empty."""
        with self._lock:
            if self._paused or self._shutdown:
                keys: tuple[str, ...] = ()
            elif self._queue:
                keys = self._queue.popleft()
            else:
                keys = tuple(self._held)
        if keys:
            self._ensure_game_ready()
        self.sync_keys(keys)

    def hold_current(self, seconds: float) -> bool:
        """
        Sleep up to ``seconds`` while keys stay down.

        Returns True if pause/shutdown interrupted the hold (keys already released).
        """
        if seconds <= 0:
            return self.paused
        interrupted = self._hold_interrupt.wait(timeout=seconds)
        if interrupted:
            self.release_and_clear()
        return interrupted

    def _ensure_game_ready(self) -> None:
        if self.cfg.dry_run or self.cfg.demo:
            return
        if not self._activated:
            self.backend.activate_game_once()
            self._activated = True
            return
        self.backend.refocus_if_needed()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._paused = True
        self._hold_interrupt.set()
        self.release_and_clear()

    def status_line(self) -> str:
        held = ",".join(sorted(self.held)) or "none"
        return (
            f"backend={self.backend.name} scheme={self.cfg.move_scheme} "
            f"held={held} queue={self.queue_size}"
        )


def assert_release_covers_both_schemes() -> None:
    """Sanity helper used by tests: cleanup always includes WASD and arrows."""
    assert set(ALL_MOVE_KEYS) == {"w", "a", "s", "d", "up", "down", "left", "right"}
