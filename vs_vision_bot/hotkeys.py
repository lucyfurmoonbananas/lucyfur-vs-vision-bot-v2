"""Pause / align / quit listeners that work even if Model Vision is unfocused."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from pynput import keyboard

# pynput + OpenCV waitKey (and stdin 'p') can see the same keystroke.
_TOGGLE_DEBOUNCE_S = 0.2


class CommandBus:
    """Thread-safe flags consumed by the main loop."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        toggle_debounce_s: float = _TOGGLE_DEBOUNCE_S,
    ) -> None:
        self._lock = threading.Lock()
        self.paused = True
        self.quit = False
        self.align = False
        self.pause_event = threading.Event()
        self.pause_event.set()
        self._clock = clock
        self._toggle_debounce_s = toggle_debounce_s
        self._last_claim_at: dict[str, float] = {}

    def _claim(self, action: str) -> bool:
        """True once per keypress; pynput + waitKey (and stdin) share this gate."""
        now = self._clock()
        last = self._last_claim_at.get(action)
        if last is not None and (now - last) < self._toggle_debounce_s:
            return False
        self._last_claim_at[action] = now
        return True

    def request_quit(self) -> None:
        with self._lock:
            self._claim("quit")
            self.quit = True
            self.paused = True
        self.pause_event.set()

    def toggle_pause(self) -> bool:
        with self._lock:
            if not self._claim("pause"):
                return self.paused
            self.paused = not self.paused
            paused = self.paused
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()
        state = "PAUSED" if paused else "RUNNING"
        print(f"[hotkey] {state}")
        return paused

    def request_align(self) -> None:
        with self._lock:
            if not self._claim("align"):
                return
            self.align = True
        print("[hotkey] align capture to mouse")

    def consume_align(self) -> bool:
        with self._lock:
            flag = self.align
            self.align = False
            return flag


class HotkeyService:
    """
    Three input paths, all mapped to the same commands:

    1. Global pynput listener — works while Vampire Survivors is focused.
    2. OpenCV waitKey — works while Model Vision is focused.
    3. Stdin line fallback — type p / q / quit and Enter if no global hook.
    """

    def __init__(self, bus: CommandBus, on_pause_change: Callable[[bool], None]) -> None:
        self.bus = bus
        self.on_pause_change = on_pause_change
        self._listener: keyboard.Listener | None = None
        self._stdin_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            self._listener = keyboard.Listener(on_press=self._on_global_press)
            self._listener.daemon = True
            self._listener.start()
            print("[hotkey] global listener on (p pause, q align, ESC quit)")
        except Exception as exc:  # noqa: BLE001
            print(f"[hotkey] global listener failed ({exc}); use stdin or Model Vision focus")

        if sys.stdin.isatty():
            self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._stdin_thread.start()
            print("[hotkey] stdin fallback on (type p / q / quit + Enter)")

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass

    def handle_waitkey(self, key_code: int) -> None:
        if key_code in (27,):  # ESC
            self.bus.request_quit()
            self.on_pause_change(True)
            return
        if key_code in (ord("p"), ord("P")):
            paused = self.bus.toggle_pause()
            self.on_pause_change(paused)
            return
        if key_code in (ord("q"), ord("Q")):
            self.bus.request_align()

    def _on_global_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.esc:
            self.bus.request_quit()
            self.on_pause_change(True)
            return
        char = getattr(key, "char", None)
        if char in ("p", "P"):
            paused = self.bus.toggle_pause()
            self.on_pause_change(paused)
        elif char in ("q", "Q"):
            self.bus.request_align()

    def _stdin_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:  # noqa: BLE001
                return
            if line == "":
                return
            token = line.strip().lower()
            if token in {"p", "pause"}:
                paused = self.bus.toggle_pause()
                self.on_pause_change(paused)
            elif token in {"q", "align"}:
                self.bus.request_align()
            elif token in {"quit", "exit", "esc"}:
                self.bus.request_quit()
                self.on_pause_change(True)
