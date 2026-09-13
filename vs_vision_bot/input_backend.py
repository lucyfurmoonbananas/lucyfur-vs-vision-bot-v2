"""Keyboard backends. Default is xdotool aimed at the Vampire Survivors window."""

from __future__ import annotations

import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Iterable

from vs_vision_bot.config import ALL_MOVE_KEYS, XDOTOOL_KEY_NAMES, Config


class InputBackend(ABC):
    name: str

    @abstractmethod
    def keydown(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def keyup(self, key: str) -> None:
        raise NotImplementedError

    def activate_game_once(self) -> bool:
        return True

    def refocus_if_needed(self) -> bool:
        return True

    def release_all_move_keys(self) -> None:
        """Always lift WASD and arrows so a scheme switch cannot leave a stuck key."""
        # XTest keyup follows the focused window. Refocus first; otherwise
        # leftover Left/Right (or WASD) stay down in the game.
        try:
            self.refocus_if_needed()
        except Exception as exc:  # noqa: BLE001 — still attempt the keyups
            print(f"[input] refocus before key release failed: {exc}", file=sys.stderr)
        for key in ALL_MOVE_KEYS:
            try:
                self.keyup(key)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                print(f"[input] keyup {key} failed: {exc}", file=sys.stderr)


class NullBackend(InputBackend):
    """Records key events without touching the OS. Used by --dry-run and tests."""

    name = "null"

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.held: set[str] = set()

    def keydown(self, key: str) -> None:
        self.events.append(("down", key))
        self.held.add(key)

    def keyup(self, key: str) -> None:
        self.events.append(("up", key))
        self.held.discard(key)


def _run_xdotool(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xdotool", *args],
        check=False,
        capture_output=True,
        text=True,
    )


class XdotoolBackend(InputBackend):
    """
    Steam/Proton games usually ignore pynput. xdotool XTest events reach them
    after the Vampire Survivors window has been activated once.
    """

    name = "xdotool"

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self.window_id: str | None = None
        self._activated = False

    def find_window(self) -> str | None:
        if shutil.which("xdotool") is None:
            print("[input] xdotool is not on PATH", file=sys.stderr)
            return None
        result = _run_xdotool(["search", "--name", self.window_name])
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not ids:
            return None
        # The last mapped id is usually the real game window, not a launcher stub.
        self.window_id = ids[-1]
        return self.window_id

    def activate_game_once(self) -> bool:
        if self._activated:
            return True
        return self._activate()

    def _activate(self) -> bool:
        wid = self.window_id or self.find_window()
        if not wid:
            print(
                f"[input] no window matching {self.window_name!r}; "
                "keys will not reach the game until it is open",
                file=sys.stderr,
            )
            return False
        result = _run_xdotool(["windowactivate", "--sync", wid])
        if result.returncode != 0:
            print(
                f"[input] windowactivate failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
        self._activated = True
        self.window_id = wid
        return True

    def refocus_if_needed(self) -> bool:
        """Recover after OpenCV Model Vision steals focus. Not a startup activate."""
        wid = self.window_id or self.find_window()
        if not wid:
            return False
        active = _run_xdotool(["getactivewindow"])
        if active.returncode == 0 and active.stdout.strip() == wid:
            return True
        result = _run_xdotool(["windowactivate", "--sync", wid])
        if result.returncode == 0:
            self._activated = True
            self.window_id = wid
            return True
        return False

    def keydown(self, key: str) -> None:
        self._key_event("keydown", key)

    def keyup(self, key: str) -> None:
        self._key_event("keyup", key)

    def _key_event(self, action: str, key: str) -> None:
        xname = XDOTOOL_KEY_NAMES.get(key, key)
        result = _run_xdotool([action, xname])
        if result.returncode != 0:
            print(
                f"[input] xdotool {action} {xname} failed: {result.stderr.strip()}",
                file=sys.stderr,
            )


class PynputBackend(InputBackend):
    """Optional fallback. Often fails to reach Steam/Proton games on Linux."""

    name = "pynput"

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key, KeyCode

        self._controller = Controller()
        self._key_map = {
            "w": KeyCode.from_char("w"),
            "a": KeyCode.from_char("a"),
            "s": KeyCode.from_char("s"),
            "d": KeyCode.from_char("d"),
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
        }

    def keydown(self, key: str) -> None:
        mapped = self._key_map.get(key)
        if mapped is not None:
            self._controller.press(mapped)

    def keyup(self, key: str) -> None:
        mapped = self._key_map.get(key)
        if mapped is not None:
            self._controller.release(mapped)


def make_backend(cfg: Config) -> InputBackend:
    if cfg.dry_run or cfg.demo:
        print("[input] using null backend (demo/dry-run — no game keys)")
        return NullBackend()
    if cfg.input_backend == "pynput":
        print(
            "[input] pynput fallback enabled. Steam/Proton often ignores it; "
            "prefer the default xdotool backend."
        )
        return PynputBackend()
    if shutil.which("xdotool") is None:
        raise SystemExit(
            "xdotool is required for the default input backend. "
            "Install it (Debian/Ubuntu: sudo apt install xdotool) "
            "or set VS_INPUT_BACKEND=pynput."
        )
    print(f"[input] xdotool backend, target window {cfg.window_name!r}, scheme {cfg.move_scheme}")
    return XdotoolBackend(cfg.window_name)


def get_mouse_position() -> tuple[int, int] | None:
    if shutil.which("xdotool") is None:
        return None
    result = _run_xdotool(["getmouselocation", "--shell"])
    if result.returncode != 0:
        return None
    vals: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            if name in {"X", "Y"}:
                vals[name] = int(value)
    if "X" in vals and "Y" in vals:
        return vals["X"], vals["Y"]
    return None


def screen_size() -> tuple[int, int]:
    if shutil.which("xdotool") is None:
        return 1920, 1080
    result = _run_xdotool(["getdisplaygeometry"])
    parts = result.stdout.split()
    if result.returncode == 0 and len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 1920, 1080


def iter_key_names(keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(keys)
