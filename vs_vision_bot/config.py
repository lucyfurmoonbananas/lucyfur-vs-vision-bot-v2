"""Runtime configuration. Defaults favor Vampire Survivors on Steam/Linux."""

from __future__ import annotations

import os
from dataclasses import dataclass


MOVE_SCHEMES = ("wasd", "arrows")
INPUT_BACKENDS = ("xdotool", "pynput")

# Always released on pause/stop/exit so leftover holds cannot stick.
ALL_MOVE_KEYS = ("w", "a", "s", "d", "up", "down", "left", "right")

WASD_VECTORS: dict[tuple[int, int], tuple[str, ...]] = {
    (0, 0): (),
    (0, -1): ("w",),
    (0, 1): ("s",),
    (-1, 0): ("a",),
    (1, 0): ("d",),
    (-1, -1): ("w", "a"),
    (1, -1): ("w", "d"),
    (-1, 1): ("s", "a"),
    (1, 1): ("s", "d"),
}

ARROW_VECTORS: dict[tuple[int, int], tuple[str, ...]] = {
    (0, 0): (),
    (0, -1): ("up",),
    (0, 1): ("down",),
    (-1, 0): ("left",),
    (1, 0): ("right",),
    (-1, -1): ("up", "left"),
    (1, -1): ("up", "right"),
    (-1, 1): ("down", "left"),
    (1, 1): ("down", "right"),
}

XDOTOOL_KEY_NAMES = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


@dataclass
class Config:
    """All tunables. Env vars override defaults; CLI can override after load."""

    move_scheme: str = "wasd"
    input_backend: str = "xdotool"
    window_name: str = "Vampire Survivors"
    capture_x: int = 200
    capture_y: int = 80
    capture_w: int = 960
    capture_h: int = 540
    debug_w: int = 400
    debug_h: int = 225
    debug_x: int | None = None
    debug_y: int | None = None
    fps: float = 12.0
    hold_ms: int = 80
    player_radius: int = 28
    min_blob_area: int = 40
    max_blob_area: int = 12000
    model_path: str | None = None
    dry_run: bool = False
    demo: bool = False

    @classmethod
    def from_env(cls) -> Config:
        scheme = _env_str("VS_MOVE_KEYS", "wasd").lower()
        if scheme in ("arrow", "arrow_keys", "arrows"):
            scheme = "arrows"
        if scheme not in MOVE_SCHEMES:
            raise ValueError(
                f"VS_MOVE_KEYS must be 'wasd' (default) or 'arrows', got {scheme!r}"
            )

        backend = _env_str("VS_INPUT_BACKEND", "xdotool").lower()
        if backend not in INPUT_BACKENDS:
            raise ValueError(
                f"VS_INPUT_BACKEND must be 'xdotool' (default) or 'pynput', got {backend!r}"
            )

        model = os.environ.get("VS_MODEL_PATH", "").strip() or None
        dry = _env_str("VS_DRY_RUN", "").lower() in ("1", "true", "yes")

        debug_x_raw = os.environ.get("VS_DEBUG_X", "").strip()
        debug_y_raw = os.environ.get("VS_DEBUG_Y", "").strip()

        return cls(
            move_scheme=scheme,
            input_backend=backend,
            window_name=_env_str("VS_WINDOW_NAME", "Vampire Survivors"),
            capture_x=_env_int("VS_CAPTURE_X", 200),
            capture_y=_env_int("VS_CAPTURE_Y", 80),
            capture_w=_env_int("VS_CAPTURE_W", 960),
            capture_h=_env_int("VS_CAPTURE_H", 540),
            debug_w=_env_int("VS_DEBUG_W", 400),
            debug_h=_env_int("VS_DEBUG_H", 225),
            debug_x=int(debug_x_raw) if debug_x_raw else None,
            debug_y=int(debug_y_raw) if debug_y_raw else None,
            fps=_env_float("VS_FPS", 12.0),
            hold_ms=_env_int("VS_HOLD_MS", 80),
            player_radius=_env_int("VS_PLAYER_RADIUS", 28),
            min_blob_area=_env_int("VS_MIN_BLOB_AREA", 40),
            max_blob_area=_env_int("VS_MAX_BLOB_AREA", 12000),
            model_path=model,
            dry_run=dry,
        )

    @property
    def frame_s(self) -> float:
        return 1.0 / max(self.fps, 1.0)

    @property
    def hold_s(self) -> float:
        return max(self.hold_ms, 1) / 1000.0

    @property
    def vectors(self) -> dict[tuple[int, int], tuple[str, ...]]:
        return ARROW_VECTORS if self.move_scheme == "arrows" else WASD_VECTORS

    def keys_for_vector(self, vector: tuple[int, int]) -> tuple[str, ...]:
        return self.vectors.get(vector, ())
