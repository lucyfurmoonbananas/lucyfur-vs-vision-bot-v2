"""Main loop: start paused, watch the playfield, dodge, send WASD via xdotool."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from vs_vision_bot.capture import ScreenCapture
from vs_vision_bot.config import Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.debug_window import DebugWindow
from vs_vision_bot.hotkeys import CommandBus, HotkeyService
from vs_vision_bot.input_backend import make_backend
from vs_vision_bot.pathfind import keys_for_result
from vs_vision_bot.vision import Vision, draw_synthetic_arena


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vs-vision-bot",
        description=(
            "Vampire Survivors vision bot. Starts PAUSED. Press p to unpause, "
            "q to align capture to the mouse, ESC to quit."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic arena + no game keys. Use this to check Model Vision without Steam.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Capture and pathfind but never send keyboard events.",
    )
    parser.add_argument(
        "--backend",
        choices=("xdotool", "pynput"),
        default=None,
        help="Override VS_INPUT_BACKEND. Default is xdotool.",
    )
    parser.add_argument(
        "--keys",
        choices=("wasd", "arrows"),
        default=None,
        help="Override VS_MOVE_KEYS. Default is wasd. arrows is a fallback only.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Vision loop rate (default 12).",
    )
    return parser


def apply_cli(cfg: Config, args: argparse.Namespace) -> Config:
    if args.demo:
        cfg.demo = True
        cfg.dry_run = True
    if args.dry_run:
        cfg.dry_run = True
    if args.backend:
        cfg.input_backend = args.backend
    if args.keys:
        cfg.move_scheme = args.keys
    if args.fps is not None:
        cfg.fps = args.fps
    return cfg


def _print_banner(cfg: Config, backend_name: str | None = None) -> None:
    print("Lucyfur VS Vision Bot v2")
    backend = backend_name or cfg.input_backend
    print(f"  DISPLAY={_display()}  backend={backend}  keys={cfg.move_scheme}")
    print(f"  capture=({cfg.capture_x},{cfg.capture_y}) {cfg.capture_w}x{cfg.capture_h}")
    print("  STARTED PAUSED")
    print("  p     toggle pause / unpause (global hotkey, Model Vision, or stdin)")
    print("  q     set capture top-left to the current mouse position")
    print("  ESC   quit and release WASD + arrow keys")
    print("  Click Vampire Survivors after unpausing if Model Vision stole focus.")
    print()


def _display() -> str:
    import os

    return os.environ.get("DISPLAY", "(unset)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_cli(Config.from_env(), args)
    backend = make_backend(cfg)
    _print_banner(cfg, backend_name=backend.name)

    bus = CommandBus()
    mover = MoveController(backend, cfg)
    capture = ScreenCapture(cfg)
    vision = Vision(cfg)
    debug = DebugWindow(cfg)

    def on_pause_change(paused: bool) -> None:
        mover.set_paused(paused)

    hotkeys = HotkeyService(bus, on_pause_change)
    hotkeys.start()

    def handle_stop(signum: int, _frame: object) -> None:
        print(f"[main] signal {signum}, releasing keys")
        bus.request_quit()
        mover.shutdown()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    last_vector = (0, 0)
    t0 = time.monotonic()
    parked = False

    try:
        while not bus.quit:
            if bus.consume_align():
                capture.align_to_mouse()
                debug.repark(capture.region, capture.display_size())

            if cfg.demo:
                frame = draw_synthetic_arena(cfg.capture_w, cfg.capture_h, t=time.monotonic() - t0)
            else:
                try:
                    frame = capture.grab()
                except Exception as exc:  # noqa: BLE001
                    print(f"[capture] grab failed: {exc}", file=sys.stderr)
                    time.sleep(cfg.frame_s)
                    continue

            if not parked:
                debug.ensure(capture.region, capture.display_size())
                parked = True

            result = vision.analyze(frame)
            vector, keys = keys_for_result(result, cfg, last=last_vector)
            last_vector = vector

            if bus.paused:
                mover.set_paused(True)
            elif result.menu:
                # Level-up / pause cards: hold still without flipping the user pause flag.
                mover.release_and_clear()
            else:
                mover.set_paused(False)
                mover.set_intent(keys)
                mover.tick()

            debug.render(frame, result, bus.paused, mover, vector)
            hotkeys.handle_waitkey(debug.poll_key())

            if bus.paused:
                time.sleep(cfg.frame_s)
            else:
                mover.hold_current(min(cfg.hold_s, cfg.frame_s))
        return 0
    finally:
        mover.shutdown()
        debug.close()
        hotkeys.stop()
        print("[main] keys released, goodbye")


if __name__ == "__main__":
    raise SystemExit(main())
