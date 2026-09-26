"""Main loop: start paused, watch the playfield, dodge, send WASD via xdotool."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from vs_vision_bot.capture import ScreenCapture
from vs_vision_bot.config import Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.debug_window import DebugWindow
from vs_vision_bot.hotkeys import CommandBus, HotkeyService
from vs_vision_bot.input_backend import make_backend
from vs_vision_bot.menus import MenuPilot, detect_level_up, tesseract_available
from vs_vision_bot.pathfind import Navigator
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
        if not paused:
            pilot.reset_run()
            if pilot_new_run[0]:
                nav.reset()
                pilot.new_run()
                pilot_new_run[0] = False

    def release_before_menu_key() -> bool:
        # Lift WASD/arrows (this also re-focuses the game window) before a menu tap.
        mover.release_and_clear()
        return True

    def click_in_capture(x: int, y: int) -> None:
        backend.click(capture.region.x + x, capture.region.y + y)

    pilot = MenuPilot(
        tap=backend.tap,
        refocus=release_before_menu_key,
        click=click_in_capture,
        results_dir=os.environ.get("VS_RESULTS_DIR", "/workspace/vsbot-check/results"),
    )
    nav = Navigator()
    # The movement game clock restarts only after a finished run (QUIT + DONE),
    # not on every manual pause.
    pilot_new_run = [False]
    auto_menus = not cfg.demo and os.environ.get("VS_AUTO_MENUS", "1").strip() not in ("0", "false", "no")
    if auto_menus and not tesseract_available():
        print("[menus] tesseract not found: level-ups pick by position only, chest screens not handled")

    hotkeys = HotkeyService(bus, on_pause_change)
    hotkeys.start()

    def handle_stop(signum: int, _frame: object) -> None:
        print(f"[main] signal {signum}, releasing keys")
        bus.request_quit()
        mover.shutdown()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    last_vector = (0, 0)
    last_nav_log = [0.0]
    # Black box: the last ~8 s of gameplay frames, written out when a GAME OVER
    # (REVIVE or QUIT) is handled, to see what actually killed the character.
    blackbox: list[tuple[float, object]] = []
    blackbox_dir = os.environ.get("VS_BLACKBOX_DIR", "/workspace/vsbot-check/deaths")
    last_box = [0.0]
    dumped_for = [0]
    t0 = time.monotonic()
    parked = False
    in_menu = False

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

            menu_kind = None
            if auto_menus and not bus.paused:
                screen = detect_level_up(frame)
                if screen is not None and not in_menu:
                    mover.release_and_clear()
                menu_kind = pilot.step(frame, screen)
                if menu_kind:
                    result.menu = True
                    result.note = pilot.status
            in_menu = menu_kind is not None
            if not cfg.demo and not bus.paused and not menu_kind:
                now_box = time.monotonic()
                if now_box - last_box[0] >= 0.5:
                    last_box[0] = now_box
                    blackbox.append((now_box, frame.copy()))
                    del blackbox[:-16]
            deaths = pilot.revives + (1 if pilot._quit_pressed else 0)
            if deaths > dumped_for[0] and blackbox:
                dumped_for[0] = deaths
                try:
                    import cv2

                    stamp = time.strftime("%H%M%S")
                    os.makedirs(blackbox_dir, exist_ok=True)
                    for i, (_t, fr) in enumerate(blackbox):
                        cv2.imwrite(os.path.join(blackbox_dir, f"death_{stamp}_{i:02d}.png"), fr)
                    print(f"[blackbox] saved {len(blackbox)} frames before death to {blackbox_dir}/death_{stamp}_*.png")
                except Exception as exc:  # noqa: BLE001
                    print(f"[blackbox] failed: {exc}")
                blackbox.clear()
            if bus.paused or result.menu:
                vector, keys = (0, 0), ()
            else:
                decision = nav.step(result, hp=pilot.hp)
                vector, keys = decision.vector, cfg.keys_for_vector(decision.vector)
                hp_txt = "?" if pilot.hp is None else f"{pilot.hp:.2f}"
                result.note = f"{decision.mode} d={decision.danger_here:.1f} hp={hp_txt}"
                now_log = time.monotonic()
                if now_log - last_nav_log[0] >= 3.0:
                    last_nav_log[0] = now_log
                    n_thr = 0 if result.threats is None else len(result.threats)
                    print(
                        f"[nav] game~{nav.game_s:5.0f}s {decision.mode} v={decision.vector} "
                        f"danger={decision.danger_here:.2f} cover={nav.cover:.2f} "
                        f"cells={n_thr} gems={len(result.gems)} hp={hp_txt} breakouts={nav.breakouts}"
                    )
            last_vector = vector
            if pilot.finished and not bus.paused:
                pilot_new_run[0] = True
                dumped_for[0] = 0
                # Final GAME OVER handled (QUIT + DONE). Stop acting; the game is
                # closed / reopened outside the bot. Press p to run again.
                bus.toggle_pause()
                mover.set_paused(True)

            if bus.paused:
                mover.set_paused(True)
            elif menu_kind:
                # Level-up / chest handled by the pilot; movement stays off.
                pass
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
                interrupted = mover.hold_current(min(cfg.hold_s, cfg.frame_s))
                if interrupted:
                    continue
        return 0
    finally:
        mover.shutdown()
        debug.close()
        hotkeys.stop()
        print("[main] keys released, goodbye")


if __name__ == "__main__":
    raise SystemExit(main())
