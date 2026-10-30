"""Main loop: start paused, watch the playfield, dodge, send WASD via xdotool."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque

from vs_vision_bot.capture import ScreenCapture
from vs_vision_bot.config import Config
from vs_vision_bot.controller import MoveController
from vs_vision_bot.debug_window import DebugWindow
from vs_vision_bot.hotkeys import CommandBus, HotkeyService
from vs_vision_bot.input_backend import make_backend
from vs_vision_bot.menus import (
    MenuPilot,
    detect_level_up,
    hp_for_nav,
    read_critical_flag,
    read_hp_fraction,
    tesseract_available,
)
from vs_vision_bot.pathfind import Navigator, chest_status, gem_status, tuning_from_env
from vs_vision_bot.vision import Vision, draw_synthetic_arena
from vs_vision_bot.watchdog import (
    FRAME_CHANGE_MIN,
    LoopHeartbeat,
    ProgressWatchdog,
    Stall,
    frame_signature,
    signature_change,
)

# Exit status when a watchdog ended the run. play_run.sh (scripts/) answers it by
# killing the game, relaunching it and realigning the window; session_loop does not
# count it as a failure.
EXIT_WATCHDOG = 3
# After a REVIVE click, log the move vector and held keys this long (~4 lines/s).
REVIVE_TRACE_S = 3.5  # covers the whole ~3 s revive escape
REVIVE_TRACE_EVERY_S = 0.25
# Per-frame HP / danger / mode kept across this window and dumped on REVIVE or QUIT.
DEATH_TRACE_S = 5.0
DEATH_TRACE_FRAMES = 2  # immediate frames; a static death screen is replaced by older snaps
DEATH_TRACE_SNAP_S = 0.5
DEATH_TRACE_SNAP_KEEP_S = 8.0  # outlasts the OCR wait on the game-over screen
DEATH_KINDS = ("revive", "quit")
# During the committed post-REVIVE escape a vision-heuristic-only menu flag (no
# pilot / OCR confirmation) must hold this many consecutive frames before the
# character is frozen. Real level-ups are confirmed by the pilot within a frame.
ESCAPE_HEURISTIC_MENU_FRAMES = 3


def _trace_num(v: float | None) -> str:
    return "none" if v is None else f"{v:.2f}"


def _frames_alike(a, b) -> bool:
    """True when both frames are the same static screen.

    A GAME OVER overlay drifts well under ``FRAME_CHANGE_MIN``; live gameplay does not.
    """
    if a is None or b is None or getattr(a, "shape", None) != getattr(b, "shape", None):
        return False
    return signature_change(frame_signature(a), frame_signature(b)) < FRAME_CHANGE_MIN


class DeathTrace:
    """Last ``window_s`` of per-frame HP read, danger and nav mode, plus frames.

    ``observe`` stores numbers, the last ``prior_frames`` frame references, and a
    copy every ``DEATH_TRACE_SNAP_S`` kept for ``DEATH_TRACE_SNAP_KEEP_S``. ``dump``
    logs the window. When the frames in hand are a static death screen, it writes
    the earlier snaps; otherwise it writes the triggering frame plus those priors.
    """

    def __init__(
        self,
        save_dir: str,
        window_s: float = DEATH_TRACE_S,
        prior_frames: int = DEATH_TRACE_FRAMES,
        log=print,
    ) -> None:
        self.save_dir = save_dir
        self.window_s = window_s
        self.log = log
        self.samples: deque = deque()
        self._priors: deque = deque(maxlen=prior_frames)
        self._hold = None
        self._snaps: deque = deque()
        self._snap_at: float | None = None
        self._seq = 0

    def observe(
        self,
        t: float,
        hp: float | None,
        med: float | None,
        danger: float | None,
        mode: str,
        frame,
    ) -> None:
        if self._hold is not None:
            self._priors.append(self._hold)
        self._hold = frame
        now = float(t)
        self.samples.append((now, hp, med, danger, mode))
        cutoff = now - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if frame is not None and (self._snap_at is None or now - self._snap_at >= DEATH_TRACE_SNAP_S):
            self._snaps.append((now, frame.copy()))
            self._snap_at = now
            cut = now - DEATH_TRACE_SNAP_KEEP_S
            while self._snaps and self._snaps[0][0] < cut:
                self._snaps.popleft()

    def clear(self) -> None:
        self.samples.clear()
        self._priors.clear()
        self._hold = None
        self._snaps.clear()
        self._snap_at = None

    def _frames_to_save(self, frame) -> list:
        recent = [fr for fr in (frame, *reversed(self._priors)) if fr is not None]
        static = len(recent) >= 2 and all(_frames_alike(recent[0], fr) for fr in recent[1:])
        if not static:
            return recent
        older = [img for _ts, img in self._snaps if not _frames_alike(img, frame)]
        return older if older else recent

    def dump(self, kind: str, frame, t: float, game_s: float | None) -> list[str]:
        when = "?" if game_s is None else f"{game_s:.0f}"
        self.log(f"[death-trace] {kind} game~{when}s n={len(self.samples)}")
        for ts, hp, med, danger, mode in self.samples:
            self.log(
                f"[death-trace] {ts - t:6.2f}s hp={_trace_num(hp)} med={_trace_num(med)} "
                f"danger={_trace_num(danger)} mode={mode}"
            )
        paths: list[str] = []
        try:
            import cv2

            os.makedirs(self.save_dir, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            self._seq += 1
            for i, fr in enumerate(self._frames_to_save(frame)):
                path = os.path.join(
                    self.save_dir, f"death_{kind}_{stamp}_{self._seq:02d}_{i:02d}.png"
                )
                if cv2.imwrite(path, fr):
                    paths.append(path)
            self.log("[death-trace] saved " + " ".join(paths))
        except Exception as exc:  # noqa: BLE001
            self.log(f"[death-trace] frame save failed: {exc}")
        return paths


def format_logged_hp(median: float | None, acted: float | None) -> str:
    """HP text for nav logs.

    The tracker median is ``?`` while a revive is still blind. Nav may already
    be acting on 0.0 (low_held). Show that, so the line is not read as healthy.
    """
    base = "?" if median is None else f"{median:.2f}"
    if acted is None:
        return base
    if median is None or abs(float(acted) - float(median)) > 0.02:
        return f"{base} nav={float(acted):.2f}"
    return base


def _pilot_progress(pilot: object) -> tuple[int, int]:
    """(level-up picks, chest confirms) seen so far. Stubs may omit both."""
    return (len(getattr(pilot, "picks", ())), int(getattr(pilot, "chest_presses", 0)))


def sync_build_state(nav: object, pilot: object) -> None:
    """Push evolution flags to the navigator when it accepts them.

    Parallel chest gating adds ``Navigator.set_build_state(evolution_ready,
    all_maxed)``. This loop calls it after a level-up pick and after a chest
    screen. Missing the method (that change not merged yet) is a no-op.
    """
    fn = getattr(nav, "set_build_state", None)
    if not callable(fn):
        return
    fn(bool(getattr(pilot, "evolution_ready", False)), bool(getattr(pilot, "all_maxed", False)))


def gate_heuristic_menu(
    menu: bool, menu_kind: str | None, escaping: bool, streak: int,
    need: int = ESCAPE_HEURISTIC_MENU_FRAMES,
) -> tuple[bool, int]:
    """(menu flag to act on, new heuristic streak).

    Only a heuristic-only flag (``menu`` set by vision, ``menu_kind`` None)
    during a revive escape is downgraded, until ``need`` frames agree. A
    pilot-confirmed menu always freezes; outside the escape nothing changes.
    """
    if not menu or menu_kind is not None:
        return menu, 0
    if not escaping:
        return menu, 0
    streak += 1
    return streak >= need, streak


def _overlay_may_clear_latch(tracker: object) -> bool:
    """A real low bar (``low_held``) is not an overlay lie.

    test7 cleared the latch on every level-up while HP was actually ~0.17, so
    the next frame logged ``critical HP on hp=0.00`` again.
    """
    return not bool(getattr(tracker, "low_held", False))


def note_menu_overlay(nav: object, pilot: object, menu_kind: str | None, armed: bool = False) -> bool:
    """Clear a false critical / HP-crash when a chest or level-up covers the bar.

    Returns whether that overlay is still up. The frame the menu closes resumes
    the HP tracker so the darkened 0.00 samples do not latch the next step.
    A tracker that is already holding a real low reading is left latched.
    """
    tracker = getattr(pilot, "_hp", None)
    # Read before resume: resume itself drops low_held, and a real low bar
    # must stay latched on both the open and the close of the menu.
    genuine_low = not _overlay_may_clear_latch(tracker)
    if menu_kind in ("chest", "levelup"):
        suspend = getattr(tracker, "suspend_for_overlay", None)
        if callable(suspend):
            suspend()
        clear = getattr(nav, "clear_menu_overlay_latch", None)
        if callable(clear) and not genuine_low:
            clear()
        return True
    if armed:
        resume = getattr(tracker, "resume_after_overlay", None)
        if callable(resume):
            resume()
        clear = getattr(nav, "clear_menu_overlay_latch", None)
        if callable(clear) and not genuine_low:
            clear()
    return False


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


def _abort_blocked_loop(stall: Stall, release_keys) -> None:
    """Heartbeat callback (runs on the heartbeat thread): the main loop is wedged.

    Best-effort key release on a helper thread (it may itself hit a wedged X
    server; xdotool calls are bounded), then hard-exit so the session script
    sees the bot die and can recover the game.
    """
    print(f"[watchdog] STALL {stall.describe()} -> ending run (exit {EXIT_WATCHDOG})", flush=True)
    t = threading.Thread(target=release_keys, daemon=True)
    t.start()
    t.join(15.0)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_WATCHDOG)


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
                sync_build_state(nav, pilot)
                build_seen[0], build_seen[1] = _pilot_progress(pilot)

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
        game_clock=lambda: nav.game_s,  # nav is created just below; only called on a level-up
        critical_hp=lambda: read_critical_flag(nav),
        # Stage for the Mad-Forest-only Pentagram banish; play_run.sh only starts Mad Forest.
        stage=os.environ.get("VS_STAGE", ""),
    )
    nav = Navigator(tune=tuning_from_env(os.environ.get("VS_TUNE")))
    # The movement game clock restarts only after a finished run (QUIT + DONE),
    # not on every manual pause.
    pilot_new_run = [False]
    # picks and chest confirms already pushed via set_build_state.
    build_seen = list(_pilot_progress(pilot))
    sync_build_state(nav, pilot)
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

    # No-progress watchdogs (monotonic clock: a VM pause does not trip them).
    watchdog = ProgressWatchdog(
        limit_s=0 if cfg.demo else cfg.watchdog_s,
        identical_s=0 if cfg.demo else cfg.stale_frame_s,
    )
    heartbeat = LoopHeartbeat(
        lambda stall: _abort_blocked_loop(stall, mover.shutdown),
        limit_s=0 if cfg.demo else cfg.loop_stall_s,
    )
    heartbeat.start()
    exit_code = 0

    last_vector = (0, 0)
    last_nav_log = [0.0]
    # Black box: the last ~8 s of gameplay frames, written out when a GAME OVER
    # (REVIVE or QUIT) is handled, to see what actually killed the character.
    blackbox: list[tuple[float, object]] = []
    blackbox_dir = os.environ.get("VS_BLACKBOX_DIR", "/workspace/vsbot-check/deaths")
    last_box = [0.0]
    dumped_for = [0]
    trace_root = (
        os.environ.get("VS_SESSION")
        or os.environ.get("VS_RESULTS_DIR")
        or "/workspace/vsbot-check/results"
    )
    trace = DeathTrace(os.path.join(trace_root, "deaths"))
    trace_dumped = [0]
    t0 = time.monotonic()
    revive_seen = [0]
    revive_trace = [0.0, -1e9, 0.0]  # started_at, until, last_line_at
    parked = False
    in_menu = False
    prev_menu = None
    heur_streak = [0]
    overlay_armed = False

    try:
        while not bus.quit:
            heartbeat.beat()
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
                    stall = watchdog.update(None, paused=bus.paused)
                    if stall is not None:
                        print(f"[watchdog] STALL {stall.describe()} -> ending run (exit {EXIT_WATCHDOG})")
                        exit_code = EXIT_WATCHDOG
                        break
                    time.sleep(cfg.frame_s)
                    continue

            if not parked:
                debug.ensure(capture.region, capture.display_size())
                parked = True

            result = vision.analyze(frame)
            scroller = getattr(vision, "background_scroll", None)
            result.scroll = scroller(frame) if scroller is not None else None

            menu_kind = None
            if auto_menus and not bus.paused:
                screen = detect_level_up(frame)
                if screen is not None and not in_menu:
                    mover.release_and_clear()
                menu_kind = pilot.step(frame, screen)
                picks_n, chests_n = _pilot_progress(pilot)
                if picks_n != build_seen[0] or chests_n != build_seen[1]:
                    build_seen[0], build_seen[1] = picks_n, chests_n
                    sync_build_state(nav, pilot)
                if menu_kind:
                    result.menu = True
                    result.note = pilot.status
                if menu_kind == "revive":
                    # Committed ~3 s escape out of the death crowd (no gem pull,
                    # no small-gap re-picks, heuristic menu hints ignored).
                    nav.request_revive_escape()
                elif menu_kind == "chest" and prev_menu != "chest":
                    # One Treasure Found screen spans several frames. The OPEN
                    # prompt spends one free chest; the DONE reward is the same
                    # open (chest_is_reward) and must not spend a second.
                    nav.on_chest_opened(reward=bool(getattr(pilot, "chest_is_reward", False)))
            prev_menu = menu_kind
            overlay_armed = note_menu_overlay(nav, pilot, menu_kind, overlay_armed)
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
            menu_hint = ""
            if result.menu and not menu_kind:
                acted, heur_streak[0] = gate_heuristic_menu(True, None, nav.escaping, heur_streak[0])
                if not acted:
                    result.menu = False
                    menu_hint = f" (menu hint ignored x{heur_streak[0]})"
            else:
                heur_streak[0] = 0
            if bus.paused or result.menu:
                vector, keys = (0, 0), ()
                if not bus.paused:
                    now_obs = time.monotonic()
                    trace.observe(
                        now_obs, read_hp_fraction(frame), pilot.hp, None, menu_kind or "menu", frame,
                    )
            else:
                if hasattr(pilot, "hp_acted"):
                    acted_hp = pilot.hp_acted()
                else:
                    acted_hp = hp_for_nav(
                        pilot.hp,
                        getattr(pilot, "hp_blind", False),
                        getattr(pilot, "hp_low_held", False),
                    )
                decision = nav.step(result, hp=acted_hp)
                vector, keys = decision.vector, cfg.keys_for_vector(decision.vector)
                hp_txt = format_logged_hp(pilot.hp, acted_hp)
                result.note = f"{decision.mode} d={decision.danger_here:.1f} hp={hp_txt}{menu_hint}"
                now_obs = time.monotonic()
                if now_obs - last_nav_log[0] >= 3.0:
                    last_nav_log[0] = now_obs
                    n_thr = 0 if result.threats is None else len(result.threats)
                    print(
                        f"[nav] game~{nav.game_s:5.0f}s {decision.mode} v={decision.vector} "
                        f"danger={decision.danger_here:.2f} cover={nav.cover:.2f} "
                        f"cells={n_thr} gems={len(result.gems)} hp={hp_txt} breakouts={nav.breakouts} "
                        f"{chest_status(result)} {gem_status(nav)} dive={nav.dive:.1f}"
                    )
                trace.observe(
                    now_obs, read_hp_fraction(frame), pilot.hp,
                    decision.danger_here, decision.mode, frame,
                )
            if not bus.paused and deaths > trace_dumped[0]:
                trace_dumped[0] = deaths
                if menu_kind in DEATH_KINDS:
                    trace.dump(menu_kind, frame, now_obs, nav.game_s)
            last_vector = vector
            if pilot.finished and not bus.paused:
                nav.log_gem_summary()
                pilot_new_run[0] = True
                dumped_for[0] = 0
                trace_dumped[0] = 0
                trace.clear()
                # Final GAME OVER handled (QUIT + DONE). Stop acting; the game is
                # closed / reopened outside the bot. Press p to run again.
                bus.toggle_pause()
                mover.set_paused(True)

            stall = watchdog.update(frame, paused=bus.paused, in_menu=bool(result.menu))
            if stall is not None:
                nav.log_gem_summary()
                print(f"[watchdog] STALL {stall.describe()} game~{nav.game_s:.0f}s -> ending run (exit {EXIT_WATCHDOG})")
                exit_code = EXIT_WATCHDOG
                break

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

            if pilot.revives != revive_seen[0]:
                if pilot.revives > revive_seen[0]:
                    now_rv = time.monotonic()
                    revive_trace[0], revive_trace[1] = now_rv, now_rv + REVIVE_TRACE_S
                revive_seen[0] = pilot.revives
            now_tr = time.monotonic()
            if now_tr < revive_trace[1] and now_tr - revive_trace[2] >= REVIVE_TRACE_EVERY_S:
                revive_trace[2] = now_tr
                held = ",".join(sorted(mover.held)) or "-"
                print(
                    f"[revive+{now_tr - revive_trace[0]:.1f}s] v={vector} keys={','.join(keys) or '-'} "
                    f"held={held} menu={menu_kind or '-'} {result.note}"
                )

            debug.render(frame, result, bus.paused, mover, vector)
            hotkeys.handle_waitkey(debug.poll_key())

            if bus.paused:
                time.sleep(cfg.frame_s)
            else:
                interrupted = mover.hold_current(min(cfg.hold_s, cfg.frame_s))
                if interrupted:
                    continue
        return exit_code
    finally:
        nav.log_gem_summary()
        heartbeat.stop()
        mover.shutdown()
        debug.close()
        hotkeys.stop()
        print("[main] keys released, goodbye")


if __name__ == "__main__":
    raise SystemExit(main())
