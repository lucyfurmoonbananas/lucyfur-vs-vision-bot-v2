"""scripts/play_run.sh guard + scripts/session_loop.sh rc handling, against stub binaries.

Nothing here touches X, Steam or the game: xdotool / wmctrl / xwininfo / scrot / steam.sh /
pkill / pgrep are shell stubs that only record their arguments, and the real run body is
replaced by VS_INNER_CMD.
"""

import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAY_RUN = ROOT / "scripts" / "play_run.sh"
SESSION_LOOP = ROOT / "scripts" / "session_loop.sh"

pytestmark = pytest.mark.skipif(
    any(shutil.which(b) is None for b in ("bash", "setsid", "timeout")) or not Path("/proc/uptime").exists(),
    reason="needs bash, setsid, timeout and /proc",
)

STUBS = {
    # Window "exists" once steam.sh has been called (the relaunch).
    "xdotool": """
        echo "xdotool $*" >> "$STUB_LOG"
        if [ "$1" = search ] && [ -f "$STUB_DIR/win_up" ]; then echo 4242; fi
        exit 0
    """,
    "wmctrl": 'echo "wmctrl $*" >> "$STUB_LOG"',
    "xwininfo": """
        echo "xwininfo $*" >> "$STUB_LOG"
        printf '  Absolute upper-left X:  200\\n  Absolute upper-left Y:  80\\n'
        printf '  Relative upper-left X:  5\\n  Relative upper-left Y:  29\\n'
        printf '  Width: 960\\n  Height: 540\\n'
    """,
    "scrot": 'echo "scrot $*" >> "$STUB_LOG"; exit 1',
    # STUB_WEDGED=1: launches do nothing until "steam.sh -shutdown" (a wedged client).
    "steam.sh": """
        echo "steam.sh $*" >> "$STUB_LOG"
        if [ "$1" = -shutdown ]; then touch "$STUB_DIR/restarted"; exit 0; fi
        if [ -z "${STUB_WEDGED:-}" ] || [ -f "$STUB_DIR/restarted" ]; then touch "$STUB_DIR/win_up"; fi
    """,
    "pkill": 'echo "pkill $*" >> "$STUB_LOG"; exit 0',
    "pgrep": "exit 1",
    # Pretend the Steam client is up; everything else is the real ps.
    "ps": """
        if [ "$*" = "-eo cmd" ]; then echo "/home/x/ubuntu12_32/steam -nofriendsui"; exit 0; fi
        exec /bin/ps "$@"
    """,
}


@pytest.fixture
def box(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in STUBS.items():
        p = bin_dir / name
        p.write_text("#!/bin/bash\n" + textwrap.dedent(body))
        p.chmod(0o755)
    session = tmp_path / "session"
    session.mkdir()
    (session / "shots").mkdir()
    env = dict(os.environ)
    env.update(
        VS_TEST_BIN=str(bin_dir),
        STUB_DIR=str(tmp_path),
        STUB_LOG=str(tmp_path / "calls.log"),
        VS_SESSION=str(session),
        VS_STEAM_SH=str(bin_dir / "steam.sh"),
        VS_PKILL=str(bin_dir / "pkill"),
        VS_PGREP=str(bin_dir / "pgrep"),
        VS_BOT_PIDFILE=str(tmp_path / "bot.pid"),
        VS_BOT_FIFO=str(tmp_path / "bot.fifo"),
        VS_FRESH_MARK=str(tmp_path / "fresh"),
        VS_WINPOS=str(tmp_path / "winpos.txt"),
        VS_GUARD_POLL_S="1",
        VS_CUTOFF_HHMM="none",
    )
    return tmp_path, session, env


def _calls(tmp_path: Path) -> str:
    p = tmp_path / "calls.log"
    return p.read_text() if p.exists() else ""


def _play(env, *, inner: str, timeout: float = 60, **extra) -> subprocess.CompletedProcess:
    env = dict(env, VS_INNER_CMD=inner, **extra)
    return subprocess.run(["bash", str(PLAY_RUN), "7"], env=env, capture_output=True, text=True, timeout=timeout)


def test_guard_passes_through_a_normal_run(box):
    tmp_path, session, env = box
    t0 = time.monotonic()
    res = _play(env, inner='echo "[x] r$1 working" >> "$VS_SESSION/loop.log"; sleep 1; exit 0')
    assert res.returncode == 0, res.stdout + res.stderr
    assert time.monotonic() - t0 < 15
    assert "GUARD_TRIP" not in (session / "loop.log").read_text()
    assert "steam.sh" not in _calls(tmp_path)


def test_guard_passes_through_inner_exit_codes(box):
    _tmp, _session, env = box
    assert _play(env, inner="exit 2").returncode == 2
    assert _play(env, inner="exit 3").returncode == 3


def test_silent_run_is_killed_and_the_game_relaunched(box):
    tmp_path, session, env = box
    # Writes one log line, then hangs forever (like a wedged OCR / xdotool call).
    inner = 'echo "[x] r$1 boot" >> "$VS_SESSION/loop.log"; echo $$ > "$STUB_DIR/inner.pid"; sleep 1000'
    res = _play(env, inner=inner, VS_LOG_SILENCE_LIMIT_S="3", VS_RUN_WALL_LIMIT_S="300")
    assert res.returncode == 3, res.stdout + res.stderr
    log = (session / "loop.log").read_text()
    assert "GUARD_TRIP silence" in log
    assert "RECOVER done" in log
    calls = _calls(tmp_path)
    assert "pkill -TERM -f VampireSurvivors\\.exe" in calls
    assert "steam.sh steam://rungameid/1794680" in calls
    assert "xdotool windowsize 4242 960 540" in calls
    assert (tmp_path / "fresh").exists()
    assert "silence" in (session / "run7_guard.txt").read_text()
    pid = int((tmp_path / "inner.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the hung run (and its process group) is gone


def test_wedged_steam_is_restarted_when_no_window_appears(box):
    """Oct 15 3 AM: Steam up but stuck on a cloud-sync dialog -> restart it, then the game comes up."""
    tmp_path, session, env = box
    inner = 'echo "[x] r$1 boot" >> "$VS_SESSION/loop.log"; sleep 1000'
    res = _play(
        env, inner=inner, VS_LOG_SILENCE_LIMIT_S="3", VS_RUN_WALL_LIMIT_S="300",
        STUB_WEDGED="1", VS_WAIT_SLEEP_S="0.1", VS_STEAM_RESTART_AT="5",
    )
    assert res.returncode == 3, res.stdout + res.stderr
    log = (session / "loop.log").read_text()
    assert "STEAM_RESTART" in log
    assert "RECOVER done" in log
    calls = _calls(tmp_path)
    assert "steam.sh -shutdown" in calls
    assert "pkill -KILL -f ubuntu12_32/steam" in calls  # stub ps keeps "Steam up" -> forced kill
    assert calls.index("steam.sh -shutdown") < calls.rindex("steam.sh steam://rungameid/1794680")


def test_no_steam_restart_when_window_comes_up(box):
    tmp_path, session, env = box
    inner = 'echo "[x] r$1 boot" >> "$VS_SESSION/loop.log"; sleep 1000'
    res = _play(env, inner=inner, VS_LOG_SILENCE_LIMIT_S="3", VS_RUN_WALL_LIMIT_S="300",
                VS_WAIT_SLEEP_S="0.1", VS_STEAM_RESTART_AT="5")
    assert res.returncode == 3
    assert "STEAM_RESTART" not in (session / "loop.log").read_text()
    assert "-shutdown" not in _calls(tmp_path)


def test_chatty_run_over_the_wall_limit_is_killed(box):
    tmp_path, session, env = box
    inner = 'while true; do echo "[x] r$1 tick" >> "$VS_SESSION/loop.log"; sleep 0.5; done'
    res = _play(env, inner=inner, VS_LOG_SILENCE_LIMIT_S="60", VS_RUN_WALL_LIMIT_S="4")
    assert res.returncode == 3
    assert "GUARD_TRIP wall" in (session / "loop.log").read_text()
    assert "steam.sh steam://rungameid/1794680" in _calls(tmp_path)


def _session_with_fake_play_run(session: Path, codes: list[int]) -> None:
    seq = " ".join(str(c) for c in codes)
    (session / "play_run.sh").write_text(
        "#!/bin/bash\n"
        f"codes=({seq})\n"
        'i=$(( $1 - 1 )); [ $i -lt ${#codes[@]} ] || exit 2\n'
        "exit ${codes[$i]}\n"
    )
    (session / "play_run.sh").chmod(0o755)
    shutil.copy(SESSION_LOOP, session / "session_loop.sh")


def _loop(env, session: Path) -> str:
    env = dict(env, VS_RUN_GAP_S="0", VS_TEST_NO_CLEANUP="1")
    subprocess.run(["bash", str(session / "session_loop.sh")], env=env, capture_output=True, text=True, timeout=60)
    return (session / "session-log.md").read_text()


def test_session_loop_does_not_count_watchdog_rc3_as_failures(box):
    _tmp, session, env = box
    _session_with_fake_play_run(session, [3] * 10 + [0])
    md = _loop(env, session)
    assert "too many consecutive failures" not in md
    assert "### Run 11" in md  # kept going past 8 watchdog recoveries
    assert md.count("Watchdog ended the run") == 10


def test_session_loop_still_stops_after_8_real_failures(box):
    _tmp, session, env = box
    _session_with_fake_play_run(session, [1, 3, 1, 1, 1, 1, 1, 1, 1, 0])
    md = _loop(env, session)
    assert "too many consecutive failures (8)" in md
    assert "### Run 9" in md and "### Run 10" not in md


def test_session_loop_crosses_early_engage_ab_with_laurel(box):
    _tmp, session, env = box
    (session / "play_run.sh").write_text(
        "#!/bin/bash\n"
        '[ "$1" -le 4 ] || exit 2\n'
        'echo "$1 tune=${VS_TUNE-<unset>} laurel=${VS_LAUREL_DAMAGE-<unset>}" >> "$VS_SESSION/envs.txt"\n'
        'echo "[levelup] cards=[\'Whip\'] new=[True] -> pick #1 Whip (hp=1.0, weapons=1, game~=50s)" > "$VS_SESSION/run$1.log"\n'
        'echo "[levelup] picked #1 Whip (total picks 1)" >> "$VS_SESSION/run$1.log"\n'
        "exit 0\n"
    )
    (session / "play_run.sh").chmod(0o755)
    shutil.copy(SESSION_LOOP, session / "session_loop.sh")
    shutil.copy(ROOT / "scripts" / "run_metrics.py", session / "run_metrics.py")
    md = _loop(dict(env, VS_TUNE="ring_commit_s=2.5"), session)
    envs = (session / "envs.txt").read_text().splitlines()
    assert envs == [
        "1 tune=ring_commit_s=2.5 laurel=0",
        "2 tune=ring_commit_s=2.5 laurel=<unset>",
        "3 tune=ring_commit_s=2.5,early_engage_until_s=0 laurel=0",
        "4 tune=ring_commit_s=2.5,early_engage_until_s=0 laurel=<unset>",
    ]
    # run 5 (rc 2, the stub cutoff) is logged too, as ON
    assert md.count("- Early engage: ON") == 3 and md.count("- Early engage: OFF") == 2
    assert "first_levelup_s=50" in md and "lv_at_120=2" in md
