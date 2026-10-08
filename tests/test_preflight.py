"""scripts/preflight.sh and the session_loop.sh preflight gate, with stub pythons (no X, no Steam)."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "preflight.sh"
SESSION_LOOP = ROOT / "scripts" / "session_loop.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run(tmp_path, *, py_body="cat >/dev/null; echo ok; exit 0", ld32=True, tools="bash", py=True):
    ld = tmp_path / "ld-linux.so.2"
    if ld32:
        ld.write_text("")
    env = dict(os.environ)
    env["VS_LD32"] = str(ld)
    env["VS_PREFLIGHT_TOOLS"] = tools
    env["VS_PY"] = str(_exe(tmp_path / "py", py_body)) if py else str(tmp_path / "nope" / "python")
    return subprocess.run(["bash", str(PREFLIGHT)], env=env, capture_output=True, text=True, timeout=30)


def test_ok_when_everything_is_there(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    assert "PREFLIGHT: ok" in r.stdout
    assert "PREFLIGHT_FAIL" not in r.stdout


def test_missing_32bit_loader_is_reported(tmp_path):
    r = _run(tmp_path, ld32=False)
    assert r.returncode == 1
    assert "32-bit loader" in r.stdout and "i386" in r.stdout


def test_missing_tool_is_reported(tmp_path):
    r = _run(tmp_path, tools="bash definitely-not-a-real-tool-xyz")
    assert r.returncode == 1
    assert "definitely-not-a-real-tool-xyz" in r.stdout


def test_missing_venv_is_reported(tmp_path):
    r = _run(tmp_path, py=False)
    assert r.returncode == 1
    assert "venv gone" in r.stdout


def test_headless_opencv_is_reported(tmp_path):
    r = _run(tmp_path, py_body="cat >/dev/null; echo 'OpenCV 5.0.0 has no GUI backend (headless build)'; exit 3")
    assert r.returncode == 1
    assert "namedWindow" in r.stdout and "opencv-python<5" in r.stdout


def test_import_failure_is_reported(tmp_path):
    r = _run(tmp_path, py_body="cat >/dev/null; echo \"import failed: No module named 'cv2'\"; exit 2")
    assert r.returncode == 1
    assert "cannot import" in r.stdout


def test_several_problems_counted(tmp_path):
    r = _run(tmp_path, ld32=False, py=False)
    assert r.returncode == 1
    assert r.stdout.count("PREFLIGHT_FAIL") == 2
    assert "2 problem(s)" in r.stdout


def test_real_python_check_matches_this_interpreter(tmp_path):
    """The embedded check really runs under a python (this one) and agrees with cv2's build info."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mss")
    gui = [l.split(":", 1)[1].strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith("GUI:")]
    env = dict(os.environ, VS_LD32=sys.executable, VS_PREFLIGHT_TOOLS="bash", VS_PY=sys.executable)
    r = subprocess.run(["bash", str(PREFLIGHT)], env=env, capture_output=True, text=True, timeout=60)
    headless = not gui or gui[0].upper() == "NONE"
    assert (r.returncode == 1) == headless, r.stdout


def _session(tmp_path, preflight_rc):
    s = tmp_path / "session"
    s.mkdir()
    _exe(s / "play_run.sh", 'touch "$VS_SESSION/played_$1"; exit 2')  # rc 2 = cutoff, ends the loop
    _exe(s / "preflight.sh", f'echo "PREFLIGHT_FAIL: stub -> fix"; exit {preflight_rc}' if preflight_rc
         else 'echo "PREFLIGHT: ok"; exit 0')
    shutil.copy(SESSION_LOOP, s / "session_loop.sh")
    env = dict(os.environ, VS_SESSION=str(s), VS_CUTOFF_HHMM="none", VS_TEST_NO_CLEANUP="1", VS_RUN_GAP_S="0")
    env.pop("VS_SKIP_PREFLIGHT", None)
    env.pop("VS_PREFLIGHT", None)
    r = subprocess.run(["bash", str(s / "session_loop.sh")], env=env, capture_output=True, text=True, timeout=60)
    return s, r


def test_session_loop_stops_before_any_run_when_preflight_fails(tmp_path):
    s, r = _session(tmp_path, preflight_rc=1)
    assert r.returncode == 4
    assert not list(s.glob("played_*"))
    log = (s / "session-log.md").read_text()
    assert "preflight failed" in log and "PREFLIGHT_FAIL: stub" in log


def test_session_loop_runs_when_preflight_passes(tmp_path):
    s, r = _session(tmp_path, preflight_rc=0)
    assert r.returncode == 0
    assert (s / "played_1").exists()
    assert "PREFLIGHT: ok" in (s / "loop_outer.log").read_text()
