#!/bin/bash
# preflight.sh — check the box can actually run a live session before the first run.
# A box rebuild wipes system packages and /tmp (2026-09-26, 2026-10-02, 2026-10-07): the 32-bit
# loader Steam needs disappears, a /tmp venv vanishes, and a headless OpenCV kills the bot at
# cv2.namedWindow. Each of those used to burn runs as rc=1 failures; this reports them up front.
# Exit 0 = ready, 1 = something is missing (one PREFLIGHT_FAIL line per problem, with a fix hint).
set -uo pipefail
REPO=${VS_REPO:-/workspace/lucyfur-vs-vision-bot-v2}
PY=${VS_PY:-"$REPO/.venv/bin/python"}
LD32=${VS_LD32:-/lib/ld-linux.so.2}
TOOLS=${VS_PREFLIGHT_TOOLS:-"xdotool wmctrl xwininfo tesseract scrot"}
FAILS=0
fail(){ echo "PREFLIGHT_FAIL: $1 -> $2"; FAILS=$((FAILS+1)); }

[ -e "$LD32" ] || fail "32-bit loader $LD32 missing (Steam client cannot start)" \
  "sudo dpkg --add-architecture i386 && sudo apt-get update && sudo apt-get install libc6:i386 libgl1:i386 libgl1-mesa-dri:i386 libglx-mesa0:i386 libgtk2.0-0t64:i386 libnss3:i386 libxtst6:i386"

for t in $TOOLS; do
  command -v "$t" >/dev/null 2>&1 || fail "tool '$t' not on PATH" "reinstall it (see /workspace/steamfix-apt*.log)"
done

if [ ! -x "$PY" ]; then
  fail "bot python $PY missing (venv gone?)" "python3 -m venv <dir> && <dir>/bin/pip install 'opencv-python<5' numpy mss pytest; ln -sfn <dir> $REPO/.venv"
else
  OUT=$("$PY" - <<'PY' 2>&1
import sys
try:
    import cv2, numpy, mss  # noqa: F401
except Exception as e:  # pragma: no cover - reported to the shell
    print(f"import failed: {e}"); sys.exit(2)
gui = [l.split(":", 1)[1].strip() for l in cv2.getBuildInformation().splitlines() if l.strip().startswith("GUI:")]
if not gui or gui[0].upper() in ("", "NONE"):
    print(f"OpenCV {cv2.__version__} has no GUI backend (headless build)"); sys.exit(3)
print("ok")
PY
)
  case $? in
    0) : ;;
    3) fail "$OUT; the bot dies at cv2.namedWindow" "$PY -m pip uninstall -y opencv-python-headless && $PY -m pip install 'opencv-python<5'" ;;
    *) fail "bot python cannot import cv2/numpy/mss ($OUT)" "$PY -m pip install 'opencv-python<5' numpy mss" ;;
  esac
fi

if [ "$FAILS" -gt 0 ]; then echo "PREFLIGHT: $FAILS problem(s)"; exit 1; fi
echo "PREFLIGHT: ok"
