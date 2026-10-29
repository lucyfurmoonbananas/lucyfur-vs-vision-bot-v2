#!/bin/bash
# session_loop.sh — run play_run.sh RUN=1,2,... until the cutoff, too many failures, or 40 runs.
# Template: lives next to play_run.sh (+ clk.sh, ocr_results.py) in a session folder.
# play_run.sh exit codes: 0 finished, 1 failed, 2 cutoff (stop), 3 a watchdog ended the run and
# the game was relaunched — rc 3 is NOT counted toward the consecutive-failure stop (a VM pause or
# a frozen game is not the bot failing to start runs).
# The loop itself exits 4 when preflight.sh (next to it, or VS_PREFLIGHT) reports a broken box.
set -uo pipefail
SESSION=${VS_SESSION:-$(dirname "$(readlink -f "$0")")}
export VS_SESSION=$SESSION
export TZ=America/Chicago
export DISPLAY=:6
export PATH="/home/box/.local/bin:/home/box/micromamba/envs/vstools/bin:$PATH"
export TESSDATA_PREFIX=/home/box/micromamba/envs/vstools/share/tessdata
export PATH="/home/box/.local/share/Steam:$PATH"
# Manual runs do not stop at 05:45 CT. Set VS_CUTOFF_HHMM=0545 for an overnight loop.
# Exported so play_run.sh (its own default is 0545) inherits the same switch.
CUTOFF_HHMM=${VS_CUTOFF_HHMM:-none}
export VS_CUTOFF_HHMM="$CUTOFF_HHMM"
MAX_FAILS=${VS_MAX_FAILS:-8}
MAX_RUNS=${VS_MAX_RUNS:-40}
RUN_GAP_S=${VS_RUN_GAP_S:-3}
BOT_PIDFILE=${VS_BOT_PIDFILE:-/tmp/vsbot_3am.pid}
cutoff_reached(){ [ "$CUTOFF_HHMM" != none ] && [ "$((10#$(date +%H%M)))" -ge "$((10#$CUTOFF_HHMM))" ]; }
echo "PLAY session loop started $(date)" | tee -a "$SESSION/loop_outer.log" "$SESSION/session-log.md"
# Preflight (box rebuilds wipe the 32-bit loader / venv / GUI OpenCV): stop before burning runs.
PREFLIGHT=${VS_PREFLIGHT:-$SESSION/preflight.sh}
if [ -z "${VS_SKIP_PREFLIGHT:-}" ] && [ -x "$PREFLIGHT" ]; then
  if ! PF_OUT=$("$PREFLIGHT" 2>&1); then
    { echo "[$(date)] preflight failed, no runs started:"; echo "$PF_OUT"; } | tee -a "$SESSION/loop_outer.log" "$SESSION/session-log.md"
    exit 4
  fi
  echo "[$(date)] $(echo "$PF_OUT" | tail -1)" >> "$SESSION/loop_outer.log"
fi
RUN=1
FAILS=0
BASE_TUNE=${VS_TUNE:-}
METRICS=${VS_METRICS:-$SESSION/run_metrics.py}
while true; do
  if cutoff_reached; then
    echo "[$(date)] cutoff (no new runs at/after $CUTOFF_HHMM)" | tee -a "$SESSION/loop_outer.log"
    break
  fi
  # Laurel A/B (bd85a14): odd runs VS_LAUREL_DAMAGE=0 (new Laurel out of the damage tier),
  # even runs unset (current behaviour). VS_LAUREL_AB=off disables the alternation.
  if [ "${VS_LAUREL_AB:-on}" != off ] && [ $((RUN % 2)) -eq 1 ]; then
    export VS_LAUREL_DAMAGE=0; LAUREL_VARIANT="B (VS_LAUREL_DAMAGE=0)"
  else
    unset VS_LAUREL_DAMAGE; LAUREL_VARIANT="A (VS_LAUREL_DAMAGE unset)"
  fi
  # Early-gem-target A/B (Bob's Oct 16 plan), crossed with Laurel so both stay balanced:
  # odd runs OFF (VS_TUNE early_gem_target_gain=0), even runs ON (default 1.4). Any VS_TUNE the
  # caller set is kept and merged. VS_GEM_AB=off disables it.
  if [ "${VS_GEM_AB:-on}" != off ] && [ $((RUN % 2)) -eq 1 ]; then
    export VS_TUNE="${BASE_TUNE:+$BASE_TUNE,}early_gem_target_gain=0"; GEM_VARIANT="OFF (early_gem_target_gain=0)"
  else
    if [ -n "$BASE_TUNE" ]; then export VS_TUNE="$BASE_TUNE"; else unset VS_TUNE; fi
    GEM_VARIANT="ON (default 1.4)"
  fi
  echo "[$(date)] launching run $RUN laurel=$LAUREL_VARIANT gem=$GEM_VARIANT" | tee -a "$SESSION/loop_outer.log"
  "$SESSION/play_run.sh" "$RUN" >> "$SESSION/loop_outer.log" 2>&1
  RC=$?
  echo "[$(date)] run $RUN rc=$RC" | tee -a "$SESSION/loop_outer.log"
  {
    echo ""
    echo "### Run $RUN — $(date '+%H:%M CT') rc=$RC"
    echo "- Laurel variant: $LAUREL_VARIANT"
    echo "- Early gem target: $GEM_VARIANT"
    if [ -f "$METRICS" ] && [ -f "$SESSION/run${RUN}.log" ]; then
      echo "- Early levelling: $(python3 "$METRICS" "$SESSION/run${RUN}.log" 2>/dev/null | tr '\n' ' ')"
    fi
    [ "$RC" -eq 3 ] && echo "- Watchdog ended the run; game relaunched$( [ -f "$SESSION/run${RUN}_guard.txt" ] && echo " (guard: $(cat "$SESSION/run${RUN}_guard.txt"))")"
    [ -f "$SESSION/run${RUN}_char.txt" ] && echo "- Character: $(cat "$SESSION/run${RUN}_char.txt")"
    [ -f "$SESSION/run${RUN}_revives.txt" ] && echo "- Revives: $(cat "$SESSION/run${RUN}_revives.txt")"
    [ -f "$SESSION/run${RUN}_stats.txt" ] && echo "- Stats: $(tr '\n' ' ' < "$SESSION/run${RUN}_stats.txt")"
    [ -f "$SESSION/run${RUN}_results_path.txt" ] && echo "- Screenshot: $(cat "$SESSION/run${RUN}_results_path.txt")"
  } >> "$SESSION/session-log.md"
  if [ "$RC" -eq 2 ]; then break; fi
  case "$RC" in
    0) FAILS=0 ;;
    3) : ;;                      # watchdog recovery: neither a success nor a failure
    *) FAILS=$((FAILS+1)) ;;
  esac
  if [ "$FAILS" -ge "$MAX_FAILS" ]; then
    echo "[$(date)] too many consecutive failures ($FAILS); stopping early" | tee -a "$SESSION/loop_outer.log" "$SESSION/session-log.md"
    break
  fi
  RUN=$((RUN+1))
  if [ "$RUN" -gt "$MAX_RUNS" ]; then echo too_many_runs; break; fi
  sleep "$RUN_GAP_S"
done
[ -n "${VS_TEST_NO_CLEANUP:-}" ] && exit 0
echo "[$(date)] cleanup" | tee -a "$SESSION/loop_outer.log"
[ -f "$BOT_PIDFILE" ] && kill "$(cat "$BOT_PIDFILE")" 2>/dev/null || true
# kill bot processes carefully
ps -eo pid,cmd | awk '/vs_vision_bot|lucyfur-vs-vision-bot/ && !/awk/{print $1}' | while read -r p; do kill "$p" 2>/dev/null || true; done
timeout 10 xdotool keyup --clearmodifiers a s d w Left Right Up Down Return Shift_L Shift_R Control_L 2>/dev/null || true
W=$(timeout 10 xdotool search --name "^Vampire Survivors$" 2>/dev/null | head -1 || true)
[ -n "${W:-}" ] && timeout 10 wmctrl -i -c "$W"
MV=$(timeout 10 xdotool search --name "^Model Vision$" 2>/dev/null | head -1 || true)
[ -n "${MV:-}" ] && timeout 10 wmctrl -i -c "$MV"
sleep 1
# ensure exe gone
ps -eo pid,cmd | awk '/VampireSurvivors\.exe/ && !/awk/{print $1}' | while read -r p; do kill "$p" 2>/dev/null || true; done
echo "[$(date)] cleanup done" | tee -a "$SESSION/loop_outer.log"
