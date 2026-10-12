#!/bin/bash
# play_run.sh RUN — one guarded Vampire Survivors bot run (Poe Ratcho, Mad Forest, Hyper off).
#
# Template: copy this file plus clk.sh and ocr_results.py into a session folder
# (/workspace/vs-session-<date>/) and drive it from that folder's session_loop.sh.
# The versioned copy lives in the repo at scripts/play_run.sh.
#
# Structure (added 2026-10-03 after run 6 of the 3 AM session sat dead for 2 h 14 min):
#   play_run.sh RUN          GUARD: starts the real run (--inner) in its own process group
#                            and watches it. It kills the run + the bot + the game, relaunches
#                            the game via steam://rungameid/1794680 and realigns the window when
#                              * the run is older than RUN_WALL_LIMIT_S (35 min), or
#                              * the watched log has had no new output for LOG_SILENCE_LIMIT_S
#                                (3 min): the bot log while the bot is alive, else loop.log.
#                            Both clocks are the box's monotonic clock (/proc/uptime), so a host
#                            / VM pause is logged as HOST_PAUSE and is not counted as run time.
#   play_run.sh --inner RUN  the old play_run.sh body. Every X / OCR / Python helper call is
#                            bounded with `timeout`; the bot's own watchdog (exit 3) also
#                            triggers the same kill + relaunch.
# Exit codes: 0 finished, 1 failed (launch/nav/bot died), 2 cutoff (stop the session),
#             3 a watchdog ended the run and the game was relaunched.
set -uo pipefail
export DISPLAY=:6
export PATH="/home/box/.local/bin:/home/box/micromamba/envs/vstools/bin:$PATH"
export TESSDATA_PREFIX=/home/box/micromamba/envs/vstools/share/tessdata
export TZ=America/Chicago
# Test hook: stub binaries (xdotool, steam.sh, pkill, ...) first on PATH.
[ -n "${VS_TEST_BIN:-}" ] && export PATH="$VS_TEST_BIN:$PATH"
SELF=$(readlink -f "$0")
SESSION=${VS_SESSION:-$(dirname "$SELF")}
REPO=${VS_REPO:-/workspace/lucyfur-vs-vision-bot-v2}
PY=${VS_PY:-"$REPO/.venv/bin/python"}
STEAM_SH=${VS_STEAM_SH:-/home/box/.local/share/Steam/steam.sh}
VS_APPID=1794680
VS_PROC_RE='VampireSurvivors\.exe'
PKILL=${VS_PKILL:-pkill}
PGREP=${VS_PGREP:-pgrep}
BOT_PIDFILE=${VS_BOT_PIDFILE:-/tmp/vsbot_3am.pid}
BOT_FIFO=${VS_BOT_FIFO:-/tmp/vsbot_stdin_3am}
FRESH_MARK=${VS_FRESH_MARK:-/tmp/vsbot_fresh_launch}
WINPOS=${VS_WINPOS:-/tmp/winpos.txt}             # client X Y W H written by align_win
RUN_WALL_LIMIT_S=${VS_RUN_WALL_LIMIT_S:-2100}      # 35 min per run
LOG_SILENCE_LIMIT_S=${VS_LOG_SILENCE_LIMIT_S:-180} # 3 min without a new log line
GUARD_POLL_S=${VS_GUARD_POLL_S:-5}
CUTOFF_HHMM=${VS_CUTOFF_HHMM:-0545}                # no new run at/after this CT time ('none' = off)
MODE=run
if [ "${1:-}" = "--inner" ]; then MODE=inner; shift; fi
RUN=${1:?usage: play_run.sh RUN}
cd "$SESSION"
log(){ echo "[$(date +%H:%M:%S)] r$RUN $*"; echo "[$(date +%H:%M:%S)] r$RUN $*" >> "$SESSION/loop.log"; }
mono(){ cut -d. -f1 /proc/uptime; }   # seconds; does not advance while the VM is paused

# --- Bounded helpers: an X call against a wedged server / window must never hang the run. ---
xdotool(){ timeout -k 2 "${XDO_TIMEOUT_S:-10}" xdotool "$@"; }
wmctrl(){ timeout -k 2 10 wmctrl "$@"; }
xwininfo(){ timeout -k 2 10 xwininfo "$@"; }
scrot(){ timeout -k 2 15 scrot "$@"; }
clk(){ timeout -k 2 10 "$SESSION/clk.sh" "$@"; }
pyt(){ local s=$1; shift; timeout -k 2 "$s" "$PY" "$@"; }   # pyt SECONDS args... (stdin passes through)

# shot FILE: full-screen PNG. scrot first; if it fails (the ~/.local/bin/scrot shim points at a
# /tmp venv that may be gone: runs 1-2 of 2026-10-03 died that way), grab with mss directly.
shot(){
  rm -f "$1"
  scrot -o "$1" 2>/dev/null
  if [ ! -s "$1" ]; then
    pyt 15 - "$1" <<'PY' 2>/dev/null
import sys
from mss import mss
from mss.tools import to_png
with mss() as s:
    g = s.grab(s.monitors[0])
    to_png(g.rgb, g.size, output=sys.argv[1])
PY
  fi
  [ -s "$1" ]
}

cutoff_reached(){
  [ "$CUTOFF_HHMM" = none ] && return 1
  [ "$((10#$(date +%H%M)))" -ge "$((10#$CUTOFF_HHMM))" ]
}

get_win(){ xdotool search --name "^Vampire Survivors$" 2>/dev/null | head -1; }

align_win(){
  local W WIDTH HEIGHT AX AY RX RY GEOM FX FY
  W=$(get_win) || true
  [ -n "${W:-}" ] || return 1
  wmctrl -i -r "$W" -b remove,maximized_vert,maximized_horz 2>/dev/null || true
  # Frame at 195,51 → Absolute client ~200,80 (xfwm4 border 5 / titlebar 29)
  xdotool windowsize "$W" 960 540 2>/dev/null || true
  xdotool windowmove "$W" 195 51 2>/dev/null || true
  wmctrl -i -r "$W" -e 0,195,51,960,540 2>/dev/null || true
  sleep 0.4
  GEOM=$(xwininfo -id "$W" 2>/dev/null) || return 1
  AX=$(echo "$GEOM" | awk -F: '/Absolute upper-left X/{gsub(/ /,"",$2); print $2}')
  AY=$(echo "$GEOM" | awk -F: '/Absolute upper-left Y/{gsub(/ /,"",$2); print $2}')
  RX=$(echo "$GEOM" | awk -F: '/Relative upper-left X/{gsub(/ /,"",$2); print $2}')
  RY=$(echo "$GEOM" | awk -F: '/Relative upper-left Y/{gsub(/ /,"",$2); print $2}')
  WIDTH=$(echo "$GEOM" | awk -F: '/Width/{gsub(/ /,"",$2); print $2; exit}')
  HEIGHT=$(echo "$GEOM" | awk -F: '/Height/{gsub(/ /,"",$2); print $2; exit}')
  [ -n "${AX:-}" ] && [ -n "${AY:-}" ] || return 1
  RX=${RX:-5}; RY=${RY:-29}
  if [ "${WIDTH:-0}" -lt 900 ] || [ "${HEIGHT:-0}" -lt 500 ]; then
    xdotool windowsize "$W" 960 540 2>/dev/null || true
    sleep 0.3
  fi
  # Nudge using frame = Absolute - Relative (never trust xdotool getwindowgeometry here)
  if [ "$AX" -ne 200 ] || [ "$AY" -ne 80 ]; then
    FX=$((AX - RX)); FY=$((AY - RY))
    xdotool windowmove "$W" $((FX + 200 - AX)) $((FY + 80 - AY)) 2>/dev/null || true
    sleep 0.35
  fi
  GEOM=$(xwininfo -id "$W" 2>/dev/null) || return 1
  AX=$(echo "$GEOM" | awk -F: '/Absolute upper-left X/{gsub(/ /,"",$2); print $2}')
  AY=$(echo "$GEOM" | awk -F: '/Absolute upper-left Y/{gsub(/ /,"",$2); print $2}')
  WIDTH=$(echo "$GEOM" | awk -F: '/Width/{gsub(/ /,"",$2); print $2; exit}')
  HEIGHT=$(echo "$GEOM" | awk -F: '/Height/{gsub(/ /,"",$2); print $2; exit}')
  echo "$AX $AY $WIDTH $HEIGHT" > "$WINPOS"
  log "align client=$AX,$AY ${WIDTH}x${HEIGHT}"
  return 0
}

# kill_vs: TERM every Vampire Survivors process, wait up to 15 s, then KILL what is left.
kill_vs(){
  local i
  $PKILL -TERM -f "$VS_PROC_RE" 2>/dev/null || true
  for i in $(seq 1 15); do
    $PGREP -f "$VS_PROC_RE" >/dev/null 2>&1 || return 0
    sleep 1
  done
  log "kill_vs: game still alive after TERM -> KILL"
  $PKILL -KILL -f "$VS_PROC_RE" 2>/dev/null || true
  sleep 1
}

stop_bot(){
  local p
  p=$(cat "$BOT_PIDFILE" 2>/dev/null || true)
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
    kill "$p" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  fi
  xdotool keyup --clearmodifiers a s d w Left Right Up Down Return 2>/dev/null || true
  for mv in $(xdotool search --name "^Model Vision$" 2>/dev/null); do wmctrl -i -c "$mv" 2>/dev/null || true; done
}

close_game(){
  local W
  W=$(get_win || true)
  [ -n "${W:-}" ] && wmctrl -i -c "$W"
  sleep 2
  $PKILL -TERM -f "$VS_PROC_RE" 2>/dev/null || true
  sleep 2
  xdotool keyup --clearmodifiers a s d w Left Right Up Down 2>/dev/null || true
  if [ -f "$BOT_PIDFILE" ]; then
    kill "$(cat "$BOT_PIDFILE")" 2>/dev/null || true
    sleep 1
  fi
  for mv in $(xdotool search --name "^Model Vision$" 2>/dev/null); do wmctrl -i -c "$mv" 2>/dev/null || true; done
  log closed
}

steam_up(){ ps -eo cmd | awk '/ubuntu12_32\/steam / && !/awk/{found=1} END{exit !found}'; }

ensure_steam(){
  # Vampire Survivors needs a live Steam client (SteamAPI_Init). Keep Steam up across runs.
  steam_up && return 0
  log "starting Steam client"
  nohup "$STEAM_SH" -nofriendsui >/tmp/steam-3am-client.log 2>&1 &
  for i in $(seq 1 60); do
    if steam_up; then
      sleep 8
      # dismiss Special Offers popup if present
      OFF=$(xdotool search --name "Special Offers" 2>/dev/null | head -1 || true)
      [ -n "${OFF:-}" ] && wmctrl -i -c "$OFF" 2>/dev/null || true
      return 0
    fi
    sleep 2
  done
  log "STEAM_START_FAIL"
  return 1
}

# launch_vs [URI]: start the game through Steam (applaunch, or a steam:// URI for recovery).
launch_vs(){
  ensure_steam || return 1
  if [ -n "${1:-}" ]; then
    nohup "$STEAM_SH" "$1" >/tmp/steam_launch_play.log 2>&1 &
  else
    nohup "$STEAM_SH" -applaunch "$VS_APPID" >/tmp/steam_launch_play.log 2>&1 &
  fi
}

# wait_window TRIES [URI]: poll ~2 s per try for the game window, re-issuing the launch every ~20 s.
wait_window(){
  local i W
  for i in $(seq 1 "$1"); do
    W=$(get_win || true)
    [ -n "${W:-}" ] && return 0
    if [ $((i % 10)) -eq 0 ]; then
      log "relaunch steam attempt $i"
      launch_vs "${2:-}" || true
    fi
    sleep 2
  done
  return 1
}

launch_to_main(){
  local i
  # Ensure prior process fully gone before launch (avoids race with dying window)
  for i in $(seq 1 15); do
    $PGREP -f "$VS_PROC_RE" >/dev/null 2>&1 || break
    sleep 1
  done
  $PKILL -KILL -f "$VS_PROC_RE" 2>/dev/null || true
  sleep 1
  launch_vs || return 1
  wait_window 90 || { log NO_WINDOW; return 1; }
  sleep 6
  boot_to_main
}

boot_to_main(){
  local i W OUT X Y WIDTH HEIGHT
  for i in $(seq 1 80); do
    align_win || { sleep 2; continue; }
    read -r X Y WIDTH HEIGHT < "$WINPOS"
    W=$(get_win || true)
    [ -n "${W:-}" ] || { sleep 2; continue; }
    shot /tmp/b.png || log "boot: screenshot failed (scrot and mss fallback)"
    OUT=$(pyt 30 - "$X" "$Y" "$WIDTH" "$HEIGHT" <<'PY'
import cv2,sys
x,y,w,h=map(int,sys.argv[1:])
im=cv2.imread('/tmp/b.png')
if im is None: print('0 0 NOIMG'); raise SystemExit
crop=im[y:y+h,x:x+w]
hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
mean=float(crop.mean())
bp=int(cv2.inRange(hsv,(90,40,40),(140,255,255)).sum()//255)
print(f'{mean:.0f} {bp}')
if bp>100000: print('EULA')
elif 40<mean<70 and 20000<bp<80000: print('MAIN')
elif mean>75: print('CHAR')
elif mean<25: print('BLACK')
else: print('OTHER')
PY
)
    log "boot $OUT"
    if echo "$OUT" | grep -q EULA; then
      xdotool windowactivate --sync "$W" 2>/dev/null || true
      xdotool mousemove --window "$W" --sync 605 45 2>/dev/null; sleep 0.2; xdotool click 1
      sleep 1; continue
    fi
    if echo "$OUT" | grep -qE 'MAIN|CHAR'; then
      shot "$SESSION/shots/r${RUN}_main.png"
      return 0
    fi
    xdotool windowactivate --sync "$W" 2>/dev/null || true
    clk $((X+WIDTH/2)) $((Y+HEIGHT*2/3))
    sleep 2
  done
  log BOOT_TIMEOUT
  return 1
}

# --- Stage-select helpers (added 2026-09-30: runs 20-39 on Sep 29 started on The Coop) ---
# screen_state: classify /tmp/pl.png (client crop also saved to /tmp/ingc.png). Prints one line:
#   INGAME grass=N            Mad Forest gameplay (grass > MF_GRASS_MIN)
#   STAGE sel=I name=.. top=.. mf=0|1 hyper=0|1 armed=0|1 sat=..   (Stage Selection screen)
#   CHAR / MAIN / LOADING / OTHER  (+ grass/mean)
MF_GRASS_MIN=120000   # Mad Forest reads ~221k; The Coop read 29,812
screen_state(){
  pyt 40 - "$1" "$2" "$3" "$4" "$MF_GRASS_MIN" <<'PY'
import cv2,subprocess,sys,re
x,y,w,h,gmin=map(int,sys.argv[1:6])
im=cv2.imread('/tmp/pl.png')
if im is None: print('OTHER noimg'); raise SystemExit
c=im[y:y+h,x:x+w]
cv2.imwrite('/tmp/ingc.png',c)
hsv=cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
grass=int(cv2.inRange(hsv,(35,40,40),(85,200,200))[270:,:].sum()//255)
mean=float(c.mean())
def ocr(img,psm='7',scale=3):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.resize(g,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    _,g=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (g>0).mean()<0.5: g=255-g
    cv2.imwrite('/tmp/st_ocr.png',g)
    r=subprocess.run(['tesseract','/tmp/st_ocr.png','stdout','--psm',psm],capture_output=True,text=True)
    return re.sub(r'\s+',' ',r.stdout).strip()
def tag(s): return (re.sub(r'[^A-Za-z0-9]+','_',s).strip('_')[:32]) or '-'
if grass>gmin:
    print(f'INGAME grass={grass}'); raise SystemExit
if mean<25:
    print(f'LOADING mean={mean:.0f} grass={grass}'); raise SystemExit
title=ocr(c[58:98,330:630]).lower()
if 'stage' in title and 'selec' in title:
    # Stage rows (list scrolled to top): tile x 290-410, description x 414-660, rows 88 px apart.
    # The selected row's description panel is grey (saturation ~31); unselected rows are blue (~76).
    sat=[float(cv2.cvtColor(c[102+88*i+5:102+88*i+70,420:655],cv2.COLOR_BGR2HSV)[...,1].mean()) for i in range(4)]
    order=sorted(range(4),key=lambda i:sat[i])
    sel=order[0] if sat[order[0]]<50 and sat[order[1]]>60 else -1
    top=ocr(c[108:124,295:410])
    selname=ocr(c[102+88*sel+6:102+88*sel+22,295:410]) if sel>=0 else ''
    low=selname.lower()
    mf=int(sel==0 and 'forest' in low and 'coop' not in low)
    box=cv2.cvtColor(c[488:512,306:330],cv2.COLOR_BGR2HSV)
    hyper=int(int((box[...,2]>150).sum())>30)      # unchecked Hyper box interior is dark (0 bright px)
    pan=cv2.cvtColor(c[458:528,292:668],cv2.COLOR_BGR2HSV)
    armed=int(int(cv2.inRange(pan,(15,120,100),(40,255,255)).sum()//255)>10000)  # yellow panel after 1st START
    print(f'STAGE sel={sel} name={tag(selname)} top={tag(top)} mf={mf} hyper={hyper} armed={armed} sat='+','.join(str(int(v)) for v in sat))
elif 'character' in title:
    print(f'CHAR grass={grass}')
elif 'ADVENTURE' in ocr(c[440:470,380:580]).upper() or 'POWER' in ocr(c[498:528,425:535]).upper():
    print('MAIN')
else:
    print(f'OTHER mean={mean:.0f} grass={grass}')
PY
}

# Scroll the stage list to the top and click the Mad Forest tile (stage 1).
select_mad_forest(){
  local WX=$1 WY=$2 i
  xdotool mousemove $((WX+470)) $((WY+270)); sleep 0.2
  for i in $(seq 1 15); do xdotool click 4; sleep 0.05; done
  sleep 0.6
  clk $((WX+350)) $((WY+140))
  sleep 1.0
}

# Leave without playing: BACK out of char/stage select, or pause (Esc, once) + QUIT/YES/DONE a started
# run, until the main menu shows. If the main menu is not reached, close the game so the wrong stage is
# never played (the next run relaunches it). Never touches Reroll/Banish/Skip.
quit_to_menu(){
  local WX WY WW WH W ST HIT i ESC=0 MISS=0
  for i in $(seq 1 10); do
    align_win || sleep 1
    read -r WX WY WW WH < "$WINPOS" || break
    W=$(get_win); [ -n "${W:-}" ] || { log "quit_to_menu: no game window"; return 1; }
    xdotool windowactivate --sync "$W" 2>/dev/null || true
    shot /tmp/pl.png
    ST=$(screen_state "$WX" "$WY" "$WW" "$WH")
    log "quit_to_menu step$i: $ST"
    case "$ST" in
      MAIN*) cp /tmp/pl.png "$SESSION/shots/r${RUN}_abort_menu.png" 2>/dev/null || true; log "quit_to_menu: at main menu"; return 0 ;;
      STAGE*|CHAR*) clk $((WX+641)) $((WY+26)); sleep 1.5; continue ;;   # red BACK button
      LOADING*) sleep 1.5; continue ;;
    esac
    HIT=$(pyt 40 - "$WX" "$WY" "$WW" "$WH" <<'PY'
import cv2,subprocess,sys
x,y,w,h=map(int,sys.argv[1:])
c=cv2.imread('/tmp/pl.png')[y:y+h,x:x+w]
g=cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2GRAY),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
_,t=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
best=None
for img in (g,t,255-t):
    cv2.imwrite('/tmp/q_ocr.png',img)
    r=subprocess.run(['tesseract','/tmp/q_ocr.png','stdout','--psm','11','tsv'],capture_output=True,text=True).stdout
    for ln in r.splitlines()[1:]:
        p=ln.split('\t')
        if len(p)<12: continue
        tx=p[11].strip().strip('.,:!?').lower()
        if tx in ('quit','yes','done'):
            L,T,W_,H_=(int(v)//2 for v in p[6:10])
            pr={'done':0,'yes':1,'quit':2}[tx]
            if best is None or pr<best[0]: best=(pr,tx,L+W_//2,T+H_//2)
    if best: break
print(f'{best[1]} {best[2]} {best[3]}' if best else 'none')
PY
)
    if [ -n "$HIT" ] && [ "${HIT%% *}" != "none" ]; then
      set -- $HIT
      log "quit_to_menu: click $1 at client $2,$3"
      clk $((WX+$2)) $((WY+$3)); sleep 2; MISS=0
    elif [ "$ESC" -eq 0 ]; then
      xdotool key --clearmodifiers Escape; ESC=1; sleep 1.5   # pause once; pressing again would resume
    else
      MISS=$((MISS+1)); [ "$MISS" -ge 2 ] && break; sleep 1
    fi
  done
  log "quit_to_menu: main menu not reached -> closing the game so the wrong stage is not played"
  close_game
  return 1
}

nav_poe_start(){
  local WX WY WW WH W MEAN SX SY attempt GRASS AURA OFF tries N INFO_OCR ST STAGE_OK SEL_TRIES POST step
  align_win || return 1
  read -r WX WY WW WH < "$WINPOS"
  W=$(get_win); [ -n "$W" ] || return 1
  xdotool windowactivate --sync "$W" 2>/dev/null || true
  shot /tmp/n.png
  MEAN=$(pyt 20 -c "import cv2; print(int(cv2.imread('/tmp/n.png')[$WY:$WY+$WH,$WX:$WX+$WW].mean()))")
  if [ "$MEAN" -lt 75 ]; then
    log "main START mean=$MEAN"
    # START button center ~ client (479,351) on 960x540 (v1.16); old 474,320 was above button
    for try in 1 2 3 4 5; do
      xdotool windowactivate --sync "$W" 2>/dev/null || true
      clk $((WX+479)) $((WY+351))
      sleep 2.5
      shot /tmp/n.png
      MEAN2=$(pyt 20 -c "import cv2; print(int(cv2.imread('/tmp/n.png')[$WY:$WY+$WH,$WX:$WX+$WW].mean()))")
      log "after START try$try mean=$MEAN2"
      # Character select is brighter than main (~45)
      if [ "${MEAN2:-0}" -gt 70 ]; then break; fi
      # alternate click slightly lower / on quick-start neighbor
      clk $((WX+479)) $((WY+360))
      sleep 2
    done
  fi
  align_win || return 1
  read -r WX WY WW WH < "$WINPOS"
  W=$(get_win); [ -n "$W" ] || return 1
  xdotool windowactivate --sync "$W" 2>/dev/null || true
  shot /tmp/n.png
  MEAN3=$(pyt 20 -c "import cv2; print(int(cv2.imread('/tmp/n.png')[$WY:$WY+$WH,$WX:$WX+$WW].mean()))")
  if [ "${MEAN3:-0}" -lt 70 ]; then log "FAIL_STILL_MAIN mean=$MEAN3"; return 1; fi
  log "select Poe via mouse+arrows mean=$MEAN3"
  # Click character grid area then use arrows (Right from Porta)
  clk $((WX+100)) $((WY+100))
  sleep 0.3
  for _ in 1 2 3 4 5 6 7 8; do xdotool key --delay 40 Left; done
  sleep 0.2
  # Porta is leftmost; Poe is typically one Right
  xdotool key --delay 120 Right
  sleep 0.6
  # Also mouse-click known Poe tile offsets (from prior sessions)
  for OFF in "425 151" "420 148" "430 155" "415 145" "435 160"; do
    set -- $OFF
    clk $((WX+$1)) $((WY+$2))
    sleep 0.35
  done
  sleep 0.8
  shot /tmp/poe.png
  cp /tmp/poe.png "$SESSION/shots/r${RUN}_poe.png"
  rm -f /tmp/poe_ok.txt   # a stale ok=1 from an earlier run must never pass the Poe check
  # Poe check: OCR the info-panel NAME row (client y 452-472, x 286-500) under the grid.
  # The old y 390-500 crop started on the grid's last row and read the "Concetta" tile -> always ok=0.
  INFO_OCR=$(pyt 40 - "$WX" "$WY" <<'PY'
import cv2,subprocess,sys,re
wx,wy=map(int,sys.argv[1:])
im=cv2.imread('/tmp/poe.png')
ok=0; txts=[]
for y0,y1,x0,x1 in [(452,472,286,500),(450,474,286,640)]:
    info=im[wy+y0:wy+y1, wx+x0:wx+x1]
    if info.size==0: continue
    g=cv2.resize(cv2.cvtColor(info,cv2.COLOR_BGR2GRAY),None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    _,g=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if (g>0).mean()<0.5: g=255-g
    cv2.imwrite('/tmp/poi.png',g)
    r=subprocess.run(['tesseract','/tmp/poi.png','stdout','--psm','7'],capture_output=True,text=True)
    txt=re.sub(r'\s+',' ',r.stdout).strip()
    txts.append(txt[:60])
    if 'Poe' in txt or 'Ratcho' in txt:
        ok=1; break
print(' | '.join(txts)[:160])
open('/tmp/poe_ok.txt','w').write(str(ok))
PY
)
  log "poe OCR: $INFO_OCR ok=$(cat /tmp/poe_ok.txt 2>/dev/null || echo 0)"
  # Never play anyone but Poe Ratcho: if the info-panel NAME row did not read Poe/Ratcho, back out
  # to the main menu (same as the wrong-stage abort) instead of confirming the character.
  if [ "$(cat /tmp/poe_ok.txt 2>/dev/null || echo 0)" != "1" ]; then
    cp /tmp/poe.png "$SESSION/shots/r${RUN}_char_abort.png" 2>/dev/null || true
    log "WRONG_CHAR_ABORT: Poe name check failed (ok=0; ocr=$INFO_OCR); quitting to menu"
    echo "wrong_char" > "$SESSION/run${RUN}_char.txt"
    quit_to_menu || true
    return 1
  fi
  # Character Confirm/START -> Stage Selection -> explicitly select + OCR-verify Mad Forest -> START.
  # Never press START on the stage screen unless that same frame shows Mad Forest selected and Hyper off.
  GRASS=0; STAGE_OK=0; SEL_TRIES=0; POST=0; ST=""
  for step in $(seq 1 16); do
    align_win || sleep 1
    read -r WX WY WW WH < "$WINPOS" || true
    W=$(get_win); [ -n "${W:-}" ] || { sleep 2; continue; }
    xdotool windowactivate --sync "$W" 2>/dev/null || true
    shot /tmp/pl.png
    ST=$(screen_state "$WX" "$WY" "$WW" "$WH")
    log "start step$step $ST"
    case "$ST" in
      INGAME*)
        GRASS=$(echo "$ST" | sed -n 's/.*grass=\([0-9]*\).*/\1/p'); break ;;
      CHAR*)
        # Character Confirm, then START (same button, client ~789,502)
        clk $((WX+789)) $((WY+502)); sleep 2.2 ;;
      STAGE*)
        if echo "$ST" | grep -q 'hyper=1'; then
          cp /tmp/pl.png "$SESSION/shots/r${RUN}_stage_abort.png" 2>/dev/null || true
          log "STAGE_ABORT: Hyper looks ON on stage select; refusing to start ($ST)"
          quit_to_menu || true; return 1
        fi
        if echo "$ST" | grep -q 'mf=1'; then
          if [ "$STAGE_OK" -eq 0 ]; then cp /tmp/pl.png "$SESSION/shots/r${RUN}_stage.png" 2>/dev/null || true; fi
          STAGE_OK=1
          log "stage verified Mad Forest -> START"
          clk $((WX+789)) $((WY+502)); sleep 2.2
        else
          SEL_TRIES=$((SEL_TRIES+1))
          if [ "$SEL_TRIES" -gt 3 ]; then
            cp /tmp/pl.png "$SESSION/shots/r${RUN}_stage_abort.png" 2>/dev/null || true
            log "STAGE_ABORT: could not select/verify Mad Forest after 3 tries ($ST)"
            quit_to_menu || true; return 1
          fi
          log "select Mad Forest try$SEL_TRIES (scroll list to top, click stage 1)"
          select_mad_forest "$WX" "$WY"
        fi ;;
      *)
        # Loading / in a run that is not Mad Forest / unknown
        # after a verified START, give loading ~6 frames to reach Mad Forest gameplay, else abort
        if [ "$STAGE_OK" -eq 1 ]; then POST=$((POST+1)); fi
        if [ "$STAGE_OK" -eq 1 ] && [ "$POST" -ge 6 ]; then break; fi
        sleep 1.5 ;;
    esac
  done
  if [ "${GRASS:-0}" -le "$MF_GRASS_MIN" ]; then
    cp /tmp/pl.png "$SESSION/shots/r${RUN}_start_abort.png" 2>/dev/null || true
    if [ "$STAGE_OK" -eq 1 ]; then
      log "WRONG_STAGE_ABORT: not Mad Forest in game (grass=${GRASS:-0} <= $MF_GRASS_MIN; last=$ST); quitting to menu"
    else
      log "FAIL_START: stage never verified (last=$ST); quitting to menu"
    fi
    quit_to_menu || true
    return 1
  fi
  log "in-game Mad Forest grass=$GRASS (> $MF_GRASS_MIN)"
  AURA=$(pyt 20 -c "
import cv2
c=cv2.imread('/tmp/ingc.png'); h,w=c.shape[:2]; ring=c[h//2-50:h//2+50,w//2-50:w//2+50]
hsv=cv2.cvtColor(ring,cv2.COLOR_BGR2HSV)
print(int(cv2.inRange(hsv,(35,50,50),(85,255,255)).sum()//255))
")
  log "aura=$AURA poe_ocr=$(cat /tmp/poe_ok.txt 2>/dev/null || echo ?)"
  echo "$AURA" > "$SESSION/shots/r${RUN}_aura.txt"
  cp /tmp/ingc.png "$SESSION/shots/r${RUN}_ingame.png"
  # (AURA < 200 abort removed 2026-10-01: that box is mostly grass and always reads ~9234; the
  #  character is enforced by the Poe name check above.)
  echo "Poe Ratcho" > "$SESSION/run${RUN}_char.txt"
  return 0
}

start_bot(){
  local WX WY WW WH LOG MV
  read -r WX WY WW WH < "$WINPOS"
  LOG=$SESSION/run${RUN}.log
  if [ -f "$BOT_PIDFILE" ]; then kill "$(cat "$BOT_PIDFILE")" 2>/dev/null || true; sleep 1; fi
  cd "$REPO"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export DISPLAY=:6 PYTHONUNBUFFERED=1
  export PATH="/home/box/.local/bin:/home/box/micromamba/envs/vstools/bin:$PATH"
  export TESSDATA_PREFIX=/home/box/micromamba/envs/vstools/share/tessdata
  export VS_CAPTURE_X=$WX VS_CAPTURE_Y=$WY VS_CAPTURE_W=960 VS_CAPTURE_H=540
  export VS_DEBUG_X=0 VS_DEBUG_Y=560 VS_DEBUG_W=190 VS_DEBUG_H=107
  # Stage verified above (Mad Forest only): enables the bot's one Banish exception (new Pentagram).
  export VS_STAGE=mad_forest
  rm -f "$BOT_FIFO"
  mkfifo "$BOT_FIFO"
  exec 9<>"$BOT_FIFO"
  log "bot env VS_STAGE=${VS_STAGE:-} VS_LAUREL_DAMAGE=${VS_LAUREL_DAMAGE-<unset>}"
  nohup ./run.sh < "$BOT_FIFO" > "$LOG" 2>&1 &
  echo $! > "$BOT_PIDFILE"
  for i in $(seq 1 50); do grep -q 'STARTED PAUSED' "$LOG" 2>/dev/null && break; sleep 0.3; done
  printf 'p\n' >&9 || true
  sleep 0.4
  MV=$(xdotool search --name "^Model Vision$" 2>/dev/null | head -1 || true)
  [ -n "${MV:-}" ] && xdotool windowactivate --sync "$MV" 2>/dev/null; sleep 0.2
  xdotool key --clearmodifiers p
  sleep 1.2
  if ! grep '\[hotkey\]' "$LOG" | tail -1 | grep -q RUNNING; then
    printf 'p\n' >&9 || true
    xdotool key --clearmodifiers p
    sleep 1
  fi
  xdotool search --name "^Vampire Survivors$" windowactivate 2>/dev/null || true
  log "bot $(grep '\[hotkey\]' "$LOG" | tail -1) capture=$WX,$WY"
  cd "$SESSION"
}

game_alive(){ $PGREP -f "$VS_PROC_RE" >/dev/null 2>&1 && [ -n "$(get_win || true)" ]; }

# Returns 0 finished, 1 bot died, 3 bot watchdog / game gone after a resume / timeout (recover).
wait_done(){
  local LOG S E L REV NOW_M NOW_W PREV_M PREV_W JUMP STALL
  LOG=$SESSION/run${RUN}.log
  # Monotonic (/proc/uptime) timing: a VM pause does not count as run time, so a resumed
  # healthy run is not TIMEOUT'ed (2026-10-03 run 6: wall clock jumped 2 h 14 min).
  S=$(mono); PREV_M=$S; PREV_W=$(date +%s)
  while true; do
    NOW_M=$(mono); NOW_W=$(date +%s)
    JUMP=$(( (NOW_W - PREV_W) - (NOW_M - PREV_M) ))
    if [ "$JUMP" -gt 60 ]; then
      if game_alive && kill -0 "$(cat "$BOT_PIDFILE")" 2>/dev/null; then
        log "BOX_RESUMED: wall clock jumped ${JUMP}s past the monotonic clock (VM was paused); game + bot alive -> continuing"
      else
        log "BOX_RESUMED: wall clock jumped ${JUMP}s (VM was paused); game or bot gone -> recover"
        return 3
      fi
    fi
    PREV_M=$NOW_M; PREV_W=$NOW_W
    E=$((NOW_M - S))
    if ! kill -0 "$(cat "$BOT_PIDFILE")" 2>/dev/null; then
      STALL=$(grep '\[watchdog\] STALL' "$LOG" 2>/dev/null | tail -1)
      if [ -n "$STALL" ]; then log "BOT_WATCHDOG: $STALL"; return 3; fi
      log BOT_DIED; return 1
    fi
    if grep -q 'run finished (QUIT + DONE)' "$LOG"; then
      log finished
      L=$(ls -t /workspace/vsbot-check/results/results_*.png 2>/dev/null | head -1 || true)
      if [ -n "${L:-}" ]; then
        cp "$L" "$SESSION/results/"
        echo "$L" > "$SESSION/run${RUN}_results_path.txt"
        cp "$L" "$SESSION/results/run${RUN}_results.png"
      fi
      REV=$(grep -c '\[revive\] clicked' "$LOG" || true)
      echo "$REV" > "$SESSION/run${RUN}_revives.txt"
      grep -E '\[revive\]|\[results\]' "$LOG" | tail -8 | tee -a "$SESSION/loop.log" || true
      if [ -f "$SESSION/results/run${RUN}_results.png" ]; then
        pyt 120 "$SESSION/ocr_results.py" "$SESSION/results/run${RUN}_results.png" "$SESSION/run${RUN}_stats.txt" || true
      fi
      return 0
    fi
    if [ $((E%60)) -lt 6 ]; then
      log "t=${E}s $(grep '\[nav\] game' "$LOG" | tail -1 | cut -c1-90) rev=$(grep -c '\[revive\] clicked' "$LOG" || true)"
      shot "$SESSION/shots/r${RUN}_t${E}.png" || true
    fi
    # Backstop only; the guard (RUN_WALL_LIMIT_S) normally fires first.
    if [ "$E" -gt $((RUN_WALL_LIMIT_S + 120)) ]; then log TIMEOUT; return 3; fi
    sleep 5
  done
}

# recover_game WHY: stop the bot, kill the game, relaunch via steam://rungameid and realign
# the window to client 960x540 at (200,80). The next run reuses the fresh game (FRESH_MARK).
recover_game(){
  local i
  log "RECOVER ($1): stopping bot, killing the game, relaunching steam://rungameid/$VS_APPID"
  rm -f "$FRESH_MARK"
  stop_bot
  kill_vs
  launch_vs "steam://rungameid/$VS_APPID" || { log "RECOVER: Steam not available"; return 1; }
  wait_window 90 "steam://rungameid/$VS_APPID" || { log "RECOVER: NO_WINDOW"; return 1; }
  sleep 6
  for i in $(seq 1 10); do align_win && break; sleep 2; done
  date +%s > "$FRESH_MARK"
  log "RECOVER done: game relaunched and aligned"
}

# fresh_game_ready: a recovery relaunched the game < 10 min ago and its window is up.
fresh_game_ready(){
  local t
  t=$(cat "$FRESH_MARK" 2>/dev/null || echo 0)
  rm -f "$FRESH_MARK"
  [ $(( $(date +%s) - ${t:-0} )) -lt 600 ] && [ -n "$(get_win || true)" ]
}

preflight(){
  shot /tmp/vsbot_preflight.png &&
    pyt 20 -c "import cv2,sys; im=cv2.imread('/tmp/vsbot_preflight.png'); sys.exit(0 if im is not None and im.size else 1)" 2>/dev/null
}

inner_main(){
  local RC
  log BEGIN
  if cutoff_reached; then log cutoff; exit 2; fi
  if ! preflight; then
    log "FAIL_PREFLIGHT: cannot screenshot the display (scrot shim + mss fallback both failed)"
    sleep 20; exit 1
  fi
  if fresh_game_ready; then
    log "reusing the game relaunched by the last recovery"
    boot_to_main || { log FAIL_LAUNCH; exit 1; }
  else
    close_game
    launch_to_main || { log FAIL_LAUNCH; exit 1; }
  fi
  nav_poe_start || { log FAIL_NAV; exit 1; }
  start_bot
  wait_done
  RC=$?
  if [ "$RC" -eq 3 ]; then recover_game "watchdog" || true; fi
  log END rc=$RC
  exit $RC
}

# descendants PID: every process below PID (fallback when the process group is not our own).
descendants(){
  local c
  for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done
}

guard_main(){
  local INNER PGID START LAST_CHANGE PREV_SIG SIG MON NOW_M NOW_W PREV_M PREV_W JUMP REASON="" RC P BP STAMP
  STAMP=$(mktemp /tmp/vsbot_guard_r${RUN}.XXXXXX)
  if [ -n "${VS_INNER_CMD:-}" ]; then
    setsid bash -c "$VS_INNER_CMD" inner "$RUN" &      # test hook
  else
    setsid "$SELF" --inner "$RUN" &
  fi
  INNER=$!
  PGID=$(ps -o pgid= -p "$INNER" 2>/dev/null | tr -d ' ')
  START=$(mono); LAST_CHANGE=$START; PREV_M=$START; PREV_W=$(date +%s); PREV_SIG=""
  while kill -0 "$INNER" 2>/dev/null; do
    sleep "$GUARD_POLL_S"
    NOW_M=$(mono); NOW_W=$(date +%s)
    JUMP=$(( (NOW_W - PREV_W) - (NOW_M - PREV_M) ))
    [ "$JUMP" -gt 60 ] && log "BOX_RESUMED (guard): wall clock jumped ${JUMP}s; VM pause not counted toward the guard limits"
    PREV_M=$NOW_M; PREV_W=$NOW_W
    # Watch the bot log while the bot is alive, else loop.log (boot / nav / recovery steps).
    MON="$SESSION/loop.log"
    BP=$(cat "$BOT_PIDFILE" 2>/dev/null || true)
    if [ -f "$SESSION/run${RUN}.log" ] && [ "$SESSION/run${RUN}.log" -nt "$STAMP" ] \
       && [ -n "${BP:-}" ] && kill -0 "$BP" 2>/dev/null; then
      MON="$SESSION/run${RUN}.log"
    fi
    SIG="$MON $(stat -c '%s %Y' "$MON" 2>/dev/null || echo none)"
    if [ "$SIG" != "$PREV_SIG" ]; then PREV_SIG=$SIG; LAST_CHANGE=$NOW_M; fi
    if [ $((NOW_M - START)) -ge "$RUN_WALL_LIMIT_S" ]; then
      REASON="wall: run older than ${RUN_WALL_LIMIT_S}s"; break
    fi
    if [ $((NOW_M - LAST_CHANGE)) -ge "$LOG_SILENCE_LIMIT_S" ]; then
      REASON="silence: no new output in $(basename "$MON") for $((NOW_M - LAST_CHANGE))s"; break
    fi
  done
  if [ -z "$REASON" ]; then
    wait "$INNER"; RC=$?
    rm -f "$STAMP"
    return $RC
  fi
  log "GUARD_TRIP $REASON -> killing run + bot + game, relaunching"
  echo "$(date '+%F %T %Z') $REASON" > "$SESSION/run${RUN}_guard.txt"
  if [ -n "${PGID:-}" ] && [ "$PGID" = "$INNER" ]; then
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 3; kill -KILL -- "-$PGID" 2>/dev/null
  else
    for P in $(descendants "$INNER") "$INNER"; do kill -TERM "$P" 2>/dev/null; done
    sleep 3
    for P in $(descendants "$INNER") "$INNER"; do kill -KILL "$P" 2>/dev/null; done
  fi
  wait "$INNER" 2>/dev/null
  rm -f "$STAMP"
  recover_game "guard: $REASON" || true
  return 3
}

if [ "$MODE" = inner ]; then
  inner_main
else
  guard_main
  RC=$?
  log "guard exit rc=$RC"
  exit $RC
fi
