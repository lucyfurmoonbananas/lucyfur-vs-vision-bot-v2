# Lucyfur VS Vision Bot v2
A clean Python bot for **Vampire Survivors** on Steam/Linux. It watches a capture
region, shows a small parked **Model Vision** debug window, pathfinds away from
detected monsters, and steers the character with **WASD** through **xdotool**.

This is a new tree, not a patch of the old arrow-key fork. pynput is an optional
fallback only. Arrow keys are an optional fallback only.

## Defaults (do not invert these)

| Setting | Default | Fallback |
| --- | --- | --- |
| Movement | **WASD** (Vampire Survivors default binds) | `VS_MOVE_KEYS=arrows` |
| Input | **xdotool** `keydown` / `keyup` after one `windowactivate` on the `Vampire Survivors` window | `VS_INPUT_BACKEND=pynput` |
| Start state | **Paused** | Press `p` to unpause |
| Cleanup | Releases **WASD and arrows**, clears the move queue | On pause, stop, ESC, signal, and `atexit` |

pynput often fails to reach Steam/Proton games. xdotool XTest events are the
reliable path on Linux.

## Install

You need Python 3.10+, a working X11 `DISPLAY`, and `xdotool`.

```bash
# Debian / Ubuntu / Steam Deck-ish
sudo apt install xdotool python3-venv python3-pip

git clone <this-repo>
cd lucyfur-vs-vision-bot-v2   # or whatever you named the checkout

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional TorchScript detector (only if you have a model file):

```bash
pip install torch
export VS_MODEL_PATH=/path/to/model.pt
```

If `VS_MODEL_PATH` is unset, the bot uses the built-in OpenCV heuristic (no
weights required).

## Run

Lucyfur’s game session is often on **`DISPLAY=:6`**. Export that in the same
shell that launches the bot, then start Steam / Vampire Survivors on that display.

```bash
export DISPLAY=:6
source .venv/bin/activate
./run.sh
# or: python -m vs_vision_bot
```

The bot **starts paused**. Nothing moves until you unpause.

### How to unpause

`p` toggles pause. Three paths are wired so this still works when Model Vision
is not focused:

1. **Global pynput listener** — works while Vampire Survivors is focused.
2. **OpenCV `waitKey`** — works while the Model Vision window is focused.
3. **stdin** — type `p` and Enter in the terminal (also `q`, `quit`).

ESC (global or Model Vision) quits and releases keys. `q` snaps the capture
**top-left** to the current mouse position so you can point at the game content
and press `q`.

### Useful modes

```bash
./run.sh --demo          # synthetic arena, no game keys (check Model Vision)
./run.sh --dry-run       # real capture + pathfind, no key events
./run.sh --keys arrows   # last-resort bind fallback; not the default
./run.sh --backend pynput
```

## Steam / Linux focus tips

- Launch the bot **after** Vampire Survivors is on the desktop.
- The bot activates the window titled `Vampire Survivors` before sending keys
  (`xdotool windowactivate --sync`), and retries if the first activate fails.
  Override with `VS_WINDOW_NAME` if your title differs.
- Key events are **global XTest** (`xdotool keydown` / `keyup`, no `--window`).
  Proton often ignores window-targeted keys. The bot re-activates the game
  before press and release so Model Vision cannot eat a `keyup`. If keys still
  miss, click the game once or unpause again.
- **Level-up and pause cards** block walking. The bot holds still when it sees
  those layouts. Press `p` if you want the whole loop paused while you pick a
  weapon.
- Keep Model Vision parked off the playfield. The bot places it in a screen
  corner that does not overlap the capture rect. After you press `q`, it
  re-parks. Force a spot with `VS_DEBUG_X` / `VS_DEBUG_Y` if needed.
- Do not put the debug window over the capture region. That would feed the
  overlay back into vision and can cover the playfield.
- Proton games need the X11 display that actually hosts the window. If the bot
  captures a black frame, `DISPLAY` is wrong (try `:6` on Lucyfur).

## Controls

| Key | Action |
| --- | --- |
| `p` | Toggle pause. Starts paused. Unpause with `p`. |
| `q` | Align capture top-left to the mouse. |
| ESC | Quit. Releases WASD **and** arrows. Clears the move queue. |

On pause, stop, ESC, SIGINT/SIGTERM, and process exit (`atexit`):

- the move queue is cleared
- any in-flight key-hold is interrupted
- `w a s d Up Down Left Right` are all sent `keyup`

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `DISPLAY` | (your session) | X display. Often `:6` on Lucyfur. |
| `VS_MOVE_KEYS` | `wasd` | `arrows` is a fallback only. |
| `VS_INPUT_BACKEND` | `xdotool` | `pynput` is a fallback only. |
| `VS_WINDOW_NAME` | `Vampire Survivors` | xdotool search name. |
| `VS_CAPTURE_X` / `Y` / `W` / `H` | `200 80 960 540` | Capture rect. `q` updates X/Y. |
| `VS_DEBUG_X` / `Y` / `W` / `H` | auto-park, `400x225` | Model Vision geometry. |
| `VS_FPS` | `12` | Vision loop rate. |
| `VS_HOLD_MS` | `80` | Max key-hold per tick (interrupted by pause). |
| `VS_MODEL_PATH` | unset | Optional TorchScript model. |
| `VS_DRY_RUN` | unset | `1` to disable key sends. |

## Layout

```
vs_vision_bot/
  main.py            entry + loop
  config.py          env / defaults (WASD + xdotool)
  capture.py         mss grab + mouse align + park math
  vision.py          heuristic blobs + optional torch + menu detect
  pathfind.py        8-way repulsion
  input_backend.py   xdotool (default), pynput, null
  controller.py      queue, pause interrupt, atexit release
  debug_window.py    parked Model Vision overlay
  hotkeys.py         global + waitKey + stdin
run.sh
requirements.txt
```

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

## Publish note

This Origin tree is the source of truth for the rewrite. Mirror it to GitHub
under **lucyfurmoonbananas** as `lucyfur-vs-vision-bot-v2` (or similar) when you
are ready. No GitHub credentials are stored here.
