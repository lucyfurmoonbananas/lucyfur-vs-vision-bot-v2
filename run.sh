#!/usr/bin/env bash
# Launch the Vampire Survivors vision bot.
# Does not override DISPLAY — export DISPLAY=:6 on Lucyfur if that is the game screen.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
exec python -m vs_vision_bot "$@"
