#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential xdotool fonts-dejavu-core
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
FONT_DIR=$(.venv/bin/python -c "import pathlib, cv2; print(pathlib.Path(cv2.__file__).resolve().parent / 'qt' / 'fonts')")
mkdir -p "$FONT_DIR"
ln -sfn /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf "$FONT_DIR/DejaVuSans.ttf"
ln -sfn /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf "$FONT_DIR/DejaVuSans-Bold.ttf"
