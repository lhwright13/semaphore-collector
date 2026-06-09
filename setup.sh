#!/usr/bin/env bash
# One-time setup for the semaphore data collector (macOS / Linux).
set -e

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Python 3.10 is required but was not found."
  echo "mediapipe does not yet ship wheels for 3.13/3.14, so 3.10 is needed."
  echo "Install it from https://www.python.org/downloads/release/python-31011/"
  echo "or with: brew install python@3.10   (macOS)"
  exit 1
fi

echo "Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
echo "Installing dependencies (this downloads ~150 MB the first time)..."
pip install -r requirements.txt

echo ""
echo "Done. To start collecting:"
echo "  source venv/bin/activate"
echo "  python collector.py"
