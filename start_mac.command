#!/bin/bash
# Double-click this file to set up (first time) and start collecting.
cd "$(dirname "$0")"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Python 3.10 is not installed yet."
  echo ""
  echo "Please install it from this page (get the macOS installer):"
  echo "  https://www.python.org/downloads/release/python-31011/"
  echo ""
  echo "Then double-click this file again."
  read -p "Press Return to close this window."
  exit 1
fi

if [ ! -d venv ]; then
  echo "Setting things up for the first time. This takes a few minutes..."
  python3.10 -m venv venv
  source venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

echo "Starting the collector..."
python collector.py
