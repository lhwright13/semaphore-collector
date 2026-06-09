@echo off
REM Double-click this file to set up (first time) and start collecting.
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 is not installed yet.
  echo.
  echo Please install it from this page ^(check "Add Python to PATH" during install^):
  echo   https://www.python.org/downloads/release/python-31011/
  echo.
  echo Then double-click this file again.
  pause
  exit /b 1
)

if not exist venv (
  echo Setting things up for the first time. This takes a few minutes...
  py -3.10 -m venv venv
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate.bat
)

echo Starting the collector...
python collector.py
pause
