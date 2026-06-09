@echo off
REM One-time setup for the semaphore data collector (Windows).

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.10 from
  echo https://www.python.org/downloads/release/python-31011/
  exit /b 1
)

echo Creating virtual environment...
py -3.10 -m venv venv
if errorlevel 1 (
  echo Could not create a 3.10 venv. Make sure Python 3.10 is installed.
  exit /b 1
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Done. To start collecting:
echo   venv\Scripts\activate.bat
echo   python collector.py
