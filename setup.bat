@echo off
title Career Compass setup
cd /d "%~dp0"
echo ============================================
echo   Career Compass - one-click setup
echo ============================================
echo.

set PY=
python --version >nul 2>nul && set PY=python
if not defined PY (
    py --version >nul 2>nul && set PY=py
)
if not defined PY (
    echo Python was not found.
    echo 1. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
    echo 2. Turn off python.exe in Settings ^> Apps ^> Advanced app settings ^> App execution aliases.
    echo 3. Restart your computer, then double-click setup.bat again.
    pause
    exit /b 1
)
echo Found:
%PY% --version
echo.

if not exist venv\Scripts\python.exe (
    echo [1/4] Creating virtual environment...
    %PY% -m venv venv || goto :fail
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/4] Installing libraries (this can take 5-15 minutes)...
venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail

echo [4/4] Downloading NLP models (needs internet)...
venv\Scripts\python.exe scripts\download_models.py

echo.
echo ============================================
echo   Setup complete! Starting the website...
echo   Open http://localhost:5000 in your browser.
echo   Close this window to stop the website.
echo ============================================
venv\Scripts\python.exe app.py
pause
exit /b 0

:fail
echo.
echo Something went wrong. Copy the error messages above and share them.
pause
exit /b 1
