@echo off
title Career Compass
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo Open http://localhost:5000 in your browser. Close this window to stop.
venv\Scripts\python.exe app.py
pause
