@echo off
REM One-click launcher for the viva laptop: starts the demo and opens the browser.
REM Works offline. Close this window (or Ctrl+C) to stop.
cd /d "%~dp0.."
".venv\Scripts\python.exe" demo\app.py
pause
