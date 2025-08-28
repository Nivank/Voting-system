@echo off
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"
echo Starting AI Voting System server...

REM Optional: ensure deps (comment out to skip)
REM python -m pip install -r requirements.txt --disable-pip-version-check

REM Launch server
start "Server" cmd /c "python server.py"

REM Give it a moment to start
timeout /t 2 /nobreak >nul

REM Open in browser
start "" http://127.0.0.1:5000

echo Server launched. You can close this window.
exit /b 0


