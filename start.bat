@echo off
rem Starts the Pokerdice trainer (opens the browser). Options: start.bat --help
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv yet - run first:  powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" trainer\assistant\app.py %*
if errorlevel 1 pause
