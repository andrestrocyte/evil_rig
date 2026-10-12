@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run powershell -ExecutionPolicy Bypass -File scripts\install_rig.ps1 first.
  pause
  exit /b 1
)
start "evil_rig" ".venv\Scripts\pythonw.exe" -m hf_dm.operator_app
