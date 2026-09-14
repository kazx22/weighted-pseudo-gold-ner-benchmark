@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo BioRED DISEASE + CHEMICAL: CORE EXTERNAL VALIDATION
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_BIORED.ps1" -Mode Core
if errorlevel 1 (
  echo.
  echo BIORED CORE FAILED. Read the newest logs\biored_disease_chemical_*.log
  pause
  exit /b 1
)
echo.
echo BIORED CORE COMPLETED.
pause
